import csv
from collections import defaultdict, Counter

import datasets
from tango import Step
from tango.format import DillFormat
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer

import pandas as pd

from datasets import Dataset, DatasetDict

from data_utils import make_data_block
from utils import fix_padding_token

DEPTH = 15


def column(name):
    return [name] + [f'{name}.l{i}' for i in range(1, DEPTH)]


@Step.register('download_cad')
class DownloadCAD(Step[pd.DataFrame]):
    DETERMINISTIC = True
    FORMAT = DillFormat

    def run(self) -> pd.DataFrame:
        from io import BytesIO
        from zipfile import ZipFile
        from urllib.request import urlopen

        URL = "https://zenodo.org/record/4881008/files/data.zip?download=1"
        FILE = "data/cad_v1_1.tsv"
        resp = urlopen(URL)
        myzip = ZipFile(BytesIO(resp.read()))
        cad_df = pd.read_csv(myzip.open(FILE), delimiter="\t", quoting=csv.QUOTE_NONE, keep_default_na=False)
        return cad_df


@Step.register('flatten_cad')
class FlattenCAD(Step[pd.DataFrame]):
    DETERMINISTIC = True
    FORMAT = DillFormat

    @classmethod
    def _flatten_dataset(
        cls,
        df: pd.DataFrame,
        depth=5,
        parent_columns=['meta_text'],
        require_full_depth=False,
        groupby_agg_fns=None
    ):
        """

        Parameters:
            depth (int): how many levels to go up in the hierarchy.
            parent_columns (List[str]): which columns to select from the ancestry of the comment.
            require_full_depth (bool): if true, only keep comments with as many ancestors as `depth';
                                       if false, allow comments with fewer ancestors.
        """

        if not groupby_agg_fns:
            groupby_agg_fns = {col: 'first' for col in parent_columns}
        for col in parent_columns:
            if col not in groupby_agg_fns:
                groupby_agg_fns[col] = 'first'

        flat_df = df.rename(columns={
            'info_id': 'info_id.l0',
            'info_id.parent': 'info_id.l1',
        })

        for level in range(1, depth + 1):
            foreign_key = f'info_id.l{level}'  # key to join on
            foreign_key_p1 = f'info_id.l{level + 1}'
            # which columns to select
            selection = df[['info_id', 'info_id.parent'] + parent_columns].rename(
                columns={
                    'info_id': foreign_key,
                    'info_id.parent': foreign_key_p1,
                }
            ).groupby([foreign_key, foreign_key_p1], as_index=False).agg(groupby_agg_fns)

            assert selection[foreign_key].is_unique

            # join
            join_type = 'inner' if require_full_depth else 'left'
            flat_df = flat_df.merge(selection, on=foreign_key, suffixes=['', f'.l{level}'], how=join_type)
            flat_df = flat_df.rename(columns={
                f'info_id.parent.l{level}': f'info_id.l{level + 1}',
            })

        flat_df.drop(columns=f'info_id.l{depth + 1}', inplace=True)
        return flat_df

    def run(self, cad_df: pd.DataFrame) -> pd.DataFrame:
        IMPORTANT_COLUMNS = ['info_id', 'info_id.parent', 'info_thread.id', 'info_subreddit', 'info_image.saved',
                             'annotation_Primary',
                             'annotation_Secondary', 'annotation_Context', 'annotation_Target', 'split', 'meta_text',
                             'meta_author', 'meta_date']

        # Rows that are posts have an `info_id` that ends with 'post',
        # if the `info_id` ends with 'title' it is the title of a post,
        # all other rows are comments.
        def process_types(row):
            if row.info_id.endswith('post'):
                info_id = row.info_id[:-5]
                return 'post', info_id, f'{info_id}-title'
            elif row.info_id.endswith('title'):
                return 'title', row.info_id, "NA"
            else:
                return 'comment', row.info_id, getattr(row, 'info_id.parent')

        # Split original 'info_id' into 'type' (post/title/comment) and 'info_id' (the rest of the identifier).
        # Also change 'info_id.parent' such that comment -> ... -> comment -> post -> title
        cad_df[['type', 'info_id', 'info_id.parent']] = cad_df.apply(process_types, axis=1, result_type='expand')

        # will return all comments with hierarchy up to DEPTH levels (with NA values where there is no more hierarchy)
        flat_cad_df = self._flatten_dataset(
            cad_df[IMPORTANT_COLUMNS + ['type', 'id']],
            depth=DEPTH,
            parent_columns=['info_image.saved', 'meta_text', 'meta_author', 'meta_date', 'type', 'annotation_Primary'],
            require_full_depth=False,
            groupby_agg_fns={'annotation_Primary': list}
        )
        flat_cad_df['depth'] = flat_cad_df[column('type')].notna().sum(axis=1)
        flat_cad_df['hierarchy'] = flat_cad_df[column('type')].apply(
            lambda row: ".".join([str(typ) for typ in row if str(typ) != 'nan']), axis=1
        )

        return flat_cad_df


@Step.register('prepare_cad')
class PrepareCAD(Step[datasets.DatasetDict]):
    DETERMINISTIC = True
    FORMAT = DatasetsFormat
    VERSION = "2"

    @classmethod
    def _annotations_to_class(cls, annotations):
        if type(annotations) is list:
            if len(annotations) > 1:
                return 'Toxic'  #
            else:
                annotations = annotations[0]
        if annotations in ('Neutral', 'CounterSpeech', 'Slur'):
            return 'OK'  # 'Approved' #'Neutral'
        else:
            return 'Toxic'

    @classmethod
    def _construct_sample(cls, idx, df, include_context=True):
        row = df.iloc[idx]

        columns = [
            item for item in zip(
                row[column('type')],
                row[column('meta_text')],
                row[column('meta_author')],
                row[column('meta_date')],
                row[column('annotation_Primary')],
            ) if isinstance(item[1], str)
        ]
        parts = [
            text.replace('[linebreak]', '<br/>')
            + f'\n<pre>|---- This {_type.upper()} by {author} posted on {date}'
            + (f' was marked as {cls._annotations_to_class(annotations)} ----|</pre>'
               if i == 0 else
               f' was marked as ????? ----|</pre>')
            for i, (_type, text, author, date, annotations) in enumerate(columns)
        ]
        parts.reverse()
        split_idx = -12 - len(cls._annotations_to_class(row['annotation_Primary']))
        completion = parts[-1][split_idx:-12]
        parts[-1] = parts[-1][:split_idx - 1]

        title = f"# THREAD in r/{row['info_subreddit']}"

        if include_context:
            full_message = title + "<br/>\n\n" + "<br/>\n\n".join(parts)
        else:
            full_message = parts[-1]

        return {
            'text': full_message,
            'completion': completion,
            'classes': row['annotation_Primary']
        }

    def run(self, flat_cad_df: pd.DataFrame, tokenizer: Tokenizer,
            sample_max_len=512, block_max_len=512, include_context=True, **kwargs) -> datasets.DatasetDict:
        dataset = {}
        for split in ('train', 'dev', 'test'):
            relevant_cad_df = flat_cad_df[flat_cad_df['split'] == split]

            dataset[split] = Dataset.from_list([
                self._construct_sample(i, relevant_cad_df, include_context=include_context)
                for i in range(len(relevant_cad_df))
            ])

        hf_dataset = DatasetDict(dataset)

        # specify loss weights, amplify loss of minority class(es)
        loss_weights = defaultdict(lambda: 1)

        labels = hf_dataset['train']['completion']
        label_counts = Counter(labels)
        majority_label, _ = label_counts.most_common(1)[0]
        for label, count in label_counts.items():
            if label == majority_label:
                continue
            token_ids = tokenizer.encode(label)
            for token_id in token_ids[1:]:
                loss_weights[token_id] = len(labels) / count

        # make sure the tokenizer has a token for padding
        tokenizer = fix_padding_token(tokenizer)

        data_blocks = hf_dataset.map(
            make_data_block,
            batched=True,
            batch_size=None,
            num_proc=1,
            remove_columns=hf_dataset['test'].column_names,
            keep_in_memory=True,
            load_from_cache_file=False,
            fn_kwargs={
                "prompt_col_name": "text",
                "label_col_name": "completion",
                "tokenizer": tokenizer,
                "token_loss_weights": loss_weights,
                "sample_max_len": sample_max_len,
                "block_max_len": block_max_len,
                "add_eos_token": False,
                "truncate_prompt": True,
                "merge_prompt_label": True
            }
        )

        return data_blocks
