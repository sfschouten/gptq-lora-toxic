import os
import sys
import warnings
import math
from functools import partial

import torch
from torch.utils.data import DataLoader

from tango import Step

from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd

from transformers import get_linear_schedule_with_warmup
from datasets import DatasetDict

from tango.integrations.torch import Model, TorchFormat
from tango.integrations.transformers import Tokenizer

from data_utils import collate_data
from utils import fix_padding_token


def _eval(model, eval_dataloader, tokenizer, samples_eval=sys.maxsize):
    model.eval()
    label_map = {}
    eval_labels = []
    eval_preds = []
    eval_idxs = []
    eval_loss = 0
    nr_correct = 0
    nr_labels = 0
    eval_progbar = tqdm(eval_dataloader)
    for step, batch in enumerate(eval_progbar):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        weights = batch.pop('weights')
        eval_idxs.extend(batch.pop('idxs'))
        with torch.no_grad():
            outputs = model(**batch)
            pred_tokens = outputs.logits[:, :, :].argmax(dim=-1)    # B x T
        loss = (weights * outputs.loss).sum()
        eval_loss += loss.detach().float()

        for b_i in range(len(batch['labels'])):
            idxr = batch['labels'][b_i] != -100                     # positions in batch for which we have labels

            label_tokens = batch['input_ids'][b_i][idxr]
            label_fulltext = tokenizer.decode(label_tokens)
            label_token = label_tokens[0].item()                    # first token

            # add label token to map
            if label_token not in label_map:
                label_map[label_token] = label_fulltext
            else:
                assert label_map[label_token] == label_fulltext
            eval_labels.append(label_token)

            pred_token = pred_tokens[b_i][idxr.roll(-1)][0].item()  # first token
            eval_preds.append(pred_token)

            nr_correct += int(label_token == pred_token)
            nr_labels += 1

        eval_progbar.set_postfix(accuracy=nr_correct/nr_labels, num_samples=nr_labels)

        if nr_labels >= samples_eval:
            break

    eval_epoch_loss = eval_loss / nr_labels
    print(f"{eval_epoch_loss=}")

    labels = list(label_map.keys())
    target_names = [label_map[lbl] for lbl in labels]
    print(classification_report(eval_labels, eval_preds, labels=labels, target_names=target_names))
    report_dict = classification_report(
        eval_labels, eval_preds, labels=labels, target_names=target_names, output_dict=True)

    return report_dict, label_map, eval_idxs, eval_preds, eval_labels


@Step.register('finetune')
class Finetune(Step):
    FORMAT = TorchFormat
    CACHEABLE = True

    def run(
        self, model: Model, data: DatasetDict, tokenizer: Tokenizer,
        lr=0.00005, num_epochs=20, min_acc_samples=12, epoch_train_steps=25, min_samples_eval=1000,
        **kwargs
    ):
        tokenizer = fix_padding_token(tokenizer)

        collate_fn = partial(collate_data, pad_token_id=tokenizer.pad_token_id)
        train_dataloader = DataLoader(data['train'], batch_size=1, shuffle=True, collate_fn=collate_fn)
        valid_dataloader = DataLoader(data['dev'], batch_size=1, shuffle=True, collate_fn=collate_fn)

        train_steps = min(min_acc_samples * epoch_train_steps, len(train_dataloader))

        # prepare model for training with peft
        model.base_model.peft_config["default"].total_step = train_steps * num_epochs
        model.gradient_checkpointing_enable()
        model.print_trainable_parameters()

        # optimizer and lr scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(train_steps * num_epochs),
        )
        # training and evaluation
        with torch.cuda.amp.autocast():
            def _train():
                model.train()
                total_loss = 0
                step = 0
                nr_samples = 0
                progress_bar = tqdm(train_dataloader)
                for batch_i, batch in enumerate(progress_bar):
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    weights = batch.pop('weights')
                    batch.pop('idxs')
                    nr_samples += len(weights)

                    outputs = model(**batch)
                    loss = weights.mean() * outputs.loss

                    if math.isnan(loss):
                        warnings.warn(f"Warning! loss became nan, skipping batch.")
                        continue

                    loss.backward()

                    if nr_samples >= min_acc_samples:
                        optimizer.step()
                        lr_scheduler.step()
                        if hasattr(model.base_model, 'update_and_allocate'):
                            # need gradients for this update, so do before zero_grad
                            model.base_model.update_and_allocate(step)
                        optimizer.zero_grad()
                        step += 1
                        nr_samples = 0

                    total_loss += loss.detach().float()
                    progress_bar.set_postfix(loss=loss.item(), step=step, num_samples=nr_samples)
                    # wandb.log({"loss": loss.item()})
                    if step == epoch_train_steps:
                        break

                train_epoch_loss = total_loss / step
                print(f"{epoch=}: {train_epoch_loss=}")

            workdir = self.workspace.work_dir(step=self)
            best_score = None
            for epoch in range(num_epochs):
                _train()
                report, _, _, _, _ = _eval(model, valid_dataloader, tokenizer, samples_eval=min_samples_eval)
                score = report['Toxic']['f1-score'] if 'Toxic' in report else -1

                if best_score is None or score > best_score:
                    best_score = score
                    model.save_pretrained(workdir)

            model.load_state_dict(torch.load(os.path.join(workdir, 'adapter_model.bin')), strict=False)
            # _eval(model, valid_dataloader, tokenizer)

        return model


@Step.register('test')
class Test(Step):

    def run(self, model: Model, tokenized_data: DatasetDict, original_data: pd.DataFrame, tokenizer: Tokenizer,
            **kwargs):
        test_data = original_data[original_data['split'] == 'test'][['annotation_Primary', 'annotation_Context']]

        collate_fn = partial(collate_data, pad_token_id=tokenizer.pad_token_id)
        test_dataloader = DataLoader(tokenized_data['test'], batch_size=1, shuffle=False, collate_fn=collate_fn)
        report, label_map, test_idxs, test_preds, test_labels = _eval(model, test_dataloader, tokenizer)
        inv_label_map = {v: k for k, v in label_map.items()}

        test_data['predictions'] = [v for _, v in sorted(enumerate(test_preds), key=lambda kv: test_idxs[kv[0]])]
        test_data['labels'] = [v for _, v in sorted(enumerate(test_labels), key=lambda kv: test_idxs[kv[0]])]
        test_data['support'] = [1] * len(test_data)

        def add_f1(df, cls):
            df[f'{cls}_f1'] = 2 * df[f'{cls}_tp'] / (2 * df[f'{cls}_tp'] + df[f'{cls}_fp'] + df[f'{cls}_fn'])
            df[f'{cls}_pr'] = df[f'{cls}_tp'] / (df[f'{cls}_tp'] + df[f'{cls}_fp'])
            df[f'{cls}_re'] = df[f'{cls}_tp'] / (df[f'{cls}_tp'] + df[f'{cls}_fn'])
            return df

        results = {}
        for cls, token_id in inv_label_map.items():
            test_data[f'{cls}_true'] = test_data['labels'] == token_id
            test_data[f'{cls}_pos'] = test_data['predictions'] == token_id
            test_data[f'{cls}_tp'] = test_data[f'{cls}_true'] & test_data[f'{cls}_pos']
            test_data[f'{cls}_fp'] = ~test_data[f'{cls}_true'] & test_data[f'{cls}_pos']
            test_data[f'{cls}_fn'] = test_data[f'{cls}_true'] & ~test_data[f'{cls}_pos']

            columns = ['support'] + [f'{cls}_tp', f'{cls}_fp', f'{cls}_fn']
            by_primary = test_data[columns+['annotation_Primary']].groupby(by='annotation_Primary').sum()
            by_context = test_data[columns+['annotation_Context']].groupby(by='annotation_Context').sum()

            columns.remove('support')
            by_primary = add_f1(by_primary, cls).drop(columns=columns)
            by_context = add_f1(by_context, cls).drop(columns=columns)
            results[f'{cls}_scores_by_primary'] = by_primary
            results[f'{cls}_scores_by_context'] = by_context

            print(by_primary)
            print()
            print(by_context)
            print()

        return test_preds, results
