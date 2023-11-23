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
    eval_loss = 0
    eval_labels = []
    eval_preds = []
    nr_correct = 0
    nr_labels = 0
    eval_progbar = tqdm(eval_dataloader)
    for step, batch in enumerate(eval_progbar):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        weights = batch.pop('weights')
        with torch.no_grad():
            outputs = model(**batch)
        loss = (weights * outputs.loss).sum()
        eval_loss += loss.detach().float()

        with torch.no_grad():
            # assuming single-token labels
            pred_tokens = outputs.logits[:, :, :].argmax(dim=-1)  # B x T

        for b_i in range(len(batch['labels'])):
            idxr = batch['labels'][b_i] != -100
            i_label_token = batch['input_ids'][b_i][idxr][0]
            i_pred_token = pred_tokens[b_i][idxr.roll(-1)][0]
            label_text = tokenizer.decode(i_label_token)
            eval_labels.append(label_text)

            pred_text = tokenizer.decode(i_pred_token)
            eval_preds.append(pred_text)

            correct = int(label_text == pred_text)
            nr_correct += correct
            nr_labels += 1

        eval_progbar.set_postfix(accuracy=nr_correct / nr_labels, num_samples=nr_labels)

        if nr_labels >= samples_eval:
            break

    eval_epoch_loss = eval_loss / nr_labels
    print(f"{eval_epoch_loss=}")
    print(classification_report(eval_labels, eval_preds))

    return eval_epoch_loss, eval_preds, eval_labels


@Step.register('finetune')
class Finetune(Step):
    FORMAT = TorchFormat

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
                progress_bar = tqdm(train_dataloader)
                step = 0
                nr_samples = 0
                for batch_i, batch in enumerate(progress_bar):
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    weights = batch.pop('weights')
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
                score = _eval(model, valid_dataloader, tokenizer, samples_eval=min_samples_eval)

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
        test_loss, test_preds, test_labels = _eval(model, test_dataloader, tokenizer)

        test_data['predictions'] = test_preds
        test_data['labels'] = test_labels
        test_data['correct'] = test_data['predictions'] == test_data['labels']

        acc_by_primary = test_data[['annotation_Primary', 'correct']].groupby(by='annotation_Primary').mean()
        acc_by_context = test_data[['annotation_Context', 'correct']].groupby(by='annotation_Context').mean()

        print(acc_by_primary)
        print(acc_by_context)

        return test_preds, acc_by_primary, acc_by_context
