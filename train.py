import os
import warnings
import math
from functools import partial

import torch
from torch.utils.data import DataLoader

from tango import Step

from tqdm import tqdm
from sklearn.metrics import classification_report

from transformers import get_linear_schedule_with_warmup
from datasets import DatasetDict

from tango.integrations.torch import Model
from tango.integrations.transformers import Tokenizer

from data_utils import collate_data
from utils import fix_padding_token


@Step.register('finetune')
class Finetune(Step):

    def run(
        self, model: Model, data: DatasetDict, tokenizer: Tokenizer,
        batch_size=1, lr=0.00005, num_epochs=20, acc_steps=8, max_steps_train=200, max_steps_eval=1000,
        **kwargs
    ):
        tokenizer = fix_padding_token(tokenizer)

        collate_fn = partial(collate_data, pad_token_id=tokenizer.pad_token_id)
        train_dataloader = DataLoader(data['train'], batch_size=1, shuffle=True, collate_fn=collate_fn)
        valid_dataloader = DataLoader(data['dev'], batch_size=1, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(data['test'], batch_size=1, shuffle=False, collate_fn=collate_fn)

        train_steps = min(max_steps_train, len(train_dataloader))
        eval_steps = min(max_steps_eval, len(valid_dataloader))

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

                for step, batch in enumerate(progress_bar):
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    weights = batch.pop('weights')
                    outputs = model(**batch)

                    weight = weights.mean()
                    loss = weight * outputs.loss

                    if math.isnan(loss):
                        warnings.warn(f"Warning! loss became nan, skipping batch.")
                        continue

                    loss.backward()

                    if (step % acc_steps) == (acc_steps - 1):
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    nonlocal global_step
                    global_step += 1

                    total_loss += loss.detach().float()
                    progress_bar.set_postfix(loss=loss.item())
                    # wandb.log({"loss": loss.item()})
                    if step >= max_steps_train:
                        break

                train_epoch_loss = total_loss / train_steps
                train_ppl = torch.exp(train_epoch_loss)
                print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

            def _eval(eval_dataloader, calc_accuracy=False, max_steps=eval_steps):
                max_steps = len(eval_dataloader) if max_steps < 0 else max_steps
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
                    loss = outputs.loss
                    eval_loss += loss.detach().float()

                    if calc_accuracy:
                        with torch.no_grad():
                            # assuming single-token labels
                            pred_tokens = outputs.logits[:, :, :].argmax(dim=-1)    # B x T

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

                        eval_progbar.set_postfix(accuracy=nr_correct / nr_labels)

                    if step >= max_steps:
                        break

                eval_epoch_loss = eval_loss / eval_steps
                eval_ppl = torch.exp(eval_epoch_loss)
                print(f"{eval_ppl=} {eval_epoch_loss=}")
                if calc_accuracy:
                    print(classification_report(eval_labels, eval_preds))

                return eval_ppl

            workdir = self.workspace.work_dir(step=self)
            global_step = 0
            best_score = None
            for epoch in range(num_epochs):
                _train()
                score = _eval(valid_dataloader, calc_accuracy=True)

                if best_score is None or score > best_score:
                    best_score = score
                    model.save_pretrained(workdir)

            model.load_state_dict(torch.load(os.path.join(workdir, 'adapter_model.bin')), strict=False)
            _eval(valid_dataloader, calc_accuracy=True, max_steps=-1)
            _eval(test_dataloader, calc_accuracy=True, max_steps=-1)

        return model
