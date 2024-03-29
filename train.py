import os
import json

import logging
import shutil
from datetime import datetime

import numpy as np
import random

from models import t5_utils, bert_utils, bart_utils

from tqdm.auto import tqdm

from transformers import (
    get_scheduler,
    DataCollatorForSeq2Seq,
    default_data_collator,
)

from accelerate import Accelerator

# from sentence_transformers.losses import CosineSimilarityLoss

# from setfit import SetFitTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import Dataset

import wandb


class BaseTester:
    """Tester base class, inference only"""

    def __init__(self, config):  # @fix...
        """basic trainer class for all models"""
        self.name = config.name

        self.tokenizer = config.tokenizer
        self.model = config.model

        self.test_dataset = config.test_dataset

        self.test_batches = config.test_batches

        self.padding = config.padding
        self.stride = config.stride
        self.seed = config.seed

        # for logging purposes...
        self.tokenizer_checkpoint = config.tokenizer_checkpoint
        self.model_checkpoint = config.model_checkpoint

        # defined locally
        self.test_dataloader = None

        # set all seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.cuda.manual_seed(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)


class BaseTrainer:
    def __init__(self, config):  # @fix...
        """basic trainer class for all models"""
        self.name = config.name
        self.lr = config.lr
        self.lr_scheduler = config.lr_scheduler
        self.num_epochs = config.n_epochs

        self.train_batch_size = config.train_batch_size
        self.val_batch_size = config.val_batch_size

        self.tokenizer = config.tokenizer
        self.model = config.model

        # for logging purposes...
        self.tokenizer_checkpoint = config.tokenizer_checkpoint
        self.model_checkpoint = config.model_checkpoint

        self.train_dataset = config.train_dataset
        self.val_dataset = config.val_dataset
        self.test_dataset = config.test_dataset

        self.train_batches = config.train_batches
        self.val_batches = config.val_batches
        self.test_batches = config.test_batches

        self.checkpoint_savedir = config.checkpoint_savedir

        self.load_from_checkpoint = config.load_from_checkpoint
        self.checkpoint_state = config.checkpoint_state
        self.checkpoint_step = config.checkpoint_step

        self.padding = config.padding
        self.stride = config.stride
        self.seed = config.seed

        # set all seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.cuda.manual_seed(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)

        FORMAT = "[%(levelname)s] :: %(asctime)s @ %(name)s :: %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger("trainer")
        self.logger.setLevel(logging.DEBUG)  # change level if debug or not

        # defined locally
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.output_dir = os.path.abspath(
            "./outputs/{}-{}".format(self.name, self.timestamp)
        )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            raise OSError("Output directory already exists")  # delete?

        logfile = os.path.join(self.output_dir, "train_info.log")
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def save_model(self, path):
        """basic model saving where the path is overwrote if"""
        if os.path.isfile(path):
            shutil.rmtree(path)
        self.model.save_pretrained(path)


class FinetuneT5(BaseTrainer):
    """

    Everything that is dynamic is defined here: dataloader, training and evaluation loop, model export, etc.

    """

    def __init__(self, config):
        super().__init__(config)
        self.max_ans_length = config.max_ans_length
        self.max_seq_length = config.max_seq_length
        s = """Training run initialized

T5-like fine tuning configuration
************************************
        Name : {}
        Model checkpoint : {}
        Tokenizer checkpoint : {}
        Max sequence length : {}
        Max answer length : {}
        Padding : {}
        Stride : {}
        Hyperparameters :
                lr={}
                lr_scheduler={}
                num_epochs={}
                batch_size={}
        Output directory : {}
************************************
        """.format(
            self.name,
            self.model_checkpoint,
            self.tokenizer_checkpoint,
            self.max_seq_length,
            self.max_ans_length,
            self.padding,
            self.stride,
            self.lr,
            self.lr_scheduler,
            self.num_epochs,
            self.train_batch_size,
            self.output_dir,
        )

        self.logger.info(s)

    def get_dataloaders(self):
        train_tensor = self.train_batches.remove_columns(["example_id"])
        val_tensor = self.val_batches.remove_columns(["example_id"])
        test_tensor = self.val_batches.remove_columns(["example_id"])

        train_tensor.set_format("torch")
        val_tensor.set_format("torch")
        test_tensor.set_format("torch")

        # create the dataloaders
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        self.train_dataloader = DataLoader(
            train_tensor,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=self.train_batch_size,
            num_workers=0,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        self.val_dataloader = DataLoader(
            val_tensor,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=self.val_batch_size,
        )

        self.logger.info("Training, validation and test dataloaders created")
        # shuffle only train...

    @torch.no_grad()
    def _eval(self, accelerator):
        """ """
        self.model.eval()
        answer_batch = []
        for i, batch in enumerate(tqdm(self.val_dataloader)):
            outputs = self.model.generate(
                **batch,
                max_length=self.max_ans_length,
                num_beams=1,
            )
            for i in outputs:
                answer_batch.append(i)

        predicted_answers = [
            t5_utils.clean_outputs(i, self.tokenizer) for i in answer_batch
        ]

        eval_outputs = list(zip(self.val_batches["example_id"], predicted_answers))

        score, predictions, targets = t5_utils.evaluate(eval_outputs, self.val_dataset)

        return score, predictions

    def __call__(self):
        """simply call the finetuning"""

        # We start the fine-tuning code here, essentially we feed it a model and some data and it trains it and
        # logs the loss/results/weigths that is all....
        self.get_dataloaders()
        checkpoint_path_bestf1 = os.path.join(self.output_dir, "checkpoint-bestf1")
        checkpoint_path_end = os.path.join(self.output_dir, "checkpoint-end")

        wandb.init(
            project="o-nlp_experiments",
            config={
                "learning_rate": self.lr,
                "architecture": self.name,
                "dataset": "oqa",
                "epochs": self.num_epochs,
            },
        )

        # training loop **************************************************

        best_f1 = -1
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = self.num_epochs * num_update_steps_per_epoch
        if self.lr_scheduler:
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0.1 * num_training_steps,
                num_training_steps=num_training_steps,
            )

        if torch.device != "cpu":
            # @BUG mixed precision breaks t5
            # mixed_precision="bf16" ? issues witht T5 models...
            # accelerator = Accelerator(mixed_precision="fp16")
            accelerator = Accelerator()
            (
                self.model,
                optimizer,
                self.train_dataloader,
                self.val_dataloader,
            ) = accelerator.prepare(
                self.model, optimizer, self.train_dataloader, self.val_dataloader
            )

        progressbar = tqdm(range(num_training_steps))

        # val_targets = self.__get_val_answers()

        losses = {"train": []}
        for epoch in range(self.num_epochs):
            self.model.train()
            for steps, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss

                if torch.device != "cpu":
                    accelerator.backward(loss)
                else:
                    loss.backward()

                losses["train"].append(loss.detach().float().cpu().numpy())

                optimizer.step()
                if self.lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
                progressbar.update(1)

            score, predictions = self._eval(accelerator)
            f1_score = score["f1"]

            self.logger.info(
                "Epoch: {}, Loss: {}, Validation F1: {}".format(
                    epoch, float(loss.cpu()), f1_score
                )
            )
            wandb.log(
                {"val_f1": f1_score, "train_loss": np.array(losses["train"]).mean()}
            )

            # save the best model
            if f1_score > best_f1:
                best_f1 = f1_score
                self.save_model(checkpoint_path_bestf1)
                self.logger.info("New save with f1 = {}".format(best_f1))

        self.save_model(checkpoint_path_end)
        self.logger.info(
            "Best {} f1 = {} localted at {}".format(
                self.name, best_f1, checkpoint_path_bestf1
            )
        )

        return {
            "best_val_f1": best_f1,
            "checkpoint_bestf1": checkpoint_path_bestf1,
            "checkpoint_end": checkpoint_path_end,
            "n_step": steps,
        }


class TaskDistillationBERT(BaseTrainer):
    """@TODO ::  implement knowledge distillation from QA experts (teacher) to domain experts (student)"""

    def __init__(self, config):
        """ """
        super().__init__(config)
        self.temperature = config.temperature  # from the KD paper, higher for
        self.alpha = config.alpha
        self.teacher_batches = config.teacher_batches
        self.KD_loss = nn.KLDivLoss(reduction="batchmean")
        self.teacher_model = config.teacher_model
        self.teacher_tokenizer = config.teacher_tokenizer

        self.teacher_slogits = []
        self.teacher_elogits = []
        self.teacher_input_ids = []

    @torch.no_grad()
    def __eval(self, accelerator):
        """ """
        start_logits = []
        end_logits = []

        self.model.eval()
        val_losses = []

        for batch in tqdm(self.val_dataloader):
            outputs = self.model(**batch)
            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
            val_losses.append(outputs.loss.detach().cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(self.val_batches)]
        end_logits = end_logits[: len(self.val_batches)]
        metrics, unused, unused = bert_utils.answer_from_logits(
            start_logits,
            end_logits,
            self.val_batches,
            self.val_dataset,
            self.tokenizer,
        )
        f1_score = metrics["f1"]
        val_loss = np.array(val_losses).mean()
        print(val_loss)
        accelerator.wait_for_everyone()
        return f1_score, val_loss

    def __kd_loss(
        self,
        student_loss,
        student_start_logits,
        student_end_logits,
        teacher_start_logits,
        teacher_end_logits,
    ):
        assert student_start_logits.size() == teacher_start_logits.size()
        assert student_end_logits.size() == teacher_end_logits.size()

        # NOTE :: idk why the log_softmax for the student?...
        start_loss = self.KD_loss(
            input=F.log_softmax(student_start_logits / self.temperature, dim=-1),
            target=F.softmax(teacher_start_logits / self.temperature, dim=-1),
        ) * (self.temperature**2)

        end_loss = self.KD_loss(
            input=F.log_softmax(student_end_logits / self.temperature, dim=-1),
            target=F.softmax(teacher_end_logits / self.temperature, dim=-1),
        ) * (self.temperature**2)

        loss_ce = (start_loss + end_loss) / 2.0
        print(loss_ce)

        total_loss = student_loss * self.alpha + (1 - self.alpha) * loss_ce

        return total_loss

    def get_dataloaders(self):
        """"""
        train_tensor = self.train_batches.remove_columns(
            ["example_id", "offset_mapping"]
        )
        train_tensor.set_format("torch")
        self.train_dataloader = DataLoader(
            train_tensor,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.train_batch_size,
            num_workers=0,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )

        # teacher_tensor = self.teacher_batches.remove_columns(
        #     ["example_id", "offset_mapping"]
        # )
        # teacher_tensor.set_format("torch")
        # self.teacher_dataloader = DataLoader(
        #     train_tensor,
        #     shuffle=True,
        #     collate_fn=default_data_collator,
        #     batch_size=self.train_batch_size,
        #     num_workers=0,
        #     worker_init_fn=self.seed_worker,
        #     generator=self.g,
        # )

        val_tensor = self.val_batches.remove_columns(["example_id", "offset_mapping"])
        val_tensor.set_format("torch")
        self.val_dataloader = DataLoader(
            val_tensor,
            collate_fn=default_data_collator,
            batch_size=self.val_batch_size,
            shuffle=False,
        )

    @torch.no_grad()
    def get_teacher_logits(self):
        # compute the logits from the teacher and save to an array for training
        accelerator = Accelerator(mixed_precision="fp16")
        (
            self.teacher_model,
            self.train_dataloader,
        ) = accelerator.prepare(self.teacher_model, self.train_dataloader)

        self.teacher_model.eval()
        for i, batch in enumerate(tqdm(self.train_dataloader)):
            outputs = self.teacher_model(**batch)
            self.teacher_slogits.append(accelerator.gather(outputs.start_logits))
            self.teacher_elogits.append(accelerator.gather(outputs.end_logits))
            self.teacher_input_ids.append(batch["input_ids"][0])

        del self.teacher_model
        del accelerator
        # @TODO :: free up the memory

    def __call__(self):
        self.get_dataloaders()
        self.get_teacher_logits()

        # dataloaders

        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        save_path = os.path.abspath(
            "{}{}-{}".format(self.checkpoint_savedir, self.name, timestamp)
        )

        # experiment tracking
        wandb.init(
            project="o-nlp_experiments",
            config={
                "learning_rate": self.lr,
                "architecture": self.name,
                "dataset": "oqa",
                "epochs": self.num_epochs,
            },
        )

        best_f1 = -1
        lowest_val_loss = 100
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = self.num_epochs * num_update_steps_per_epoch

        # accelerator
        if self.lr_scheduler:
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0.1 * num_training_steps,
                num_training_steps=num_training_steps,
            )
        if torch.device != "cpu":
            # @BUG mixed precision breaks generation
            accelerator = Accelerator(mixed_precision="fp16")
            (
                self.model,
                optimizer,
                self.train_dataloader,
                self.val_dataloader,
            ) = accelerator.prepare(
                self.model, optimizer, self.train_dataloader, self.val_dataloader
            )
            if self.lr_scheduler:
                accelerator.register_for_checkpointing(lr_scheduler)

        losses = {"train": []}

        # training loop
        progressbar = tqdm(range(num_training_steps))
        for epoch in range(self.num_epochs):
            self.model.train()
            for steps, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)

                loss = self.__kd_loss(
                    outputs.loss,
                    outputs.start_logits,
                    outputs.end_logits,
                    self.teacher_slogits[steps],
                    self.teacher_elogits[steps],
                )
                assert torch.equal(batch["input_ids"][0], self.teacher_input_ids[steps])

                accelerator.backward(loss)
                losses["train"].append(loss.detach().cpu().numpy())

                optimizer.step()
                if self.lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
                progressbar.update(1)
                # eval
                # if steps % 50 == 0:
            f1_score, val_loss = self.__eval(accelerator)
            self.logger.info("steps {} : f1 {}".format(steps, f1_score))
            wandb.log(
                {
                    "val_f1": f1_score,
                    "val_loss": val_loss,
                    "train_loss": np.array(losses["train"]).mean(),
                    "n_step": steps,
                }
            )

            # checkpointing (only best_val)
            if val_loss < lowest_val_loss:
                self.save_model(save_path)
                lowest_val_loss = val_loss
                best_f1 = f1_score
                self.logger.info(
                    "New save with f1 = {} at lowest val loss".format(best_f1)
                )

        self.save_model(
            "{}FINAL-{}-{}".format(self.checkpoint_savedir, self.name, timestamp)
        )
        self.logger.info(
            "Best {} f1 = {}, saved at {}".format(self.name, best_f1, save_path)
        )


class PretrainT5(BaseTrainer):
    """ """

    def __init__(self, config):
        super().__init__(config)
        self.max_ans_length = config.max_ans_length
        self.max_seq_length = config.max_seq_length

    def __get_val_answers(self):
        """ """
        val_targets = []
        for sample in self.val_batches["labels"]:
            val_targets.append(
                self.tokenizer.decode(
                    [tok for tok in sample if tok != -100], skip_special_tokens=True
                )
            )

        return Dataset.from_dict(
            {"id": self.val_batches["example_id"], "target_strings": val_targets}
        )

    def __get_dataloaders(self):
        train_tensor = self.train_batches.remove_columns(["example_id"])
        val_tensor = self.val_batches.remove_columns(["example_id"])

        train_tensor.set_format("torch")
        val_tensor.set_format("torch")

        # create the dataloaders
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        self.train_dataloader = DataLoader(
            train_tensor,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=4,
            num_workers=0,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        self.val_dataloader = DataLoader(
            val_tensor,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=4,
        )

        self.logger.info("Training, validation dataloaders created")
        # shuffle only train...

    def __save_checkpoint(self, accelerator, n_masked_tokens, n_step):
        ckpt_path = "./ckpts/"
        ckpt_max = 5

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        dirs = [
            os.path.relpath(ckpt_path + f.name)
            for f in os.scandir(ckpt_path)
            if f.is_dir()
        ]
        dirs.sort(key=os.path.getctime)

        if len(dirs) >= ckpt_max:
            shutil.rmtree(dirs[0])

        accelerator.save_state(
            "./ckpts/{}-{}mt-{}s".format(self.name, n_masked_tokens, n_step)
        )

    @torch.no_grad()
    def __eval(self, losses):
        self.model.eval()

        for i, batch in enumerate(tqdm(self.val_dataloader)):
            outputs = self.model(**batch)
            losses["val"].append(outputs.loss.item())
            # accumulate val_losses losses["val"]
        self.model.train()
        return losses

    def __call__(self):
        """ """
        self.__get_dataloaders()
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        local_path = os.path.abspath(
            "{}{}-{}".format(self.checkpoint_savedir, self.name, timestamp)
        )

        best_val_loss = 100

        wandb.init(
            project="o-nlp_experiments",
            config={
                "learning_rate": self.lr,
                "architecture": "t5-pretraining-test",
                "dataset": "oqa-alpha",
                "epochs": self.num_epochs,
            },
        )

        best_f1 = -1
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = self.num_epochs * num_update_steps_per_epoch

        if self.lr_scheduler:
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0.1 * num_training_steps,
                num_training_steps=num_training_steps,
            )

        if torch.device != "cpu":
            accelerator = Accelerator()
            (
                self.model,
                optimizer,
                self.train_dataloader,
                self.val_dataloader,
            ) = accelerator.prepare(
                self.model, optimizer, self.train_dataloader, self.val_dataloader
            )
            if self.lr_scheduler:
                accelerator.register_for_checkpointing(lr_scheduler)

        progressbar = tqdm(range(num_training_steps))

        val_targets = self.__get_val_answers()
        n_masked_tokens = 0
        save_threshold = 0
        losses = {"train": [], "val": []}
        n_step = 0

        if self.load_from_checkpoint:
            accelerator.load_state(self.checkpoint_state)
            # get the number of masked tokens to the reloaded step...
            for steps, batch in enumerate(self.train_dataloader):
                if n_step <= self.checkpoint_step:
                    flabels = batch["labels"].flatten().cpu()
                    n_masked_tokens += len(flabels[flabels >= 0]) - 2
                    save_threshold = int(n_masked_tokens / 20000)
                    n_step += 1
                else:
                    break
            n_step = self.checkpoint_step
            accelerator.skip_first_batches(self.train_dataloader, self.checkpoint_step)
            self.logger.info(
                "Checkpoint: {} reloaded! Step: {}, Number of masked tokens: {}".format(
                    self.checkpoint_state, n_step, n_masked_tokens
                )
            )

        self.model.train()
        for epoch in range(self.num_epochs):
            for steps, batch in enumerate(self.train_dataloader):
                flabels = batch["labels"].flatten().cpu()
                n_masked_tokens += len(flabels[flabels >= 0]) - 2
                # logger.info(
                #     "epoch {} : n_masked_tokens {} ".format(epoch, n_masked_tokens)
                # )

                outputs = self.model(**batch)

                loss = outputs.loss
                losses["train"].append(loss.item())

                if torch.device != "cpu":
                    accelerator.backward(loss)
                else:
                    loss.backward()

                optimizer.step()
                if self.lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
                progressbar.update(1)

                if (
                    int(n_masked_tokens / 20000) > save_threshold
                ):  # save initial checkpoint then each 1k masked tokens
                    # (ckpt_num * 1000)
                    save_threshold = int(n_masked_tokens / 20000)
                    self.__save_checkpoint(accelerator, n_masked_tokens, n_step)
                    self.logger.info(
                        "chekpoint saved at step {} after {} masked tokens".format(
                            n_step, n_masked_tokens
                        )
                    )
                    losses = self.__eval(losses)
                    # @TODO :: save best model according to lowest validation loss!
                    if np.array(losses["val"]).mean() < best_val_loss:
                        best_val_path = os.path.abspath(
                            "{}/bestval-{}-{}".format(
                                self.checkpoint_savedir, self.name, timestamp
                            )
                        )
                        self.save_model(best_val_path)
                        best_val_loss = np.array(losses["val"]).mean()

                    # get mean training and val losses, reset the losses arrays, log n_masked_tokens, step and more
                    wandb.log(
                        {
                            "val_loss": np.array(losses["val"]).mean(),
                            "train_loss": np.array(losses["train"]).mean(),
                            "n_step": n_step,
                            "num_masked_tokens": n_masked_tokens,
                        }
                    )
                    losses = {"train": [], "val": []}
                n_step += 1

        self.model.save_pretrained(local_path)


class PretrainBERT(BaseTrainer):
    """ """

    def __init__(self, config):
        super().__init__(config)
        self.mask_questions = True
        self.eval_interval = 100

    @torch.no_grad()
    def __eval(self, losses):
        self.model.eval()
        for i, batch in enumerate(tqdm(self.val_dataloader)):
            outputs = self.model(**batch)
            losses["val"].append(outputs.loss.item())
        self.model.train()
        return losses

    def __get_dataloader(self):
        train_tensor = self.train_batches
        train_tensor.set_format("torch")

        tacoma_data_collator = TacomaCollator(
            self.tokenizer,
            mask_questions=self.mask_questions,
            _lambda=17,  # mean answer length (tokens) from OQA
        )

        self.train_dataloader = DataLoader(
            train_tensor,
            shuffle=True,
            collate_fn=tacoma_data_collator,
            batch_size=16,
            num_workers=0,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )

        val_tensor = self.val_batches
        val_tensor.set_format("torch")
        self.val_dataloader = DataLoader(
            val_tensor,
            collate_fn=tacoma_data_collator,
            batch_size=16,
            shuffle=False,
        )

        self.logger.info("Training and validation dataloaders created")

    def __save_checkpoint(self, accelerator, n_step):
        ckpt_max = 5

        if not os.path.exists(self.checkpoint_savedir):
            os.makedirs(self.checkpoint_savedir)

        dirs = [
            os.path.relpath(self.checkpoint_savedir + "/" + f.name)
            for f in os.scandir(self.checkpoint_savedir)
            if f.is_dir()
        ]
        dirs.sort(key=os.path.getctime)

        if len(dirs) >= ckpt_max:
            shutil.rmtree(dirs[0])

        accelerator.save_state(
            "{}/{}-{}steps".format(self.checkpoint_savedir, self.name, n_step)
        )

    def __call__(self):
        self.__get_dataloader()
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        save_path = os.path.abspath(
            "{}/final-{}-{}".format(self.checkpoint_savedir, self.name, timestamp)
        )
        tokenizer_path = os.path.abspath(
            "{}/tokenizer-{}-{}".format(self.checkpoint_savedir, self.name, timestamp)
        )
        best_val_loss = 100

        # experiment tracking
        wandb.init(
            project="o-nlp_experiments",
            config={
                "learning_rate": self.lr,
                "architecture": self.name,
                "dataset": "tacoma-angle_oqa",
                "epochs": self.num_epochs,
            },
        )

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = self.num_epochs * num_update_steps_per_epoch

        # accelerator
        if self.lr_scheduler:
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0.1 * num_training_steps,
                num_training_steps=num_training_steps,
            )

        if torch.device != "cpu":
            # @BUG mixed precision breaks generation
            accelerator = Accelerator(mixed_precision="fp16")
            (
                self.model,
                optimizer,
                self.train_dataloader,
                self.val_dataloader,
            ) = accelerator.prepare(
                self.model, optimizer, self.train_dataloader, self.val_dataloader
            )
            if self.lr_scheduler:
                accelerator.register_for_checkpointing(lr_scheduler)

        losses = {"train": [], "val": []}

        # training loop
        progressbar = tqdm(range(num_training_steps))
        for epoch in range(self.num_epochs):
            self.model.train()
            for steps, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                losses["train"].append(loss.item())

                optimizer.step()
                if self.lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
                progressbar.update(1)

                if (steps % self.eval_interval) == 0:
                    losses = self.__eval(losses)
                    if np.array(losses["val"]).mean() < best_val_loss:
                        best_val_path = os.path.abspath(
                            "{}/bestval-{}-{}".format(
                                self.checkpoint_savedir, self.name, timestamp
                            )
                        )
                        self.save_model(best_val_path)
                        best_val_loss = np.array(losses["val"]).mean()

                    wandb.log(
                        {
                            "val_loss": np.array(losses["val"]).mean(),
                            "train_loss": np.array(losses["train"]).mean(),
                            "n_step": steps,
                        }
                    )
                    losses = {"train": [], "val": []}

                    self.__save_checkpoint(accelerator, n_step=steps)

            losses = self.__eval(losses)
            if losses["train"] != []:
                train_loss = None
            else:
                train_loss = np.array(losses["train"]).mean()

            wandb.log(
                {
                    "val_loss": np.array(losses["val"]).mean(),
                    "train_loss": train_loss,
                    "n_step": steps,
                }
            )
            self.save_model(save_path)
            self.tokenizer.save_pretrained(tokenizer_path)
            self.logger.info(
                "Pretraining finished with best validation loss at {}!".format(
                    best_val_loss
                )
            )


class FinetuneBERT(BaseTrainer):
    """ """

    def __init__(self, config):
        super().__init__(config)
        self.max_length = config.max_length
        self.bitfit = config.bitfit
        s = """Training run initialized

BERT-like fine tuning configuration
************************************
        Name : {}
        Model checkpoint : {}
        Tokenizer checkpoint : {}
        Max sequence length : {}
        Padding : {}
        Stride : {}
        Hyperparameters :
                bitfit={}
                lr={}
                lr_scheduler={}
                num_epochs={}
                batch_size={}
        Output directory : {}
************************************
        """.format(
            self.name,
            self.model_checkpoint,
            self.tokenizer_checkpoint,
            self.max_length,
            self.padding,
            self.stride,
            self.bitfit,
            self.lr,
            self.lr_scheduler,
            self.num_epochs,
            self.train_batch_size,
            self.output_dir,
        )

        self.logger.info(s)

    @torch.no_grad()
    def __eval(self, accelerator):
        """ """
        start_logits = []
        end_logits = []

        self.model.eval()
        val_losses = []

        for batch in tqdm(self.val_dataloader):
            outputs = self.model(**batch)
            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
            val_losses.append(outputs.loss.detach().cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(self.val_batches)]
        end_logits = end_logits[: len(self.val_batches)]
        metrics, _, _, _ = bert_utils.answer_from_logits(
            start_logits,
            end_logits,
            self.val_batches,
            self.val_dataset,
            self.tokenizer,
        )
        f1_score = metrics["f1"]
        val_loss = np.array(val_losses).mean()

        accelerator.wait_for_everyone()
        return f1_score, val_loss

    def __get_dataloaders(self):
        """"""
        train_tensor = self.train_batches.remove_columns(
            ["example_id", "offset_mapping"]
        )
        train_tensor.set_format("torch")
        self.train_dataloader = DataLoader(
            train_tensor,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.train_batch_size,
            num_workers=0,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )

        val_tensor = self.val_batches.remove_columns(["example_id", "offset_mapping"])
        val_tensor.set_format("torch")
        self.val_dataloader = DataLoader(
            val_tensor,
            collate_fn=default_data_collator,
            batch_size=self.val_batch_size,
            shuffle=False,
        )

        self.logger.info("Training and validation dataloaders created")

    def __call__(self):
        # dataloaders
        self.__get_dataloaders()
        checkpoint_path_bestf1 = os.path.join(self.output_dir, "checkpoint-bestf1")
        checkpoint_path_end = os.path.join(self.output_dir, "checkpoint-end")
        # experiment tracking
        wandb.init(
            project="o-nlp_experiments",
            config={
                "learning_rate": self.lr,
                "architecture": self.name,
                "dataset": "oqa",
                "epochs": self.num_epochs,
            },
        )

        best_f1 = -1
        lowest_val_loss = 100
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = self.num_epochs * num_update_steps_per_epoch

        # accelerator
        if self.lr_scheduler:
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0.1 * num_training_steps,
                num_training_steps=num_training_steps,
            )

        if torch.device != "cpu":
            # @BUG mixed precision breaks generation
            accelerator = Accelerator(mixed_precision="fp16")
            (
                self.model,
                optimizer,
                self.train_dataloader,
                self.val_dataloader,
            ) = accelerator.prepare(
                self.model, optimizer, self.train_dataloader, self.val_dataloader
            )
            if self.lr_scheduler:
                accelerator.register_for_checkpointing(lr_scheduler)

        losses = {"train": []}

        # training loop
        progressbar = tqdm(range(num_training_steps))
        for epoch in range(self.num_epochs):
            self.model.train()
            for steps, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                losses["train"].append(loss.detach().cpu().numpy())

                optimizer.step()
                if self.lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()

                progressbar.update(1)
            # eval
            f1_score, val_loss = self.__eval(accelerator)
            self.logger.info(
                "steps {} : f1 {}, val_loss {}".format(steps, f1_score, val_loss)
            )
            wandb.log(
                {
                    "val_f1": f1_score,
                    "val_loss": val_loss,
                    "train_loss": np.array(losses["train"]).mean(),
                    "n_step": steps,
                }
            )

            if f1_score > best_f1:
                best_f1 = f1_score
                self.save_model(checkpoint_path_bestf1)
                self.logger.info("New save with f1 = {}".format(best_f1))

        self.save_model(checkpoint_path_end)

        self.logger.info(
            "Best {} f1 = {} localted at {}".format(
                self.name, best_f1, checkpoint_path_bestf1
            )
        )

        return {
            "best_val_f1": best_f1,
            "checkpoint_bestf1": checkpoint_path_bestf1,
            "checkpoint_end": checkpoint_path_end,
            "n_step": steps,
        }


class FinetuneBART(BaseTrainer):
    """>>> tuner = FinetuneBART(config)
    desired output"""

    def __init__(self, config):
        super().__init__(config)
        self.max_ans_length = config.max_ans_length
        self.max_seq_length = config.max_seq_length
        s = """Training run initialized

BART-like fine tuning configuration
************************************
        Name : {}
        Model checkpoint : {}
        Tokenizer checkpoint : {}
        Max sequence length : {}
        Max answer length : {}
        Padding : {}
        Stride : {}
        Hyperparameters :
                lr={}
                lr_scheduler={}
                num_epochs={}
                batch_size={}
        Output directory : {}
************************************
        """.format(
            self.name,
            self.model_checkpoint,
            self.tokenizer_checkpoint,
            self.max_seq_length,
            self.max_ans_length,
            self.padding,
            self.stride,
            self.lr,
            self.lr_scheduler,
            self.num_epochs,
            self.train_batch_size,
            self.output_dir,
        )

        self.logger.info(s)

    def __get_dataloaders(self):
        train_tensor = self.train_batches.remove_columns(["example_id"])
        train_tensor.set_format("torch")

        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        self.train_dataloader = DataLoader(
            train_tensor,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=self.train_batch_size,
            num_workers=0,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )

        val_tensor = self.val_batches.remove_columns(["example_id"])
        val_tensor.set_format("torch")
        self.val_dataloader = DataLoader(
            val_tensor,
            collate_fn=data_collator,
            batch_size=self.val_batch_size,
            shuffle=False,
        )

        self.logger.info("Training and validation dataloaders created")

    @torch.no_grad()
    def __eval(self, accelerator):
        """ """
        self.model.eval()
        answer_batches = []
        for batch in tqdm(self.val_dataloader):
            outputs = self.model.generate(
                **batch, max_length=self.max_ans_length, num_beams=1
            )
            for i in outputs:
                answer_batches.append(i)

        predicted_answers = [
            bart_utils.clean_outputs(i, tokenizer=self.tokenizer)
            for i in answer_batches
        ]
        eval_outputs = list(zip(self.val_batches["example_id"], predicted_answers))

        score, predictions, targets = bart_utils.evaluate(
            eval_outputs, self.val_dataset
        )
        f1_score = score["f1"]

        return f1_score

    def __call__(self):
        self.__get_dataloaders()

        checkpoint_path_bestf1 = os.path.join(self.output_dir, "checkpoint-bestf1")
        checkpoint_path_end = os.path.join(self.output_dir, "checkpoint-end")

        best_f1 = -1
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = self.num_epochs * num_update_steps_per_epoch

        # experiment tracking
        wandb.init(
            project="o-nlp_experiments",
            config={
                "learning_rate": self.lr,
                "architecture": self.name,
                "dataset": "oqa",
                "epochs": self.num_epochs,
            },
        )

        # accelerator
        if self.lr_scheduler:
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0.1 * num_training_steps,
                num_training_steps=num_training_steps,
            )

        if torch.device != "cpu":
            accelerator = Accelerator(mixed_precision="fp16")
            (
                self.model,
                optimizer,
                self.train_dataloader,
                self.val_dataloader,
            ) = accelerator.prepare(
                self.model, optimizer, self.train_dataloader, self.val_dataloader
            )
            if self.lr_scheduler:
                accelerator.register_for_checkpointing(lr_scheduler)

        losses = {"train": []}

        # training loop
        progressbar = tqdm(range(num_training_steps))
        for epoch in range(self.num_epochs):
            self.model.train()
            for steps, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                losses["train"].append(loss.detach().cpu().numpy())

                optimizer.step()
                if self.lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
                progressbar.update(1)

            # eval
            f1_score = self.__eval(accelerator)
            self.logger.info("epoch {} : f1 {}".format(epoch, f1_score))
            wandb.log(
                {"val_f1": f1_score, "train_loss": np.array(losses["train"]).mean()}
            )

            # save the best model
            if f1_score > best_f1:
                best_f1 = f1_score
                self.save_model(checkpoint_path_bestf1)
                self.logger.info("New save with f1 = {}".format(best_f1))

        self.save_model(checkpoint_path_end)
        self.logger.info(
            "Best {} f1 = {} localted at {}".format(
                self.name, best_f1, checkpoint_path_bestf1
            )
        )

        return {
            "best_val_f1": best_f1,
            "checkpoint_bestf1": checkpoint_path_bestf1,
            "checkpoint_end": checkpoint_path_end,
            "n_step": steps,
        }


class EvaluateBERT(BaseTester):
    """Evaluate model and print results to stdout and export as a log/txt file artifact"""

    def __init__(self, config, output_dir=None):
        super().__init__(config)
        self.max_length = config.max_length
        self.bitfit = config.bitfit
        if output_dir == None:
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            self.logfile = os.path.abspath(
                "evaluation-{}-{}.log".format(self.name, timestamp)
            )
            self.resultfile = os.path.join("./", "results.json")
        else:
            self.logfile = os.path.join(output_dir, "evaluation.log")
            self.resultfile = os.path.join(output_dir, "results.json")
        # create logger with 'spam_application'
        self.logger = logging.getLogger("eval_logger")
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(self.logfile)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        s = """Evaluation run initialized

BERT-like evaluation configuration
************************************
        Name : {}
        Model checkpoint : {}
        Tokenizer checkpoint : {}
        Max sequence length : {}
        Padding : {}
        Stride : {}
        Output directory : {}
************************************
        """.format(
            self.name,
            self.model_checkpoint,
            self.tokenizer_checkpoint,
            self.max_length,
            self.padding,
            self.stride,
            output_dir,
        )

        self.logger.info(s)

    def __get_dataloader(self):
        test_tensor = self.test_batches.remove_columns(["example_id", "offset_mapping"])
        test_tensor.set_format("torch")
        self.test_dataloader = DataLoader(
            test_tensor,
            collate_fn=default_data_collator,
            batch_size=4,
            shuffle=False,
        )

        self.logger.info("Test dataloader created")

    @torch.no_grad()
    def __call__(self, multiple_answers=False):
        # dataloader for test
        self.__get_dataloader()

        # run inference on batch
        accelerator = Accelerator(mixed_precision="fp16")
        (
            self.model,
            self.test_dataloader,
        ) = accelerator.prepare(self.model, self.test_dataloader)

        # eval
        start_logits = []
        end_logits = []

        self.model.eval()

        for batch in tqdm(self.test_dataloader):
            outputs = self.model(**batch)
            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(self.test_batches)]
        end_logits = end_logits[: len(self.test_batches)]
        (
            metrics,
            predicted_answers,
            theoretical_answers,
            sampled_answers,
        ) = bert_utils.answer_from_logits(
            start_logits,
            end_logits,
            self.test_batches,
            self.test_dataset,
            self.tokenizer,
            multiple_answers=multiple_answers,
        )
        f1_score = metrics["f1"]
        em = metrics["exact_match"]

        self.logger.info(
            "\n{} \nEvaluation results \nmodel : {} \n > F1 = {} \n > EM = {} \n{}".format(
                "*" * 50, self.name, f1_score, em, "*" * 50
            )
        )

        results = {
            "em": em,
            "f1": f1_score,
            "predicted_answers": predicted_answers,
            "sampled_answers": sampled_answers,
        }
        print(results)

        # SAVE results as json
        with open(self.resultfile, "w") as fp:
            json.dump(results, fp)

        self.logger.info("Evaluation results saved at {}".format(self.resultfile))

        return results


class EvaluateBART(BaseTester):
    """ """

    def __init__(self, config, output_dir=None):
        super().__init__(config)
        self.max_ans_length = config.max_ans_length
        self.max_seq_length = config.max_seq_length

        if output_dir == None:
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            self.logfile = os.path.abspath(
                "evaluation-{}-{}.log".format(self.name, timestamp)
            )
            self.resultfile = os.path.join("./", "results.json")
        else:
            self.logfile = os.path.join(output_dir, "evaluation.log")
            self.resultfile = os.path.join(output_dir, "results.json")

        self.logger = logging.getLogger("eval_logger")
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(self.logfile)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        s = """Evaluation run initialized

BART-like evaluation configuration
************************************
        Name : {}
        Model checkpoint : {}
        Tokenizer checkpoint : {}
        Max sequence length : {}
        Max answer length : {}
        Padding : {}
        Stride : {}
        Output directory : {}
************************************
        """.format(
            self.name,
            self.model_checkpoint,
            self.tokenizer_checkpoint,
            self.max_seq_length,
            self.max_ans_length,
            self.padding,
            self.stride,
            output_dir,
        )

        self.logger.info(s)

    def get_dataloaders(self):
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        test_tensor = self.test_batches.remove_columns(["example_id"])
        test_tensor.set_format("torch")
        self.test_dataloader = DataLoader(
            test_tensor,
            collate_fn=data_collator,
            batch_size=4,
            shuffle=False,
        )

        self.logger.info("Test and validation dataloaders created")

    @torch.no_grad()
    def __call__(self, return_answers=False):
        self.get_dataloaders()

        accelerator = Accelerator(mixed_precision="fp16")
        (
            self.model,
            self.test_dataloader,
        ) = accelerator.prepare(self.model, self.test_dataloader)

        # run eval
        self.model.eval()
        answer_batches = []
        for batch in tqdm(self.test_dataloader):
            outputs = self.model.generate(
                **batch, max_length=self.max_ans_length, num_beams=1
            )
            for i in outputs:
                answer_batches.append(i)

        predicted_answers = [
            bart_utils.clean_outputs(i, tokenizer=self.tokenizer)
            for i in answer_batches
        ]
        eval_outputs = list(zip(self.test_batches["example_id"], predicted_answers))

        score, predictions, targets = bart_utils.evaluate(
            eval_outputs, self.test_dataset
        )
        f1_score = score["f1"]
        em = score["exact_match"]

        # logging
        self.logger.info(
            "\n{} \nEvaluation results \nmodel : {} \n > F1 = {} \n > EM = {} \n{}".format(
                "*" * 50, self.name, f1_score, em, "*" * 50
            )
        )

        results = {
            "em": em,
            "f1": f1_score,
            "predicted_answers": predictions,
        }

        # SAVE results as json
        with open(self.resultfile, "w") as fp:
            json.dump(results, fp)

        self.logger.info("Evaluation results saved at {}".format(self.resultfile))

        return results


class EvaluateT5(BaseTester):
    """ """

    def __init__(self, config, output_dir=None):
        super().__init__(config)
        self.max_ans_length = config.max_ans_length
        self.max_seq_length = config.max_seq_length

        if output_dir == None:
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            self.logfile = os.path.abspath(
                "evaluation-{}-{}.log".format(self.name, timestamp)
            )
            self.resultfile = os.path.join("./", "results.json")
        else:
            self.logfile = os.path.join(output_dir, "evaluation.log")
            self.resultfile = os.path.join(output_dir, "results.json")

        self.logger = logging.getLogger("eval_logger")
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(self.logfile)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        s = """Evaluation run initialized

T5-like evaluation configuration
************************************
        Name : {}
        Model checkpoint : {}
        Tokenizer checkpoint : {}
        Max sequence length : {}
        Max answer length : {}
        Padding : {}
        Stride : {}
        Output directory : {}
************************************
        """.format(
            self.name,
            self.model_checkpoint,
            self.tokenizer_checkpoint,
            self.max_seq_length,
            self.max_ans_length,
            self.padding,
            self.stride,
            output_dir,
        )

        self.logger.info(s)

    def get_dataloaders(self):
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        test_tensor = self.test_batches.remove_columns(["example_id"])
        test_tensor.set_format("torch")
        self.test_dataloader = DataLoader(
            test_tensor,
            collate_fn=data_collator,
            batch_size=4,
            shuffle=False,
        )

        self.logger.info("Test and validation dataloaders created")

    @torch.no_grad()
    def __call__(self):
        self.get_dataloaders()
        accelerator = Accelerator()
        (
            self.model,
            self.test_dataloader,
        ) = accelerator.prepare(self.model, self.test_dataloader)

        # run eval
        self.model.eval()
        answer_batch = []
        for i, batch in enumerate(tqdm(self.test_dataloader)):
            outputs = self.model.generate(
                **batch,
                max_length=self.max_ans_length,
                num_beams=1,
            )
            for i in outputs:
                answer_batch.append(i)

        predicted_answers = [
            t5_utils.clean_outputs(i, self.tokenizer) for i in answer_batch
        ]

        eval_outputs = list(zip(self.test_batches["example_id"], predicted_answers))

        score, predictions, targets = t5_utils.evaluate(eval_outputs, self.test_dataset)

        f1_score = score["f1"]
        em = score["exact_match"]
        # logging
        self.logger.info(
            "\n{} \nEvaluation results \nmodel : {} \n > F1 = {} \n > EM = {} \n{}".format(
                "*" * 50, self.name, f1_score, em, "*" * 50
            )
        )

        results = {
            "em": em,
            "f1": f1_score,
            "predicted_answers": predictions,
        }

        # SAVE results as json
        with open(self.resultfile, "w") as fp:
            json.dump(results, fp)

        self.logger.info("Evaluation results saved at {}".format(self.resultfile))

        return results
