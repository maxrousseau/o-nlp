import os
import logging
import shutil

import numpy as np
import random

from t5_utils import *

from tqdm.auto import tqdm

from transformers import get_scheduler, DataCollatorForSeq2Seq

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb


class BaseTrainer:
    FORMAT = "[%(levelname)s] :: %(asctime)s @ %(name)s :: %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)  # change level if debug or not

    def __init__(self, config):  # @fix...
        """basic trainer class for all models"""
        self.name = config.name
        self.lr = config.lr
        self.lr_scheduler = config.lr_scheduler
        self.num_epochs = config.n_epochs

        self.tokenizer = config.tokenizer
        self.model = config.model

        self.train_dataset = config.train_dataset
        self.test_dataset = config.test_dataset
        self.train_batches = config.train_batches
        self.val_batches = config.val_batches
        self.test_batches = config.test_batches

        self.checkpoint_savedir = config.checkpoint_savedir

        self.load_from_checkpoint = config.load_from_checkpoint
        self.checkpoint_state = config.checkpoint_state
        self.checkpoint_step = config.checkpoint_step

        self.max_seq_length = config.max_seq_length
        self.max_ans_length = config.max_ans_length
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

        # defined locally
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


class FineTuneT5(BaseTrainer):
    """

    Everything that is dynamic is defined here: dataloader, training and evaluation loop, model export, etc.

    """

    def __init_(self):
        super().__init__(config)

    def __get_dataloaders(self):
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
        self.test_dataloader = DataLoader(
            test_tensor,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=4,
        )
        logger.info("Training, validation and test dataloaders created")
        # shuffle only train...

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
            {"id": self.val_batches["example_id"], "answer_strings": val_targets}
        )

    def __call__(self):
        """simply call the finetuning"""

        # We start the fine-tuning code here, essentially we feed it a model and some data and it trains it and
        # logs the loss/results/weigths that is all....
        self.__get_dataloaders()
        local_path = os.path.abspath("{}-{}".format(self.checkpoint_savedir, self.name))
        wandb.init(
            project="o-nlp",
            config={
                "learning_rate": self.lr,
                "architecture": "t5-small-test",
                "dataset": "oqa-alpha",
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
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )

        if torch.device != "cpu":
            # @BUG mixed precision breaks generation
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

        val_targets = self.__get_val_answers()

        for epoch in range(self.num_epochs):
            self.model.train()
            for steps, batch in enumerate(self.train_dataloader):

                outputs = self.model(**batch)
                loss = outputs.loss
                if torch.device != "cpu":
                    accelerator.backward(loss)
                else:
                    loss.backward()

                optimizer.step()
                if self.lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
                progressbar.update(1)

            answer_batch = []
            for i, batch in enumerate(tqdm(self.val_dataloader)):
                self.model.eval()
                with torch.no_grad():
                    batch.pop("labels")
                    batch.pop("decoder_input_ids")
                    batch.pop("attention_mask")

                    outputs = self.model.generate(
                        **batch,
                        max_length=25,
                        num_beams=1,
                    )
                    for i in outputs:
                        answer_batch.append(i)

            predicted_answers = [clean_outputs(i, self.tokenizer) for i in answer_batch]
            # print(predicted_answers)

            eval_outputs = list(zip(self.val_batches["example_id"], predicted_answers))

            score, predictions, targets = evaluate(eval_outputs, val_targets)

            # print(list(zip(predictions, targets)))
            f1_score = score["f1"]

            self.logger.info(
                "Epoch: {}, Loss: {}, Validation F1: {}".format(
                    epoch, float(loss.cpu()), score
                )
            )
            wandb.log({"loss": loss, "val_f1": f1_score})

            # @HERE :: TODO -- hook up wandb and then refactor BART in this way...

            # save the best model
            if f1_score > best_f1:
                best_f1 = f1_score
                if os.path.isfile(local_path):
                    shutil.rmtree(local_path)
                self.model.save_pretrained(local_path)
                self.logger.info("new best model saved!")

        # @TODO :: next re

        self.logger.info("Best model f1 = {}".format(best_f1))

        return None


class PretrainT5(BaseTrainer):
    """ """

    def __init_(self):
        super().__init__(config)

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

        logger.info("Training, validation dataloaders created")
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
        # @BUG for some reason this is about 4x slower than training ? Ok Fixed
        for i, batch in enumerate(tqdm(self.val_dataloader)):
            outputs = self.model(**batch)
            losses["val"].append(outputs.loss.item())
            # accumulate val_losses losses["val"]
        self.model.train()
        return losses

    def __call__(self):
        """ """
        self.__get_dataloaders()
        local_path = os.path.abspath(
            "{}-{}-fullrun".format(self.checkpoint_savedir, self.name)
        )

        wandb.init(
            project="o-nlp",
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
                num_warmup_steps=100,
                num_training_steps=num_training_steps,
            )

        if torch.device != "cpu":
            # @BUG mixed precision breaks generation
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
            logger.info(
                "Checkpoint: {} reloaded! Step: {}, Number of masked tokens: {}".format(
                    self.checkpoint_state, n_step, n_masked_tokens
                )
            )

        # @TODO :: eval here before we start training then only eval every 100k masked tokens otherwise it is probably
        # going to take too long to train (right now over 8hours for 86k samples)
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

                # __import__("IPython").embed()

                if torch.device != "cpu":
                    accelerator.backward(loss)
                else:
                    loss.backward()

                # @TODO :: accumulate training losses in numpy array losses["train"]
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
                    logger.info(
                        "chekpoint saved at step {} after {} masked tokens".format(
                            n_step, n_masked_tokens
                        )
                    )
                    losses = self.__eval(losses)

                    # @TODO :: save best model according to lowest validation loss!

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

        return None
