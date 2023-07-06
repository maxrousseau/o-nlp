from train import BaseTrainer

from tacoma.collator import TacomaCollator, tokenize_with_mask_mapping

import os
import wandb

from datasets import Dataset, load_dataset

from dataclasses import dataclass
from typing import Any

from tqdm.auto import tqdm

import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    get_scheduler,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    default_data_collator,
)

from accelerate import Accelerator


@dataclass
class TACOMA_CFG:
    name: str = "tacoma-default"
    lr: float = 2e-5
    n_epochs: int = 12
    lr_scheduler: bool = True
    model_checkpoint: str = ""
    tokenizer_checkpoint: str = ""
    checkpoint_savedir: str = ""
    gradient_accumulation_steps: int = 4

    max_seq_length: int = 512
    max_ans_length: int = 128
    stride: int = 128
    padding: str = "max_length"
    seed: str = 0

    load_from_checkpoint: bool = False
    checkpoint_state: str = None
    checkpoint_step: int = None

    train_batch_size: int = 4
    val_batch_size: int = 16

    train_dataset: Dataset = None
    val_dataset: Dataset = None
    test_dataset: Dataset = None

    val_batches: Any = None
    train_batches: Any = None
    test_batches: Any = None

    model: Any = None
    tokenizer: Any = None

    def __repr__(self) -> str:
        s = """
Tacoma model configuration
************************************
        Name : {}
        Model checkpoint : {}
        Tokenizer checkpoint : {}
        Max sequence length : {}
        Max answer length : {}
        Hyperparameters :
                lr={},
                lr_scheduler={},
                num_epochs={},
                batch_size={}
************************************
        """.format(
            self.name,
            self.model_checkpoint,
            self.tokenizer_checkpoint,
            self.max_seq_length,
            self.max_ans_length,
            self.lr,
            self.lr_scheduler,
            self.n_epochs,
            self.train_batch_size,
        )
        return s


def t5_init(model_checkpoint, tokenizer_checkpoint):
    """initialize model and tokenizer, mode=Default, LoRA"""

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_checkpoint,  # torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    return model, tokenizer


def setup_tacoma_training(data_repo, config):
    config.model, config.tokenizer = t5_init(
        config.model_checkpoint, config.tokenizer_checkpoint
    )

    masking_dataset = load_dataset(data_repo)
    masking_dataset = masking_dataset["train"]
    masking_dataset = masking_dataset.shuffle(seed=config.seed).train_test_split(
        test_size=0.05
    )

    masking_dataset = masking_dataset.map(
        lambda example: tokenize_with_mask_mapping(
            example, config.tokenizer, max_sequence_length=config.max_seq_length
        )
    )

    config.train_dataset = masking_dataset["train"].filter(
        lambda example: example["valid"] == True
    )
    config.val_dataset = masking_dataset["test"].filter(
        lambda example: example["valid"] == True
    )

    # __import__("IPython").embed()

    config.train_batches = config.train_dataset.remove_columns(
        [
            "question",
            "text",
            "target",
            "target_start",
            "valid",
        ]
    )
    config.val_batches = config.val_dataset.remove_columns(
        [
            "question",
            "text",
            "target",
            "target_start",
            "valid",
        ]
    )

    return config


class TacomaT5(BaseTrainer):
    """

    Everything that is dynamic is defined here: dataloader, training and evaluation loop, model export, etc.

    """

    def __init__(self, config, eval_steps=100):
        super().__init__(config)
        self.max_ans_length = config.max_ans_length
        self.max_seq_length = config.max_seq_length
        self.eval_steps = eval_steps
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        s = """Training run initialized

TaCoMa training configuration
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
        self.train_batches.set_format("torch")
        self.val_batches.set_format("torch")

        # create the dataloaders
        label_pad_token_id = -100
        tacoma_collator = TacomaCollator(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        self.train_dataloader = DataLoader(
            self.train_batches,
            shuffle=True,
            collate_fn=tacoma_collator,
            batch_size=self.train_batch_size,
            num_workers=0,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        self.val_dataloader = DataLoader(
            self.val_batches,
            shuffle=False,
            collate_fn=tacoma_collator,
            batch_size=self.val_batch_size,
        )

        self.logger.info("Training, validation and test dataloaders created")
        # shuffle only train...

    @torch.no_grad()
    def __eval(self):
        self.model.eval()

        val_losses = []
        for i, batch in enumerate(tqdm(self.val_dataloader)):
            batch.pop("word_ids")
            batch.pop("mask_mappings")
            batch.pop("offset_mapping")
            outputs = self.model(**batch)
            val_losses.append(outputs.loss.detach().cpu().numpy())

        self.model.train()

        val_loss = np.array(val_losses).mean()
        return val_loss

    def __call__(self):
        """simply call the finetuning"""

        # We start the fine-tuning code here, essentially we feed it a model and some data and it trains it and
        # logs the loss/results/weigths that is all....
        self.get_dataloaders()
        checkpoint_path_bestval = os.path.join(self.output_dir, "checkpoint-bestval")
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

        best_val_loss = 100

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_update_steps_per_epoch = (
            len(self.train_dataloader) / self.gradient_accumulation_steps
        )
        num_training_steps = (
            self.num_epochs * num_update_steps_per_epoch
        )  # total number of updates
        num_total_steps = len(self.train_dataloader) * self.num_epochs
        if self.lr_scheduler:
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0.1 * num_training_steps,
                num_training_steps=num_training_steps,
            )

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )
        (
            self.model,
            optimizer,
            self.train_dataloader,
            self.val_dataloader,
        ) = accelerator.prepare(
            self.model, optimizer, self.train_dataloader, self.val_dataloader
        )

        progressbar = tqdm(range(num_total_steps))

        train_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()

            for steps, batch in enumerate(self.train_dataloader):
                batch.pop("word_ids")
                batch.pop("mask_mappings")
                batch.pop("offset_mapping")
                with accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    train_losses.append(loss.detach().cpu().numpy())
                    optimizer.step()
                    if self.lr_scheduler:
                        lr_scheduler.step()
                    optimizer.zero_grad()
                progressbar.update(1)

                if steps % self.eval_steps == 0:
                    # run eval
                    train_loss = np.array(train_losses).mean()
                    val_loss = self.__eval()

                    train_losses = []

                    self.logger.info(
                        "Epoch: {}, step {}, training loss: {}, validation loss: {}".format(
                            epoch, steps, train_loss, val_loss
                        )
                    )
                    wandb.log(
                        {
                            "epoch": epoch,
                            "step": steps,
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                        }
                    )

                    # save the best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(checkpoint_path_bestval)
                        self.logger.info(
                            "New save with best_val_loss = {}".format(best_val_loss)
                        )

        self.save_model(checkpoint_path_end)
        self.logger.info(
            "Best {} val_loss = {} localted at {}".format(
                self.name, best_val_loss, checkpoint_path_bestval
            )
        )

        return {
            "best_val_loss": best_val_loss,
            "checkpoint_bestf1": checkpoint_path_bestval,
            "checkpoint_end": checkpoint_path_end,
            "n_step": num_training_steps,
        }


def sanitize(train_batches, tokenizer, model):
    train_batches.set_format("torch")
    tacoma_collator = TacomaCollator(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    dataloader = DataLoader(
        train_batches,
        collate_fn=tacoma_collator,
        batch_size=4,
    )
    for steps, batch in enumerate(tqdm(dataloader)):
        None
        # print("ok")
        # print(type(batch))
