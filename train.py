import logging

import numpy as np
import random

from t5_utils import *

from tqdm.auto import tqdm

from transformers import get_scheduler, DataCollatorForSeq2Seq

from accelerate import Accelerator

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


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

    def get_val_answers(self):
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

        # @HERE :: finish setup of dataloaders and all, then get to the training and eval loops

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

        # TODO setup for GPU accelerate fp16 :: vs CPU (vanilla pytorch)
        # __import__("IPython").embed()
        # print(torch.device())
        # torch.device = "cpu"
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

        progressbar = tqdm(range(num_training_steps))

        val_targets = self.get_val_answers()

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

            eval_outputs = list(zip(self.val_batches["example_id"], predicted_answers))

            score, predictions, targets = evaluate(eval_outputs, val_targets)

            print(list(zip(predictions, targets)))
            f1_score = score["f1"]

            self.logger.info(
                "Epoch: {}, Loss: {}, Validation F1: {}".format(
                    epoch, float(loss.cpu()), score
                )
            )

        #    # save the best model
        #    if f1_score > best_f1:
        #        best_f1 = f1_score
        #        if os.path.isfile(local_path):
        #            shutil.rmtree(local_path)
        #        if self.train_prefix:
        #            self.model.save_adapter(local_path, "prefix_tuning")
        #            self.logger.info("new best prefix adapter saved!")
        #        else:
        #            self.model.save_pretrained(local_path)
        #            self.logger.info("new best model saved!")
        #
        # self.logger.info("Best model f1 = {}".format(best_f1))
        # return best_f1

        # HERE - TODO
        # NEXT! -> Return the best model after 35 epochs based on the top validation F1 like in the paper

        # @TODO :: IMPORTANT setup wandb in this class

        return None
