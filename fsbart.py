import os

import shutil
import time
import random
import logging

import numpy as np

from tqdm.auto import tqdm

from transformers import (
    get_scheduler,
    DataCollatorForSeq2Seq,
    BartTokenizerFast,
    BartForConditionalGeneration,
)
from transformers.adapters import PrefixTuningConfig

from accelerate import Accelerator

import datasets
from datasets import load_metric

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

###############################################################################
#                          FewShotBART Implementation                         #
###############################################################################


bart_default_config = {
    "name": "bart-default",
    "lr": 2e-5,
    "num_epochs": 12,
    "lr_scheduler": True,
    "checkpoint": "facebook/bart-base",
    "checkpoint_savedir": "./ckpt",
    "train_dev_dataset": None,
    "val_dev_dataset": None,
    "train_dataset": None,
    "test_dataset": None,
    "max_seq_length": 384,
    "max_ans_length": 128,
    "stride": 128,
    "padding": "max_length",
    "seed": 0,
    "prefix": False,
    "train_prefix": False,
    "unfreeze": False,
}


class FsBART:
    """snippet to finetune with this class

    bart_default_config["train_dataset"] = train_raw
    bart_default_config["test_dataset"] = test_raw

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(bart_default_config)

    fsbart = FsBART(**bart_default_config)
    fsbart.finetune()
    """

    FORMAT = "[%(levelname)s] :: %(asctime)s @ %(name)s :: %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("bart")
    logger.setLevel(logging.DEBUG)  # change level if debug or not

    def __init__(self, **kwargs):
        """base configuration for the few-shot qa with generative BART model"""
        # model/training/inference configuration
        self.name = kwargs["name"]
        self.lr = kwargs["lr"]
        self.num_epochs = kwargs["num_epochs"]
        self.checkpoint = kwargs["checkpoint"]
        self.train_dev_dataset = kwargs["train_dev_dataset"]
        self.val_dev_dataset = kwargs["val_dev_dataset"]
        self.train_dataset = kwargs["train_dataset"]
        self.test_dataset = kwargs["test_dataset"]
        self.checkpoint_savedir = kwargs["checkpoint_savedir"]
        self.max_seq_length = kwargs["max_seq_length"]
        self.max_ans_length = kwargs["max_ans_length"]
        self.padding = kwargs["padding"]
        self.stride = kwargs["stride"]
        self.seed = kwargs["seed"]
        self.prefix = kwargs["prefix"]
        self.train_prefix = kwargs["train_prefix"]
        self.unfreeze = kwargs["unfreeze"]

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
        self.tokenizer = None
        self.model = None
        self.proc_train_dataset = None
        self.proc_test_dataset = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.metric = load_metric("squad")

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def model_initialization(self, mode):
        if mode == "mi":
            self.tokenizer = BartTokenizerFast.from_pretrained(self.checkpoint)
            self.model = BartForConditionalGeneration.from_pretrained(self.checkpoint)
            self.logger.info("tokenizer initialized")

            # TODO :: implement the prefix version with transformers-adapers and training code for SQuAD MI
            if self.prefix:
                prefix_config = PrefixTuningConfig(flat=False, prefix_length=10)
                self.model = BartForConditionalGeneration.from_pretrained(
                    self.checkpoint
                )
                self.model.add_adapter("prefix_tuning", config=prefix_config)
                self.logger.info("prefix bart for mask infilling model initialized")
            else:
                self.model = BartForConditionalGeneration.from_pretrained(
                    self.checkpoint
                )
                self.logger.info("bart model initialized")

        if self.train_prefix:
            self.model.set_active_adapters("prefix_tuning")
            self.logger.info("training prefix parameters")

        # TODO see how to incorporate unfreezing of model params and various combinations of adapters vs full model (frozen/unfrozen)
        # if self.unfreeze:
        #     self.model.freeze_model(False)
        #     self.logger.info("model parameters unfrozen")

    def preprocess(self, examples):
        source, target = examples["masked_strings"], examples["qa_strings"]
        source_tokenized = self.tokenizer(
            source,
            padding=self.padding,
            max_length=self.max_seq_length,
            truncation=True,
        )

        target_tokenized = self.tokenizer(
            target,
            padding=self.padding,
            max_length=self.max_seq_length,
            truncation=True,
        )

        batch = {k: v for k, v in source_tokenized.items()}

        batch["example_id"] = examples["id"]

        # Ignore padding in the loss
        batch["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in l]
            for l in target_tokenized["input_ids"]
        ]
        return batch

    def clean_output(self, output):
        """take the logit outputs from a sample of the seq2seq LM and turn it into a string for evaluation!"""
        seq = output
        top_tokens = []
        for i in seq:
            top_tokens.append(
                self.tokenizer.decode(np.argmax(i), skip_special_tokens=True)
            )
        # NOTE :: when I eval the model I need to actually remove all the stuff
        # that is not part of the answer, also - some samples may not contain
        # Context in the sequence... then go for the dot?

        answer_tokens = [""]
        if " Answer" in top_tokens:
            start_idx = (
                top_tokens.index(" Answer") + 2
            )  # skip the ":" token, maybe check if its there before?
            if " Context" in top_tokens:
                end_idx = (
                    top_tokens.index(" Context") - 1
                )  # skip the "." token, maybe check if its there before...
            elif "." in top_tokens:
                end_idx = top_tokens.index(".")
                answer_tokens = top_tokens[start_idx:end_idx]
            else:
                answer_tokens = [""]
        else:
            answer_tokens = [""]

        answer_str = "".join(answer_tokens).strip()
        return answer_str

    def eval(self, eval_outputs, answers):

        theoretical_answers = []
        predicted_answers = []
        datasets.disable_progress_bar()

        for idx, predicted_answer in eval_outputs:
            label_answer = answers.filter(lambda sample: sample["id"] == idx)[
                "answer_strings"
            ]

            theoretical_answers.append(
                {"id": idx, "answers": {"answer_start": [], "text": label_answer}}
            )

            predicted_answers.append({"id": idx, "prediction_text": predicted_answer})

        m = self.metric.compute(
            predictions=predicted_answers, references=theoretical_answers
        )

        # BUG -- exact match metric doesn't seem to be working, I don't think
        # it can bc this is a generative model?

        return m, predicted_answers, theoretical_answers

    def prepare(self, examples, shuffle, type=None):
        # stuff

        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        tokenized_dataset = examples.map(
            self.preprocess,
            batched=True,
            remove_columns=examples.column_names,
        )

        tensor = tokenized_dataset.remove_columns(["example_id"])
        tensor.set_format("torch")

        dataloader = DataLoader(
            tensor,
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=4,
            num_workers=0,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )

        self.logger.info("{} dataset processed and dataloaders created".format(type))
        return dataloader, tokenized_dataset

    # def __post_processing():
    #     ''' TODO :: maybe I need this for bert but not necessarily for bart unless I go beyond the length.... (see later)'''

    def finetune(self, mode=None, gpu=True):
        local_path = os.path.abspath("./{}-{}".format(self.name, int(time.time())))

        self.model_initialization("mi")

        if mode == "dev":
            training_set = self.train_dev_dataset
            eval_set = self.val_dev_dataset
        elif mode == "run":
            training_set = self.train_dataset
            eval_set = self.test_dataset
        else:
            raise NameError('Please specify mode for fine-tuning: "dev" or "run"')

        self.train_dataloader, self.proc_train_dataset = self.prepare(
            training_set, shuffle=True, type="training"
        )
        self.test_dataloader, self.proc_test_dataset = self.prepare(
            eval_set, shuffle=False, type="validation"
        )

        best_f1 = -1
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = self.num_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        # TODO setup for GPU accelerate fp16 :: vs CPU (vanilla pytorch)
        if torch.device != "cpu":
            accelerator = Accelerator(fp16=True)
            (
                self.model,
                optimizer,
                self.train_dataloader,
                self.test_dataloader,
            ) = accelerator.prepare(
                self.model, optimizer, self.train_dataloader, self.test_dataloader
            )

        progressbar = tqdm(range(num_training_steps))
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
                lr_scheduler.step()
                optimizer.zero_grad()
                progressbar.update(1)

            predicted_answers = []
            answer_batch = None
            for i, batch in enumerate(tqdm(self.test_dataloader)):
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(
                        **batch
                    )  # BUG idk why but evaluation is extremely slow....
                    if torch.device != "cpu":
                        if i == 0:
                            answer_batch = (
                                accelerator.gather(outputs.logits).cpu().numpy()
                            )
                        elif (i % 4 == 0) & (i != 0):
                            predicted_answers += [
                                self.clean_output(i) for i in answer_batch
                            ]
                            answer_batch = (
                                accelerator.gather(outputs.logits).cpu().numpy()
                            )
                        else:
                            answer_batch = np.concatenate(
                                (
                                    answer_batch,
                                    accelerator.gather(outputs.logits).cpu().numpy(),
                                ),
                                axis=0,
                            )

                    else:
                        # have the same as above for cpu (if needed)
                        predicted_answers += [
                            self.clean_output(i) for i in outputs.logits.cpu().numpy()
                        ]

            predicted_answers += [self.clean_output(i) for i in answer_batch]

            eval_outputs = list(
                zip(self.proc_test_dataset["example_id"], predicted_answers)
            )
            score, predictions, targets = self.eval(eval_outputs, self.test_dataset)
            f1_score = score["f1"]
            self.logger.info(
                "Epoch: {}, Loss: {}, Validation F1: {}".format(
                    epoch, float(loss.cpu()), score
                )
            )

            # save the best model
            if f1_score > best_f1:
                best_f1 = f1_score
                if os.path.isfile(local_path):
                    shutil.rmtree(local_path)
                self.model.save_pretrained(local_path)
                self.logger.info("new best model saved!")

        self.model = BartForConditionalGeneration.from_pretrained(
            local_path, local_files_only=True
        )
        self.logger.info("best model reloaded!")
        # final eval
        print("Best model f1 = {}".format(best_f1))
        return best_f1
        # HERE - TODO
        # NEXT! -> Return the best model after 35 epochs based on the top validation F1 like in the paper

    def run(self, mode, inputs=None, init=False, evaluate=True):
        """run model for inference only"""
        try:
            if init:
                self.model_initialization(mode)
                # preprocessing only validation
                self.preprocess(self.test_dataset, "validation")
                self.logger.info("new model initialized")
            else:
                self.logger.info("using current model")
        except:
            self.logger.error("model initialization error")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # TODO if eval true then simply run on validation? otherwise use new samples from input
        # if eval:
        self.model.eval()
        eval_outputs = []
        for i, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                batch.pop("offset_mapping")
                idx = int(batch["example_id"].cpu().numpy())
                batch.pop("example_id")
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = self.model(**batch)
                eval_outputs.append((idx, outputs.logits.cpu().numpy()))

        metrics, predictions, targets = self.eval(eval_outputs, self.test_dataset)
        f1_score = metrics["f1"]

        return f1_score, predictions, targets
