import random
import logging

import numpy as np

from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import get_scheduler

from accelerate import Accelerator

import datasets
from datasets import load_metric

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import default_data_collator


###############################################################################
#                          FewShotBART Implementation                         #
###############################################################################

# note :: set seed from cli
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


bart_default_config = {
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

        # defined locally
        self.tokenizer = None
        self.model = None
        self.proc_train_dataset = None
        self.proc_test_dataset = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.metric = load_metric("squad")

    def __model_initialization(self, mode):
        if mode == "qa":
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = BartForConditionalGeneration.from_pretrained(self.checkpoint)
            self.logger.info("tokenizer and QA model initialized")

    def __training_preprocessing(self, examples):
        """examples have all three types of string"""

        model_inputs = self.tokenizer(
            examples["masked_strings"],
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=False,
        )
        labels = self.tokenizer(
            text_target=examples["qa_strings"],
            max_length=self.max_ans_length,
            padding=self.padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def __evaluation_preprocessing(self, examples):
        """ """
        model_inputs = self.tokenizer(
            examples["masked_strings"],
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=False,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # BUG :: truncation of the input will offset the answer and create a
        # tensor of inequal length were we would have a greater number of spans
        # for a given number of labels, should I include a dummy label - that
        # would not work?, i think the bug comes from returning the overflowing
        # tokens....
        # NOTE :: temporarily set to false to try out the model

        labels = self.tokenizer(
            text_target=examples["qa_strings"],
            max_length=self.max_ans_length,
            padding=self.padding,
            truncation=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        model_inputs["example_id"] = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def __clean_output(self, output):
        """take the logit outputs from a sample of the seq2seq LM and turn it into a string for evaluation!"""
        seq = output[0]
        top_tokens = []
        for i in seq:
            top_tokens.append(
                self.tokenizer.decode(np.argmax(i), skip_special_tokens=True)
            )
        # NOTE :: when I eval the model I need to actually remove all the stuff
        # that is not part of the answer, also - some samples may not contain
        # Context in the sequence... then go for the dot?

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

    def __eval(self, eval_outputs, answers):

        theoretical_answers = []
        predicted_answers = []
        model_outputs = []
        datasets.disable_progress_bar()

        for idx, outputs in eval_outputs:
            label_answer = answers.filter(lambda sample: sample["id"] == idx)[
                "answer_strings"
            ]
            predicted_answer = self.__clean_output(outputs)

            theoretical_answers.append(
                {"id": idx, "answers": {"answer_start": [], "text": label_answer}}
            )
            predicted_answers.append({"id": idx, "prediction_text": predicted_answer})
            # print("PRED {} ANS {}".format(predicted_answer, label_answer[0]))

        m = self.metric.compute(
            predictions=predicted_answers, references=theoretical_answers
        )

        # BUG -- exact match metric doesn't seem to be working, I don't think
        # it can bc this is a generative model!

        return m

    def __preprocess(self, examples, type=None):
        # stuff

        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        if type == "training":

            self.proc_train_dataset = examples.map(
                self.__training_preprocessing,
                batched=True,
                remove_columns=train_raw.column_names,
            )
            train_tensor = self.proc_train_dataset
            train_tensor.set_format("torch")
            self.train_dataloader = DataLoader(
                train_tensor,
                shuffle=True,
                collate_fn=data_collator,
                batch_size=4,
            )
            self.logger.info("training dataset processed and dataloaders created")

        elif type == "validation":
            self.proc_test_dataset = examples.map(
                self.__evaluation_preprocessing,
                batched=True,
                remove_columns=test_raw.column_names,
            )
            test_tensor = self.proc_test_dataset
            test_tensor.set_format("torch")
            self.test_dataloader = DataLoader(
                test_tensor,
                collate_fn=data_collator,
                batch_size=1,
            )
            self.logger.info("validation dataset processed and dataloaders created")

        else:
            self.logger.error(
                "preprocessing requires valide type training or validation"
            )

    def finetune(self):
        self.__model_initialization("qa")

        self.__preprocess(self.train_dataset, "training")
        self.__preprocess(self.test_dataset, "validation")

        best_f1 = 0

        optimizer = AdamW(self.model.parameters(), lr=self.lr)

        # setup for GPU
        accelerator = Accelerator(fp16=True)
        (
            self.model,
            optimizer,
            self.train_dataloader,
            self.test_dataloader,
        ) = accelerator.prepare(
            self.model, optimizer, self.train_dataloader, self.test_dataloader
        )

        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = self.num_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for epoch in range(self.num_epochs):
            self.model.train()
            for steps, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            eval_outputs = []
            for i, batch in enumerate(self.test_dataloader):
                self.model.eval()
                with torch.no_grad():
                    batch.pop("offset_mapping")
                    idx = int(batch["example_id"].cpu().numpy())
                    batch.pop("example_id")
                    outputs = self.model(
                        **batch
                    )  # BUG idk why but evaluation is extremely slow....
                    eval_outputs.append(
                        (idx, accelerator.gather(outputs.logits).cpu().numpy())
                    )

            score = self.__eval(eval_outputs, self.test_dataset)
            print(
                "Epoch: {}, Loss: {}, Validation F1: {}".format(
                    epoch, float(loss.cpu()), score
                )
            )

            # save the best model
            if f1_score > best_f1:
                best_f1 = f1_score
                self.model.save_pretrained("./top_bart.bin")
                self.logger.info("new best model saved!")

        # final eval
        print("Best model f1 = {}".format(best_f1))
        return best_f1
        # HERE - TODO
        # NEXT! -> Return the best model after 35 epochs based on the top validation F1 like in the paper

        # def run(self, ):
        """run model for inference only"""
