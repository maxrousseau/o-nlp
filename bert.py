import logging
import collections
import random

import numpy as np

from tqdm.auto import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import default_data_collator
from transformers import get_scheduler

from accelerate import Accelerator

from datasets import load_metric


###############################################################################
#                             BERT Implementation                             #
###############################################################################


# note :: set seed from cli
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


bert_default_config = {
    "lr": 2e-5,
    "num_epochs": 2,
    "lr_scheduler": True,
    "checkpoint": "dmis-lab/biobert-base-cased-v1.1-squad",
    "checkpoint_savedir": "./ckpt",
    "train_dev_dataset": None,
    "val_dev_dataset": None,
    "train_dataset": None,
    "test_dataset": None,
    "stride": 128,
    "max_length": 384,
}


class OrthoBert:
    """"""

    FORMAT = "[%(levelname)s] :: %(asctime)s @ %(name)s :: %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("bert")
    logger.setLevel(logging.DEBUG)  # change level if debug or not

    def __init__(self, **kwargs):
        """base configuration for the BERT model, include default training parameters as well"""
        # model/training/inference configuration
        self.lr = kwargs["lr"]
        self.num_epochs = kwargs["num_epochs"]
        self.checkpoint = kwargs["checkpoint"]
        self.train_dev_dataset = kwargs["train_dev_dataset"]
        self.val_dev_dataset = kwargs["val_dev_dataset"]
        self.train_dataset = kwargs["train_dataset"]
        self.test_dataset = kwargs["test_dataset"]
        self.checkpoint_savedir = kwargs["checkpoint_savedir"]
        self.max_length = kwargs["max_length"]
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
        """
        Initialize the model with the desired mode (MLM or QA)
        """
        if mode == "qa":
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.checkpoint)
            self.logger.info("qa model initialized")

        if mode == "mlm":
            None

    def __training_preprocessing(self, examples):
        """
        preprocessing for training examples
                return:
                inputs:
                features: ['example_id', 'offset_mapping', 'attention_mask', 'token_type_id']
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_lengt",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context (question = 0, context = 1,  special token = None)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)

            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        return inputs

    def __evaluation_preprocessing(self, examples):
        """
        preprocessing for evalutation samples (validation and test)
        return:
        inputs:
        features ['example_id', 'offset_mapping', 'attention_mask', 'token_type_id']
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,  # what is this for?
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    def __answerFromLogit(self, start_logits, end_logits, features, examples):
        """
        from the HF tutorial, this function takes the logits as input and returns the score from the metric (EM and F1)
        @TODO - separate the best answer code from the metric computation -- keeping them in separate functions would make it easier to
        check out the output
        """
        n_best = 20
        max_answer_length = 30

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[
                                offsets[start_index][0] : offsets[end_index][1]
                            ],
                            "logit_score": start_logit[start_index]
                            + end_logit[end_index],
                        }
                        answers.append(answer)

            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in examples
        ]
        return self.metric.compute(
            predictions=predicted_answers, references=theoretical_answers
        )

    def preprocess(self, examples, type=None):
        """specify type of preprocessing (train or eval)"""

        if type == "training":
            self.proc_train_dataset = examples.map(
                self.__training_preprocessing,
                batched=True,
                remove_columns=examples.column_names,
            )

            train_tensor = self.proc_train_dataset
            train_tensor.set_format("torch")
            self.train_dataloader = DataLoader(
                train_tensor,
                shuffle=True,
                collate_fn=default_data_collator,
                batch_size=8,
            )

            self.logger.info("training dataset processed and dataloaders created")

        elif type == "validation":
            self.proc_test_dataset = examples.map(
                self.__evaluation_preprocessing,
                batched=True,
                remove_columns=examples.column_names,
            )

            test_tensor = self.proc_test_dataset.remove_columns(
                ["example_id", "offset_mapping"]
            )
            test_tensor.set_format("torch")
            self.test_dataloader = DataLoader(
                test_tensor,
                collate_fn=default_data_collator,
                batch_size=1,
            )

            self.logger.info("validation dataset processed and dataloaders created")

        else:
            self.logger.error(
                "preprocessing requires valide type training or validation"
            )

    def finetune(self, mode):
        # load the model
        self.__model_initialization(mode)

        # preprocessing
        self.preprocess(self.train_dataset, "training")
        self.preprocess(self.test_dataset, "validation")

        best_f1 = 0

        # fine-tuning
        optimizer = AdamW(self.model.parameters(), lr=self.lr)

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

        progressbar = tqdm(range(num_training_steps))
        for epoch in range(self.num_epochs):
            self.model.train()  # this just sets the torch model to train
            for steps, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)  # backprop here

                # zero gradient and update Lr and optimizer
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progressbar.update(1)

            self.logger.info("epoch {} :: Ok".format(epoch))
            # eval after epoch
            self.model.eval()
            start_logits = []
            end_logits = []
            print("Evaluation")  # use accelarator.print() if you only want to
            # print once when training on multiple machines

            for batch in tqdm(self.test_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch)

                start_logits.append(
                    accelerator.gather(outputs.start_logits).cpu().numpy()
                )
                end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

                # hf bert models use classes format outputs
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#:~:text=return%20QuestionAnsweringModelOutput(,)
            # ? what is the point of this code...
            start_logits = np.concatenate(start_logits)
            end_logits = np.concatenate(end_logits)
            start_logits = start_logits[: len(self.proc_test_dataset)]
            end_logits = end_logits[: len(self.proc_test_dataset)]
            metrics = self.__answerFromLogit(
                start_logits, end_logits, self.proc_test_dataset, self.test_dataset
            )
            print(f"epoch {epoch}:", metrics)
            f1_score = metrics["f1"]
            accelerator.wait_for_everyone()

            # save the best model
            # @TODO -- model checkpoint saving needs to be refined!!!
            if f1_score > best_f1:
                best_f1 = f1_score
                self.model.save_pretrained("./top_bert.bin")
                self.logger.info("new best model saved!")

        print("Best model f1 = {}".format(best_f1))
        return best_f1

    def debug(self, mode):
        # set debug parameters for epochs
        self.num_epochs = 1

        # load the model Ok
        self.__model_initialization(mode)

        # preprocessing :: Ok
        self.preprocess(self.train_dev_dataset, "training")
        self.preprocess(self.val_dev_dataset, "validation")

        # check fine-tuning
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.logger.info("optimizer set")

        # @NOTE :: implement pretraining/eval/run, then do the same for BART(read fewshot paper and code in detail beforehand)

    # def pretrain():

    # def evaluate():

    # def run():
