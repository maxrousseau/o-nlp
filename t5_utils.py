#!/usr/bin/env python

###############################################################################
#                      T5 Implementation for OQA                              #
###############################################################################
import logging

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    get_scheduler,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from load_data import load_mini_oqa, formatToMI

from datasets import Dataset, load_metric

from dataclasses import dataclass
from typing import Any


FORMAT = "[%(levelname)s] :: %(asctime)s @ %(name)s :: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("t5-utils")
logger.setLevel(logging.DEBUG)  # change level if debug or not


@dataclass
class T5CFG:
    name: str = "t5-default"
    lr: float = 2e-5
    n_epochs: int = 12
    lr_scheduler: bool = True
    model_checkpoint: str = "google/t5-v1_1-base"
    tokenizer_checkpoint: str = "google/t5-v1_1-base"
    checkpoint_savedir: str = "./t5-ckpt"
    max_seq_length: int = 384
    max_ans_length: int = 128
    stride: int = 128
    padding: str = "max_length"
    seed: str = 0

    train_dataset: Dataset = None
    test_dataset: Dataset = None

    val_batches: Any = None
    train_batches: Any = None
    test_batches: Any = None

    model: Any = None
    tokenizer: Any = None
    runmode: str = None

    # TBD add a print/export function to the config when we save model...
    # def __repr__() -> str


def t5_init(model_checkpoint, tokenizer_checkpoint, mode=None):
    """initialize model and tokenizer, mode=Default, LoRA"""

    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_checkpoint)

    if mode == "lora":
        # TODO add lora
        None
    else:
        None

    return model, tokenizer


def preprocess_validation(examples, tokenizer, padding, max_seq_length):
    """ """
    source = examples["masked_strings"]
    source_tokenized = tokenizer(
        source,
        padding=padding,
        max_length=max_seq_length,
        truncation=True,
    )

    batch = {k: v for k, v in source_tokenized.items()}

    batch["example_id"] = examples["id"]

    return batch


def preprocess_training(examples, tokenizer, padding, max_seq_length, max_ans_length):
    """preprocess vaildation for mask infilling QA"""

    # @TODO :: look at fsbart paper for the t5 preprocessing...
    source, target = examples["masked_strings"], examples["answer_strings"]
    source_tokenized = tokenizer(
        source,
        padding=padding,
        max_length=max_seq_length,
        truncation=True,
    )

    batch = {k: v for k, v in source_tokenized.items()}

    target_tokenized = tokenizer(
        target,
        padding=padding,
        max_length=max_ans_length,
        truncation=True,
    )

    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]

    batch["example_id"] = examples["id"]

    return batch


def prepare_inputs(
    examples,
    tokenizer,
    padding=8,
    max_seq_length=None,
    max_ans_length=None,
    subset=None,
    test_size=0.2,
):
    """prepare either the training, validation or test"""

    if subset == "train":
        # @TODO :: check if this works
        tokenized_dataset = examples.map(
            lambda example: preprocess_training(
                example, tokenizer, padding, max_seq_length, max_ans_length
            ),
            batched=True,
            remove_columns=examples.column_names,
        )

        tokenized_dataset = tokenized_dataset.train_test_split(
            test_size=test_size, shuffle=False
        )  # training/test were
        # shuffled prior to loading

        logger.info(
            "{} dataset processed and tokenized, n-train = {}, n-val = {}".format(
                subset, len(tokenized_dataset["train"]), len(tokenized_dataset["test"])
            )
        )

        return tokenized_dataset["train"], tokenized_dataset["test"]

    elif subset == "test":
        tokenized_dataset = examples.map(
            lambda example: preprocess_validation(
                example, tokenizer, padding, max_seq_length
            ),
            batched=True,
            remove_columns=examples.column_names,
        )
        logger.info(
            "{} dataset processed and tokenized, n-test = {}".format(
                subset, len(tokenized_dataset)
            )
        )

        return tokenized_dataset

    else:
        Exception("Specify subset for data preparation")


def clean_outputs(output_ids, tokenizer):
    """take the logit outputs from a sample of the seq2seq LM and turn it into a string for evaluation!"""
    out = tokenizer.decode(output_ids, skip_special_tokens=True)

    # @BUG :: T5 tries to generate too many different masked sequences??, maybe fix with post-processing

    try:
        answer_start = out.find("Answer: ") + 8
        answer_end = out.find("Context")
        answer = out[answer_start:answer_end]
    except:
        answer = ""

    return answer


def evaluate(outputs, target_answers):
    """..."""
    theoretical_answers = []
    predicted_answers = []

    for idx, predicted_answer in outputs:
        label_answer = target_answers.filter(lambda sample: sample["id"] == idx)[
            "answer_strings"
        ]
        theoretical_answers.append(
            {"id": idx, "answers": {"answer_start": [], "text": label_answer}}
        )
        predicted_answers.append({"id": idx, "prediction_text": predicted_answer})

    metric = load_metric("squad")
    m = metric.compute(predictions=predicted_answers, references=theoretical_answers)

    return m, predicted_answers, theoretical_answers


def setup_finetune_t5(train_path, test_path, config):
    """call t5 setup from config, return everything that is necessary for fine-tuning"""
    oqa_dataset = load_mini_oqa(train_path, test_path)

    config.train_dataset = Dataset.from_dict(formatToMI(oqa_dataset[2]))
    config.test_dataset = Dataset.from_dict(formatToMI(oqa_dataset[3]))
    logger.info("Masked QA datasets loaded from file")

    config.model, config.tokenizer = t5_init(
        config.model_checkpoint, config.tokenizer_checkpoint
    )
    logger.info("Model and tokenizers loaded")

    # @TODO :: implement val split here and return both training and validation!!!!
    config.train_batches, config.val_batches = prepare_inputs(
        config.train_dataset,
        config.tokenizer,
        padding=config.padding,
        max_seq_length=config.max_seq_length,
        max_ans_length=config.max_ans_length,
        subset="train",
    )

    config.test_batches = prepare_inputs(
        config.test_dataset,
        config.tokenizer,
        padding=config.padding,
        max_seq_length=config.max_seq_length,
        subset="test",
    )

    return config
