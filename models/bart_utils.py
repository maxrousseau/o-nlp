#!/usr/bin/env python

###############################################################################
#                            BART helper functions                            #
###############################################################################
import os
import logging
from dataclasses import dataclass
from typing import Any
import collections
import gc

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    get_scheduler,
    DataCollatorForSeq2Seq,
    BartTokenizerFast,
    BartForConditionalGeneration,
)

from datasets import Dataset
import datasets

import pyarrow as pa

import evaluate

metric = evaluate.load("squad")

datasets.utils.logging.set_verbosity_warning

FORMAT = "[%(levelname)s] :: %(asctime)s @ %(name)s :: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("bart-utils")
logger.setLevel(logging.DEBUG)  # change level if debug or not


@dataclass
class BARTCFG:
    name: str = "bart-default"
    lr: float = 2e-5
    n_epochs: int = 12
    lr_scheduler: bool = True
    model_checkpoint: str = ""
    tokenizer_checkpoint: str = ""
    checkpoint_savedir: str = "./bart-ckpt"
    max_seq_length: int = 384
    max_ans_length: int = 128
    stride: int = 128
    padding: str = "max_length"
    seed: str = 0

    load_from_checkpoint: bool = False
    checkpoint_state: str = None
    checkpoint_step: int = None

    train_dataset: Dataset = None
    val_dataset: Dataset = None
    test_dataset: Dataset = None

    val_batches: Any = None
    train_batches: Any = None
    test_batches: Any = None

    model: Any = None
    tokenizer: Any = None

    # TBD add a print/export function to the config when we save model...
    # def __repr__() -> str


def bart_init(model_checkpoint, tokenizer_checkpoint):
    """ """
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = BartTokenizerFast.from_pretrained(tokenizer_checkpoint)

    return model, tokenizer


def bart_format_mi(dataset):
    """take a squad-like qa dataset and transform into MLM format specified in the fewshotBART paper
    "Question: a question? Answer: <mask>. Context: this is the context"
    USAGE:
        train_raw = Dataset.from_dict(formatToMI(dset[2]))
        test_raw = Dataset.from_dict(formatToMI(dset[3]))
        # then you can feed those to the FsBART model class at initialization to run
    """
    gc.disable()
    contexts = pa.array(dataset["context"])
    questions = pa.array(dataset["question"])
    answers = pa.array([i["text"][0] for i in dataset["answers"]])

    masked_strings = pa.compute.binary_join_element_wise(
        "Question: ", questions, " Answer: <mask>. Context: ", contexts, ""
    )
    full_strings = pa.compute.binary_join_element_wise(
        "Question: ", questions, " Answer: ", answers, ". Context: ", contexts, ""
    )
    qa_strings = pa.compute.binary_join_element_wise(
        "Question: ", questions, " Answer: ", answers, ".", ""
    )

    gc.enable()

    return Dataset.from_dict(
        {
            "masked_strings": masked_strings.to_pylist(),
            "full_strings": full_strings.to_pylist(),
            "qa_strings": qa_strings.to_pylist(),
            "answer_strings": answers.to_pylist(),
            "id": dataset["id"],
        }
    )


def preprocess_training(
    examples,
    tokenizer=None,
    padding="max_length",
    max_seq_length=512,
    max_ans_length=256,
):
    """ """
    source, target = examples["masked_strings"], examples["qa_strings"]
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


def preprocess_validation(
    examples, tokenizer=None, padding="max_length", max_seq_length=512
):
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


def clean_outputs(output_ids, tokenizer=None):
    """take the logit outputs from a sample of the seq2seq LM and turn it into a string for evaluation!"""
    out = tokenizer.decode(output_ids, skip_special_tokens=True)
    out = out.lower()

    try:
        answer_start = out.find("answer: ") + 8
        answer_end = out.find(". context")
        answer = out[answer_start:answer_end].strip()
    except:
        answer = ""

    return answer


def prepare_inputs(
    dataset,
    tokenizer=None,
    subset=None,
    padding="max_length",
    max_seq_length=512,
    max_ans_length=None,
):
    if subset == "train":
        # do this
        tokenized_dataset = dataset.map(
            lambda example: preprocess_training(
                example,
                tokenizer=tokenizer,
                padding=padding,
                max_seq_length=max_seq_length,
            ),
            batched=True,
            remove_columns=dataset.column_names,
            keep_in_memory=True,
        )
        logger.info(
            "Training dataset processed and tokenized : n = {}".format(
                len(tokenized_dataset)
            )
        )
    elif subset == "eval":
        tokenized_dataset = dataset.map(
            lambda example: preprocess_validation(
                example,
                tokenizer=tokenizer,
                padding=padding,
                max_seq_length=max_seq_length,
            ),
            batched=True,
            remove_columns=dataset.column_names,
            keep_in_memory=True,
        )
        logger.info(
            "Validation/evaluation dataset processed and tokenized : n = {}".format(
                len(tokenized_dataset)
            )
        )

    else:
        raise Exception("Specify subset for data preparation")

    return tokenized_dataset


def evaluate(eval_outputs, answers):
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

    m = metric.compute(predictions=predicted_answers, references=theoretical_answers)

    return m, predicted_answers, theoretical_answers


def setup_finetune_bart(train_path, val_path, config):
    """"""
    config.train_dataset = bart_format_mi(Dataset.load_from_disk(train_path))
    config.val_dataset = bart_format_mi(Dataset.load_from_disk(val_path))

    logger.info("Training and validation datasets loaded from disk")

    config.model, config.tokenizer = bart_init(
        config.model_checkpoint, config.tokenizer_checkpoint
    )

    logger.info("Model and tokenizer initialized")

    config.train_batches = prepare_inputs(
        config.train_dataset,
        config.tokenizer,
        max_seq_length=config.max_seq_length,
        max_ans_length=config.max_ans_length,
        padding=config.padding,
        subset="train",
    )

    config.val_batches = prepare_inputs(
        config.val_dataset,
        config.tokenizer,
        max_seq_length=config.max_seq_length,
        padding=config.padding,
        subset="eval",
    )

    return config


def setup_evaluate_bart(test_path, config):
    config.test_dataset = bart_format_mi(Dataset.load_from_disk(test_path))

    logger.info("datasets loaded from disk")

    config.model, config.tokenizer = bart_init(
        config.model_checkpoint, config.tokenizer_checkpoint
    )

    logger.info("model and tokenizer initialized")

    config.test_batches = prepare_inputs(
        config.test_dataset,
        config.tokenizer,
        max_seq_length=config.max_seq_length,
        padding=config.padding,
        subset="eval",
    )

    return config
