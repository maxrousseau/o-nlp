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
from peft import LoraConfig, get_peft_model, TaskType

from load_data import load_mini_oqa, t5_format_mi

from datasets import Dataset, load_metric
import datasets

datasets.utils.logging.set_verbosity_warning

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
    lora: bool = False

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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(
        "trainable params: {} || all params: {} || trainable%: {}".format(
            trainable_params, all_param, (100 * trainable_params / all_param)
        )
    )


def t5_init(model_checkpoint, tokenizer_checkpoint, mode=None, lora=False):
    """initialize model and tokenizer, mode=Default, LoRA"""

    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_checkpoint)

    if lora:
        # TODO add lora
        # hyperparams from:
        # https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq.ipynb
        # peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32,
        # lora_dropout=0.1)
        # IMPORTANT: loss should be increased and hyperparams explored for LoRA (start with 2e-4)
        # IDK if this is truly worth exploring... maybe better to focus of further pre-training methods.
        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
        )

        model = get_peft_model(model, config)
    print_trainable_parameters(model)

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
    out = tokenizer.decode(output_ids, skip_special_tokens=False)

    # @BUG :: T5 tries to generate too many different masked sequences??, maybe fix with post-processing

    try:
        answer_start = out.find("<extra_id_0>") + 12
        answer_end = out.find("<extra_id_1>")
        # answer = answer[:answer_end]
        answer = out[answer_start:answer_end]
    except:
        answer = ""

    stop_token = answer.find("</s>")
    if stop_token != -1:
        answer = answer[:stop_token]

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

    config.train_dataset = Dataset.from_dict(t5_format_mi(oqa_dataset[2]))
    config.test_dataset = Dataset.from_dict(t5_format_mi(oqa_dataset[3]))
    logger.info("Masked QA datasets loaded from file")

    config.model, config.tokenizer = t5_init(
        config.model_checkpoint, config.tokenizer_checkpoint, lora=config.lora
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
