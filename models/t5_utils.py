#!/usr/bin/env python

###############################################################################
#                      T5 Implementation for OQA                              #
###############################################################################
import logging
import gc

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    get_scheduler,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from load_data import load_mini_oqa, t5_format_mi, denoising_format, load_tgt

from datasets import Dataset, load_metric, load_dataset
import datasets

datasets.utils.logging.set_verbosity_warning

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

import evaluate

metric = evaluate.load("squad")

datasets.utils.logging.set_verbosity_warning


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
    model_checkpoint: str = ""
    tokenizer_checkpoint: str = ""
    checkpoint_savedir: str = "./t5-ckpt"
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
    runmode: str = None

    def __repr__(self) -> str:
        s = """
T5 model configuration
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

    # TBD add a print/export function to the config when we save model...
    # def __repr__() -> str


def t5_format_mi(dataset):
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
        "Question: ", questions, " Answer: <extra_id_0> Context: ", contexts, ""
    )
    # Important not to include the "." character at the end of the answer otherwise the model generates double dots
    target_answers = pa.compute.binary_join_element_wise(
        "<extra_id_0> ", answers, "<extra_id_1>", ""
    )

    gc.enable()

    return Dataset.from_dict(
        {
            "masked_strings": masked_strings.to_pylist(),
            "answer_strings": target_answers.to_pylist(),
            "id": dataset["id"],
        }
    )


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


def t5_init(model_checkpoint, tokenizer_checkpoint):
    """initialize model and tokenizer, mode=Default, LoRA"""

    model = T5ForConditionalGeneration.from_pretrained(
        model_checkpoint, torch_dtype=torch.bfloat16
    )
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_checkpoint)

    # if lora:
    #     # TODO add lora
    #     # hyperparams from:
    #     # https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq.ipynb
    #     # peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32,
    #     # lora_dropout=0.1)
    #     # IMPORTANT: loss should be increased and hyperparams explored for LoRA (start with 2e-4)
    #     # IDK if this is truly worth exploring... maybe better to focus of further pre-training methods.
    #     for param in model.parameters():
    #         param.requires_grad = False  # freeze the model - train adapters later
    #         if param.ndim == 1:
    #             # cast the small parameters (e.g. layernorm) to fp32 for stability
    #             param.data = param.data.to(torch.float32)

    #     config = LoraConfig(
    #         r=8,
    #         lora_alpha=32,
    #         lora_dropout=0.1,
    #         bias="none",
    #         task_type=TaskType.SEQ_2_SEQ_LM,
    #         inference_mode=False,
    #     )

    #     model = get_peft_model(model, config)
    # print_trainable_parameters(model)

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
            keep_in_memory=True,
        )

        logger.info(
            "{} dataset processed and tokenized, n = {}".format(
                subset, len(tokenized_dataset)
            )
        )

        return tokenized_dataset

    elif subset == "eval":
        tokenized_dataset = examples.map(
            lambda example: preprocess_validation(
                example, tokenizer, padding, max_seq_length
            ),
            batched=True,
            remove_columns=examples.column_names,
            keep_in_memory=True,
        )
        logger.info(
            "{} dataset processed and tokenized, n = {}".format(
                subset, len(tokenized_dataset)
            )
        )

        return tokenized_dataset

    else:
        Exception("Specify subset for data preparation")


def preprocess_denoising(examples, tokenizer, padding, max_seq_length, max_ans_length):
    """preprocess for"""

    # @TODO :: look at fsbart paper for the t5 preprocessing...
    source, target = examples["masked_strings"], examples["target_strings"]
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


def prepare_inputs_denoising(
    examples,
    tokenizer,
    padding=8,
    max_seq_length=None,
    max_ans_length=None,
    test_size=0.05,
):
    """ """
    tokenized_dataset = examples.map(
        lambda example: preprocess_denoising(
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
        "Pretraining dataset processed and tokenized, n-train = {}, n-val = {}".format(
            len(tokenized_dataset["train"]), len(tokenized_dataset["test"])
        )
    )

    return tokenized_dataset["train"], tokenized_dataset["test"]


# @TODO :: vectorize this!!! quite slow...
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

    m = metric.compute(predictions=predicted_answers, references=theoretical_answers)

    return m, predicted_answers, theoretical_answers


def evaluate_pretraining(outputs, target_answers):
    """..."""
    theoretical_answers = []
    predicted_answers = []

    for idx, predicted_answer in outputs:
        label_answer = target_answers.filter(lambda sample: sample["id"] == idx)[
            "target_strings"
        ]
        theoretical_answers.append(
            {"id": idx, "answers": {"answer_start": [], "text": label_answer}}
        )
        predicted_answers.append({"id": idx, "prediction_text": predicted_answer})

    metric = load_metric("squad")
    m = metric.compute(predictions=predicted_answers, references=theoretical_answers)

    return m, predicted_answers, theoretical_answers


def setup_finetune_t5(dataset_repo, config):
    """call t5 setup from config, return everything that is necessary for fine-tuning"""
    # @HERE :: fix dataset loading and preprocessing to remove the __get_val_answers() method from FinetuneT5

    oqa = load_dataset(dataset_repo)
    config.train_dataset = t5_format_mi(oqa["train"])
    config.val_dataset = t5_format_mi(oqa["validation"])

    logger.info("Masked QA datasets loaded")

    config.model, config.tokenizer = t5_init(
        config.model_checkpoint, config.tokenizer_checkpoint
    )
    logger.info("Model and tokenizers loaded")

    config.train_batches = prepare_inputs(
        config.train_dataset,
        config.tokenizer,
        padding=config.padding,
        max_seq_length=config.max_seq_length,
        max_ans_length=config.max_ans_length,
        subset="train",
    )

    config.val_batches = prepare_inputs(
        config.val_dataset,
        config.tokenizer,
        padding=config.padding,
        max_seq_length=config.max_seq_length,
        subset="eval",
    )

    return config


def setup_evaluate_t5(dataset_repo, config):
    """call t5 setup from config, return everything that is necessary for fine-tuning"""
    # @HERE :: fix dataset loading and preprocessing to remove the __get_val_answers() method from FinetuneT5
    oqa = load_dataset(dataset_repo)
    config.test_dataset = t5_format_mi(oqa["test"])

    logger.info("Test dataset loaded from disk and formatted to mask-filling")

    config.model, config.tokenizer = t5_init(
        config.model_checkpoint, config.tokenizer_checkpoint
    )
    logger.info("Model and tokenizers loaded")

    config.test_batches = prepare_inputs(
        config.test_dataset,
        config.tokenizer,
        padding=config.padding,
        max_seq_length=config.max_seq_length,
        subset="eval",
    )

    return config


def setup_pretrain_t5(data_path, config):
    """call t5 setup from config, return everything that is necessary for fine-tuning"""
    config.train_dataset = Dataset.load_from_disk(data_path).shuffle(seed=0)
    config.train_dataset = denoising_format(config.train_dataset)
    # config.train_dataset = denoising_format(config.train_dataset.select(range(100)))

    logger.info("Masked tgt datasets loaded from file")

    config.model, config.tokenizer = t5_init(
        config.model_checkpoint, config.tokenizer_checkpoint, lora=False
    )
    logger.info("Model and tokenizers loaded")

    # @TODO :: implement val split here and return both training and validation!!!!
    config.train_batches, config.val_batches = prepare_inputs_denoising(
        config.train_dataset,
        config.tokenizer,
        padding=config.padding,
        max_seq_length=config.max_seq_length,
        max_ans_length=config.max_ans_length,
    )

    config.test_batches = None

    return config
