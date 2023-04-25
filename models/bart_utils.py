#!/usr/bin/env python

###############################################################################
#                            BART helper functions                            #
###############################################################################
import os
import logging
from dataclasses import dataclass
from typing import Any
import collections

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    get_scheduler,
    DataCollatorForSeq2Seq,
)

from datasets import Dataset
import datasets

from evaluate import load

metric = load("squad")

datasets.utils.logging.set_verbosity_warning


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
    runmode: str = None

    # TBD add a print/export function to the config when we save model...
    # def __repr__() -> str


def bart_init():
    """ """
    None


def preprocess_training():
    """ """
    None


def preprocess_validation():
    """ """
    None


def clean_outputs():
    None


def prepare_inputs():
    None


def evaluate():
    None
