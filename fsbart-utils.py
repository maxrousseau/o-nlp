"""

Major rewrite needed, in here place all the general functions that will be needed for the preparation of the model

preprocess
prepare
set_seeds
clean_output
eval

Then in another file, write the pf

"""
import os

import shutil
import time
import random
import logging

from dataclasses import dataclass

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

FORMAT = "[%(levelname)s] :: %(asctime)s @ %(name)s :: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("bart")
logger.setLevel(logging.DEBUG)  # change level if debug or not


class FsbartConfig:
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


# fix seed
