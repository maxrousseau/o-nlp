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
    None
