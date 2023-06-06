import logging
from dataclasses import dataclass
from typing import Any
import collections

import numpy as np

from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import datasets
from datasets import Dataset, load_dataset

from evaluate import load


@dataclass(repr=False)
class Prompt:
    fmt: str = None  # ICL, QA, Instruct
    samples: Any  # list of sample prompts


@dataclass(repr=False)
class GpuInference:
    """initialize and run GPU inference"""
