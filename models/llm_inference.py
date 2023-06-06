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
    samples: Any = None  # list of sample prompts


@dataclass(repr=False)
class GpuInference:
    """initialize and run GPU inference"""

    model_name: str = None
    int8: bool = True
    num_samples: int = 12
    # options: T0pp, UnifiedQAv2 or FLAN for the T5 like models
    # otherwise -> GPT2-XL, GPT-J, LLaMa, MPT, etc.

    # @HERE :: start with unifiedqav2-large (int8 inference, should be enough if a good generation strategy is chosen,
    # then scale up as needed)...
