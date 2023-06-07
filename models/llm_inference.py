import logging
from dataclasses import dataclass
from typing import Any
import collections

import numpy as np

from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import datasets
from datasets import Dataset, load_dataset

from accelerate import init_empty_weights

import torch


@dataclass(repr=False)
class Prompt:
    fmt: str = None  # ICL, QA, Instruct
    samples: Any = None  # list of sample prompts

    def _t0(self):
        """ """

    def _uniqa(self):
        """ """

    def _flan(self):
        """ """

    def parse(self):
        """format inputs and return as a dataset with the correct prompt format for generation"""
        return None


@dataclass(repr=False)
class GpuInference:
    """initialize and run GPU inference"""

    model_checkpoint: str = None
    tokenizer_checkpoint: str = None
    int8: bool = True
    num_samples: int = 12
    # options: T0pp, UnifiedQAv2 or FLAN for the T5 like models
    # otherwise -> GPT2-XL, GPT-J, LLaMa, MPT, etc.

    # @HERE :: start with unifiedqav2-large (int8 inference, should be enough if a good generation strategy is chosen,
    # then scale up as needed).

    def __get_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)
        # @ init_empty_weights? # bnb!
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_checkpoint,
            device_map="auto",
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )

    def __get_dataloader(self):
        """simple dataloader for each sample"""
        return None

    def __compute_f1(self, sampled_outputs, answer):
        """get the F1 per generated batch for a given example"""

    @torch.no_grad()
    def genseq(self):
        """generate sequences per batch"""
        return None
