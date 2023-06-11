import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List
import collections

import numpy as np

from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import datasets
from datasets import Dataset, load_dataset

from accelerate import init_empty_weights

import torch


# https://huggingface.co/spaces/evaluate-metric/f1 just implement a simple evaluate metric
"""
short term goal:
1. implement t5 int8 inference (dataloader, prompt formating, f1)
2. find a good generation strategy
3. figure out maximum F1 on the training set for n=5, 10, 20 with unifiedqav2-large (If close to 100%, move to next step)
"""


class Prompt:
    def __init__(self, fmt, dataset):
        self.fmt = fmt  # ICL, QA, Instruc
        self.dataset = dataset
        self.samples = {"answer": [], "prompt": []}

    def _t0(self, example):
        """
        https://github.com/bigscience-workshop/promptsource/blob/main/promptsource/templates/squad/templates.yaml"""
        question = example["question"]
        context = example["context"]
        self.samples["answer"].append(example["answers"]["text"][0])
        self.samples["prompt"].append(
            f"""
Refer to the passage below and answer the following question:\n\nPassage: {context}\n\nQuestion: {question}
        """
        )

    def _uniqa(self, example):
        """ """
        question = example["question"]
        context = example["context"]
        self.samples["answer"].append(example["answers"]["text"][0])
        self.samples["prompt"].append(f"{question}\n{context}")

    def _flan(self, example):
        """ """
        question = example["question"]
        context = example["context"]
        self.samples["answer"].append(example["answers"]["text"][0])
        self.samples["prompt"].append(
            f"Read this and answer the question.\n\n{context}\n\n{question}"
        )

    def parse(self):
        """format inputs and return as a dataset with the correct prompt format for generation"""
        for e in self.dataset:
            eval("self._" + self.fmt + "(e)")

        return Dataset.from_dict(self.samples)


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
