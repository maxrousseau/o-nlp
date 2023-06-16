import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List
import collections

import numpy as np

from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

import datasets
from datasets import Dataset, load_dataset

from accelerate import init_empty_weights, Accelerator

import torch
from torch.utils.data import DataLoader

import re
import string
from collections import Counter


# https://huggingface.co/spaces/evaluate-metric/f1 just implement a simple evaluate metric
"""
short term goal:
1. implement t5 int8 inference (dataloader, prompt formating, f1)
2. find a good generation strategy
3. figure out maximum F1 on the training set for n=5, 10, 20 with unifiedqav2-large (If close to 100%, move to next step)
"""


class GpuInference:
    """initialize and run GPU inference


    takes a dataset as input, transforms it into a set of prompts, perform inference with the given strategy and returns
    the input dataset with the generated results


    """

    def __init__(
        self,
        model_checkpoint=None,
        tokenizer_checkpoint=None,
        dataset=None,
        int8=True,
        num_samples=4,
        prompt_fmt="uniqa",
    ):

        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.int8 = int8
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prompt_fmt = prompt_fmt  # ICL, QA, Instruc
        self.dataset = dataset
        self.samples = {"answer": [], "prompt": [], "id": []}

        # options: T0pp, UnifiedQAv2 or FLAN for the T5 like models
        # otherwise -> GPT2-XL, GPT-J, LLaMa, MPT, etc.

        # @HERE :: start with unifiedqav2-large (int8 inference, should be enough if a good generation strategy is chosen,
        # then scale up as needed).

    def _t0(self, example):
        """
        https://github.com/bigscience-workshop/promptsource/blob/main/promptsource/templates/squad/templates.yaml"""
        question = example["question"]
        context = example["context"]
        self.samples["id"].append(example["id"])
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
        self.samples["id"].append(example["id"])
        self.samples["answer"].append(example["answers"]["text"][0])
        self.samples["prompt"].append(f"{question}\n{context}")

    def _flan(self, example):
        """ """
        question = example["question"]
        context = example["context"]
        self.samples["id"].append(example["id"])
        self.samples["answer"].append(example["answers"]["text"][0])
        self.samples["prompt"].append(
            f"Read this and answer the question.\n\n{context}\n\n{question}"
        )

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

    def __tokenize(
        self, examples, tokenizer=None, padding="max_length", max_seq_length=512
    ):
        """preprocess vaildation for mask infilling QA"""

        # @TODO :: look at fsbart paper for the t5 preprocessing...
        source = examples["prompt"]
        source_tokenized = tokenizer(
            source,
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )

        batch = {k: v for k, v in source_tokenized.items()}

        batch["example_id"] = examples["id"]

        return batch

    def __get_dataloader(self, input_data, tokenizer, model):
        """simple dataloader for each sample"""

        input_tensor = input_data.remove_columns(["example_id"])
        input_tensor.set_format("torch")
        # create the dataloaders

        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        dataloader = DataLoader(
            input_tensor,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=1,
        )

        return dataloader

    def __compute_f1(self, sampled_outputs, answer):
        """get the F1 per generated batch for a given example"""

    def get_prompts(self):
        """ """
        for e in self.dataset:
            eval("self._" + self.prompt_fmt + "(e)")

        return Dataset.from_dict(self.samples)

    def __normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.__normalize_answer(prediction).split()
        ground_truth_tokens = self.__normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @torch.no_grad()
    def genseq(self, prompts):
        """generate sequences per batch"""

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map="auto",
        )

        tokenized_dataset = prompts.map(
            lambda example: self.__tokenize(
                example,
                tokenizer=self.tokenizer,
                padding="max_length",
                max_seq_length=1024,
            ),
            batched=True,
            remove_columns=prompts.column_names,
            keep_in_memory=True,
        )

        dataloader = self.__get_dataloader(
            tokenized_dataset, tokenizer=self.tokenizer, model=self.model
        )

        seq_outputs = []

        accelerator = Accelerator()
        (self.model, dataloader) = accelerator.prepare(self.model, dataloader)

        self.model.eval()
        for steps, batch in enumerate(tqdm(dataloader)):
            outputs = self.model.generate(
                **batch,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=8,
            )

            seq_outputs.append(outputs)

        seqs = {"answer" : [], "predictions" : []}

        for i in range(len(seq_outputs)):
            answer = self.samples["answer"][i]
            predictions = [self.tokenizer.decode(x, skip_special_tokens=True) for x in seq_outputs[i]]
            scores = []
            for p in predictions:
                scores.append(self.f1_score(p, answer))
            seqs["answer"].append(answer)
            seqs["predictions"].append(list(zip(predictions, scores)))


        return seqs

        # __run()

        # seqs = None

        # return seqs
