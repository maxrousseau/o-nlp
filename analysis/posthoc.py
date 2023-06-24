import os
import logging

from models import bert_utils

from train import *

from datasets import Dataset

test_ds_path = "../tmp/oqa_shuffled_split/bin/test"
# C:\Users\roum5\source\o-nlp\tmp\content\pubmedbert-sft-pubmedbert-sft-27-05-2023_15-59-03


def get_answer_length(example):
    a_len = len(example["answers"]["text"][0].split())
    example["answer_length"] = a_len
    return example


# group by answer length and evaluate subgroups
def get_length_sets(dataset):
    # <5 words, 5<=x<=12, 12>
    dataset = dataset.map(get_answer_length)

    short_ans = dataset.filter(lambda example: example["answer_length"] < 5)
    medium_ans = dataset.filter(lambda example: 5 <= example["answer_length"] <= 12)
    long_ans = dataset.filter(lambda example: 12 < example["answer_length"])

    assert len(dataset) == len(short_ans) + len(medium_ans) + len(long_ans)

    length_datasets = {"short": short_ans, "medium": medium_ans, "long": long_ans}

    return length_datasets


# groups by topic and evaluate
def get_topic_sets(dataset):
    topics = sorted(set(dataset["topic"]))
    topic_datasets = {}
    for t in topics:
        ds_topic = dataset.filter(lambda example: example["topic"] == t)
        topic_datasets[t] = ds_topic

    return topic_datasets


# export dataset with answers
def get_predictions(
    dataset_path,
    model_checkpoint="../tmp/content/pubmedbert-sft-pubmedbert-sft-27-05-2023_15-59-03",
):
    config = bert_utils.BERTCFG(
        name="pubmedbert-squad-presft",
        model_checkpoint=model_checkpoint,
        tokenizer_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        max_length=512,
        seed=0,
    )
    # eval
    config = bert_utils.setup_evaluate_oqa(dataset_path, config)
    evaluater = EvaluateBERT(config)

    answers = evaluater(return_answers=True)

    return answers


# implement *SHAP* and explore successes/failures
