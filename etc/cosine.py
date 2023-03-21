import os

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parents[1]))
from load_data import load_oqa, load_tgt

from datasets import Dataset

from sklearn.metrics.pairwise import cosine_similarity


def init_model(checkpoint="distilbert-base-uncased"):
    """intialize model and tokenizer"""

    model = AutoModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return model, tokenizer


def embed_string(target, tokenizer, model):
    tokenized_target = tokenizer(
        target,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    # tokenized_sentence.set_format("torch")

    attn_mask = tokenized_target["attention_mask"]

    outputs = model(**tokenized_target)
    embeddings = outputs["last_hidden_state"]

    # apply attention masks
    attn_mask = attn_mask.unsqueeze(-1).expand(embeddings.size()).float()

    masked_embeddings = embeddings * attn_mask

    # mean pooling... (https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1)
    # figure out what is actually happening here... not sure I understand this operation
    summed_embeddings = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(attn_mask.sum(1), min=1e-9)
    mean_pooled = summed_embeddings / summed_mask

    return tokenized_target, mean_pooled.detach().numpy()


def embed_sentence(examples, model, tokenizer):
    targets = examples["targets"]
    tokenized_sentence, mean_pooled = embed_string(targets, tokenizer, model)

    batch = {k: v for k, v in tokenized_sentence.items()}
    batch["targets"] = examples["targets"]
    batch["embeddings"] = mean_pooled
    batch["id"] = examples["id"]
    batch["context"] = examples["context"]

    return batch


def get_cosine_similarity(answers, sample):
    """take a model and a tuple of sentences as input and return the cosine similarity"""
    sample = sample.reshape(1, -1)

    sim = cosine_similarity(
        sample,  # (1, 768)
        answers,  # (n_answer, 768)
    )

    return sim.max()


def loop_cos(answers, samples):
    c = []
    for s in samples:
        s = s.reshape(1, -1)
        c.append(get_cosine_similarity(answers, s))
    return c


def filter_patterns(answers, target_dataset):
    """for each target pattern, filter by the cosine similarity for a given threshold"""

    # @NOTE :: start with a ratio of 0.7 and look at the output to re-adjust

    model, tokenizer = init_model()
    embedded_targets_dataset = target_dataset.map(
        lambda example: embed_sentence(example, model, tokenizer)
    )

    target_embeddings = np.array(embedded_targets_dataset["embeddings"]).squeeze()

    # slow...
    answers_embeddings = []
    for a in tqdm(answers):
        _, embed = embed_string(a, tokenizer, model)
        answers_embeddings.append(embed.squeeze())

    answers_embeddings = np.array(answers_embeddings)

    # map the cosine similarity to the target embeddings then add the max value column to the dataset and save for
    # filtering according to threshold later

    f = lambda x: get_cosine_similarity(answers_embeddings, x)

    max_cosine_sim = np.apply_along_axis(f, 1, target_embeddings)
    embedded_targets_dataset = embedded_targets_dataset.add_column(
        "cosine_similarity", max_cosine_sim
    )

    # embedded_targets_dataset["max_cos_sim"] = max_cosine_sim

    # @add this to a column
    return embedded_targets_dataset

    # return answers_embeddings, embedded_targets_dataset
    # filter dataset from threshold


def main():
    qa = load_oqa("c:/Users/roum5/source/data/oqa/oqa-v0.4-26feb2023.json")
    tgt_patterns_ds = Dataset.from_dict(
        load_tgt("c:/Users/roum5/source/o-nlp/tmp/ngram-tgt-test.json", n_samples=-1)
    )
    all_answers = qa["answers"]
    cos_fil_ds = filter_patterns(all_answers, tgt_patterns_ds)

    cos_fil_ds.save_to_disk("./targets-cosine")
    # The technique seems to work!!, the threshold may need to be set at around 0.80


if __name__ == "__main__":
    main()
