import os, json

import numpy as np

from collections import Counter

import pyarrow as pa

from nltk.tokenize import word_tokenize

# @TODO :: implement and compute those statistics


def ngram_overlap():
    """compute the overlap in topK ngrams between OQA and SQuAD"""


def type_token_ratio(tokens):
    """compute the complexity of a given answer
    https://en.wikipedia.org/wiki/Lexical_density"""
    types = Counter(tokens)
    return (len(types) / len(tokens)) * 100


def compute_ari():
    """https://en.wikipedia.org/wiki/Automated_readability_index"""
    return None


def qa_stats(contexts, questions, answers):

    q_len = [len(word_tokenize(q.lower())) for q in questions]
    a_len = [len(word_tokenize(a.lower())) for a in answers]
    c_len = [len(word_tokenize(c.lower())) for c in contexts]

    q_len_mu, q_len_sd = np.mean(q_len), np.std(q_len)
    a_len_mu, a_len_sd = np.mean(a_len), np.std(a_len)
    c_len_mu, c_len_sd = np.mean(c_len), np.std(c_len)

    # q_ttr = [type_token_ratio(word_tokenize(q.lower())) for q in questions]
    # a_ttr = [type_token_ratio(word_tokenize(a.lower())) for a in answers]
    # c_ttr = [type_token_ratio(word_tokenize(c.lower())) for c in contexts]

    # q_ttr_mu, q_ttr_sd = np.mean(q_ttr), np.std(q_ttr)
    # a_ttr_mu, a_ttr_sd = np.mean(a_ttr), np.std(a_ttr)
    # c_ttr_mu, c_ttr_sd = np.mean(c_ttr), np.std(c_ttr)

    stats = {
        "questions": (q_len_mu, q_len_sd),
        "answers": (a_len_mu, a_len_sd),
        "contexts": (c_len_mu, c_len_sd),
        # "question_ttr": (q_ttr_mu, q_ttr_sd),
        # "answer_ttr": (a_ttr_mu, a_ttr_sd),
        # "context_ttr": (c_ttr_mu, c_ttr_sd),
    }

    return stats


def main():
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).absolute().parents[1]))

    # from load_data import loadSquadMI
    from datasets import load_dataset
    from load_data import load_oqa_full

    results = {}

    # squad stats
    squad = load_dataset("squad")
    contexts = pa.array(squad["train"]["context"]).to_pylist()
    questions = pa.array(squad["train"]["question"]).to_pylist()
    answers = pa.array([i["text"][0] for i in squad["train"]["answers"]]).to_pylist()
    results["squad"] = qa_stats(contexts, questions, answers)

    # oqa stats
    oqa = load_oqa_full("c:/Users/roum5/source/data/oqa/oqa-v0.4-26feb2023.json")
    contexts = oqa["context"]
    questions = oqa["question"]
    answers = oqa["answers"]

    results["oqa"] = qa_stats(contexts, questions, answers)

    import pprint

    pprint.pprint(results)

    # dump
    with open("./stats-test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
