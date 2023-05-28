from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
import numpy as np


oqa = Dataset.load_from_disk("../tmp/oqa_shuffled_split/bin/train")

questions = oqa["question"]


def get_tfidf_similarity(oqa):
    qc_sim = []
    for i, sample in enumerate(oqa):
        tfidf_corpus = questions + [sample["context"]]
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(tfidf_corpus)
        pairwise_similarity = tfidf * tfidf.T
        sim_array = pairwise_similarity.toarray()
        np.fill_diagonal(sim_array, np.nan)

        qc_sim.append(sim_array[-1, i])
    return qc_sim


# np.array(sim).mean() -> 0.2953277673385341
# np.array(sim).min() -> 0.047923200009277425
# np.array(sim).max() -> 0.6541878987241408
# np.array(sim).std() -> 0.1198281853894279


def get_max_tfidf_similarity(oqa):
    qc_sim = []
    for i, sample in enumerate(oqa):
        tfidf_corpus = questions + [sample["context"]]
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(tfidf_corpus)
        pairwise_similarity = tfidf * tfidf.T
        sim_array = pairwise_similarity.toarray()
        np.fill_diagonal(sim_array, np.nan)

        max_idx = np.nanargmax(sim_array[-1])
        max_val = sim_array[-1, max_idx]

        qc_sim.append(max_val)
    return qc_sim
