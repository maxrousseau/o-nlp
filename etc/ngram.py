import os
import collections
import random
import json

# import load_data
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.util import ngrams

from tqdm.auto import tqdm

from bs4 import BeautifulSoup
import requests

import numpy as np


def find_topk_ngrams(sentences, n, k):
    """extract recurring top K n-grams from a list of sentences"""
    sentences = " ".join(sentences)
    tokenized_sentences = [word for word in word_tokenize(sentences) if word.isalpha()]
    ng = ngrams(tokenized_sentences, n)

    ng_freq = collections.Counter(ng)

    return ng_freq.most_common(k)


def find_k_occur_ngrams(sentences, n, k):
    """extract recurring top K n-grams from a list of sentences"""
    sentences = " ".join(sentences)
    tokenized_sentences = [word for word in word_tokenize(sentences) if word.isalpha()]
    ng = ngrams(tokenized_sentences, n)

    ng_freq = collections.Counter(ng)
    ng_k_occur = [
        (key, val) for key, val in dict(ng_freq).items() if val >= k
    ]  # get items n-grams that occur 3 times or more

    return ng_k_occur


def occurs(corpus, ngram):
    """count how many times an n-gram occurs in the reference corpus"""
    occ = 0
    ngram = ngram.lower()
    for p in tqdm(corpus):
        p = p.lower()
        occ += p.count(ngram)

    print("ngram \{}\ occurs {} times in corpus".format(ngram, occ))


def ngrams_to_csv(ngram_dict, path):
    """ """
    for k, v in ngram_dict.items():
        ngram_dict[k] = [" ".join(i[0]) for i in v]

    ng_df = pd.DataFrame.from_dict(ngram_dict)
    ng_df.to_csv(path)


def mask_sample(n_grams, text, max_len):
    """ """


def fulltextFromId(pmcid):
    url = "https://www.ncbi.nlm.nih.gov/pmc/articles/{}/".format(pmcid)
    r = requests.get(url)
    print(r.text)
    soup = BeautifulSoup(r.text, "html.parser")
    article_body = soup.findall("p")
    print(article_body.text)
    # ["documents"][idx]["infons"]["type"] == "paragraph"


def apply_random_mask(inputstr, ngram, seed=0, lam=8):
    """gross but works"""
    # @TODO :: set seed
    n_words = np.random.poisson(lam)
    direction = random.randint(0, 1)

    sentences = sent_tokenize(inputstr)
    tok_ng = word_tokenize(ngram)

    masked_context = []

    for s in sentences:
        if s.find("TGT") >= 0:
            s = word_tokenize(s)
            idx = s.index("TGT")
            del s[idx]

            if idx == 0:
                direction = 0

            if direction == 0:
                if len(s[idx + len(tok_ng) :]) > n_words:
                    m = s[idx + len(tok_ng) : idx + len(tok_ng) + n_words]
                    s = (
                        s[: idx + len(tok_ng)]
                        + ["[MASK]"]
                        + s[idx + len(tok_ng) + n_words :]
                    )
                else:
                    m = s[idx + len(tok_ng) : -1]
                    s = s[: idx + len(tok_ng)] + ["[MASK]"] + s[-1:]

            else:
                if len(s[:idx]) > n_words:
                    m = s[idx - n_words : idx]
                    s = s[: idx - n_words] + ["[MASK]"] + s[idx:]
                else:
                    m = s[:idx]
                    s = ["[MASK]"] + s[idx:]

            mask_target = TreebankWordDetokenizer().detokenize(m)
            masked_context.append(TreebankWordDetokenizer().detokenize(s))
        else:
            masked_context.append(s)

    masked_context = " ".join(masked_context)

    return (masked_context, mask_target)


def get_target_patterns(corpus, kgrams=None, avg_len=12):
    """
    Create target pattern dataset. For each article search for all occurences of patterns, for each occurence mask n-tokens
        according to the poisson distribution (avg_len, sd_len). Mask either the following of preceding tokens.
    """

    # @BUG :: a single chunk may contain several patterns, should there only be one?...
    max_context_len = 384

    chunks = []  # tokenize sentences
    chunk_ids = []
    count = 0

    for d in tqdm(corpus):
        chunk = []
        sentences = sent_tokenize(d)
        t = 0
        for s in sentences:
            t += len(word_tokenize(s))
            if t < max_context_len:
                chunk.append(s)
            else:
                chunks.append(" ".join(chunk))
                chunk_ids.append(count)
                count += 1
                t = 0
                chunk = []

        # if len(chunks) == 40:
        #     print("done")
        #     break

    masked_patterns = []
    masked_ids = []

    for ng in tqdm(kgrams):
        for c, c_id in list(zip(chunks, chunk_ids)):
            gms = list(
                " ".join(grams)
                for grams in ngrams(word_tokenize(c), len(word_tokenize(ng)))
            )
            if ng in gms:
                # apply random mask
                idx = c.find(ng)
                c.lower()
                c = c[:idx] + " TGT " + c[idx:]
                try:
                    masked_c, mask_tgt = apply_random_mask(c, ng, lam=avg_len)
                except:
                    print(ng)
                    print(c)
                masked_patterns.append(
                    {"context": masked_c, "target": mask_tgt, "id": c_id}
                )
            else:
                continue

    return masked_patterns


def cosine_sim():
    """filter targets with cosine similarity"""


def main():
    # get corpus path
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).absolute().parents[1]))
    from load_data import loadCorpus

    corpus, _ = loadCorpus(
        "c:/Users/roum5/source/data/angle_corpus_v1.0/parsed_papers_6063.json"
    )

    ngram_list = pd.read_csv("./etc/ngrams.csv")["ngrams"].to_list()

    tgts = get_target_patterns(corpus, kgrams=ngram_list, avg_len=12)

    with open("./ngram-tgt-test.json", "w", encoding="utf-8") as f:
        json.dump(tgts, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
