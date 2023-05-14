import torch

import datasets
from datasets import Dataset

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

from transformers import AutoTokenizer

from tqdm.auto import tqdm

import json

from dataclasses import dataclass

# from ..models import custom_datacollator

"""
given a corpus, a QA dataset and a sentence classifier: generate a masked span filling dataset

1. chunk corpus into max_context_length
2. process each chunk and segment sentences (Spacy), concat with questions and use setfit to create samples (append to target_dataset)
3. tokenize dataset (pair question and text, generate the mask_mappings)
4. apply mask with the mage_collator
5. return the processed dataset (pytorch format or dataset format)


Do for 100 papers from the corpus and then scale when cleaned up. Allow to save the generated datasets.
"""


@dataclass
class TadamDatasetGenerator:
    corpus: datasets.arrow_dataset.Dataset
    qa_dataset: datasets.arrow_dataset.Dataset
    tokenizer: Any = None  # change to HF tokenizer
    sentence_classifier: Any = None  # change to HF classifier!

    _lambda: int = 0

    max_context_length: int = 384
    max_question_length: int = 128
    max_sequence_length: int = 512

    mask_sentence_only: bool = True

    queries = {"target": [], "question": [], "qs": [], "text": []}

    target_dataset = {"question": [], "text": [], "target": [], "target_start": []}

    def tokenize_corpus(self, examples):
        """tokenize the pretraining corpora"""
        inputs = self.tokenizer(examples["text"], padding=False, truncation=False)
        if self.tokenizer.is_fast:
            inputs["word_ids"] = [
                inputs.word_ids(i) for i in range(len(inputs["input_ids"]))
            ]
        return inputs

    def chunk_corpus(self, examples):
        """chunking the corpus for pretraining"""

        chunk_size = self.max_context_length

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size

        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column

        result["labels"] = result["input_ids"].copy()
        # we need this when we corrupt our input_ids
        return result

    def find_targets(self, sentences):

        text = " ".join(sentences)
        questions = self.qa_dataset["question"]
        tfidf_corpus = questions + [text]

        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(tfidf_corpus)
        pairwise_similarity = tfidf * tfidf.T
        sim_array = pairwise_similarity.toarray()
        np.fill_diagonal(sim_array, np.nan)

        max_idx = np.nanargmax(sim_array[-1])
        max_val = sim_array[-1, max_idx]

        if max_val > 0.2:
            for s in sentences:
                self.queries["qs"].append(" ".join((questions[max_idx], s)))
                self.queries["question"].append(questions[max_idx])
                self.queries["target"].append(s)
                self.queries["text"].append(text)

    def get_relevant_pairs(self):
        is_relevant = self.sentence_classifier(self.queries["qs"])
        for i, rel in enumerate(is_relevant):
            if rel:
                self.target_dataset["question"].append(self.queries["question"][i])
                self.target_dataset["text"].append(self.queries["text"][i])
                self.target_dataset["target_start"].append(
                    self.queries["text"][i].find(self.queries["target"][i])
                )
                self.target_dataset["target"].append(self.queries["target"][i])

    def process_chunks(self, chunks):
        for chunk in tqdm(chunks):
            chunk = self.parser(
                self.tokenizer.decode(chunk["input_ids"], skip_special_tokens=True)
            )
            chunk_sentences = [s.text for s in chunk.doc.sents]
            self.find_targets(chunk_sentences)
            # self.find_targets(chunk, self.qa_dataset["question"], self._lambda)

    def find_offset(self, offset_mapping, k, idx):
        for p, i in enumerate(offset_mapping):
            if (i[0] + i[1]) != 0 and i[k] == idx:
                return p

    def mask_mapping(self, example):
        """Here I need the the sentence start position in the context"""

        inputs = self.tokenizer(
            example["question"],
            example["text"],
            max_length=self.max_sequence_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
        )

        if self.tokenizer.is_fast:
            inputs["word_ids"] = [i for i in range(len(inputs["input_ids"]))]

        inputs["mask_mappings"] = []

        start_pos_sentence = example["target_start"]
        end_pos_sentence = start_pos_sentence + len(example["target"])

        start_pos_question = 0
        end_pos_question = len(example["question"])

        start_sentence_mapping = self.find_offset(
            inputs["offset_mapping"], 0, start_pos_sentence
        )

        end_sentence_mapping = self.find_offset(
            inputs["offset_mapping"], 1, end_pos_sentence
        )

        start_question_mapping = self.find_offset(
            inputs["offset_mapping"], 0, start_pos_question
        )
        end_question_mapping = self.find_offset(
            inputs["offset_mapping"], 1, end_pos_question
        )

        mask_mappings = [0] * 512
        num_sentence_tokens = end_sentence_mapping - start_sentence_mapping
        num_question_tokens = end_question_mapping - start_question_mapping

        mask_mappings = (
            mask_mappings[:start_sentence_mapping]
            + [1] * (num_sentence_tokens + 1)
            + mask_mappings[end_sentence_mapping + 1 :]
        )
        mask_mappings = (
            mask_mappings[:start_question_mapping]
            + [1] * (num_question_tokens + 1)
            + mask_mappings[end_question_mapping + 1 :]
        )
        assert len(mask_mappings) == 512

        inputs["mask_mappings"].append(mask_mappings)

        return inputs

    def __call__(self):
        # returns the tokenized/collated mask_dataset { "id", "text", "labels", "word_ids", ...}

        tokenized_corpus = self.corpus.map(
            self.tokenize_corpus,
            batched=True,
            remove_columns=["text", "article_id", "page"],
        )

        chunked_corpus = tokenized_corpus.map(self.chunk_corpus, batched=True)

        import spacy

        self.parser = spacy.load("en_core_web_sm")
        self.process_chunks(chunked_corpus)
        print("{} possible masking targets".format(len(self.queries["question"])))
        self.get_relevant_pairs()
        print(
            "{} masking samples generated".format(len(self.target_dataset["question"]))
        )

        self.target_dataset = Dataset.from_dict(self.target_dataset)
        self.target_dataset.save_to_disk("tacoma-angle_oqa")

        tokenized_mask_dataset = self.target_dataset.map(
            self.mask_mapping,
            batched=False,
            remove_columns=self.target_dataset.column_names,
        )

        return tokenized_mask_dataset


def test_case():
    import sys

    sys.path.append("../")
    from models.setfit_utils import SetfitModelAccelerate
    from load_data import load_corpus

    corpus = load_corpus("../tmp/angle_orthodontist_corpus.json")
    corpus = Dataset.from_dict(
        {
            "article_id": [x.get("article_id") for x in corpus],
            "page": [x.get("page") for x in corpus],
            "text": [x.get("text") for x in corpus],
        }
    )
    corpus = corpus
    oqa = Dataset.load_from_disk("../tmp/oqa_v1.0_shuffled_split/bin/train")

    sentence_classifier = SetfitModelAccelerate.from_pretrained(
        "../tmp/setfit-pubmedbert-07-05-2023", batch_size=16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )

    # chunk it up
    tadam = TadamDatasetGenerator(
        corpus=corpus,
        qa_dataset=oqa,
        sentence_classifier=sentence_classifier,
        tokenizer=tokenizer,
    )
    tadam()


#    return tadam


if __name__ == "__main__":
    test_case()
