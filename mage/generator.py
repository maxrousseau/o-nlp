import torch

import datasets
from datasets import Dataset

from typing import Any

from transformers import AutoTokenizer

from tqdm.auto import tqdm

import json

from dataclasses import dataclass

from setfit import SetFitModel

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

    target_dataset = {
        "question": [],
        "text": [],
        "target": [],
    }

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
        queries = {"target": [], "question": [], "qs": []}
        text = " ".join(sentences)
        for s in tqdm(sentences):
            self.qa_dataset.map(
                lambda x: queries["qs"].append(" ".join((x["question"], s))),
                batched=False,
            )
            for q in self.qa_dataset["question"]:
                queries["qs"].append(" ".join((q, s)))
                queries["question"].append(q)
                queries["target"].append(s)

        is_relevant = self.sentence_classifier(queries["qs"])
        for i, rel in enumerate(is_relevant):
            if rel:
                self.target_dataset["question"].append(queries["question"][i])
                self.target_dataset["text"].append(text)
                self.target_dataset["target"].append(queries["target"][i])

    def process_chunks(self, chunks):
        for chunk in tqdm(chunks):
            chunk = self.parser(
                self.tokenizer.decode(chunk["input_ids"], skip_special_tokens=True)
            )
            chunk_sentences = [s.text for s in chunk.doc.sents]
            self.find_targets(chunk_sentences)
            break
            # self.find_targets(chunk, self.qa_dataset["question"], self._lambda)

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

        return chunked_corpus


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
    corpus = corpus.select(range(100))
    oqa = Dataset.load_from_disk("../tmp/bin/train")

    sentence_classifier = SetfitModelAccelerate.from_pretrained(
        "../tmp/setfit-pubmedbert/setfit-pubmedbert-07-05-2023-85vacc"
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
    test_c = tadam()


# if __name__ == "__main__":
#    test_case()
