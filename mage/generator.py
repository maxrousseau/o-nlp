import torch

from dataclasses import dataclass

from ..models import custom_datacollator

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
class MageDatasetGenerator:
    corpus = None
    qa_dataset = None
    sentence_classifier = None

    _lambda = None

    max_context_length = 384
    max_question_length = 128
    max_sequence_length = 512

    mask_sentence_only = True

    tokenizer = None

    target_dataset = {
        "id": [],
        "question": [],
        "text": [],
        "target": [],
    }

    def chunk_corpus(self):
        """ """
        self.chunks
        None

    def find_targets(self):
        None

    def process_chunks(self):
        for chunk in self.chunks:
            self.find_targets(chunk, self.qa_dataset["question"], self._lambda)

    def __call__(self):
        # returns the tokenized/collated mask_dataset { "id", "text", "labels", "word_ids", ...}
        None
