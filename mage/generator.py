import torch

import datasets
from datasets import Dataset

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from collections.abc import Mapping
from tqdm.auto import tqdm
import random
from dataclasses import dataclass
from transformers.data.data_collator import _torch_collate_batch, tolist

from transformers import (
    DataCollatorForWholeWordMask,
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
)
import torch

import warnings

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


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
class TacomaDatasetGenerator:
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
        # self.target_dataset.save_to_disk("tacoma-angle_oqa") TODO implement a save/export function
        # NOTE :: dataset generation is extremely slow... might need to figure out a way to quantize the model to speed
        # it up. furthermore, the tfidf threshold should be estimated from the target dataset to be more
        # reprensentative... test out for now then clean this up
        return self.target_dataset


def find_offset(offset_mapping, k, idx):
    for p, i in enumerate(offset_mapping):
        if (i[0] + i[1]) != 0 and i[k] == idx:
            return p


def mask_mapping(example, tokenizer, max_sequence_length=512):
    """Here I need the the sentence start position in the context"""

    inputs = tokenizer(
        tokenizer.mask_token + example["question"],
        tokenizer.mask_token + example["text"],
        max_length=max_sequence_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )

    if tokenizer.is_fast:
        inputs["word_ids"] = [i for i in range(len(inputs["input_ids"]))]

    inputs["valid"] = True
    inputs["mask_mappings"] = []

    # add 5 bc len of "[OQA]" special token

    start_pos_sentence = example["target_start"] + 5
    end_pos_sentence = start_pos_sentence + len(example["target"]) + 5

    start_pos_question = 5
    end_pos_question = len(example["question"]) + 5

    start_question_mapping = find_offset(
        inputs["offset_mapping"], 0, start_pos_question
    )
    end_question_mapping = find_offset(inputs["offset_mapping"], 1, end_pos_question)

    question_offset = end_question_mapping + 1  # sep token

    # @BUG :: because the .find method was used I have a bad start position for a particular sentence!! this causes
    # issues with start of sentence/end of sentence find offset returning None, hacky fix now to just return empty
    # mask_mapping and remove those samples post-tokenization
    start_sentence_offset = find_offset(
        inputs["offset_mapping"][question_offset:], 0, start_pos_sentence
    )
    if start_sentence_offset == None:
        inputs["mask_mappings"].append([])
        inputs["valid"] = False
        return inputs

    start_sentence_mapping = start_sentence_offset + question_offset

    # if the end of the sentence is truncated (and that there is no padding)
    if end_pos_sentence > inputs["offset_mapping"][-2][1] and inputs["offset_mapping"][
        -2
    ] != (0, 0):
        end_sentence_mapping = len(inputs["offset_mapping"]) - 1

    else:
        end_sentence_offset = find_offset(
            inputs["offset_mapping"][question_offset:], 1, end_pos_sentence
        )
        if end_sentence_offset == None:
            inputs["mask_mappings"].append([])
            inputs["valid"] = False
            return inputs
        end_sentence_mapping = end_sentence_offset + question_offset

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
        + [2] * (num_question_tokens + 1)
        + mask_mappings[end_question_mapping + 1 :]
    )

    assert len(mask_mappings) == 512

    inputs["mask_mappings"].append(mask_mappings)

    return inputs


def tokenize_tacoma(dataset, tokenizer):
    tokenized_mask_dataset = dataset.map(
        lambda example: mask_mapping(
            example, tokenizer=tokenizer, max_sequence_length=512
        ),
        batched=False,
        remove_columns=dataset.column_names,
    )

    return tokenized_mask_dataset


class TacomaCollator(DataCollatorForWholeWordMask):
    def __init__(self, tokenizer, mask_questions=False, _lambda=12):
        super().__init__(tokenizer)
        self.mask_questions = mask_questions
        self._lambda = _lambda

    # @TODO :: will need to modify call function as well to change _whole_word_mask -> _mask_targets
    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(
            input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [ , ]-> [  ## ]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(
                self._mask_targets(ref_tokens, mask_mappings=e["mask_mappings"])
            )  # @HERE :: apply mask targets
        batch_mask = _torch_collate_batch(
            mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _mask_targets(
        self, input_tokens: List[str], max_predictions=512, mask_mappings=None
    ):
        """ """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        if self.mask_questions:
            mapping_id = random.randint(1, 2)
        else:
            mapping_id = 1  # always mask within target sentence

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            cand_indexes.append([i])

        cand_indexes = np.array(cand_indexes)
        cand_indexes = torch.flatten(torch.Tensor(cand_indexes).int())

        mask_mapping_bool = (np.array(mask_mappings[0]) == mapping_id).astype(int)

        mask_mapping_bool = torch.Tensor(mask_mapping_bool).int().bool()
        targets = torch.masked_select(cand_indexes, mask_mapping_bool).numpy()

        span_length = np.random.poisson(self._lambda)

        # resample from distrib if span too long
        if span_length >= len(targets):
            while span_length >= len(targets):
                span_length -= 1

        cand_indexes = targets[:-span_length]
        random.shuffle(cand_indexes)

        try:
            start_position = cand_indexes[0]
        except:
            print(targets)
            if len(targets) == 1:
                span_length = 1
                start_position = targets[0]

        end_position = start_position + span_length

        # print(start_position)
        # print(end_position)

        mask_labels = []
        for i in range(len(input_tokens)):
            if i >= start_position and i <= end_position:
                mask_labels.append(1)
            else:
                mask_labels.append(0)

        return mask_labels

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
                Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
                'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.

        MODIFICATIONS :: apply the mask token 100% of the time!!!
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # @HERE everything is masked unlike in the original implementation!
        indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels


def test_collator():
    return None


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
    tadam = TacomaDatasetGenerator(
        corpus=corpus,
        qa_dataset=oqa,
        sentence_classifier=sentence_classifier,
        tokenizer=tokenizer,
    )
    tadam()


# USAGE:
# tacoma_tgt = Dataset.load_from_disk("../tmp/tacoma-angle_oqa")
# tk = tokenize_tacoma(tacoma_tgt, tokenizer)
# tacoma_collator = TacomaCollator(tokenizer)
# tacoma_test = tacoma_collator(tk) (this will be for the training function! / dataloader)


if __name__ == "__main__":
    test_case()
