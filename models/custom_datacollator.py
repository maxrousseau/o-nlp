from datasets import Dataset
from transformers import (
    DataCollatorForWholeWordMask,
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
)
import torch

import warnings

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np

import random

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)

dataset = Dataset.load_from_disk("../tmp/bin/train")


class DataCollatorForWholeWordSpan(DataCollatorForWholeWordMask):
    """
    Modify above to get continuous spans of masked tokens"""

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        # modifications start here ****************************************
        # > the whole word mask to span next tokens when sampling start points and then add them to the the

        random.shuffle(cand_indexes)
        num_to_predict = min(
            max_predictions,
            max(1, int(round(len(input_tokens) * self.mlm_probability))),
        )
        masked_lms = []
        covered_indexes = set()
        flat_cand_indexes = sum(cand_indexes, [])
        print(type(flat_cand_indexes))
        print(flat_cand_indexes[:20])
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                # a) sample from the poisson distribution
                # b) mask continuous tokens IF they are present in the cand_indexes
                # c) append the covered indixes to covered_indexes and masked_lms
                n_tokens = np.random.poisson(12)
                m_index = index

                while n_tokens > 1:
                    if m_index in flat_cand_indexes and m_index not in covered_indexes:
                        if len(masked_lms) >= num_to_predict:
                            n_tokens = 0
                        else:
                            covered_indexes.add(m_index)
                            masked_lms.append(m_index)
                            n_tokens = n_tokens - 1
                            m_index += 1
                    else:
                        n_tokens = 0
        print(len(covered_indexes))
        print(len(masked_lms))

        if len(covered_indexes) != len(masked_lms):
            raise ValueError(
                "Length of covered_indexes is not equal to length of masked_lms."
            )
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len(input_tokens))
        ]
        return mask_labels


def tokenize_corpus(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


def chunk_corpus(examples, chunk_size=512):
    """chunking the corpus for pretraining"""

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


def view_span_masks(masked_dataset, tokenizer, index=0):
    labels = masked_dataset["labels"][index].numpy()
    labels = [x for x in labels if x > 0]
    label_string = tokenizer.decode(labels[:7], skip_special_tokens=False)
    return label_string


def test_case():

    # let's first create a dummy dataset
    dataset = Dataset.load_from_disk("../tmp/bin/train")
    dataset = dataset.remove_columns(
        ["answers", "answer_sentence", "topic", "reference", "question"]
    )
    dataset = dataset.rename_column("context", "text")

    # so we can tokenize this using our pretrained bert-like tokenizer, we don't truncate the input as we will split it
    # into chunks at preprocessing
    tokenized_dataset = dataset.map(
        tokenize_corpus, batched=True, remove_columns=["text", "id"]
    )

    chunked_dataset = tokenized_dataset.map(chunk_corpus, batched=True)

    # apply whole word mask
    collator = DataCollatorForWholeWordMask(tokenizer, mlm=True, mlm_probability=0.5)

    span_collator = DataCollatorForWholeWordSpan(
        tokenizer, mlm=True, mlm_probability=0.5
    )
    # tensor = tokenizer.encode(test_string, return_tensors="pt")
    mlm_dataset = collator(
        chunked_dataset
    )  # gives us masked input_ids and our original labels

    span_dataset = span_collator(chunked_dataset)

    # @HERE :: stop, redo above code but adapt for 512 token chunks of the training dataset with questions for now?
    # done :: whole word masking is OK : each mask replaces a whole word (not a subword),

    # @HERE :: now we want to be able to mask a continuous span

    # ... and then begin implementation of the span datacollator which should be a simple modification of the call
    # function from the whole word mask collator

    tensor = tokenizer(
        dummy_ds["text"],
        max_length=32,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    print(tensor)
    print(type(tensor))
    print(len(tensor))
    print(tensor[0])

    output = collator([tensor[i] for i in range(2)])

    # decode to visualize mask
    masked_output = tokenizer.decode(output[0])

    # start trying to modify above
    # print("{}\n{}".format(test_string, masked_output))

    # go through the HF tutorial and colab -- https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt
    # it should not be too complicated to implement a custo collator/masking function but their code is too confusing
    # for me to figure it out without the tutorial...


if __name__ == "__main__":
    test_case()
