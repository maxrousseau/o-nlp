from datasets import Dataset
import datasets
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

        if len(covered_indexes) != len(masked_lms):
            raise ValueError(
                "Length of covered_indexes is not equal to length of masked_lms."
            )
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len(input_tokens))
        ]
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


def tokenize_corpus(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


def replace_items(item, i, start, end, value):
    if i >= start and i <= end:
        return value
    else:
        return item


def find_offset(offset_mapping, k, idx):
    for p, i in enumerate(offset_mapping):
        if (i[0] + i[1]) != 0 and i[k] == idx:
            return p


# @HERE --- begin figuring out the maskable_mappings
# @BUG --- questions will need to be of length max 128tokens and corpus will need to be chunked into 384
def tokenize_question_context(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )

    if tokenizer.is_fast:
        inputs["word_ids"] = [i for i in range(len(inputs["input_ids"]))]

    inputs["mask_mappings"] = []

    start_pos_sentence = example["sentence_start"]
    end_pos_sentence = start_pos_sentence + len(example["answer_sentence"])

    start_pos_question = 0
    end_pos_question = len(example["question"])

    start_sentence_mapping = find_offset(
        inputs["offset_mapping"], 0, start_pos_sentence
    )

    end_sentence_mapping = find_offset(inputs["offset_mapping"], 1, end_pos_sentence)

    start_question_mapping = find_offset(
        inputs["offset_mapping"], 0, start_pos_question
    )
    end_question_mapping = find_offset(inputs["offset_mapping"], 1, end_pos_question)

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
    label_string = tokenizer.decode(labels[:12], skip_special_tokens=False)
    return label_string


def get_answer_sentence(dataset):
    ans = []
    ans_pos = []
    for example in dataset:
        sentences = example["context"].split(". ")
        for s in sentences:
            if example["answers"]["text"][0] in s:
                ans.append(s)
                ans_pos.append(example["context"].find(s))

    dset_ans = datasets.Dataset.from_dict(
        {"answer_sentence": ans, "sentence_start": ans_pos}
    )
    dataset = datasets.concatenate_datasets([dataset, dset_ans], axis=1)
    return dataset


def test_case():
    # let's first create a dummy dataset
    dataset = Dataset.load_from_disk("../tmp/bin/train").select(range(10))
    dataset = dataset.remove_columns(["answer_sentence"])
    dataset = get_answer_sentence(dataset)
    # dataset = dataset.remove_columns(
    #    ["answers", "answer_sentence", "topic", "reference", "question"]
    # )
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

    # randmoly... basically I can replace 'random.shuffle(cand_indexes)' by a function which calls a sentence classifier!

    # ... and then begin implementation of the span datacollator which should be a simple modification of the call
    # function from the whole word mask collator
    tokenized_dataset = dataset.map(
        tokenize_question_context, batched=False, remove_columns=dataset.column_names
    )
    # verify that the masking_mappings are properly applied to the target sentence and the question
    # apply mappings to question and to sentence then verify that they match with the ones from the dataset

    input_ids = torch.Tensor(tokenized_dataset["input_ids"][0]).int()
    mask = torch.Tensor(tokenized_dataset["mask_mappings"][0]).int()
    mask = mask.bool()

    # np.array([0, 1, 1, 0, 0, 2, 2])
    # (a == 1).astype(int)

    targets = torch.masked_select(input_ids, mask).numpy()

    # @HERE - apply Bool based mask based on the value 1 or 2

    # after we get the maskable ids, determine possible start points by looking at lenght of the list - span
    # length. then randomize that sublist, sample the mask tokens and apply masks as in the whole word span collator.

    # @NOTE: final format for DataCollatorMageTuning -> dataset {question, chunk, } -> tokenized_dataset { "input_ids", "mask_mapping",
    # "attention_mask", "word_ids", "token_type_ids" }


def DataCollatorMageTuning():
    """
        modify from the span masking collator

    #@HERE - STEP 1 preprocessing the questions and text chunks identify maskable sequences
        I need to add a column for maskable sequences [0, 1, 2] # 0 no mask, 1 target sentence, 2 question :: tokenize
        sample dataset above with question and context to identify how to apply this...

        Algo:
        for each sample:

            if mask_question:
                    maskable_ids = rand(1 or 2) (convert to bool matrix)
            else:
                    maskable_ids = 1 (convert to bool matrix)

        -> cand_word_ids = get the word_ids of of maskable_ids
        -> span_length = poisson(12)

        if span_length >= len_cand_word_ids:
            while span_len >= len_cand_word_ids:
                    span_length = poisson(12)

        -> cand_start_ids = cand_word_ids[:span_length]
        -> shuffle(cand_start_ids)
        -> select mask word ids (see code above)

        endfor

    """
