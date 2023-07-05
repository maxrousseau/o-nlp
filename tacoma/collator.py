import torch

from transformers import (
    DataCollatorForSeq2Seq,
)

import numpy as np

"""
Here we create the TacomaCollator which is built on the seq2seq collator but applies a dynamic
mask with the specified target sentence, _lambda, ...

To begin with this T5 refactor, we will only do mask-filling

1) sample from lambda (resample if > len(mappings))
2) find possible start positions given lambda and select at random
3) slice attention_mask, input_ids, and create labels column
4) output are as follows (attention_mask, input_ids, labels)

For now, simply test if we see an improvement with T5-base with a single epoch and 5K denoising samples without any
prompting. If promising, design templates `a la FLAN` and use the FLAN-T5 checkpoints.

"""

from torch.utils.data import DataLoader
import random
from torch.optim import AdamW


class TacomaCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        model=None,
        padding=True,
        max_length=None,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
        max_label_length=128,
        _lambda=18,
    ):
        super().__init__(
            tokenizer,
            model=model,
            padding=padding,
            max_length=None,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors,
        )
        self.max_label_length = max_label_length
        self._lambda = _lambda
        self.special_0 = tokenizer.encode(
            tokenizer.special_tokens_map["additional_special_tokens"][0]
        )[0]
        self.special_1 = tokenizer.encode(
            tokenizer.special_tokens_map["additional_special_tokens"][1]
        )[0]

    def _get_labels(self, example, l=None):
        # count the ones
        n_maskable_tokens = torch.bincount(example["mask_mappings"].flatten())[1].item()
        span_length = np.random.poisson(l)

        if n_maskable_tokens == 0:
            assert False

        # resample if over limit of maskable tokens
        while span_length >= n_maskable_tokens:
            span_length = np.random.poisson(l)

        # get the possible word ids
        mask_mapping_bool = torch.Tensor(example["mask_mappings"]).int().bool()
        maskable_word_ids = torch.masked_select(example["word_ids"], mask_mapping_bool)

        cand_indexes = maskable_word_ids[:-span_length].numpy()

        if len(cand_indexes) > 1:
            start_idx = random.randint(0, len(cand_indexes) - 1)
            start_word_id = cand_indexes[start_idx]
        else:
            start_word_id = cand_indexes[0]

        input_ids = example["input_ids"].numpy()
        label = input_ids[start_word_id : start_word_id + span_length]
        label = np.insert(label, 0, self.special_0)
        label = np.append(label, self.special_1)

        return torch.LongTensor(label), (start_word_id, start_word_id + span_length)

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # modifications-start HERE
        # labels are generated dynamically!, labels is a list of longtensor
        labels = []
        f_keys = features[0].keys()
        for i, f in enumerate(features):
            label_tensor, splice_idx = self._get_labels(f, l=self._lambda)

            features[i]["labels"] = label_tensor
            labels.append(label_tensor)

            # slice across all feature keys except
            features[i]["input_ids"] = torch.cat(
                (
                    features[i]["input_ids"][: splice_idx[0]],
                    torch.LongTensor([self.special_0]),
                    features[i]["input_ids"][splice_idx[1] :],
                ),
                0,
            )
            features[i]["attention_mask"] = torch.cat(
                (
                    features[i]["attention_mask"][: splice_idx[0]],
                    torch.LongTensor([1]),
                    features[i]["attention_mask"][splice_idx[1] :],
                ),
                0,
            )
        # modifications-end HERE
        # return features

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


# @BUG :: temporarily don't forget to take out invalid samples!!!
def tokenize_with_mask_mapping(example, tokenizer, max_sequence_length=768):
    """Here I need the the sentence start position in the context"""

    def find_offset(offset_mapping, k, idx):
        for p, i in enumerate(offset_mapping):
            if (i[0] + i[1]) != 0 and i[k] == idx:
                return p

    inputs = tokenizer(
        example["text"],
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

    start_pos_sentence = example["target_start"]
    end_pos_sentence = start_pos_sentence + len(example["target"])

    # @BUG :: because the .find method was used I have a bad start position for a particular sentence!! this causes
    # issues with start of sentence/end of sentence find offset returning None, hacky fix now to just return empty
    # mask_mapping and remove those samples post-tokenization

    start_sentence_offset = find_offset(inputs["offset_mapping"], 0, start_pos_sentence)

    if start_sentence_offset == None:  # see bug above
        inputs["mask_mappings"].append(None)
        inputs["valid"] = False
        return inputs

    # if the end of the sentence is truncated (and that there is no padding)
    if end_pos_sentence > inputs["offset_mapping"][-2][1] and inputs["offset_mapping"][
        -2
    ] != (0, 0):
        end_sentence_offset = len(inputs["offset_mapping"]) - 1

    else:
        end_sentence_offset = find_offset(inputs["offset_mapping"], 1, end_pos_sentence)
        if end_sentence_offset == None:
            inputs["mask_mappings"].append(None)
            inputs["valid"] = False
            return inputs

    mask_mappings = [0] * max_sequence_length

    num_sentence_tokens = end_sentence_offset - start_sentence_offset

    mask_mappings = (
        mask_mappings[:start_sentence_offset]
        + [1] * (num_sentence_tokens + 1)
        + mask_mappings[end_sentence_offset + 1 :]
    )

    assert len(mask_mappings) == max_sequence_length

    inputs["mask_mappings"].append(mask_mappings)

    return inputs


def test_mapping(features, sample, tokenizer):
    """ """
    mask_mapping_bool = torch.Tensor(features["mask_mappings"]).int().bool()
    input_ids = torch.Tensor(features["input_ids"]).int()
    mask_target = torch.masked_select(input_ids, mask_mapping_bool).numpy()

    is_equal = tokenizer.decode(mask_target) == sample["target"]
    if is_equal:
        print("Mappings are Ok!")
    else:
        print(
            "Mask mapping error!\nMapping: {}\nTarget: {}".format(
                mask_target, sample["target"]
            )
        )


def collator_test(batch_tensor, tokenizer):
    tacoma_collator = TacomaCollator(tokenizer)
    tdl = DataLoader(batch_tensor, collate_fn=tacoma_collator, batch_size=4)
    return next(
        iter(tdl)
    )  # batch.pop("word_ids", "offset_mappings", "mask_mappings") before the forward pass!


# make sure we have decreasing loss Ok
def miniloop(model, batch):
    optimizer = AdamW(model.parameters(), lr=1e-4)
    for i in range(5):
        outputs = model(**batch)
        loss = outputs.loss
        print(f"loss {loss}")
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
