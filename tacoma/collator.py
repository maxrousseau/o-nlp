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


def collator_test(batch_tensor, tokenizer):
    tacoma_collator = TacomaCollator(tokenizer)
    tdl = DataLoader(batch_tensor, collate_fn=tacoma_collator, batch_size=4)
    a = next(iter(tdl))
    print(a)


class TacomaCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, mask_questions=True, _lambda=18):
        super().__init__(tokenizer)
        self.mask_questions = mask_questions
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
        while span_length > n_maskable_tokens:
            span_length = np.random.poisson(l)

        # get the possible word ids
        mask_mapping_bool = torch.Tensor(example["mask_mappings"]).int().bool()
        maskable_word_ids = torch.masked_select(example["word_ids"], mask_mapping_bool)

        cand_indexes = maskable_word_ids[:-span_length].numpy()

        start_idx = random.randint(0, len(cand_indexes) - 1)

        start_word_id = cand_indexes[start_idx] - 1
        input_ids = example["input_ids"].numpy()
        label = input_ids[start_word_id : start_word_id + span_length]
        label = np.insert(label, 0, self.special_0)
        label = np.append(label, self.special_1)

        # torch.LongTensor(label)

        return label

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # modifications HERE
        # labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        labels = []
        for i, f in enumerate(features):
            span_length = 5
            labels.append(self._get_labels(f, l=span_length))

        return labels
        # re arrange array into LongTensor as it would be used in the default_collator for seq2seq
        assert False

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


'''

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
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
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

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
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

class TacomaCollatorOld(DataCollatorForWholeWordMask):
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
            mapping_id = np.random.choice([1, 2], p=[0.8, 0.2])
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

'''


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

    # start_sentence_mapping = start_sentence_offset + question_offset

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
