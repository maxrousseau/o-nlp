#!/usr/bin/env python

###############################################################################
#                               BERT-like utils                               #
###############################################################################
import logging
from dataclasses import dataclass
from typing import Any
import collections

import numpy as np

from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
)

import datasets
from datasets import Dataset, load_dataset

from evaluate import load


metric = load("squad")

datasets.utils.logging.set_verbosity_warning

FORMAT = "[%(levelname)s] :: %(asctime)s @ %(name)s :: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("bert-utils")
logger.setLevel(logging.DEBUG)  # change level if debug or not


@dataclass
class BERTCFG:
    name: str = "bert-default"
    lr: float = 2e-5
    n_epochs: int = 12
    lr_scheduler: bool = False
    model_checkpoint: str = ""
    tokenizer_checkpoint: str = ""
    checkpoint_savedir: str = "./bert-ckpt"
    max_length: int = 512
    stride: int = 128
    padding: str = "max_length"
    seed: int = 0
    append_special_token: bool = False

    train_batch_size: int = 4
    val_batch_size: int = 16

    load_from_checkpoint: bool = False
    checkpoint_state: str = None
    checkpoint_step: int = None

    train_dataset: Dataset = None
    val_dataset: Dataset = None
    test_dataset: Dataset = None

    val_batches: Any = None
    train_batches: Any = None
    test_batches: Any = None

    bitfit: bool = False
    model: Any = None
    tokenizer: Any = None

    def __repr__(self) -> str:
        s = """
BERT-like model configuration
************************************
        Name : {}
        Model checkpoint : {}
        Tokenizer checkpoint : {}
        Max sequence length : {}
        Hyperparameters :
                bitfit={},
                lr={},
                lr_scheduler={},
                num_epochs={},
                batch_size={}
************************************
        """.format(
            self.name,
            self.model_checkpoint,
            self.tokenizer_checkpoint,
            self.max_length,
            self.bitfit,
            self.lr,
            self.lr_scheduler,
            self.n_epochs,
            self.train_batch_size,
        )
        return s


def tacoma_mlm_init(
    model_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    tokenizer_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    dataset_token = "[OQA]"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    # add new tokenizer
    # special_tokens_dict = {"additional_special_tokens": dataset_token}
    special_tokens_dict = {"mask_token": dataset_token}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    # resize the model embeddings
    model.resize_token_embeddings(len(tokenizer))

    # @TODO :: modify datacollator to insert the masks
    return model, tokenizer


def bert_init(model_checkpoint, tokenizer_chekpoint):
    """Iinitialize BERT model for extractive question answering"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_chekpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    return model, tokenizer


def bert_mlm_init(model_checkpoint, tokenizer_chekpoint):
    """Iinitialize BERT model for masked language modelling"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_chekpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    return model, tokenizer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def apply_bitfit(model):
    for name, param in model.named_parameters():
        if name.startswith("qa_outputs"):
            param.requires_grad = True
        elif name.endswith("bias"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print_trainable_parameters(model)

    return model


def preprocess_training(
    examples,
    tokenizer,
    padding="max_length",
    stride=128,
    max_len=512,
    append_special_token=False,
):
    """
    preprocessing for training examples
            return:
            inputs:
            features: ['example_id', 'offset_mapping', 'attention_mask', 'token_type_id', 'start_position', 'end_position']
    """
    if append_special_token:
        questions = tokenizer.mask_token + [q.strip() for q in examples["question"]]
        contexts = tokenizer.mask_token + examples["context"]
    else:
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=max_len,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context (question = 0, context = 1,  special token = None)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)

        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids

    return inputs


def preprocess_validation(
    examples,
    tokenizer,
    padding="max_length",
    stride=128,
    max_len=512,
    append_special_token=False,
):
    """preprocessing for evalutation samples (validation and test)
    return:
    inputs:
    features ['example_id', 'offset_mapping', 'attention_mask', 'token_type_id']
    """
    if append_special_token:
        questions = tokenizer.mask_token + [q.strip() for q in examples["question"]]
        contexts = tokenizer.mask_token + examples["context"]
    else:
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=max_len,  # what is this for?
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids

    return inputs


def answer_from_logits(start_logits, end_logits, features, examples, tokenizer):
    """
    from the HF tutorial, this function takes the logits as input and returns the score from the metric (EM and F1)
    @TODO - separate the best answer code from the metric computation -- keeping them in separate functions would make it easier to
    check out the output
    """
    metric = load("squad")
    n_best = 20
    max_answer_length = 50

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    # @TODO ::  make sure this all works even when we don't have the answers
    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]

    # @TODO :: add option for metric only or verbose output (predicted, theoretical, and input)
    metrics = metric.compute(
        predictions=predicted_answers, references=theoretical_answers
    )

    return metrics, predicted_answers, theoretical_answers


def prepare_inputs(
    dataset,
    tokenizer,
    stride,
    max_len,
    subset=None,
    padding="max_length",
    append_special_token=False,
):
    """ """
    if subset == "train":
        tokenized_dataset = dataset.map(
            lambda example: preprocess_training(
                example,
                tokenizer,
                padding=padding,
                max_len=max_len,
                stride=stride,
                append_special_token=append_special_token,
            ),
            batched=True,
            remove_columns=dataset.column_names,
            # keep_in_memory=True,
        )
        logger.info(
            "Training dataset processed and tokenized : n = {}".format(
                len(tokenized_dataset)
            )
        )
        return tokenized_dataset
    elif subset == "eval":
        tokenized_dataset = dataset.map(
            lambda example: preprocess_validation(
                example,
                tokenizer,
                padding=padding,
                max_len=max_len,
                stride=stride,
                append_special_token=append_special_token,
            ),
            batched=True,
            remove_columns=dataset.column_names,
            # keep_in_memory=True,
        )
        logger.info(
            "Test dataset processed and tokenized : n = {}".format(
                len(tokenized_dataset)
            )
        )
        return tokenized_dataset

    else:
        raise Exception("Specify subset for data preparation")


def setup_finetuning_oqa(train_path, val_path, config):
    """
        Setup function for fine-tuning BERT-like models on OQA-v1.0

        Load and preprocess the training and validation data. Initialize the model and tokenizer. Returns the config
    object which contains everything needed to instantiate a trainer and run.
    """

    config.train_dataset = Dataset.load_from_disk(train_path)
    config.val_dataset = Dataset.load_from_disk(val_path)

    logger.info("datasets loaded from disk")

    config.model, config.tokenizer = bert_init(
        config.model_checkpoint, config.tokenizer_checkpoint
    )

    logger.info("model and tokenizer initialized")

    if config.bitfit:
        config.model = apply_bitfit(config.model)

    config.train_batches = prepare_inputs(
        config.train_dataset,
        config.tokenizer,
        stride=config.stride,
        max_len=config.max_length,
        padding=config.padding,
        subset="train",
    )
    config.val_batches = prepare_inputs(
        config.val_dataset,
        config.tokenizer,
        stride=config.stride,
        max_len=config.max_length,
        padding=config.padding,
        subset="train",
    )

    return config


def setup_finetuning_squad(val_path, config):
    squad = load_dataset(
        "squad", download_mode="force_redownload"
    )  # @BUG remove for caching
    config.train_dataset = squad["train"]
    config.val_dataset = Dataset.load_from_disk(val_path)

    # !bert

    logger.info("datasets loaded from disk")

    config.model, config.tokenizer = bert_init(
        config.model_checkpoint, config.tokenizer_checkpoint
    )

    logger.info("model and tokenizer initialized")

    if config.bitfit:
        config.model = apply_bitfit(config.model)

    config.train_batches = prepare_inputs(
        config.train_dataset,
        config.tokenizer,
        stride=config.stride,
        max_len=config.max_length,
        padding=config.padding,
        subset="train",
    )
    config.val_batches = prepare_inputs(
        config.val_dataset,
        config.tokenizer,
        stride=config.stride,
        max_len=config.max_length,
        padding=config.padding,
        subset="eval",
    )

    return config


def setup_pretrain_bert(train_path, config):
    """Implement pretraining and test out using TAPT described in the "don't stop pretraining" paper

        This should yield performance improvements in theory...

    For the masking strategy, copy SpanBERT MLM for now
    """
    import sys

    sys.path.append("../")
    from mage import generator

    # load the datasets, shuffle and split
    pretraining_dataset = load_dataset("m-rousseau/tacoma22k")
    pretraining_dataset = pretraining_dataset["train"]

    pretraining_dataset = pretraining_dataset.shuffle(
        seed=config.seed
    ).train_test_split(test_size=0.05)

    logger.info("datasets loaded from disk, shuffled, training/validation split")

    config.train_dataset = pretraining_dataset["train"]
    config.val_dataset = pretraining_dataset["test"]

    config.model, config.tokenizer = tacoma_mlm_init(
        model_checkpoint=config.model_checkpoint,
        tokenizer_checkpoint=config.tokenizer_checkpoint,
    )
    logger.info("model and tokenizer initialized")

    tokenized_train_dataset = generator.tokenize_tacoma(
        config.train_dataset, config.tokenizer
    )
    tokenized_val_dataset = generator.tokenize_tacoma(
        config.val_dataset, config.tokenizer
    )

    config.train_batches = tokenized_train_dataset.filter(
        lambda example: example["valid"] == True
    )
    config.val_batches = tokenized_val_dataset.filter(
        lambda example: example["valid"] == True
    )
    logger.info("dataset cleaned and tokenized")

    return config


def setup_evaluate_oqa(test_path, config):
    """ """
    config.test_dataset = Dataset.load_from_disk(test_path)

    logger.info("datasets loaded from disk")

    config.model, config.tokenizer = bert_init(
        config.model_checkpoint, config.tokenizer_checkpoint
    )

    logger.info("model and tokenizer initialized")

    config.test_batches = prepare_inputs(
        config.test_dataset,
        config.tokenizer,
        stride=config.stride,
        max_len=config.max_length,
        padding=config.padding,
        subset="eval",
    )

    return config
