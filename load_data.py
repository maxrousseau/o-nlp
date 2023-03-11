import json
import gc

from datasets import Dataset
from datasets import load_dataset
import pyarrow as pa


def load_bioasq(train_path, test_path):
    """
    Load the bioasq dataset from json, split and transform into pytorch/hf Dataset objects import typer

    """
    # note :: we consider that the dataset to be loaded will contain the following fields
    #   also it will be pre-shuffled to ease the splitting processs
    f_train = open(train_path, "rb")
    train_raw = json.load(f_train)
    f_train.close()
    f_test = open(test_path, "rb")
    test_raw = json.load(f_test)
    f_test.close()

    train_val_split = 0.6

    train_c = [x.get("context") for x in train_raw]
    train_q = [x.get("question") for x in train_raw]
    train_a = [x.get("answers") for x in train_raw]
    train_id = [x.get("id") for x in train_raw]
    train_dict = {
        "question": train_q,
        "context": train_c,
        "answers": train_a,
        "id": train_id,
    }

    # split training and validation
    s_train_idx = int(len(train_id) * train_val_split)
    dev_train_set = {k: train_dict[k][:s_train_idx] for k in train_dict}
    val_set = {k: train_dict[k][s_train_idx:] for k in train_dict}

    full_train_set = Dataset.from_dict(train_dict, split="full_training")
    dev_train_set = Dataset.from_dict(dev_train_set, split="small_training")
    val_set = Dataset.from_dict(val_set, split="validation")

    test_c = [x.get("context") for x in test_raw]
    test_q = [x.get("question") for x in test_raw]
    test_a = [x.get("answers") for x in test_raw]
    test_id = [x.get("id") for x in test_raw]
    test_dict = {
        "question": test_q,
        "context": test_c,
        "answers": test_a,
        "id": test_id,
    }
    test_set = Dataset.from_dict(test_dict, split="test")

    return (dev_train_set, val_set, full_train_set, test_set)


def load_oqa_full(fpath):
    f = open(fpath, "rb")
    raw = json.load(f)
    f.close()

    c = [x.get("context") for x in raw]
    q = [x.get("question") for x in raw]
    a = [x.get("answer") for x in raw]
    s = [x.get("answer_sentences") for x in raw]
    uuid = [x.get("uuid") for x in raw]
    return {
        "question": q,
        "context": c,
        "answers": a,
        "id": uuid,
    }


def load_mini_oqa(train_path, test_path):
    """
    Load the mini oqa dataset from json, split and transform into pytorch/hf Dataset objects import typer

    """
    # note :: we consider that the dataset to be loaded will contain the following fields
    #   also it will be pre-shuffled to ease the splitting processs
    f_train = open(train_path, "rb")
    train_raw = json.load(f_train)
    f_train.close()
    f_test = open(test_path, "rb")
    test_raw = json.load(f_test)
    f_test.close()

    train_val_split = 0.5

    train_c = [x.get("context") for x in train_raw]
    train_q = [x.get("question") for x in train_raw]
    train_a = [x.get("answers") for x in train_raw]
    train_id = [x.get("id") for x in train_raw]
    train_dict = {
        "question": train_q,
        "context": train_c,
        "answers": train_a,
        "id": train_id,
    }

    # split training and validation
    s_train_idx = int(len(train_id) * train_val_split)
    dev_train_set = {k: train_dict[k][:s_train_idx] for k in train_dict}
    val_set = {k: train_dict[k][s_train_idx:] for k in train_dict}

    full_train_set = Dataset.from_dict(train_dict, split="full_training")
    dev_train_set = Dataset.from_dict(dev_train_set, split="small_training")
    val_set = Dataset.from_dict(val_set, split="validation")

    test_c = [x.get("context") for x in test_raw]
    test_q = [x.get("question") for x in test_raw]
    test_a = [x.get("answers") for x in test_raw]
    test_id = [x.get("id") for x in test_raw]
    test_dict = {
        "question": test_q,
        "context": test_c,
        "answers": test_a,
        "id": test_id,
    }
    test_set = Dataset.from_dict(test_dict, split="test")

    return (dev_train_set, val_set, full_train_set, test_set)


def formatToMI(dataset):
    """take a squad-like qa dataset and transform into MLM format specified in the fewshotBART paper
        "Question: a question? Answer: <mask>. Context: this is the context"

        USAGE:
            train_raw = Dataset.from_dict(formatToMI(dset[2]))
            test_raw = Dataset.from_dict(formatToMI(dset[3]))

            # then you can feed those to the FsBART model class at initialization to run

    #@BUG :: formatting is different for T5 and BART implement both!!!
    """
    gc.disable()
    contexts = pa.array(dataset["context"])
    questions = pa.array(dataset["question"])
    answers = pa.array([i["text"][0] for i in dataset["answers"]])

    # masked_strings = pa.compute.binary_join_element_wise(
    #    "Question: ", questions, " Answer: <mask>. Context: ", contexts, ""
    # )
    masked_strings = pa.compute.binary_join_element_wise(
        "Question: ", questions, " Answer: <extra_id_0> Context: ", contexts, ""
    )

    full_strings = pa.compute.binary_join_element_wise(
        "Question: ", questions, " Answer: ", answers, ". Context: ", contexts, ""
    )
    qa_strings = pa.compute.binary_join_element_wise(
        "Question: ", questions, " Answer: ", answers, ".", ""
    )

    target_answers = pa.compute.binary_join_element_wise(
        "<extra_id_0> ", answers, "." ""
    )

    gc.enable()

    return {
        "masked_strings": masked_strings.to_pylist(),
        "full_strings": full_strings.to_pylist(),
        "qa_strings": qa_strings.to_pylist(),
        "answer_strings": target_answers.to_pylist(),
        "id": dataset["id"],
    }


# @TODO -- determine if more elaborate function needed...
def loadOQAforRetrieval(path):
    """ """
    f = open(path, "rb")
    samples = json.load(f)
    f.close()
    return samples


def loadCorpus(corpus_path):
    """ """
    corpus = []  # just a list of strings, each string is a page from an article
    corpus_keys = []  # list of dicts correstponding to the passage above
    # {article_id : "", page: 0}

    with open(corpus_path, "rb") as f:
        raw_corpus = json.load(f)
        for p in raw_corpus:
            key = p["article_id"]
            page = 1

            for c in p["content"]:
                corpus.append(c)
                corpus_keys.append({"article_id": key, "page": page})
                page += 1

    return (corpus, corpus_keys)


def loadSquadMI(n=None, set=None):
    """create a dataloader for SQuAD"""

    raw_datasets = load_dataset("squad")

    if n is not None:
        squad_subset = formatToMI(raw_datasets[set][:n])
        return squad_subset
    else:
        squad_subset = formatToMI(raw_datasets[set])
        return squad_subset
