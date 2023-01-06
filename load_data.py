import json
from datasets import Dataset


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
    """
    masked_strings = []
    full_strings = []
    qa_strings = []
    answer_strings = []

    for i in range(len(dataset["question"])):
        question = dataset["question"][i]
        answer = dataset["answers"][i]["text"][0]
        context = dataset["context"][i]

        masked_strings.append(
            "Question: {} Answer: <mask>. Context: {}".format(question, context)
        )
        full_strings.append(
            "Question: {} Answer: {}. Context: {}".format(question, answer, context)
        )
        qa_strings.append("Question: {} Answer: {}.".format(question, answer))
        answer_strings.append(answer)

    return {
        "masked_strings": masked_strings,
        "full_strings": full_strings,
        "qa_strings": qa_strings,
        "answer_strings": answer_strings,
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


def loadSquadMI(n=None):
    """create a dataloader for SQuAD"""
    from datasets import load_dataset

    raw_datasets = load_dataset("squad")

    if n is not None:
        squad_subset = formatToMI(raw_datasets["train"][:n])
        return squad_subset
    else:
        return 0
