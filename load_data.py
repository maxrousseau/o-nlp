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
