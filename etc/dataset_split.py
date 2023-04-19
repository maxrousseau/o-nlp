import json

from datasets import Dataset, concatenate_datasets


def load_from_json(fpath):
    f = open(fpath, "rb")
    raw = json.load(f)
    f.close()

    c = [x.get("context") for x in raw]
    t = [x.get("topic") for x in raw]
    q = [x.get("question") for x in raw]
    s = [x.get("answer_sentences") for x in raw]
    m = [x.get("meta") for x in raw]
    uuid = [x.get("uuid") for x in raw]

    answer = [x.get("answer") for x in raw]

    answers = []

    for i in range(len(answer)):
        answers.append({"answer_start": [c[i].find(answer[i])], "text": [answer[i]]})

    return Dataset.from_dict(
        {
            "question": q,
            "context": c,
            "answers": answers,
            "answer_sentence": s,
            "topic": t,
            "reference": m,
            "id": uuid,
        }
    )


def split_by_topic(dataset, val_split=0.12, test_split=0.18):
    """ """

    topics = set(dataset["topic"])

    train = []
    val = []
    test = []

    for topic in topics:
        # mak
        topic_set = dataset.filter(lambda example: example["topic"] == topic)
        topic_set = topic_set.shuffle(seed=0)
        test_index = int(test_split * len(topic_set))
        val_index = int((val_split * (len(topic_set) - test_index)) + test_index)

        test.append(topic_set.select(range(test_index)))
        val.append(topic_set.select(range(test_index, val_index)))
        train.append(topic_set.select(range(val_index, len(topic_set))))

    train = concatenate_datasets(train).shuffle(seed=0)
    val = concatenate_datasets(val).shuffle(seed=0)
    test = concatenate_datasets(test).shuffle(seed=0)

    return train, val, test


def create_dataset(fpath, save_dir):
    """ """
    full_dataset = load_from_json(fpath)
    train_dataset, val_dataset, test_dataset = split_by_topic(full_dataset)

    train_dataset.save_to_disk(os.path.join(save_dir, "bin/train"))
    val_dataset.save_to_disk(os.path.join(save_dir, "bin/val"))
    test_dataset.save_to_disk(os.path.join(save_dir, "bin/test"))

    train_dataset.to_json(os.path.join(save_dir, "json/train.json"))
    val_dataset.to_json(os.path.join(save_dir, "json/val.json"))
    test_dataset.to_json(os.path.join(save_dir, "json/test.json"))
