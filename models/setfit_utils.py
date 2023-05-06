import random
import torch
from datasets import Dataset

from tqdm.auto import tqdm

import spacy


nlp = spacy.load("en_core_web_sm")

# begin by setting up the dataset for sentence classification


def get_answer_sentence(example):
    random.seed(0)
    a = example["answers"]["text"][0]
    other_sentences = []
    is_unique = True
    count = 0
    for s in example["sentences"]:
        if a in s:
            count += 1
            example["answer_sentence"] = s
            if count > 1:
                is_unique = False

        else:
            other_sentences.append(s)

    if is_unique:
        example["other_sentences"] = other_sentences
        example["random_sentence"] = random.choice(other_sentences)
    else:
        example["answer_sentence"] = None
        example["other_sentences"] = None
        example["random_sentence"] = None
    return example


def get_sentence_list(dataset):
    sent_col = []
    for i in tqdm(range(len(dataset))):
        c = dataset["context"][i]
        c = nlp(c)
        sentences = []
        for s in c.sents:
            sentences.append(s.text)
        sent_col.append(sentences)
    dataset = dataset.add_column("sentences", sent_col)
    return dataset


def make_classification_datset(dataset):
    texts = []
    labels = []
    label_texts = []
    for example in dataset:
        if example["answer_sentence"] != None:
            texts.append(
                "{} {}".format(example["question"], example["answer_sentence"])
            )
            labels.append(1)
            label_texts.append("answer")

            texts.append(
                "{} {}".format(example["question"], example["random_sentence"])
            )
            labels.append(0)
            label_texts.append("other")
    ds = Dataset.from_dict({"text": texts, "label": labels, "label_text": label_texts})
    return ds


train_dataset = Dataset.load_from_disk(
    "/content/drive/MyDrive/onlp/oqa_v1.0_shuffled_split/bin/train"
)
val_dataset = Dataset.load_from_disk(
    "/content/drive/MyDrive/onlp/oqa_v1.0_shuffled_split/bin/val"
)

# sentences, add a list of sentences
train_dataset = get_sentence_list(train_dataset)
val_dataset = get_sentence_list(val_dataset)

# create answer sent and non-answer sent dataset
train_set = train_dataset.map(get_answer_sentence)
val_set = val_dataset.map(
    get_answer_sentence
)  # BUG some None values for answer sentence given that they are over
# two sentences

cls_train_dataset = make_classification_datset(train_set)
cls_val_dataset = make_classification_datset(val_set)
# read the setfit blog and paper, take notes


# even simpler! perform the contrastive learning with PubmedBERT on the following:
# question + relevant answer sent vs question + irrelevant answer sent

biomodel = SetFitModel.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)  # NICE

trainer = SetFitTrainer(
    model=biomodel,
    train_dataset=cls_train_dataset,
    eval_dataset=cls_val_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20,  # The number of text pairs to generate for contrastive learning
    num_epochs=1,  # The number of epochs to use for contrastive learning
    column_mapping={
        "text": "text",
        "label": "label",
    },  # Map dataset columns to text/label expected by trainer
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()
