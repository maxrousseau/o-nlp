import random
from datasets import Dataset
from dataclasses import dataclass

from tqdm.auto import tqdm

import spacy


@dataclass
class SFCFG:
    name: str = "setfit-default"
    lr: float = 2e-5
    n_epochs: int = 1
    lr_scheduler: bool = False
    model_checkpoint: str = ""
    tokenizer_checkpoint: str = ""
    checkpoint_savedir: str = "./setfit-ckpt"
    max_length: int = 256
    padding: str = "max_length"
    seed: int = 0

    train_dataset: Dataset = None
    val_dataset: Dataset = None
    test_dataset: Dataset = None

    val_batches: Any = None
    train_batches: Any = None
    test_batches: Any = None

    model: Any = None
    tokenizer: Any = None
    runmode: str = None

    # TBD add a print/export function to the config when we save model...
    # def __repr__() -> str


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


def get_sentence_list(dataset, parser=None):
    sent_col = []
    for i in tqdm(range(len(dataset))):
        c = dataset["context"][i]
        c = parser(c)
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


def setup_setfit_training(train_path, val_path, config):
    spacy_nlp = spacy.load("en_core_web_sm")
    # begin by setting up the dataset for sentence classification

    train_dataset = Dataset.load_from_disk(train_path)
    val_dataset = Dataset.load_from_disk(val_path)

    # sentences, add a list of sentences
    train_dataset = get_sentence_list(train_dataset, parser=spacy_nlp)
    val_dataset = get_sentence_list(val_dataset, parser=spacy_nlp)

    # create answer sent and non-answer sent dataset
    train_set = train_dataset.map(get_answer_sentence)
    val_set = val_dataset.map(get_answer_sentence)
    # BUG some None values for answer sentence given that they are over two sentences -- discard if so

    config.train_dataset = make_classification_datset(train_set)
    config.val_dataset = make_classification_datset(val_set)

    # @HERE :: customize the setfit trainer class to fit with the rest of the codebase
    config.model = SetFitModel.from_pretrained(config.model_checkpoint)
    return config

    # trainer = SetFitTrainer(
    #     model=biomedl,
    #     train_dataset=cls_train_dataset,
    #     eval_dataset=cls_val_dataset,
    #     loss_class=CosineSimilarityLoss,
    #     metric="accuracy",
    #     batch_size=16,
    #     num_iterations=20,  # The number of text pairs to generate for contrastive learning
    #     num_epochs=1,  # The number of epochs to use for contrastive learning
    #     column_mapping={
    #         "text": "text",
    #         "label": "label",
    #     },  # Map dataset columns to text/label expected by trainer
    # )
    #
    # # Train and evaluate
    # trainer.train()
    # metrics = trainer.evaluate()
