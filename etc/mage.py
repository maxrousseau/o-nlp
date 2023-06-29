import os
import random
import collections

import json

import numpy as np

from datasets import Dataset

from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    DistilBertForQuestionAnswering,
    AutoModelForQuestionAnswering,
)
from transformers import default_data_collator

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import sent_tokenize

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# @TODO :: custom loss function
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# cos_model = AutoModel.from_pretrained("distilbert-base-uncased")


# only works if answer is a single sentence
def get_answer_sentence(example):
    slist = sent_tokenize(example["context"])
    for s in slist:
        if example["answers"]["text"][0] in s:
            example["sentence"] = s
            example["answers"]["answer_start"][0] = s.find(
                example["answers"]["text"][0]
            )
            return example
        else:
            None


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

    train_c = [x.get("context") for x in train_raw]
    train_q = [x.get("question") for x in train_raw]
    train_a = [x.get("answers") for x in train_raw]
    train_id = [x.get("id") for x in train_raw]
    train_dict = {
        "question": train_q,
        "context": train_c,
        "sentence": ["" for i in range(len(train_a))],
        "answers": train_a,
        "id": train_id,
    }

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

    train_set = Dataset.from_dict(train_dict, split="train")
    train_set = train_set.map(get_answer_sentence)
    test_set = Dataset.from_dict(test_dict, split="test")
    test_set = test_set.map(get_answer_sentence)

    return train_set, test_set


# compute cosine sim mean for batch
# @torch.no_grad
def batch_cosine_sim():
    None

    # cosine sim mean huawei-noah/TinyBERT_General_4L_312D
    # distilbert-base-uncased


def embed_string(target):
    tokenized_target = tokenizer(
        target,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    # tokenized_sentence.set_format("torch")

    attn_mask = tokenized_target["attention_mask"]

    outputs = cos_model(**tokenized_target)
    embeddings = outputs["last_hidden_state"]

    # apply attention masks
    attn_mask = attn_mask.unsqueeze(-1).expand(embeddings.size()).float()

    masked_embeddings = embeddings * attn_mask

    # mean pooling... (https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1)
    # figure out what is actually happening here... not sure I understand this operation
    summed_embeddings = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(attn_mask.sum(1), min=1e-9)
    mean_pooled = summed_embeddings / summed_mask

    return tokenized_target, mean_pooled.detach().numpy()


def get_cosine_similarity(targets, sample):
    """take a model and a tuple of sentences as input and return the top cosine similarity"""
    sample = sample.reshape(1, -1)

    sim = cosine_similarity(
        sample,  # (1, 768)
        targets,  # (n_answer, 768)
    )

    return sim.max()


def training_preprocessing(examples, tokenizer):
    """
    preprocessing for training examples
            return:
            inputs:
            features: ['example_id', 'offset_mapping', 'attention_mask', 'token_type_id']
    """
    questions = ["?" for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["sentence"],
        max_length=256,
        truncation=True,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
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

    return inputs


def prepare_inputs(examples):
    """ """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )

    inputs["example_id"] = examples["id"]
    return inputs


def answer_from_logits(start_logits, end_logits, features):
    """
    @NOTE :: for now will just use the example as we did not return the offset mapppings, if this makes it to an actual
        experiments, we'll crop/stride to maximize the number of passages"""
    n_best = 20
    max_answer_length = 50

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_spans = []
    for example in features:
        example_id = example["example_id"]
        context = example["text"]
        answers = []

        for feature_index in example_to_features[example_id]:
            # __import__("IPython").embed()
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            # print(start_indexes)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # print(start_logit)
                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            # longest_answer = max(answers, key=lambda x: len(x["text"]))
            predicted_spans.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_spans.append({"id": example_id, "prediction_text": ""})
    return predicted_spans


def postprocessing():
    """turn predictions into spans of text"""

    return None


def train():
    DistilBertForQuestionAnswering.from_pretrained(
        "distilbert-base-uncased-distilled-squad"
    )

    None


def get_tfidf(corpus, answers):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer = vectorizer.fit(corpus)

    answer_vectors = vectorizer.transform(answers)

    return vectorizer, answer_vectors


class NItemsIterator:
    def __init__(self, iterable, n):
        self.iterable = iterable
        self.n = n
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        items = []
        for i in range(self.n):
            try:
                items.append(self.iterable[self.index])
                self.index += 1
            except IndexError:
                break
        if items:
            return items
        else:
            raise StopIteration


def main():
    # load 1000 random contexts

    from pathlib import Path

    # __import__("sys").path.append(str(Path(__file__).absolute().parents[1]))
    __import__("sys").path.append(str(Path("../").absolute().parents[1]))
    from load_data import load_corpus, load_mini_oqa

    corpus = load_corpus(
        "c:/Users/roum5/source/data/angle_corpus_v1.0/parsed_papers_6063.json"
    )
    random.seed(1)
    random.shuffle(corpus)
    # corpus = corpus[:100]
    corpus_dataset = {}
    corpus_dataset["context"] = [x.get("text") for x in corpus]
    corpus_dataset["question"] = ["?" for x in range(len(corpus))]
    corpus_dataset["id"] = [x.get("article_id") + str(x.get("page")) for x in corpus]
    corpus_dataset = Dataset.from_dict(corpus_dataset)

    # load oqa-alpha
    oqa_small = load_mini_oqa(
        "c:/Users/roum5/source/o-nlp/tmp/oqa_v0.1_train.json",
        "c:/Users/roum5/source/o-nlp/tmp/oqa_v0.1_test.json",
    )

    oqa_small_train = oqa_small[2]
    train_dataset = oqa_small_train.map(
        prepare_inputs,
        batched=True,
        remove_columns=oqa_small_train.column_names,
    )

    corpus_tokenized = corpus_dataset.map(
        prepare_inputs, batched=True, remove_columns=corpus_dataset.column_names
    )
    corpus_tensor = corpus_tokenized.remove_columns(["example_id", "offset_mapping"])
    corpus_tensor.set_format("torch")

    # @TODO: currently embedding one string at a time, apply for batches
    target_answers = [x.get("text")[0] for x in oqa_small_train["answers"]]
    tfidf, target_vectors = get_tfidf(corpus_dataset["context"], target_answers)
    target_vectors = target_vectors.toarray()
    # target_answer_embeddings = []
    # for a in tqdm(target_answers):
    #    _, embed = embed_string(a)
    #    target_answer_embeddings.append(embed.squeeze())
    # target_answer_embeddings = np.array(target_answer_embeddings)  # (n_answers, 768)

    # goal for today (get the cosine sim working for a batch and figure out the loss)
    # training loop LATER, let's get some BS outputs for
    qamodel = DistilBertForQuestionAnswering.from_pretrained(
        # "distilbert-base-uncased-distilled-squad"
        "distilbert-base-uncased"
    )

    optimizer = AdamW(qamodel.parameters(), lr=3e-4)
    for param in qamodel.parameters():
        param.requires_grad_()

    # make the dataloader
    dataloader = DataLoader(
        corpus_tensor,
        collate_fn=default_data_collator,
        batch_size=16,
        num_workers=0,
    )

    # __import__("IPython").embed()
    feature_dataset = corpus_tokenized.add_column("text", corpus_dataset["context"])
    feature_iterator = NItemsIterator(feature_dataset, 16)
    # for epoch in range(self.num_epochs):
    for steps, batch in tqdm(enumerate(dataloader)):
        out = qamodel(**batch)

        # out = qamodel(corpus_tensor["input_ids"])

        start_logits = out.start_logits.detach().numpy()
        end_logits = out.end_logits.detach().numpy()
        # print(start_logits.shape)
        # print(start_logits.shape)
        # start_logits = start_logits[: len(corpus_dataset)]
        # end_logits = end_logits[: len(corpus_dataset)]

        # @BUG :: here we have to find a way to retrieve the batch samples in the tokenized corpus
        features = next(feature_iterator)
        answers = answer_from_logits(start_logits, end_logits, features)
        # print(corpus_dataset)
        # print(answers)

        test_spans = [x.get("prediction_text") for x in answers]

        # __import__("IPython").embed()
        span_vectors = tfidf.transform(test_spans)
        # prediction_embeddings = []
        # for s in tqdm(test_spans):
        #    _, embed = embed_string(s)
        #    prediction_embeddings.append(embed.squeeze())
        # prediction_embeddings = np.array(prediction_embeddings)

        span_vectors = span_vectors.toarray()
        # __import__("IPython").embed()
        f = lambda x: get_cosine_similarity(target_vectors, x)
        # compute the loss for the batch
        max_cosine_sim = np.apply_along_axis(f, 1, span_vectors).tolist()
        # print(max_cosine_sim)

        for i in range(len(max_cosine_sim)):
            if len(test_spans[i].split()) > 0:
                max_cosine_sim[i] = max_cosine_sim[i]
            else:
                max_cosine_sim[i] = 0.01
        print(max_cosine_sim)
        qamodel.train()

        sim_loss = torch.sum((1 - torch.tensor(max_cosine_sim)) ** 2)
        sim_loss.requires_grad_()
        sim_loss.backward()
        print(sim_loss)
        optimizer.step()
        optimizer.zero_grad()

    # this is not working so well, I think instead of bert for cosine similarity I should go with a tf-idf method!!!,
    # simpler and may give better results... there is very little signal from what I am getting right now...
    return qamodel, answers


def tuneSpan():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "distilbert-base-uncased-distilled-squad"
    )
    train, test = load_mini_oqa(
        "../tmp/oqa_v0.1_train.json", "../tmp/oqa_v0.1_train.json"
    )

    train_tokenized = train.map(
        lambda example: training_preprocessing(example, tokenizer),
        batched=True,
        remove_columns=train.column_names,
    )

    n_epochs = 3
    lr = 1e-3
    optimizer = AdamW(model.parameters(), lr=lr)

    train_tensor = train_tokenized
    train_tensor.set_format("torch")

    # @HERE :: GOAL for the night - see if I can at least have a QA model converge on the training set (use miniLMv2 squad)
    # or whatever.. also check if token classification may be a better approach?...

    """
Ok so this model converges, now I need to check the performance on test sentences... looking at F1 is good but maybe
    also I should look at cosine/tfidf similarity?
"""

    dataloader = DataLoader(
        train_tensor,
        collate_fn=default_data_collator,
        batch_size=8,
        num_workers=0,
    )
    n_step = 0
    for epoch in range(n_epochs):
        for steps, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()  # backprop here
            # zero gradient and update Lr and optimizer
            optimizer.step()
            optimizer.zero_grad()
            print("step {} :: loss :: {}".format(n_step, loss))
            n_step += 1

        print("epoch {} :: Ok".format(epoch))


if __name__ == "__main__":
    # main()
    tuneSpan()


# main()
