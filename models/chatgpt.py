import os
import json
from datetime import datetime

from tqdm.auto import tqdm

import openai

import datasets
from datasets import Dataset, load_dataset

import evaluate

metric = evaluate.load("squad")
OPENAI_KEY = os.environ.get("OPENAI_KEY")


def prompt_fmt(example):
    """ """

    instruction = "Answer the following question using the provided context. The answer must correspond to a span of text from the context.\n\n"
    question = "Question: {}\n\n".format(example["question"])
    context = "Context: {}".format(example["context"])

    example["prompt"] = instruction + question + context
    return example


def eval_chatgpt(dataset, model="gpt-3.5-turbo", output_dir="./dir"):
    # build prompts
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    name = f"{model}-{timestamp}-results.json"
    output_file = os.path.join("./", name)
    dataset = dataset.map(prompt_fmt)
    theoretical_answers = []
    predicted_answers = []

    for example in tqdm(dataset):
        chat_completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": example["prompt"]}],
        )
        response = chat_completion.choices[0].message.content
        theoretical_answers.append({"id": example["id"], "answers": example["answers"]})
        predicted_answers.append({"id": example["id"], "prediction_text": response})

    # @TODO :: add option for metric only or verbose output (predicted, theoretical, and input)
    metrics = metric.compute(
        predictions=predicted_answers, references=theoretical_answers
    )

    f1_score = metrics["f1"]
    em = metrics["exact_match"]

    print(
        "\n{} \nEvaluation results \nmodel : {} \n > F1 = {} \n > EM = {} \n{}".format(
            "*" * 50, model, f1_score, em, "*" * 50
        )
    )

    results = {
        "em": em,
        "f1": f1_score,
        "predicted_answers": predicted_answers,
    }

    # SAVE results as json
    with open(output_file, "w") as fp:
        json.dump(results, fp)

    return metrics, predicted_answers, theoretical_answers
