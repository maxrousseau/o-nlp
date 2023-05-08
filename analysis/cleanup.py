import os
import re
import json


def answer_sentence(answer, context):
    """for a given answer or context extract the sentence(s) containing the answer from the context"""
    start = context.find(answer)
    end = start + len(answer)

    s_start = context[:start].rfind(". ")
    if s_start < 0:  # beginning of context
        s_start = 0
    else:
        s_start += 2

    s_end = context[end:].find(". ")
    if s_end < 0:  # end of context
        s_end = len(context)
    else:
        s_end += len(context[:end]) + 1

    try:
        assert context[s_start:s_end] != ""
    except:
        print(answer)
        print(context)

    return context[s_start:s_end]


def clean_string(s):
    s = s.lower()
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)  # replace non-ascii chars by space
    s = re.sub(r"\s\s+", " ", s)
    s = s.strip()  # remove leading trailing whitespace
    return s


def clean_sample(sample):
    """for each sample,"""

    sample["answer"] = clean_string(sample["answer"])
    sample["question"] = clean_string(sample["question"])
    sample["context"] = clean_string(sample["passage"])

    sample["answer_sentence"] = answer_sentence(sample["answer"], sample["context"])
    sample["answer_sentence"] = clean_string(sample["answer_sentence"])

    sample.pop("subtopic")
    sample.pop("passage")
    sample.pop("reference_text")

    try:
        assert sample["context"].find(sample["answer"]) != -1
    except:
        print(sample["id"])

    return sample


def export(fpath, array):
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(array, f, ensure_ascii=False, indent=4)


def main(ds_path):
    ds = open(ds_path, "rb")
    raw_ds = json.load(ds)
    ds.close()

    clean_ds = []
    for i in raw_ds:
        i = clean_sample(i)
        clean_ds.append(i)

    return clean_ds
