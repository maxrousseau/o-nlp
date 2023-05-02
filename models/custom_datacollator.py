from datasets import Dataset
from transformers import DataCollatorForWholeWordMask, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)


class DataCollatorForWholeWordSpan(DataCollatorForWholeWordMask):
    """
    Modify above to get continuous spans of masked tokens"""

    def __call__():
        None


def tokenize_corpus(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


def chunk_corpus(examples, chunk_size=4):
    """chunking the corpus for pretraining"""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size

    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    # we need this when we corrupt our input_ids

    return result


def test_case():

    # let's first create a dummy dataset
    dummy_ds = Dataset.from_dict(
        {
            "text": [
                "The goal of orthodontics is to achieve ideal class I occlusion",
                "This is another dummy text sentece",
            ]
        }
    )  # these will serve as our unsupervised corpus

    # so we can tokenize this using our pretrained bert-like tokenizer, we don't truncate the input as we will split it
    # into chunks at preprocessing
    tokenized_dataset = dummy_ds.map(
        tokenize_corpus, batched=True, remove_columns=["text"]
    )
    chunk_size = 128
    mlm_dataset = tokenized_dataset.map(chunk_corpus, batched=True)

    # apply whole word mask
    collator = DataCollatorForWholeWordMask(tokenizer, mlm=True, mlm_probability=0.5)
    # tensor = tokenizer.encode(test_string, return_tensors="pt")
    masked = collator(mlm_dataset)  # gives us masked input_ids and our original labels

    # @HERE :: stop, redo above code but adapt for 512 token chunks of the training dataset with questions for now?
    # ... and then begin implementation of the span datacollator which should be a simple modification of the call
    # function from the whole word mask collator

    tensor = tokenizer(
        dummy_ds["text"],
        max_length=32,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    print(tensor)
    print(type(tensor))
    print(len(tensor))
    print(tensor[0])

    output = collator([tensor[i] for i in range(2)])

    # decode to visualize mask
    masked_output = tokenizer.decode(output[0])

    # start trying to modify above
    # print("{}\n{}".format(test_string, masked_output))

    # go through the HF tutorial and colab -- https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt
    # it should not be too complicated to implement a custo collator/masking function but their code is too confusing
    # for me to figure it out without the tutorial...


if __name__ == "__main__":
    test_case()
