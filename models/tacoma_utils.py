from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)


def get_tokenizer(
    checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset_token = "[OQA]"
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    # add new tokenizer
    # special_tokens_dict = {"additional_special_tokens": dataset_token}
    special_tokens_dict = {"mask_token": dataset_token}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    # resize the model embeddings
    model = model.resize_token_embeddings(len(tokenizer))

    # @TODO :: modify datacollator to insert the masks
    return model, tokenizer
