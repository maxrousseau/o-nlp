# code from: https://github.com/tlkh/t5-fp16-surgery

import torch
import numpy as np
from random import choice
import string


def count_param(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def gen_random_sequence(length=128):
    chars = string.ascii_letters + "     .!?'"
    return "".join([choice(chars) for i in range(length)])


def test_no_nan(model, tokenizer, iterations=100):
    model_half = model.cuda().half()
    model_half.train()
    text = " Hello world! The Earth is a nice place to be. "
    train_text = " translate English to German: The house is wonderful. "
    train_label = " Das Haus ist wunderbar. "
    for idx in range(iterations):
        inputs = tokenizer.encode(
            str(idx) + text + str(idx), return_tensors="pt"
        ).cuda()
        out = model_half(input_ids=inputs, decoder_input_ids=inputs)
        if torch.isnan(out[0]).any():
            return False
        train_input_ids = tokenizer(
            str(idx) + train_text + str(idx), return_tensors="pt"
        ).input_ids.cuda()
        train_labels = tokenizer(
            str(idx) + train_label + str(idx), return_tensors="pt"
        ).input_ids.cuda()
        loss = model_half(input_ids=train_input_ids, labels=train_labels).loss
        if torch.isnan(loss):
            return False
        inputs = tokenizer.encode(gen_random_sequence(), return_tensors="pt").cuda()
        out = model_half(input_ids=inputs, decoder_input_ids=inputs)
        if torch.isnan(out[0]).any():
            return False
    return True


def scale_weights(layer, scale_down_factor):
    old_weights = layer.weight
    layer.weight = torch.nn.Parameter(old_weights / scale_down_factor)
    return old_weights


def search_and_reset_layers(model, tokenizer, scale_down_factor=10, revert_old=False):
    model = model.float()
    total_params = count_param(model)
    param_reset_count = 0

    print("Testing encoder")
    for i, layer in enumerate(model.encoder.block[::-1]):
        fflayer1 = layer.layer[1].DenseReluDense.wi
        fflayer2 = layer.layer[1].DenseReluDense.wo

        # fflayer2.reset_parameters()
        old_weights = scale_weights(fflayer2, scale_down_factor)
        param_reset_count += count_param(fflayer2)
        if test_no_nan(model, tokenizer):
            print("Success at encoder", len(model.encoder.block) - i, "FF2")
            return model.float(), int(param_reset_count / total_params * 100)
        else:
            if revert_old:
                param_reset_count -= count_param(fflayer2)
                fflayer2.weight = old_weights

        # fflayer1.reset_parameters()
        old_weights = scale_weights(fflayer1, scale_down_factor)
        param_reset_count += count_param(fflayer1)
        if test_no_nan(model, tokenizer):
            print("Success at encoder", len(model.encoder.block) - i, "FF1")
            return model.float(), int(param_reset_count / total_params * 100)
        else:
            if revert_old:
                param_reset_count -= count_param(fflayer1)
                fflayer1.weight = old_weights

    print("Testing decoder")
    for i, layer in enumerate(model.decoder.block[::-1]):
        fflayer1 = layer.layer[2].DenseReluDense.wi
        fflayer2 = layer.layer[2].DenseReluDense.wo

        # fflayer2.reset_parameters()
        old_weights = scale_weights(fflayer2, scale_down_factor)
        param_reset_count += count_param(fflayer2)
        if test_no_nan(model, tokenizer):
            print("Success at decoder", len(model.decoder.block) - i, "FF2")
            return model.float(), int(param_reset_count / total_params * 100)
        else:
            if revert_old:
                param_reset_count -= count_param(fflayer2)
                fflayer2.weight = old_weights

        # fflayer1.reset_parameters()
        old_weights = scale_weights(fflayer1, scale_down_factor)
        param_reset_count += count_param(fflayer1)
        if test_no_nan(model, tokenizer):
            print("Success at decoder", len(model.decoder.block) - i, "FF1")
            return model.float(), int(param_reset_count / total_params * 100)
        else:
            if revert_old:
                param_reset_count -= count_param(fflayer1)
                fflayer1.weight = old_weights

    return model.float(), False
