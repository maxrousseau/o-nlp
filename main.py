#!/usr/bin/env python
import os
import logging

import t5_utils
from train import FineTuneT5


from absl import app
from absl import flags

from datasets import Dataset

FLAGS = flags.FLAGS

runmode = [
    "t5-finetune",
    "t5-pretrain",
    "bart-finetune",
    "bart-pretrain",
    "bert-finetune",
    "bert-pretrain",
]

flags.DEFINE_string(
    "name", None, "model name for saving checkpoints and tracking loss w/ wandb"
)
flags.DEFINE_enum(
    "runmode", None, ["finetune", "pretrain"], "type of training to run model-type"
)
flags.DEFINE_float("lr", 2e-5, "learning rate (AdamW optimizer)")
flags.DEFINE_integer("epochs", 2, "number of training epochs")
flags.DEFINE_bool("lr_scheduler", True, "use learning rate scheduler")


flags.DEFINE_integer("max_seq_len", 384, "maximum length (tokens) of input sequence")
flags.DEFINE_integer("max_ans_len", 128, "maximum length (tokens) of output sequence")


flags.DEFINE_integer("seed", 0, "random seed")


flags.DEFINE_string("train_dataset", None, "training dataset")
flags.DEFINE_string("test_dataset", None, "test/evaluation dataset")


flags.DEFINE_string(
    "model_checkpoint", None, "model checkpoint for loading pretrained weights"
)
flags.DEFINE_string(
    "tokenizer_checkpoint", None, "model checkpoint for loading pretrained weights"
)
flags.DEFINE_string("savedir", None, "directory for model and checkpoint export")
# @TODO :: absl-py seems quite slow, try radicli...


def main(argv):
    """get args and call"""

    train_ds_path = FLAGS.train_dataset
    test_ds_path = FLAGS.test_dataset

    runmode = FLAGS.runmode

    if runmode == "finetune":
        config = t5_utils.T5CFG(
            name=FLAGS.name,
            lr=FLAGS.lr,
            lr_scheduler=FLAGS.lr_scheduler,
            n_epochs=FLAGS.epochs,
            model_checkpoint=FLAGS.model_checkpoint,
            tokenizer_checkpoint=FLAGS.tokenizer_checkpoint,
            checkpoint_savedir=FLAGS.savedir,
            max_seq_length=FLAGS.max_seq_len,
            max_ans_length=FLAGS.max_ans_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = t5_utils.setup_finetune_t5(train_ds_path, test_ds_path, config)
        tuner = FineTuneT5(config)
        tuner()

    else:
        None

    # @TODO cleanup the load function to only output train, val and test datasets
    # @TODO add logging here

    # test the data preprocessing


if __name__ == "__main__":
    app.run(main)