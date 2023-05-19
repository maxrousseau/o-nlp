#!/usr/bin/env python
import os
import logging

from models import t5_utils, bert_utils, bart_utils, setfit_utils

from train import *


from absl import app
from absl import flags

from datasets import Dataset

FLAGS = flags.FLAGS

runmode = [
    "t5-finetune",
    "t5-evaluate",
    "t5-pretrain",
    "bart-finetune",
    "bart-pretrain",
    "bart-evaluate",
    "bert-finetune",
    "bert-squad-finetune",
    "bert-evaluate",
    "bert-pretrain",
    "splinter-finetune",
    "train-classifier",
    "bert-metatune",
]

flags.DEFINE_string(
    "name", None, "model name for saving checkpoints and tracking loss w/ wandb"
)
flags.DEFINE_enum("runmode", None, runmode, "type of training to run model-type")
flags.DEFINE_float("lr", 2e-5, "learning rate (AdamW optimizer)")
flags.DEFINE_integer("epochs", 2, "number of training epochs")
flags.DEFINE_bool("lr_scheduler", True, "use learning rate scheduler")

flags.DEFINE_integer("checkpoint_step", None, "current step at last checkpoint save")
flags.DEFINE_string(
    "checkpoint_state", None, "path to the checkpoint directory to load from"
)
flags.DEFINE_bool("load_from_checkpoint", False, "load from a checkpoint?")

flags.DEFINE_bool(
    "only_cls_head", False, "freeze all parameters except the classification head"
)


flags.DEFINE_integer("max_seq_len", 384, "maximum length (tokens) of input sequence")
flags.DEFINE_integer("max_ans_len", 128, "maximum length (tokens) of output sequence")


flags.DEFINE_integer("seed", 0, "random seed")


flags.DEFINE_string("train_dataset", None, "training dataset")
flags.DEFINE_string("val_dataset", None, "validation dataset")
flags.DEFINE_string("test_dataset", None, "test/evaluation dataset")


flags.DEFINE_string(
    "model_checkpoint", None, "model checkpoint for loading pretrained weights"
)
flags.DEFINE_string(
    "tokenizer_checkpoint", None, "model checkpoint for loading pretrained weights"
)
flags.DEFINE_string("savedir", None, "directory for model and checkpoint export")

flags.DEFINE_bool("lora", False, "finetune model with LoRA")
# @TODO :: absl-py seems quite slow, try radicli...


def main(argv):
    """get args and call"""

    train_ds_path = FLAGS.train_dataset
    val_ds_path = FLAGS.val_dataset
    test_ds_path = FLAGS.test_dataset

    runmode = FLAGS.runmode

    if runmode == "t5-finetune":
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
        config = t5_utils.setup_finetune_t5(train_ds_path, val_ds_path, config)
        tuner = FinetuneT5(config)
        tuner()

    elif runmode == "t5-evaluate":
        config = t5_utils.T5CFG(
            name=FLAGS.name,
            model_checkpoint=FLAGS.model_checkpoint,
            tokenizer_checkpoint=FLAGS.tokenizer_checkpoint,
            max_seq_length=FLAGS.max_seq_len,
            max_ans_length=FLAGS.max_ans_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = t5_utils.setup_evaluate_t5(test_ds_path, config)
        # @HERE :: make sure setup function is good, then finish training loop
        evaluator = EvaluateT5(config)
        evaluator()

    elif runmode == "bart-finetune":
        config = bart_utils.BARTCFG(
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
        config = bart_utils.setup_finetune_bart(train_ds_path, val_ds_path, config)
        # @HERE :: make sure setup function is good, then finish training loop
        tuner = FinetuneBART(config)
        tuner()

    elif runmode == "bart-evaluate":
        config = bart_utils.BARTCFG(
            name=FLAGS.name,
            model_checkpoint=FLAGS.model_checkpoint,
            tokenizer_checkpoint=FLAGS.tokenizer_checkpoint,
            max_seq_length=FLAGS.max_seq_len,
            max_ans_length=FLAGS.max_ans_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = bart_utils.setup_evaluate_bart(test_ds_path, config)
        # @HERE :: make sure setup function is good, then finish training loop
        evaluater = EvaluateBART(config)
        evaluater()

    elif runmode == "bert-pretrain":
        config = bert_utils.BERTCFG(
            name=FLAGS.name,
            lr=FLAGS.lr,
            lr_scheduler=FLAGS.lr_scheduler,
            n_epochs=FLAGS.epochs,
            model_checkpoint=FLAGS.model_checkpoint,
            tokenizer_checkpoint=FLAGS.tokenizer_checkpoint,
            checkpoint_savedir=FLAGS.savedir,
            max_length=FLAGS.max_seq_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = bert_utils.setup_pretrain_bert(train_ds_path, config)

        tuner = PretrainBERT(config)
        tuner()

    elif runmode == "bert-finetune":
        config = bert_utils.BERTCFG(
            name=FLAGS.name,
            lr=FLAGS.lr,
            lr_scheduler=FLAGS.lr_scheduler,
            n_epochs=FLAGS.epochs,
            model_checkpoint=FLAGS.model_checkpoint,
            tokenizer_checkpoint=FLAGS.tokenizer_checkpoint,
            checkpoint_savedir=FLAGS.savedir,
            max_length=FLAGS.max_seq_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = bert_utils.setup_finetuning_oqa(train_ds_path, val_ds_path, config)

        tuner = FinetuneBERT(config)
        tuner()
    elif runmode == "splinter-finetune":
        config = bert_utils.BERTCFG(
            name=FLAGS.name,
            lr=FLAGS.lr,
            lr_scheduler=FLAGS.lr_scheduler,
            n_epochs=FLAGS.epochs,
            model_checkpoint=FLAGS.model_checkpoint,
            tokenizer_checkpoint=FLAGS.tokenizer_checkpoint,
            checkpoint_savedir=FLAGS.savedir,
            max_length=FLAGS.max_seq_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = bert_utils.setup_finetuning_splinter_oqa(
            train_ds_path, val_ds_path, config
        )
        tuner = FinetuneSplinter(config)
        tuner()

    elif runmode == "bert-metatune":
        config = bert_utils.BERTCFG(
            name=FLAGS.name,
            lr=FLAGS.lr,
            lr_scheduler=FLAGS.lr_scheduler,
            n_epochs=FLAGS.epochs,
            model_checkpoint=FLAGS.model_checkpoint,
            tokenizer_checkpoint=FLAGS.tokenizer_checkpoint,
            checkpoint_savedir=FLAGS.savedir,
            max_length=FLAGS.max_seq_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = bert_utils.setup_metatune(
            train_ds_path, val_ds_path, config, only_head=FLAGS.only_cls_head
        )
        tuner = MetatuneBERT(
            config, n_step_eval=200, stagnation_threshold=3, n_steps_nudge=4
        )
        tuner()

    elif runmode == "bert-squad-finetune":
        config = bert_utils.BERTCFG(
            name=FLAGS.name,
            lr=FLAGS.lr,
            lr_scheduler=FLAGS.lr_scheduler,
            n_epochs=FLAGS.epochs,
            model_checkpoint=FLAGS.model_checkpoint,
            tokenizer_checkpoint=FLAGS.tokenizer_checkpoint,
            checkpoint_savedir=FLAGS.savedir,
            max_length=FLAGS.max_seq_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = bert_utils.setup_finetuning_squad(
            val_ds_path, config, only_head=FLAGS.only_cls_head
        )
        tuner = FinetuneBERT(config)
        tuner()

    elif runmode == "bert-evaluate":
        config = bert_utils.BERTCFG(
            name=FLAGS.name,
            model_checkpoint=FLAGS.model_checkpoint,
            tokenizer_checkpoint=FLAGS.tokenizer_checkpoint,
            max_length=FLAGS.max_seq_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = bert_utils.setup_evaluate_oqa(test_ds_path, config)

        evaluater = EvaluateBERT(config)
        evaluater()

    elif runmode == "t5-pretrain":
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
            checkpoint_step=FLAGS.checkpoint_step,
            checkpoint_state=FLAGS.checkpoint_state,
            load_from_checkpoint=FLAGS.load_from_checkpoint,
        )
        config = t5_utils.setup_pretrain_t5(train_ds_path, config)
        pretrainer = PretrainT5(config)
        pretrainer()

    elif runmode == "train-classifier":
        config = setfit_utils.SFCFG(
            name=FLAGS.name,
            lr=FLAGS.lr,
            n_epochs=FLAGS.epochs,
            model_checkpoint=FLAGS.model_checkpoint,
            checkpoint_savedir=FLAGS.savedir,
            max_length=FLAGS.max_seq_len,
            seed=FLAGS.seed,
            runmode=FLAGS.runmode,
        )
        config = setfit_utils.setup_setfit_training(train_ds_path, val_ds_path, config)
        trainer = Setfit(config)
        trainer()

    else:
        None

    # @TODO cleanup the load function to only output train, val and test datasets
    # @TODO add logging here

    # test the data preprocessing


if __name__ == "__main__":
    app.run(main)
