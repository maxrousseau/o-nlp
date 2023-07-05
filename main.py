#!/usr/bin/env python
import os
import logging
import argparse
import tomli

from models import t5_utils, bert_utils, bart_utils
from train import *

from tacoma import tacoma


from datasets import Dataset


def main():
    """get args and call"""
    parser = argparse.ArgumentParser(
        description="Read TOML configuration file for training and inference"
    )
    parser.add_argument(
        "config",
        metavar="path",
        type=str,
        nargs=1,
        help="path to the configuration file",
    )
    args = parser.parse_args()

    config_path = args.config[0]
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    runmode = config["mode"]["runmode"]
    model_config = config["model"]
    dataset_config = config["dataset"]
    tokenizer_config = config["tokenizer"]
    hyperparameter_config = config["hyperparameters"]
    misc_config = config["misc"]

    if runmode == "t5-finetune":
        t5_config = t5_utils.T5CFG(
            name=model_config["name"],
            lr=hyperparameter_config["learning_rate"],
            lr_scheduler=hyperparameter_config["learning_rate_scheduler"],
            n_epochs=hyperparameter_config["num_epochs"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            checkpoint_savedir=misc_config["save_dir"],
            max_seq_length=model_config["max_seq_len"],
            max_ans_length=model_config["max_ans_len"],
            seed=hyperparameter_config["seed"],
        )
        t5_config = t5_utils.setup_finetune_t5(
            dataset_config["repository"],
            t5_config,
        )
        tuner = FinetuneT5(t5_config)
        run_state = tuner()

        if config["mode"]["eval_after_training"]:
            t5_config.model_checkpoint = run_state["checkpoint_bestf1"]
            t5_config = t5_utils.setup_evaluate_t5(
                dataset_config["repository"], t5_config
            )
            evaluater = EvaluateT5(t5_config, output_dir=tuner.output_dir)
            evaluater()

    elif runmode == "t5-evaluate":
        t5_config = t5_utils.T5CFG(
            name=model_config["name"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            max_seq_length=model_config["max_seq_len"],
            max_ans_length=model_config["max_ans_len"],
            seed=hyperparameter_config["seed"],
        )
        t5_config = t5_utils.setup_evaluate_t5(dataset_config["repository"], t5_config)
        evaluator = EvaluateT5(t5_config)
        evaluator()

    elif runmode == "bart-finetune":
        bart_config = bart_utils.BARTCFG(
            name=model_config["name"],
            lr=hyperparameter_config["learning_rate"],
            lr_scheduler=hyperparameter_config["learning_rate_scheduler"],
            n_epochs=hyperparameter_config["num_epochs"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            checkpoint_savedir=misc_config["save_dir"],
            max_seq_length=model_config["max_seq_len"],
            max_ans_length=model_config["max_ans_len"],
            seed=hyperparameter_config["seed"],
        )
        bart_config = bart_utils.setup_finetune_bart(
            dataset_config["repository"],
            bart_config,
        )
        tuner = FinetuneBART(bart_config)
        run_state = tuner()

        if config["mode"]["eval_after_training"]:
            bart_config.model_checkpoint = run_state["checkpoint_bestf1"]
            bart_config = bart_utils.setup_evaluate_bart(
                dataset_config["repository"], bart_config
            )
            evaluater = EvaluateBART(bart_config, output_dir=tuner.output_dir)
            evaluater()

    elif runmode == "bart-evaluate":
        bart_config = bart_utils.BARTCFG(
            name=model_config["name"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            max_seq_length=model_config["max_seq_len"],
            max_ans_length=model_config["max_ans_len"],
            seed=hyperparameter_config["seed"],
        )
        bart_config = bart_utils.setup_evaluate_bart(
            dataset_config["repository"], bart_config
        )
        # @HERE :: make sure setup function is good, then finish training loop
        print(bart_config)
        evaluater = EvaluateBART(bart_config)
        evaluater()

    elif runmode == "bert-pretrain":
        bert_config = bert_utils.BERTCFG(
            name=model_config["name"],
            lr=hyperparameter_config["learning_rate"],
            lr_scheduler=hyperparameter_config["learning_rate_scheduler"],
            n_epochs=hyperparameter_config["num_epochs"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            checkpoint_savedir=misc_config["save_dir"],
            max_length=model_config["max_seq_len"],
            seed=hyperparameter_config["seed"],
        )
        bert_config = bert_utils.setup_pretrain_bert(
            dataset_config["repository"], bert_config
        )
        tuner = PretrainBERT(bert_config)
        tuner()

    elif runmode == "bert-finetune":
        bert_config = bert_utils.BERTCFG(
            name=model_config["name"],
            lr=hyperparameter_config["learning_rate"],
            lr_scheduler=hyperparameter_config["learning_rate_scheduler"],
            n_epochs=hyperparameter_config["num_epochs"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            checkpoint_savedir=misc_config["save_dir"],
            max_length=model_config["max_seq_len"],
            seed=hyperparameter_config["seed"],
            bitfit=model_config["bitfit"],
            append_special_token=tokenizer_config["append_special_tokens"],
        )
        bert_config = bert_utils.setup_finetuning_oqa(
            dataset_config["repository"],
            bert_config,
        )
        tuner = FinetuneBERT(bert_config)
        run_state = tuner()
        if config["mode"]["eval_after_training"]:
            bert_config.model_checkpoint = run_state["checkpoint_bestf1"]
            bert_config = bert_utils.setup_evaluate_oqa(
                dataset_config["repository"], bert_config
            )
            evaluater = EvaluateBERT(bert_config, output_dir=tuner.output_dir)
            evaluater()

    elif runmode == "bert-taskdistil":
        bert_config = bert_utils.TaskDistillationCFG(
            name=model_config["name"],
            lr=hyperparameter_config["learning_rate"],
            lr_scheduler=hyperparameter_config["learning_rate_scheduler"],
            n_epochs=hyperparameter_config["num_epochs"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            checkpoint_savedir=misc_config["save_dir"],
            max_length=model_config["max_seq_len"],
            seed=hyperparameter_config["seed"],
            bitfit=model_config["bitfit"],
            append_special_token=tokenizer_config["append_special_tokens"],
            teacher_model_checkpoint=model_config["teacher_checkpoint"],
            teacher_tokenizer_checkpoint=tokenizer_config["teacher_checkpoint"],
            temperature=hyperparameter_config["temperature"],
            alpha=hyperparameter_config["alpha"],
            train_batch_size=hyperparameter_config["training_batch_size"],
            val_batch_size=hyperparameter_config["validation_batch_size"],
        )
        bert_config = bert_utils.setup_finetuning_oqa(
            dataset_config["repository"],
            bert_config,
        )
        print(bert_config)
        tuner = TaskDistillationBERT(bert_config)
        tuner()

    elif runmode == "bert-squad-finetune":
        bert_config = bert_utils.BERTCFG(
            name=model_config["name"],
            lr=hyperparameter_config["learning_rate"],
            lr_scheduler=hyperparameter_config["learning_rate_scheduler"],
            n_epochs=hyperparameter_config["num_epochs"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            checkpoint_savedir=misc_config["save_dir"],
            max_length=model_config["max_seq_len"],
            seed=hyperparameter_config["seed"],
        )
        bert_config = bert_utils.setup_finetuning_squad(
            dataset_config["repository"],
            bert_config,
        )
        print(bert_config)
        tuner = FinetuneBERT(bert_config)
        tuner()

    elif runmode == "bert-evaluate":
        bert_config = bert_utils.BERTCFG(
            name=model_config["name"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            max_length=model_config["max_seq_len"],
            seed=hyperparameter_config["seed"],
        )
        bert_config = bert_utils.setup_evaluate_oqa(
            dataset_config["repository"], bert_config
        )

        print(bert_config)
        evaluater = EvaluateBERT(bert_config)
        evaluater()

    elif runmode == "t5-pretrain":
        t5_config = t5_utils.T5CFG(
            name=model_config["name"],
            lr=hyperparameter_config["learning_rate"],
            lr_scheduler=hyperparameter_config["learning_rate_scheduler"],
            n_epochs=hyperparameter_config["num_epochs"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            checkpoint_savedir=misc_config["save_dir"],
            max_seq_length=model_config["max_seq_len"],
            max_ans_length=model_config["max_ans_len"],
            seed=hyperparameter_config["seed"],
            checkpoint_step=model_config["checkpoint_step"],
            checkpoint_state=model_config["checkpoint_state"],
            load_from_checkpoint=model_config["load_from_last_checkpoint"],
        )
        t5_config = t5_utils.setup_pretrain_t5(dataset_config["repository"], t5_config)
        print(t5_config)
        pretrainer = PretrainT5(t5_config)
        pretrainer()

    elif runmode == "tacoma":
        tacoma_config = tacoma.TACOMA_CFG(
            name=model_config["name"],
            lr=hyperparameter_config["learning_rate"],
            lr_scheduler=hyperparameter_config["learning_rate_scheduler"],
            n_epochs=hyperparameter_config["num_epochs"],
            model_checkpoint=model_config["checkpoint"],
            tokenizer_checkpoint=tokenizer_config["checkpoint"],
            checkpoint_savedir=misc_config["save_dir"],
            max_seq_length=model_config["max_seq_len"],
            max_ans_length=model_config["max_ans_len"],
            seed=hyperparameter_config["seed"],
        )
        tacoma_config = tacoma.setup_tacoma_training(
            dataset_config["repository"],
            tacoma_config,
        )
        tuner = tacoma.TacomaT5(tacoma_config)
        run_state = tuner()

    # elif runmode == "train-classifier":
    #     setfit_config = setfit_utils.SFCFG(
    #         name=model_config["name"],
    #         lr=hyperparameter_config["learning_rate"],
    #         n_epochs=hyperparameter_config["num_epochs"],
    #         model_checkpoint=model_config["checkpoint"],
    #         checkpoint_savedir=misc_config["save_dir"],
    #         max_length=model_config["max_seq_len"],
    #         seed=hyperparameter_config["seed"],
    #     )
    #     setfit_config = setfit_utils.setup_setfit_training(
    #         dataset_config["repository"],
    #         setfit_config,
    #     )
    #     trainer = Setfit(setfit_config)
    #     trainer()

    # else:
    #     assert TypeError


if __name__ == "__main__":
    main()
