python main.py^
 --name="t5-default"^
 --lr=2e-5^
 --epochs=1^
 --lr_scheduler=True^
 --model_checkpoint="google/t5-v1_1-small"^
 --tokenizer_checkpoint="google/t5-v1_1-small"^
 --train_dataset="./tmp/oqa_v0.1_train.json"^
 --test_dataset="./tmp/oqa_v0.1_test.json"^
 --max_seq_len=384^
 --max_ans_len=128^
 --seed=0^
 --runmode="finetune"^
 --savedir="./test_path"

::  for training datastet just oqa or similar formatted dataset will work (prefinetuning with SQuAD is hardcoded)
