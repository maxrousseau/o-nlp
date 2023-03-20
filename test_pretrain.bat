python main.py^
 --name="t5-default-pretrain"^
 --lr=2e-5^
 --epochs=1^
 --lr_scheduler=False^
 --model_checkpoint="google/t5-v1_1-small"^
 --tokenizer_checkpoint="google/t5-v1_1-small"^
 --train_dataset="./tmp/ngram-tgt-test.json"^
 --max_seq_len=384^
 --max_ans_len=128^
 --seed=0^
 --runmode="pretrain"^
 --lora=True^
 --savedir="./test_path"^
