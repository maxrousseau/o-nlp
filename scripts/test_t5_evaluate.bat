python main.py^
 --name="t5-test-evaluate"^
 --model_checkpoint="test_path-t5-test-28-04-2023_14-05-47"^
 --tokenizer_checkpoint="t5-small"^
 --test_dataset="./tmp/bin/val"^
 --max_seq_len=512^
 --max_ans_len=128^
 --seed=0^
 --runmode="t5-evaluate"^
 --savedir="./test_path"^
