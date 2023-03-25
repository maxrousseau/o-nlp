python main.py \
 --name="t5-default-pretrain" \
 --lr=1e-4 \
 --epochs=1 \
 --lr_scheduler=False \
 --model_checkpoint="google/t5-v1_1-base" \
 --tokenizer_checkpoint="google/t5-v1_1-base" \
 --train_dataset="/datastores/tgt-beta-0-85t-86k" \
 --max_seq_len=512 \
 --max_ans_len=128 \
 --seed=0 \
 --runmode="pretrain" \
 --lora=False \
 --savedir="./test_path"
