pip install transformers
pip install absl-py
pip install peft
pip install accelerate
pip install datasets
pip install sentencepiece
pip install wandb

wandb login 7a817528b63f37a967ac341e84ae3ba00dd9f18e

python main.py \
 --name="t5-default-pretrain" \
 --lr=2e-5 \
 --epochs=1 \
 --lr_scheduler=False \
 --model_checkpoint="google/t5-v1_1-base" \
 --tokenizer_checkpoint="google/t5-v1_1-base" \
 --train_dataset="/datastores/tapp-ds/tgt-beta-86k" \
 --max_seq_len=512 \
 --max_ans_len=128 \
 --seed=0 \
 --runmode="pretrain" \
 --lora=False \
 --savedir="./test_path" \
 --load_from_checkpoint = True \
 --checkpoint_state="/datastores/tapp-ds/tapp-t5-beta-200052-step2878/" \
 --checkpoint_step=2878 \
