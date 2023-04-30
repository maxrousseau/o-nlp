pip install transformers
pip install absl-py
pip install accelerate
pip install evaluate
pip install datasets
pip install sentencepiece
pip install wandb

wandb login 7a817528b63f37a967ac341e84ae3ba00dd9f18e


python main.py \
     --name="scifive-base-sft-oqa" \
     --lr=1e-4 \
     --epochs=35 \
     --lr_scheduler=False \
     --model_checkpoint="razent/SciFive-base-Pubmed_PMC"\
     --tokenizer_checkpoint="razent/SciFive-base-Pubmed_PMC" \
     --train_dataset="/datastores/oqav1/bin/train" \
     --val_dataset="/datastores/oqav1/bin/val" \
     --max_seq_len=768 \
     --max_ans_len=256 \
     --seed=0 \
     --runmode="t5-finetune" \
     --savedir="./scifive_ckpts" \

python main.py \
    --name="biobart-base-sft-oqa" \
     --lr=2e-5 \
     --epochs=35 \
     --lr_scheduler=False \
     --model_checkpoint="GanjinZero/biobart-base" \
     --tokenizer_checkpoint="GanjinZero/biobart-base" \
     --train_dataset="/datastores/oqav1/bin/train" \
     --val_dataset="/datastores/oqav1/bin/val" \
     --max_seq_len=1024 \
     --max_ans_len=256 \
     --seed=0 \
     --runmode="bart-finetune" \
     --savedir="./biobart_ckpts" \
