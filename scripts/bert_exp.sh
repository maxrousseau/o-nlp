pip install transformers
pip install absl-py
pip install accelerate
pip install evaluate
pip install datasets
pip install sentencepiece
pip install wandb

wandb login $1

python main.py \
 --name="biobert-ft" \
 --lr=2e-5 \
 --epochs=35 \
 --lr_scheduler=False \
 --model_checkpoint="dmis-lab/biobert-base-cased-v1.1" \
 --tokenizer_checkpoint="dmis-lab/biobert-base-cased-v1.1" \
 --train_dataset="/datastores/oqav1/bin/train" \
 --val_dataset="/datastores/oqav1/bin/val" \
 --max_seq_len=512 \
 --seed=0 \
 --runmode="bert-finetune" \
 --savedir="./biobert_ckpts" \

python main.py \
 --name="pubmedbert-ft" \
 --lr=2e-5 \
 --epochs=35 \
 --lr_scheduler=False \
 --model_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
 --tokenizer_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
 --train_dataset="/datastores/oqav1/bin/train" \
 --val_dataset="/datastores/oqav1/bin/val" \
 --max_seq_len=512 \
 --seed=0 \
 --runmode="bert-finetune" \
 --savedir="./pubmedbert_ckpts" \
