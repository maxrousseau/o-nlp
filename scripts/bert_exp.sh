pip install transformers
pip install absl-py
pip install accelerate
pip install evaluate
pip install datasets
pip install sentencepiece
pip install wandb

wandb login 7a817528b63f37a967ac341e84ae3ba00dd9f18e

python main.py \
 --name="pubmedbert-squad-sft" \
 --lr=3e-5 \
 --epochs=2 \
 --lr_scheduler=False \
 --model_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
 --tokenizer_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
 --max_seq_len=512 \
 --seed=0 \
 --runmode="bert-squad-finetune" \
 --savedir="./pubmedbert_squad_sft_ckpts" \
 --only_cls_head=False \

 python main.py \
 --name="pubmedbert-squad-sft-headonly" \
 --lr=3e-5 \
 --epochs=2 \
 --lr_scheduler=False \
 --model_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
 --tokenizer_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
 --max_seq_len=512 \
 --seed=0 \
 --runmode="bert-squad-finetune" \
 --savedir="./pubmedbert_squad_sft_ckpts" \
 --only_cls_head=True \
