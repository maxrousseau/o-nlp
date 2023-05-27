# keep everything as is and simply keep decreasing the learning rate, that might be the issue...

pip install transformers
pip install absl-py
pip install accelerate
pip install evaluate
pip install datasets
pip install sentencepiece
pip install wandb
pip install setfit

wandb login 7a817528b63f37a967ac341e84ae3ba00dd9f18e

python main.py --name="tacomav2" \
       --lr=6e-5 \
       --epochs=1 \
       --lr_scheduler=True \
       --model_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
       --tokenizer_checkpoint="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
       --max_seq_len=512 \
       --seed=0 \
       --runmode="bert-pretrain" \
       --savedir="./tacomav2" \
