#!/bin/bash

#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=03:00:00
#SBATCH --mail-user=maximerousseau08@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:1


module purge
module load StdEnv/2020 gcc/9.3.0 arrow/11.0.0 python/3.10

source ~/projects/def-azouaq/mrouss/onlp-env/bin/activate
pip list

cd ~/projects/def-azouaq/mrouss/o-nlp/

wandb offline
nvidia-smi

python main.py configs/t5_sft.toml

python main.py configs/scifive.toml

python main.py configs/bart_sft.toml

python main.py configs/biobart_sft.toml

python main.py configs/bert.toml

python main.py configs/biobert_sft.toml

python main.py configs/pubmedbert_sft.toml

python main.py configs/pubmedbert_squad_sft.toml
