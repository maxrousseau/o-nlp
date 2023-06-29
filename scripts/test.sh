#!/bin/bash

#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:30:0
#SBATCH --mail-user=maximerousseau08@gmail.com
#SBATCH --mail-type=ALL


source ~/projects/def-azouaq/mrouss/onlp-env/bin/activate
module purge
module load StdEnv/2020 gcc/9.3.0 arrow/11.0.0 python/3.10

cd ~/projects/def-azouaq/mrouss/o-nlp/

python main.py config/test-bert.toml
