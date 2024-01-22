# 🦷 O-NLP : orthodontic natural language processing

This repository contains the code used to conduct fine-tuning on OrthodonticQA (OQA).

## Usage

To download the assets (dataset and pretrained model weights) required to reproduce the experiments use the following command.

```
python etc/build_cache.py assests.toml
```

We use TOML to configure the hyper-parameters for the run. The ```config/```
directory contains the various configurations for the experiments from the
paper.

```scripts/``` contains examples for usage with a SLURM cluster.

To launch a training run:

```
python main.py configs/pubmedbert_sft.toml
```
