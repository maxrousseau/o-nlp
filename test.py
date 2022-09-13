from load_data import load_mini_oqa
from bert import OrthoBert
from bert import default_config

import os

train = os.path.abspath("/home/max/datasets/mini-oqa/oqa_v0.1_train.json")
test = os.path.abspath("/home/max/datasets/mini-oqa/oqa_v0.1_test.json")
# train = os.path.abspath("/home/max/datasets/bioasq9b-dummy/bioasq_9b_train.json")
# test = os.path.abspath("/home/max/datasets/bioasq9b-dummy/bioasq_9b_test.json")

dset = load_mini_oqa(train, test)

default_config["train_dev_dataset"] = dset[0]
default_config["val_dev_dataset"] = dset[1]
default_config["train_dataset"] = dset[2]
default_config["test_dataset"] = dset[3]

obert = OrthoBert(**default_config)
print(obert.train_dev_dataset)
print(obert.val_dev_dataset)
obert.debug("qa")

obert.finetune("qa")
