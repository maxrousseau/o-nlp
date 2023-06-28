import os

from huggingface_hub import snapshot_download
import tomli
import argparse


# ...
def main():
    parser = argparse.ArgumentParser(
        description="Read TOML configuration file for training and inference"
    )
    parser.add_argument(
        "config",
        metavar="path",
        type=str,
        nargs=1,
        help="path to the tomli file describing the items to be cached",
    )
    # @TODO :: absl-py seems quite slow, try radicli...
    args = parser.parse_args()

    config_path = args.config[0]
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    models = config["models"]
    datasets = config["datasets"]

    asset_dir = os.path.abspath(config["dir"])

    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)

        model_path = os.path.join(asset_dir, "models")
        data_path = os.path.join(asset_dir, "datasets")

        os.makedirs(model_path)
        os.makedirs(data_path)
    else:
        raise OSError("asset directory already exists")

    for m in models:
        m_path = os.path.join(model_path, m)
        snapshot_download(repo_id=m, local_dir=m_path, repo_type="model")
    for d in datasets:
        d_path = os.path.join(data_path, d)
        snapshot_download(repo_id=d, local_dir=d_path, repo_type="dataset")


if __name__ == "__main__":
    main()
