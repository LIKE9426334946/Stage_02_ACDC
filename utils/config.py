import argparse

import yaml


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="覆盖模型名称")
    args = parser.parse_args()

    with open("./config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.model is not None:
        config["model"]["name"] = args.model

    return config

