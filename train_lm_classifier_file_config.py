import argparse
import json
import os

import torch

from common.constants import DATASETS_DIRS
from eval_lm_classifier import setup_configs_for_eval
from pet import WrapperConfig
from pet.modeling import eval_lm_classifier
from pet.modeling import TrainConfig, EvalConfig
from pet.tasks import PROCESSORS, load_examples, TRAIN_SET, UNLABELED_SET, METRICS, DEFAULT_METRICS
from pet.wrapper import MODEL_CLASSES
from train_lm_classifier import setup_train_lm_config


def main():
    parser = argparse.ArgumentParser(description="Train MLM classifier")

    # Required parameters
    parser.add_argument("--config_path", default=None, type=str, required=True,
                        help="Path of the json configuration file to run evalutaion from.")
    parser.add_argument("--override_visible_gpu", type=int)

    args = parser.parse_args()

    override_visible_gpu = args.override_visible_gpu

    with open(args.config_path, 'r') as fp:
        args_dict = json.load(fp)

    args = argparse.Namespace(**args_dict)

    args.override_visible_gpu = override_visible_gpu

    setup_train_lm_config(args)


if __name__ == '__main__':
    main()
