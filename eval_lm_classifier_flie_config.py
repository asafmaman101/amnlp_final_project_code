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


def main():
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")

    # Required parameters
    parser.add_argument("--config_path", default=None, type=str, required=True,
                        help="Path of the json configuration file to run evalutaion from.")

    args = parser.parse_args()

    with open(args.config_path, 'r') as fp:
        args_dict = json.load(fp)

    args = argparse.Namespace(**args_dict)

    setup_configs_for_eval(args)


if __name__ == '__main__':
    main()
