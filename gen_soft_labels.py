#!/home/fodl/asafmaman/anaconda3/envs/pet/bin/python

import argparse
import os

import torch

from common.constants import DATASETS_DIRS
from pet.tasks import PROCESSORS, load_examples, UNLABELED_SET
from pet.utils import Timer

import pet
import log

logger = log.get_logger('root')


def generate_soft_labels(args):
    timer = Timer('end-to-end')

    if not os.path.exists(args.path):
        logger.info(f'Wrong path, {args.path} does not exist.')

    logger.info(f"Generating soft labels for {args.path}")

    # Setup CUDA, GPU & distributed training
    gpus_string_list = map(lambda x: str(x), args.visible_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus_string_list)
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()

    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()

    args.label_list = processor.get_labels()

    args.data_dir = DATASETS_DIRS[args.task_name]
    unlabeled_data = load_examples(args.task_name, args.data_dir, UNLABELED_SET, num_examples=args.unlabeled_examples)

    pet.generate_soft_labels(path=args.path, reduction=args.reduction,
                             unlabeled_data=unlabeled_data, n_gpu=args.n_gpu, seed=args.seed)

    logger.info(timer.elapsed_str())


def main():
    parser = argparse.ArgumentParser(description="Create soft labels from a folder trained ensemble.")

    parser.add_argument("--path", required=True, help="Path to directory with ensemble set.")

    parser.add_argument("--visible_gpus", required=False, default=[0], type=int, nargs="+",
                        help="Specify which gpus to use. ")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to evaluate on")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--unlabeled_examples", default=-1, type=int,
                        help="The total number of unlabeled examples to use, where -1 equals all examples")
    parser.add_argument("--reduction", default='wmean', choices=['wmean', 'mean'],
                        help="Reduction strategy for merging predictions from multiple PET models. Select either "
                             "uniform weighting (mean) or weighting based on train set accuracy (wmean)")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    generate_soft_labels(args)


if __name__ == "__main__":
    main()
