import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Tuple
import pandas as pd

import torch

from common.constants import DATASETS_DIRS
from common.utils import load_unlabeled_with_logits
from common.utils import trained_lm_path
from pet import LogitsList
from pet.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from pet.utils import eq_div, Timer
from pet.wrapper import WRAPPER_TYPES, MODEL_CLASSES, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig
import pet
import log
import logging

logger = log.get_logger('root')


def load_train_lm_config(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type='mlm', task_name=args.task_name, label_list=args.label_list,
                              max_seq_length=args.max_seq_length,
                              cache_dir=args.cache_dir)

    train_cfg = pet.TrainConfig(device=args.device, per_gpu_train_batch_size=args.per_gpu_train_batch_size,
                                per_gpu_unlabeled_batch_size=args.per_gpu_unlabeled_batch_size, n_gpu=args.n_gpu,
                                num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm, lm_training=args.lm_training, alpha=args.alpha,
                                use_logits=args.use_logits)

    eval_cfg = pet.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.per_gpu_eval_batch_size)

    return model_cfg, train_cfg, eval_cfg


def setup_train_lm_config(args):
    args.output_dir = args.output_dir.format(**args.__dict__)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)


    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'console.log'))
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    # Setup CUDA, GPU & distributed training
    if args.override_visible_gpu:
        args.visible_gpus = [args.override_visible_gpu]
    gpus_string_list = map(lambda x: str(x), args.visible_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus_string_list)
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    logger.info("Parameters:\n{}".format(pd.Series(args.__dict__)))

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # Prepare task
    args.task_name = args.task_name.lower()

    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()

    args.label_list = processor.get_labels()

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET

    args.data_dir = os.path.join("datasets", args.task_name)

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    model_cfg, train_cfg, eval_cfg = load_train_lm_config(args)

    eval_data = load_examples(args.task_name, args.data_dir, eval_set, num_examples=args.test_examples)

    if not train_cfg.use_logits:
        train_data = load_examples(args.task_name, args.data_dir, TRAIN_SET, num_examples=args.train_examples)
        unlabeled_data = load_examples(
            args.task_name, args.data_dir, UNLABELED_SET, num_examples=args.unlabeled_examples)
    else:
        unlabeled_data = load_unlabeled_with_logits(args.task_name, args.trained_on_examples, args.unlabeled_examples)
        train_data = unlabeled_data

    pet.train_lm_classifier(model_cfg, train_cfg, eval_cfg,
                            pattern_id=args.pattern_id, output_dir=args.output_dir,
                            train_data=train_data, unlabeled_data=unlabeled_data,
                            eval_data=eval_data, do_eval=args.do_eval,
                            seed=args.seed)


def main():
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")
    #
    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    parser.add_argument("--pattern_id", default=0, type=int,
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--lm_training", action='store_true',
                        help="Whether to use language modeling as auxiliary task (only for PET)")
    parser.add_argument("--use_logits", action='store_true',
                        help="Whether to train on unlabeled data or regular train data.")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling task (only for PET)")
    parser.add_argument("--temperature", default=2, type=float,
                        help="Temperature used for combining PVPs (only for PET)")
    parser.add_argument("--decoding_strategy", default='default', choices=['default', 'ltr', 'parallel'],
                        help="The decoding strategy for PET with multiple masks (only for PET)")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument("--per_gpu_unlabeled_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for auxiliary language modeling examples in PET.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--unlabeled_examples", default=-1, type=int,
                        help="The total number of unlabeled examples to use, where -1 equals all examples")
    parser.add_argument("--cache_dir", default=".cache/huggingface", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")
    parser.add_argument("--visible_gpus", required=False, default=[0], type=int, nargs="+",
                        help="Specify which gpus to use. ")

    args = parser.parse_args()

    setup_train_lm_config(args)


if __name__ == "__main__":
    main()
