import argparse
import json
import os
from typing import Tuple
import pandas as pd

import torch

from common.utils import load_unlabeled_with_logits
from pet.tasks import PROCESSORS, load_examples, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from pet.wrapper import MultitaskWrapperConfig
import pet
import log
import logging

logger = log.get_logger('root')


def load_train_lm_config(args) -> Tuple[MultitaskWrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    model_cfg = MultitaskWrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                                       wrapper_type='mlm', train_tasks=args.train_tasks,
                                       max_seq_length=args.max_seq_length, cache_dir=args.cache_dir,
                                       pattern_dict=args.pattern_dict)

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


def setup_train_multi_config(args):
    args.output_dir = args.output_dir.format(**args.__dict__)

    if isinstance(args.pattern_dict, list):
        args.pattern_dict = dict(zip(args.train_tasks, args.pattern_dict))
    elif isinstance(args.pattern_dict, int):
        args.pattern_dict = dict.fromkeys(args.train_tasks, args.pattern_dict)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'console.log'))
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Parameters:\n{}".format(pd.Series(args.__dict__)))

    # Setup CUDA, GPU & distributed training
    gpus_string_list = map(lambda x: str(x), args.visible_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus_string_list)
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.train_tasks = list(map(lambda x: x.lower(), args.train_tasks))

    for task_name in args.train_tasks:
        if task_name not in PROCESSORS: raise ValueError("Task '{}' not found".format(task_name))
    # processors = {task_name: PROCESSORS[task_name]() for task_name in args.train_tasks}

    args.label_list = {task_name: PROCESSORS[task_name]().get_labels() for task_name in args.train_tasks}

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET

    args.data_dirs = {task_name: os.path.join("datasets", task_name) for task_name in args.train_tasks}

    args.metrics = {task_name: METRICS.get(task_name, DEFAULT_METRICS) for task_name in args.train_tasks}

    model_cfg, train_cfg, eval_cfg = load_train_lm_config(args)

    eval_datas = {
        task_name: load_examples(task_name, args.data_dirs[task_name], eval_set, num_examples=args.test_examples)
        for task_name in args.train_tasks}
    if args.use_logits or args.lm_training:
        unlabeled_datas = {
            task_name: load_unlabeled_with_logits(task_name, args.trained_on_examples, args.unlabeled_examples)
            for task_name in args.train_tasks}
    else:
        unlabeled_datas = None
    if not train_cfg.use_logits:
        train_datas = {
            task_name: load_examples(task_name, args.data_dirs[task_name], TRAIN_SET, num_examples=args.train_examples)
            for task_name in args.train_tasks}
    else:
        train_datas = unlabeled_datas

    pet.train_multi_classifier(model_cfg, train_cfg, eval_cfg,
                            pattern_dict=args.pattern_dict, output_dir=args.output_dir,
                            train_datas=train_datas, unlabeled_datas=unlabeled_datas,
                            eval_datas=eval_datas, do_eval=args.do_eval,
                            seed=args.seed)


def main():
    parser = argparse.ArgumentParser(description="Train multitask MLM classifier")

    # Required parameters
    parser.add_argument("--config_path", default=None, type=str, required=True,
                        help="Path of the json configuration file to run experiment from.")

    args = parser.parse_args()

    with open(args.config_path, 'r') as fp:
        args_dict = json.load(fp)

    args = argparse.Namespace(**args_dict)

    setup_train_multi_config(args)


if __name__ == "__main__":
    main()