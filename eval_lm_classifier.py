import argparse
import os

import torch

from common.constants import DATASETS_DIRS
from pet import WrapperConfig
from pet.modeling import eval_lm_classifier
from pet.modeling import TrainConfig, EvalConfig
from pet.tasks import PROCESSORS, load_examples, TRAIN_SET, UNLABELED_SET, METRICS, DEFAULT_METRICS
from pet.wrapper import MODEL_CLASSES


def setup_configs_for_eval(args):
    processor = PROCESSORS[args.task_name]()
    label_list = processor.get_labels()

    gpus_string_list = map(lambda x: str(x), args.visible_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus_string_list)
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type='mlm', task_name=args.task_name, label_list=label_list, max_seq_length=256,
                              pattern_id=args.pattern_id, cache_dir=args.cache_dir)

    eval_cfg = EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                          per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size)

    eval_data = load_examples(args.task_name, data_dir=DATASETS_DIRS[args.task_name],
                              set_type=args.eval_set, num_examples=args.test_examples)

    eval_lm_classifier(model_config=model_cfg, eval_config=eval_cfg,
                       output_dir=args.output_dir, eval_data=eval_data, seed=args.seed)


def main():
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--pattern_id", default=0, type=int, required=True,
                        help="The ids of the PVPs to be used (only for PET)")

    parser.add_argument("--pet_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--cache_dir", default=".cache/huggingface", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")
    parser.add_argument("--visible_gpus", required=False, default=[0], type=int, nargs="+",
                        help="Specify which gpus to use. ")

    args = parser.parse_args()

    setup_configs_for_eval(args)


if __name__ == '__main__':
    main()
