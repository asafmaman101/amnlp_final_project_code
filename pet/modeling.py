# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import glob
import json
import os
import pickle
import random
import statistics
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

import log
from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div, \
    Timer
from pet.wrapper import TransformerModelWrapper, SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER, WrapperConfig, \
    TransformerMultitaskModelWrapper, MultitaskWrapperConfig

from pet.utils import LOG_CONST_WIDTH

logger = log.get_logger('root')


class PetConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    """Configuration for training a model."""

    def __init__(self, device: str = None, per_gpu_train_batch_size: int = 8, per_gpu_unlabeled_batch_size: int = 8,
                 n_gpu: int = 1, num_train_epochs: int = 3, max_steps: int = -1, gradient_accumulation_steps: int = 1,
                 weight_decay: float = 0.0, learning_rate: float = 5e-5, adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0, max_grad_norm: float = 1, lm_training: bool = False, use_logits: bool = False,
                 alpha: float = 0.9999, temperature: float = 1):
        """
        Create a new training config.

        :param device: the device to use ('cpu' or 'gpu')
        :param per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train for
        :param max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
        :param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the maximum learning rate to use
        :param adam_epsilon: the epsilon value for Adam
        :param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        :param max_grad_norm: the maximum norm for the gradient
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use each training example's logits instead of its label (used for distillation)
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for distillation
        """
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.per_gpu_unlabeled_batch_size = per_gpu_unlabeled_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.lm_training = lm_training
        self.use_logits = use_logits
        self.alpha = alpha
        self.temperature = temperature


class EvalConfig(PetConfig):
    """Configuration for evaluating a model."""

    def __init__(self, device: str = None, n_gpu: int = 1, per_gpu_eval_batch_size: int = 8,
                 metrics: List[str] = None, decoding_strategy: str = 'default', priming: bool = False):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr', or 'parallel')
        :param priming: whether to use priming
        """
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics
        self.decoding_strategy = decoding_strategy
        self.priming = priming


class IPetConfig(PetConfig):
    """Configuration for iterative PET training."""

    def __init__(self, generations: int = 3, logits_percentage: float = 0.25, scale_factor: float = 5,
                 n_most_likely: int = -1):
        """
        Create a new iPET config.

        :param generations: the number of generations to train
        :param logits_percentage: the percentage of models to use for annotating training sets for the next generation
        :param scale_factor: the factor by which the training set is increased for each generation
        :param n_most_likely: If >0, in the first generation the n_most_likely examples per label are chosen even
                              if their predicted label is different
        """
        self.generations = generations
        self.logits_percentage = logits_percentage
        self.scale_factor = scale_factor
        self.n_most_likely = n_most_likely


def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config)
    return model


def init_multi_model(config: MultitaskWrapperConfig) -> TransformerMultitaskModelWrapper:
    """Initialize a new model from the given config."""
    for pattern_id in config.pattern_dict.values():
        assert pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerMultitaskModelWrapper(config)
    return model


def train_ipet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
               ensemble_eval_config: EvalConfig, ipet_config: IPetConfig, final_model_config: WrapperConfig,
               final_train_config: TrainConfig, final_eval_config: EvalConfig, pattern_ids: List[int], output_dir: str,
               ensemble_repetitions: int = 3, final_repetitions: int = 1, reduction: str = 'wmean',
               train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None,
               eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True, seed: int = 42):
    """
    Train and evaluate a new iPET model for a given task.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param ipet_config: the iPET training configuration
    :param final_model_config: the model configuration for the final distilled sequence classifier
    :param final_train_config: the training configuration for the final distilled sequence classifier
    :param final_eval_config: the evaluation configuration for the final distilled sequence classifier
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ensemble_repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param final_repetitions: the number of training repetitions for the final distilled sequence classifier
    :param reduction: the reduction strategy for merging predictions, either 'mean' or 'wmean'
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """
    for gen in range(ipet_config.generations):
        gen_output_dir = os.path.join(output_dir, f'g{gen}')

        # Step 1: Train an ensemble of models corresponding to individual patterns
        ipet_data_dir = os.path.join(output_dir, f'g{gen - 1}', 'next-gen-train-data') if gen > 0 else None
        train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids,
                           gen_output_dir, ipet_data_dir=ipet_data_dir,
                           repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                           eval_data=eval_data, do_train=do_train, do_eval=do_eval, save_unlabeled_logits=True)

        # Step 2: Use the model to annotate examples for the next generation
        original_data_size = len(train_data) if train_data else 10 / ipet_config.scale_factor
        num_new_examples = int(original_data_size * (ipet_config.scale_factor ** (gen + 1)) - len(train_data))
        generate_ipet_train_sets(train_data=train_data, unlabeled_data=unlabeled_data,
                                 labels=ensemble_model_config.label_list, logits_dir=gen_output_dir,
                                 output_dir=os.path.join(gen_output_dir, 'next-gen-train-data'), reduction=reduction,
                                 num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage,
                                 n_most_likely=ipet_config.n_most_likely if gen == 0 else -1, seed=seed)

    # Step 3: Merge the annotations created by each individual model
    logits_dir = os.path.join(output_dir, f'g{ipet_config.generations - 1}')
    logits_file = os.path.join(logits_dir, 'unlabeled_logits.txt')
    merge_logits(logits_dir, logits_file, reduction)
    logits = LogitsList.load(logits_file).logits
    assert len(logits) == len(unlabeled_data)
    logger.info("Got {} logits from file {}".format(len(logits), logits_file))
    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    # Step 4: Train the final sequence classifier model
    final_model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    final_train_config.use_logits = True

    train_classifier(final_model_config, final_train_config, final_eval_config, os.path.join(output_dir, 'final'),
                     repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                     eval_data=eval_data, do_train=do_train, do_eval=do_eval)


def train_pet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
              ensemble_eval_config: EvalConfig, final_model_config: WrapperConfig, final_train_config: TrainConfig,
              final_eval_config: EvalConfig, pattern_ids: List[int], output_dir: str, ensemble_repetitions: int = 3,
              final_repetitions: int = 1, reduction: str = 'wmean', train_data: List[InputExample] = None,
              unlabeled_data: List[InputExample] = None, eval_data: List[InputExample] = None, do_train: bool = True,
              do_eval: bool = True, no_distillation: bool = False, seed: int = 42):
    """
    Train and evaluate a new PET model for a given task.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param final_model_config: the model configuration for the final distilled sequence classifier
    :param final_train_config: the training configuration for the final distilled sequence classifier
    :param final_eval_config: the evaluation configuration for the final distilled sequence classifier
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ensemble_repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param final_repetitions: the number of training repetitions for the final distilled sequence classifier
    :param reduction: the reduction strategy for merging predictions, either 'mean' or 'wmean'
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param no_distillation: if true, no distillation is performed
    :param seed: the random seed to use
    """

    # Step 1: Train an ensemble of models corresponding to individual patterns

    logger.info('')
    logger.info("".center(LOG_CONST_WIDTH, '#'))
    logger.info("STEP 1: TRAINING ENSEMBLE MODELS".center(LOG_CONST_WIDTH, '#'))
    logger.info("".center(LOG_CONST_WIDTH, '#'))

    step_one_timer = Timer('step 1')

    train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids, output_dir,
                       repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                       eval_data=eval_data, do_train=do_train, do_eval=do_eval,
                       save_unlabeled_logits=not no_distillation, seed=seed)

    logger.info(step_one_timer.elapsed_str())

    if no_distillation:
        return

    # Step 2: Merge the annotations created by each individual model

    logger.info('')
    logger.info("".center(LOG_CONST_WIDTH, '#'))
    logger.info("STEP 2: CREATING SOFT LABELS".center(LOG_CONST_WIDTH, '#'))
    logger.info("".center(LOG_CONST_WIDTH, '#'))

    step_two_timer = Timer('step 2')

    logits_file = os.path.join(output_dir, 'unlabeled_logits.txt')
    logger.info('Merging ensemble results...')
    merge_logits(output_dir, logits_file, reduction)
    logger.info('Finished merging.')
    logits = LogitsList.load(logits_file).logits
    assert len(logits) == len(unlabeled_data)
    logger.info("Got {} logits from file {}".format(len(logits), logits_file))
    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    logger.info(step_two_timer.elapsed_str())

    # Step 3: Train the final sequence classifier model

    logger.info('')
    logger.info("".center(LOG_CONST_WIDTH, '#'))
    logger.info("STEP 3: TRAIN CLASSIFIER".center(LOG_CONST_WIDTH, '#'))
    logger.info("".center(LOG_CONST_WIDTH, '#'))

    step_three_timer = Timer('step 3')

    final_model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    final_train_config.use_logits = True

    train_classifier(final_model_config, final_train_config, final_eval_config, os.path.join(output_dir, 'final'),
                     repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                     eval_data=eval_data, do_train=do_train, do_eval=do_eval, seed=seed)

    logger.info(step_three_timer.elapsed_str())


def generate_soft_labels(path: str, reduction: str = 'wmean',
                         unlabeled_data: List[InputExample] = None, n_gpu: int = 1,
                         seed: int = 42):
    # train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids, output_dir,
    #                    repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
    #                    eval_data=eval_data, do_train=do_train, do_eval=do_eval,
    #                    save_unlabeled_logits=not no_distillation, seed=seed)

    # def train_pet_ensemble(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig,
    #                        pattern_ids: List[int], output_dir: str, ipet_data_dir: str = None, repetitions: int = 3,
    #                        train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None,
    #                        eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True,
    #                        save_unlabeled_logits: bool = False, seed: int = 42):

    set_seed(seed)

    models_pathes = glob.glob(os.path.join(path, 'p*-i*'))

    models_logits = {}

    for model_path in models_pathes:
        model_str_id = os.path.basename(model_path)
        pattern_id, iteration = model_str_id[1], model_str_id[4]

        wrapper = TransformerModelWrapper.from_pretrained(model_path)
        eval_config = EvalConfig.load(os.path.join(model_path, 'eval_config.json'))
        eval_config.n_gpu = n_gpu
        logger.info(f'Evaluating unlabeled logits for p.{pattern_id}-i.{iteration}')

        models_logits[model_str_id] = evaluate(wrapper, unlabeled_data, eval_config)['logits']

        wrapper.model = None
        wrapper = None
        torch.cuda.empty_cache()

    all_logits_lists = []

    for model_path in models_pathes:
        results_file = os.path.join(model_path, 'results.txt')
        model_str_id = os.path.basename(model_path)
        logits = models_logits[model_str_id].tolist()

        if reduction == 'mean':
            result_train = 1
        else:
            if not os.path.exists(results_file):
                logger.warning(f"Skipping model_path '{model_path}' because 'results.txt' not found")
                continue

            with open(results_file, 'r') as fh:
                results = ast.literal_eval(fh.read())
                result_train = results['train_set_before_training']

        logger.info("File {}: Score = {}, #Logits = {}, #Labels = {}".format(
            results_file, result_train, len(logits), len(logits[0])))

        loglist = LogitsList(score=result_train, logits=logits)
        all_logits_lists.append(loglist)

    merged_loglist = merge_logits_lists(all_logits_lists, reduction=reduction)

    all_logits = merged_loglist.logits

    assert len(all_logits) == len(unlabeled_data)

    for example, example_logits in zip(unlabeled_data, all_logits):
        example.logits = example_logits

    with open(os.path.join(path, 'unlabeled_data_object.pkl'), 'wb') as fp:
        pickle.dump(unlabeled_data, fp)


def train_classifier(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig, output_dir: str,
                     repetitions: int = 3, train_data: List[InputExample] = None,
                     unlabeled_data: List[InputExample] = None, eval_data: List[InputExample] = None,
                     do_train: bool = True, do_eval: bool = True, seed: int = 42):
    """
    Train and evaluate a sequence classification model.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    train_pet_ensemble(model_config, train_config, eval_config, pattern_ids=[0], output_dir=output_dir,
                       repetitions=repetitions,
                       train_data=train_data, unlabeled_data=unlabeled_data, eval_data=eval_data, do_train=do_train,
                       do_eval=do_eval, seed=seed)


def train_lm_classifier(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig,
                        pattern_id: List[int], output_dir: str,
                        train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None,
                        eval_data: List[InputExample] = None, do_eval: bool = True,
                        seed: int = 42):
    set_seed(seed)
    results_dict = {}
    os.makedirs(output_dir, exist_ok=True)

    model_config.pattern_id = pattern_id
    wrapper = init_model(model_config)

    assert wrapper.config.wrapper_type == MLM_WRAPPER

    results_dict.update({'wrapper_config': model_config.__dict__})
    results_dict.update({'train_config': train_config.__dict__})
    results_dict.update({'eval_config': eval_config.__dict__})
    results_dict.update(train_single_model(wrapper, train_data, train_config, eval_config,
                                           unlabeled_data=unlabeled_data, return_train_set_results=False,
                                           labelled=True))

    wrapper.save(output_dir)
    train_config.save(os.path.join(output_dir, 'train_config.json'))
    eval_config.save(os.path.join(output_dir, 'eval_config.json'))

    if do_eval:
        eval_result = evaluate(wrapper, eval_data, eval_config)
        results_dict['test_set_after_training'] = eval_result['scores']

    with open(os.path.join(output_dir, 'results.json'), 'w') as fh:
        json.dump(results_dict, fh, indent=4)



def eval_lm_classifier(model_config: WrapperConfig, eval_config: EvalConfig, output_dir: str,
                       eval_data: List[InputExample] = None, seed: int = 42):
    set_seed(seed)
    results_dict = {}

    os.makedirs(output_dir, exist_ok=True)

    wrapper = init_model(model_config)

    assert wrapper.config.wrapper_type == MLM_WRAPPER
    eval_result = evaluate(wrapper, eval_data, eval_config)

    results_dict.update({'wrapper_config': model_config.__dict__})
    results_dict.update({'eval_config': eval_config.__dict__})
    results_dict['test_set_after_training'] = eval_result['scores']

    with open(os.path.join(output_dir, 'results.json'), 'w') as fh:
        json.dump(results_dict, fh, indent=4)

    print(json.dumps(results_dict,indent=2))


def train_multi_classifier(model_config: MultitaskWrapperConfig, train_config: TrainConfig, eval_config: EvalConfig,
                           pattern_dict: Dict[str, List[int]], output_dir: str,
                           train_datas: Dict[str, List[InputExample]] = None,
                           unlabeled_datas: Dict[str, List[InputExample]] = None,
                           eval_datas: Dict[str, List[InputExample]] = None, do_eval: bool = True,
                           seed: int = 42):
    set_seed(seed)
    results_dict = {}

    os.makedirs(output_dir, exist_ok=True)

    model_config.pattern_dict = pattern_dict
    wrapper = init_multi_model(model_config)

    assert wrapper.config.wrapper_type == MLM_WRAPPER

    results_dict.update({'wrapper_config': model_config.__dict__})
    results_dict.update({'train_config': train_config.__dict__})
    results_dict.update({'eval_config': eval_config.__dict__})
    results_dict.update(train_single_multi_model(wrapper, train_datas, train_config, eval_config,
                                           unlabeled_datas=unlabeled_datas, return_train_set_results=False,
                                           labelled=True))

    wrapper.save(output_dir)
    train_config.save(os.path.join(output_dir, 'train_config.json'))
    eval_config.save(os.path.join(output_dir, 'eval_config.json'))

    # if do_eval:
    #     eval_result = evaluate(wrapper, eval_data, eval_config)
    #     results_dict['test_set_after_training'] = eval_result['scores']
    #
    # with open(os.path.join(output_dir, 'results.json'), 'w') as fh:
    #     json.dump(results_dict, fh, indent=4)



def train_pet_ensemble(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig,
                       pattern_ids: List[int], output_dir: str, ipet_data_dir: str = None, repetitions: int = 3,
                       train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None,
                       eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True,
                       save_unlabeled_logits: bool = False, seed: int = 42):
    """
    Train and evaluate an ensemble of PET models without knowledge distillation.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ipet_data_dir: optional directory containing additional training data for iPET
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param save_unlabeled_logits: whether logits for unlabeled examples should be saved in a file ``logits.txt``. This
           is required for both iPET and knowledge distillation.
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    for pattern_id in pattern_ids:
        for iteration in range(repetitions):

            model_type_string_lower = 'classifier' if model_config.wrapper_type == SEQUENCE_CLASSIFIER_WRAPPER else \
                f'p{pattern_id}-i{iteration}'
            model_type_string_upper = model_type_string_lower.upper()

            model_config.pattern_id = pattern_id
            results_dict = {}

            pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, iteration)

            if os.path.exists(pattern_iter_output_dir):
                logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
                continue

            if not os.path.exists(pattern_iter_output_dir):
                os.makedirs(pattern_iter_output_dir)

            wrapper = init_model(model_config)

            # Training
            if do_train:
                logger.info('')
                logger.info(f'{model_type_string_upper} TRAINING PHASE'.center(LOG_CONST_WIDTH, '='))

                if ipet_data_dir:
                    p = os.path.join(ipet_data_dir, 'p{}-i{}-train.bin'.format(pattern_id, iteration))
                    ipet_train_data = InputExample.load_examples(p)
                    for example in ipet_train_data:
                        example.logits = None
                else:
                    ipet_train_data = None

                logger.info(f'STARTING {model_type_string_upper} TRAINING...'.center(LOG_CONST_WIDTH, '-'))
                training_timer = Timer(f'{model_type_string_lower} training')

                results_dict.update(train_single_model(wrapper, train_data, train_config, eval_config,
                                                       ipet_train_data=ipet_train_data,
                                                       unlabeled_data=unlabeled_data))

                logger.info(training_timer.elapsed_str())
                logger.info(f'FINISHED {model_type_string_upper} TRAINING.'.center(LOG_CONST_WIDTH, '='))

                results_txt_file = 'results.txt'
                logger.info(f'Saving results to {os.path.join(pattern_iter_output_dir, results_txt_file)}')
                with open(os.path.join(pattern_iter_output_dir, results_txt_file), 'w') as fh:
                    fh.write(str(results_dict))
                logger.info("Saving complete")

                logger.info("Saving trained model at {}...".format(pattern_iter_output_dir))
                wrapper.save(pattern_iter_output_dir)
                train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
                eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
                logger.info("Saving complete")

                if save_unlabeled_logits:
                    eval_unlabeled_timer = Timer('unlabeled logits')
                    logger.info(f'Evaluating unlabeled logits for p.{pattern_id}-i.{iteration}')
                    logits = evaluate(wrapper, unlabeled_data, eval_config)['logits']
                    save_logits(os.path.join(pattern_iter_output_dir, 'logits.txt'), logits)
                    logger.info(eval_unlabeled_timer.elapsed_str())

                if not do_eval:
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

            # Evaluation
            if do_eval:
                logger.info('')
                logger.info(f'{model_type_string_upper} EVALUATION PHASE'.center(LOG_CONST_WIDTH, '='))
                logger.info(f'STARTING {model_type_string_upper} EVALUATION...'.center(LOG_CONST_WIDTH, '-'))

                pattern_iteration_timer_eval = Timer(f'{model_type_string_lower} evaluation')

                if not wrapper:
                    wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

                eval_result = evaluate(wrapper, eval_data, eval_config, priming_data=train_data)

                save_predictions(os.path.join(pattern_iter_output_dir, 'predictions.jsonl'), wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

                scores = eval_result['scores']

                logger.info(f'RESULT (pattern_id={pattern_id}, iteration={iteration})'.center(LOG_CONST_WIDTH, '-'))
                logger.info(f'{model_type_string_lower} accuracy: {scores["acc"]:2.4%}')

                logger.info(pattern_iteration_timer_eval.elapsed_str())
                logger.info(f'FINISHED {model_type_string_upper} EVALUATION.'.center(LOG_CONST_WIDTH, '='))

                results_dict['test_set_after_training'] = scores
                results_json = 'results.json'
                logger.info(f'Saving results to {os.path.join(pattern_iter_output_dir, results_json)}')
                with open(os.path.join(pattern_iter_output_dir, results_json), 'w') as fh:
                    json.dump(results_dict, fh, indent=4)
                logger.info("Saving complete")

                for metric, value in scores.items():
                    results[metric][pattern_id].append(value)

                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

    if do_eval:
        logger.info('')
        if model_config.wrapper_type == SEQUENCE_CLASSIFIER_WRAPPER:
            logger.info("OVERALL CLASSIFIER RESULTS".center(LOG_CONST_WIDTH, '='))
        else:
            logger.info("OVERALL ENSEMBLE RESULTS".center(LOG_CONST_WIDTH, '='))
        _write_results(os.path.join(output_dir, 'result_test.txt'), results)
    else:
        logger.info("ENSEMBLE TRAINING COMPLETE".center(LOG_CONST_WIDTH, '='))


def train_single_model(model: TransformerModelWrapper, train_data: List[InputExample], config: TrainConfig,
                       eval_config: EvalConfig = None, ipet_train_data: List[InputExample] = None,
                       unlabeled_data: List[InputExample] = None, return_train_set_results: bool = True,
                       labelled=False):
    """
    Train a single model.

    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :param ipet_train_data: an optional list of iPET training examples to use
    :param unlabeled_data: an optional list of unlabeled examples to use
    :param return_train_set_results: whether results on the train set before and after training should be computed and
           returned
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
    if not ipet_train_data:
        ipet_train_data = []

    results_dict = {}

    model.model.to(device)

    if train_data and return_train_set_results:
        logger.info('Evaluating accuracy on train set prior to training')
        results_dict['train_set_before_training'] = evaluate(model, train_data, eval_config)['scores']['acc']
        logger.info(f"Finished. Result: acc={results_dict['train_set_before_training']:2.4%}")

    all_train_data = train_data + ipet_train_data

    if not all_train_data and not config.use_logits:
        logger.warning('Training method was called without training examples')
    else:
        logger.info('TRAINING...')
        global_step, tr_loss = model.train(
            all_train_data, device,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            per_gpu_unlabeled_batch_size=config.per_gpu_unlabeled_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            unlabeled_data=unlabeled_data if config.lm_training or config.use_logits else None,
            lm_training=config.lm_training,
            use_logits=config.use_logits,
            alpha=config.alpha,
            temperature=config.temperature,
            labelled=labelled
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    if train_data and return_train_set_results:
        logger.info('Evaluating accuracy on train set AFTER to training')
        results_dict['train_set_after_training'] = evaluate(model, train_data, eval_config)['scores']['acc']
        logger.info(f"Finished. Result: acc={results_dict['train_set_after_training']:2.4%}")

    return results_dict


def train_single_multi_model(model: TransformerMultitaskModelWrapper, train_datas: Dict[str, List[InputExample]],
                             config: TrainConfig,
                             eval_config: EvalConfig = None,
                             unlabeled_datas: Dict[str, List[InputExample]] = None,
                             return_train_set_results: bool = True,
                             labelled=False):

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    results_dict = {}

    model.model.to(device)

    if not train_datas and not config.use_logits:
        logger.warning('Training method was called without training examples')
    else:
        logger.info('TRAINING...')
        global_step, tr_loss = model.train(
            train_datas, device,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            per_gpu_unlabeled_batch_size=config.per_gpu_unlabeled_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            unlabeled_datas=unlabeled_datas if config.lm_training or config.use_logits else None,
            lm_training=config.lm_training,
            use_logits=config.use_logits,
            alpha=config.alpha,
            temperature=config.temperature,
            labelled=labelled
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    return results_dict


def evaluate(model: TransformerModelWrapper, eval_data: List[InputExample], config: EvalConfig,
             priming_data: List[InputExample] = None) -> Dict:
    """
    Evaluate a model.

    :param model: the model to evaluate
    :param eval_data: the examples for evaluation
    :param config: the evaluation config
    :param priming_data: an optional list of priming data to use
    :return: a dictionary containing the model's logits, predictions and (if any metrics are given) scores
    """

    if config.priming:
        for example in eval_data:
            example.meta['priming_data'] = priming_data

    metrics = config.metrics if config.metrics else ['acc']
    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    model.model.to(device)
    results = model.eval(eval_data, device, per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu, decoding_strategy=config.decoding_strategy, priming=config.priming)

    predictions = np.argmax(results['logits'], axis=1)
    scores = {}

    for metric in metrics:
        if metric == 'acc':
            scores[metric] = simple_accuracy(predictions, results['labels'])
        elif metric == 'f1':
            scores[metric] = f1_score(results['labels'], predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(results['labels'], predictions, average='macro')
        elif metric == 'em':
            scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
        else:
            raise ValueError(f"Metric '{metric}' not implemented")

    results['scores'] = scores
    results['predictions'] = predictions
    return results


def _write_results(path: str, results: Dict):
    with open(path, 'w') as fh:
        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')


def merge_logits(logits_dir: str, output_file: str, reduction: str):
    """
    Merge the logits predicted for unlabeled examples by multiple models.

    :param logits_dir: a directory for which each sub-directory corresponds to a pretrained model and contains
           both a file ``results.txt`` containing that model's results on the training set and a file ``logits.txt``
           containing that model's predictions for the unlabeled data.
    :param output_file: the file to which the merged logits for all unlabeled examples are written.
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    """
    subdirs = next(os.walk(logits_dir))[1]
    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    all_logits_lists = []

    for subdir in subdirs:
        results_file = os.path.join(logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(logits_dir, subdir, 'logits.txt')
        logits = []

        if not os.path.exists(results_file) or not os.path.exists(logits_file):
            logger.warning(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
            continue

        if reduction == 'mean':
            result_train = 1
        else:
            with open(results_file, 'r') as fh:
                results = ast.literal_eval(fh.read())
                result_train = results['train_set_before_training']

        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logger.info("File {}: Score = {}, #Logits = {}, #Labels = {}".format(
            results_file, result_train, len(logits), len(logits[0])))

        loglist = LogitsList(score=result_train, logits=logits)
        all_logits_lists.append(loglist)

    merged_loglist = merge_logits_lists(all_logits_lists, reduction=reduction)
    merged_loglist.save(output_file)


def merge_logits_lists(logits_lists: List[LogitsList], reduction: str = 'mean') -> LogitsList:
    """
    Merge a list of :class:`LogitsList` objects.

    :param logits_lists: the lists to merge
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    :return: the merged list
    """

    assert len(set(len(ll.logits) for ll in logits_lists)) == 1
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0).tolist()
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    return LogitsList(score=-1, logits=logits)


def generate_ipet_train_sets(train_data: List[InputExample], unlabeled_data: List[InputExample], labels: List[str],
                             logits_dir: str, output_dir: str, reduction: str, num_new_examples: int,
                             logits_percentage: float, n_most_likely: int = -1, seed: int = 42):
    """
    Generate training sets for the next generation of iPET models.

    :param train_data: the training examples
    :param unlabeled_data: the unlabeled examples
    :param labels: the list of all possible labels
    :param logits_dir: the directory that contains the predictions of all models in the current generation for the
           unlabeled data.
    :param output_dir: the output directory
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    :param num_new_examples: the number of new examples to create
    :param logits_percentage: the percentage of models to use for annotating training sets for the next generation
    :param n_most_likely: If >0, in the first generation the n_most_likely examples per label are chosen even
                              if their predicted label is different
    :param seed: the random seed to use
    """
    subdirs = next(os.walk(logits_dir))[1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    if train_data:
        train_examples_per_label = [sum(1 for ex in train_data if ex.label == label) for label in labels]
        multiplier = num_new_examples / len(train_data)
        examples_per_label = [int(epl * multiplier) for epl in train_examples_per_label]
        logger.info(f"Example distribution in the original dataset: {train_examples_per_label}")
    else:
        examples_per_label = eq_div(num_new_examples, len(labels))

    logger.info(f"Target distribution for the new dataset: {examples_per_label}")

    for example in unlabeled_data:
        example.label, example.logits = None, None

    logits_lists = {}

    rng = random.Random(seed)
    rng_np = np.random.RandomState(seed)

    for subdir in subdirs:
        results_file = os.path.join(logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(logits_dir, subdir, 'logits.txt')
        logits = []

        if not os.path.exists(results_file) or not os.path.exists(logits_file):
            logger.warning(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
            continue

        if reduction == 'mean':
            result_train = 1
        else:
            with open(results_file, 'r') as fh:
                results = ast.literal_eval(fh.read())
                result_train = results['train_set_before_training']

        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logger.info("File {}: Score = {}, #Logits = {}, #Labels = {}".format(
            results_file, result_train, len(logits), len(logits[0])))

        loglist = LogitsList(score=result_train, logits=logits)
        logits_lists[subdir] = loglist

    for subdir in subdirs:
        other_logits_lists = [ll for sd, ll in logits_lists.items() if sd != subdir]
        subdir_train_set = generate_ipet_train_set(
            other_logits_lists, labels=labels, original_data=unlabeled_data, examples_per_label=examples_per_label,
            logits_percentage=logits_percentage, reduction=reduction, n_most_likely=n_most_likely, rng=rng,
            rng_np=rng_np
        )

        InputExample.save_examples(subdir_train_set,
                                   os.path.join(output_dir, subdir + '-train.bin'))


def generate_ipet_train_set(logits_lists: List[LogitsList], labels: List[str], original_data: List[InputExample],
                            examples_per_label: List[int], logits_percentage: float, reduction: str = 'mean',
                            n_most_likely: int = -1, rng=None, rng_np=None) -> List[InputExample]:
    """
    Generate a single training set for the next generation of iPET models.

    :param logits_lists: predictions from the previous generation of models
    :param labels: all task labels
    :param original_data: the original training data corresponding to the logits_lists
    :param examples_per_label: the number of examples per label to create
    :param logits_percentage: the percentage of models/logits to choose
    :param reduction: the reduction strategy ('wmean' or 'mean')
    :param n_most_likely: if >0, for each label the n_most_likely examples with the highest logits are chosen
    :param rng: the random number generator to use for non-numpy operations
    :param rng_np: the random number generator to use for numpy operations
    :return: a list of input examples that serves as training set for the next generation
    """

    assert len(set(len(ll.logits) for ll in logits_lists)) == 1

    if not rng:
        rng = random.Random()
    if not rng_np:
        rng_np = np.random.RandomState()

    num_logits_lists = round(len(logits_lists) * logits_percentage)
    logits_lists = rng.sample(logits_lists, k=num_logits_lists)
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0)
        logits = softmax(logits, axis=1).tolist()
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights)
        logits = softmax(logits, axis=1).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    assert len(logits) == len(original_data)

    for lgs, example in zip(logits, original_data):
        example.logits = lgs
        example.label = labels[np.argmax(example.logits).item()]

    test_set = []

    for idx, label in enumerate(labels):

        if n_most_likely <= 0:
            examples = [ex for ex in original_data if ex.label == label]
            logger.info("There are {} examples for label {}".format(len(examples), label))
            while len(examples) < examples_per_label[idx]:
                # upsample examples if there are too few
                examples.extend(ex for ex in original_data if ex.label == label)
        else:
            examples = [(ex.logits[idx], ex_idx, ex) for ex_idx, ex in enumerate(original_data)]
            examples.sort(reverse=True)
            examples = [ex for score, ex_idx, ex in examples[:n_most_likely]]
            examples = [deepcopy(ex) for ex in examples]
            for example in examples:
                example.logits = [example.logits[idx]]
                example.label = label

        label_examples = _draw_examples_by_label_probability(
            examples=examples, num_examples=examples_per_label[idx], rng=rng_np)
        test_set.extend(label_examples)

    return test_set


def _draw_examples_by_label_probability(examples: List[InputExample], num_examples: int, rng) -> List[InputExample]:
    label_probabilities = [max(example.logits) for example in examples]
    sum_label_probabilities = sum(label_probabilities)
    label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
    return rng.choice(examples, size=num_examples, replace=False, p=label_probabilities).tolist()
