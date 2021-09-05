import json
import os
import pickle
import stat
from collections import defaultdict
from itertools import product
from typing import Union, Dict

import pandas as pd

from common.constants import DATASETS_DIRS
from pet import LogitsList
from pet.tasks import load_examples, UNLABELED_SET

DEFAULTS_PATH = 'experiment_configurations/defaults.json'


def exec_tmux_window(command: Union[str, list], window_name: str) -> None:
    if isinstance(command, list):
        command = '; '.join(command)
    os.system('tmux new-window -n %s "%s ; exec bash"' % (window_name, command))


def build_files(recipe_path: str, verbose: bool = False, build_script: bool = True,
                take_defaults: bool = False) -> None:
    bash_script_file_name = 'bash_script.sh'
    bash_script = ['#!/bin/bash\n']

    with open(recipe_path, 'r') as fp:
        recipe = json.load(fp)

    base, options = recipe['base'], recipe['options']

    if take_defaults:
        with open(DEFAULTS_PATH, 'r') as fp:
            defaults = json.load(fp)
    else:
        defaults ={}

    updated_base = dict(defaults, **base)
    gpus = updated_base['visible_gpus']

    recipe_dir = os.path.dirname(recipe_path)
    if not isinstance(options, list):
        options = [options]

    gpu = 0
    gpus_dict = defaultdict(list)

    meta_output_dir = updated_base['output_dir']

    for option in options:
        for combination in product(*option.values()):
            combination_dict = dict(zip(option.keys(), combination))

            updated_base.update(combination_dict)
            updated_base.update(dict(visible_gpus=[gpus[gpu % len(gpus)]]))

            parsed_output_dir = updated_base['output_dir'].format(**updated_base)
            updated_base.update(dict(output_dir=parsed_output_dir))

            if "pattern_dict" in combination_dict:
                if isinstance(combination_dict['pattern_dict'], list):
                    combination_dict['pattern_dict'] = dict(zip(combination_dict['train_tasks'], combination_dict['pattern_dict']))
                elif isinstance(combination_dict['pattern_dict'], int):
                    combination_dict['pattern_dict'] = dict.fromkeys(combination_dict['train_tasks'], combination_dict['pattern_dict'])

            path = os.path.join(recipe_dir, 'compiled_scripts')
            filename = []

            for var, val in combination_dict.items():
                if var == 'train_tasks' and isinstance(val, list):
                    temp_val = val.copy()
                    for idx, item in enumerate(temp_val):
                        if item == 'yelp-full':
                            temp_val[idx] = 'yf'
                        elif item == 'yelp-polarity':
                            temp_val[idx] = 'yp'
                    filename.append('_'.join(v[0:2] for v in temp_val))
                elif var == 'pattern_dict':
                    filename.append('p')
                    filename.append('_'.join(map(lambda x: str(x), val.values())))
                else:
                    parsed_val = os.path.basename(str(val))
                    filename.append(f'{var[0]}_{parsed_val}')

            filename = '_'.join(filename)

            updated_base.update(dict(output_dir=os.path.join(meta_output_dir, filename)))

            path = os.path.join(path, filename) + '.json'

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, 'w') as fp:
                json.dump(updated_base, fp, indent=2)
            if verbose: print(f'Saved {path}')

            # window_name = os.path.basename(updated_base["output_dir"].format(**updated_base))
            command = f'python eval_lm_classifier_file_config.py --config_path {path}'

            gpus_dict[gpus[gpu % len(gpus)]].append(command)
            # bash_script += [f'tmux new-window -n {window_name} "{command} ; exec bash"']

            gpu += 1

    print(f'total {gpu} runs.')
    for gpu_number, commands in gpus_dict.items():
        bash_script += [f'tmux new-window -n gpu_{gpu_number} "\\']
        for command in commands:
            bash_script.append(command + ' ; \\')
        bash_script += [f'exec bash"']
        bash_script += ['']


    if build_script:
        bash_script_path = os.path.join(recipe_dir, bash_script_file_name)

        with open(bash_script_path, 'w') as f:
            f.write('\n'.join(bash_script))

        st = os.stat(bash_script_path)
        os.chmod(bash_script_path, st.st_mode | stat.S_IEXEC)

        if verbose: print(f'\nBash script: {bash_script_path}\n')


def parse_dir_results(path: str, verbose: bool = False, save=False, override=False) -> Dict[str, Dict[str, Dict[str, float]]]:

    experiments_dict = {}

    listdir = os.listdir(path)
    file = 'parsed_results.json'
    if file in listdir:
        listdir.remove(file)
    for experiment_dir in listdir:
        results_path = os.path.join(path, experiment_dir, 'result_test.txt')
        results = defaultdict(dict)

        with open(results_path, 'r') as fp:
            lines = fp.read().split('\n')

        lines = list(map(lambda x: x.split(' '), lines))

        for model, avg, _, std in lines[:-1]:
            results['avg'][model[:-1]] = float(avg)
            results['std'][model[:-1]] = float(std)

        experiments_dict[experiment_dir] = dict(results)

        if verbose:
            print(experiment_dir.center(30, '='))
            pd.options.display.float_format = '{:.2%}'.format
            df = pd.DataFrame(results)
            df['avg'].map('{:,.2f}'.format)
            print(df)
            print('\n')

    if save:
        parsed_path = os.path.join(path, 'parsed_results.json')
        if os.path.exists(parsed_path) and not override:
            print(f"Folder already parsed. Parsed results are in {parsed_path}. To override pass --override flag to command.")
        else:
            with open(parsed_path, 'w') as fp:
                json.dump(experiments_dict, fp, indent=2)
            print(f'Parsed results were saved to {parsed_path}')

    return experiments_dict


def trained_lm_path(train_task: str, train_examples: int, pattern_id: int, iteration: int, final:bool = False):
    final_str = "final" if final else ""
    root_path = f"outputs/{train_task}/{train_task}_m_{train_examples}_s_250/"
    specifier = f"p{pattern_id}-i{iteration}"

    return os.path.join(root_path, final_str,specifier)

if __name__ == '__main__':
    # print(parse_dir_results('../outputs/yahoo'))
    # parse_dir_results('../outputs/agnews')
    # load_unlabeled_with_logits('outputs/agnews/agnews_m_10_s_250', 'agnews')
    pass


def load_unlabeled_with_logits(task_name: str, trained_on_examples: int = 1000,
                               unlabeled_examples: int = -1, use_cache=False):

    logits_path = f"outputs/PET_vanilla/{task_name}/{task_name}_m_{trained_on_examples}_s_250"

    logits_file = os.path.join(logits_path, 'unlabeled_logits.txt')
    logits = LogitsList.load(logits_file).logits

    if len(logits) < unlabeled_examples:
        raise ValueError("Not enough logits evaluated in the specified directory.")

    if unlabeled_examples == -1:
        unlabeled_examples = len(logits)

    unlabeled_data = load_examples(task_name, DATASETS_DIRS[task_name], UNLABELED_SET, num_examples=unlabeled_examples)

    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    if use_cache:
        with open(os.path.join(logits_path, 'unlabeled_data.pkl'), 'wb') as fp:
            pickle.dump(unlabeled_data, fp)

    return unlabeled_data