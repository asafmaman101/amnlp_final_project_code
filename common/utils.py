import json
import os
import stat
from collections import defaultdict
from itertools import product
from typing import Union, Dict

import pandas as pd

DEFAULTS_PATH = 'experiment_configurations/defaults.json'


def exec_tmux_window(command: Union[str, list], window_name: str) -> None:
    if isinstance(command, list):
        command = '; '.join(command)
    os.system('tmux new-window -n %s "%s ; exec bash"' % (window_name, command))


def build_files(recipe_path: str, verbose: bool = False, build_script: bool = True) -> None:
    bash_script_file_name = 'bash_script.sh'
    bash_script = ['#!/bin/bash\n']

    with open(recipe_path, 'r') as fp:
        recipe = json.load(fp)

    base, options = recipe['base'], recipe['options']

    with open(DEFAULTS_PATH, 'r') as fp:
        defaults = json.load(fp)

    updated_base = dict(defaults, **base)
    gpus = updated_base['visible_gpus']

    recipe_dir = os.path.dirname(recipe_path)
    if not isinstance(options, list):
        options = [options]

    for option in options:
        for gpu, combination in enumerate(product(*option.values())):
            combination_dict = dict(zip(option.keys(), combination))

            updated_base.update(combination_dict)
            updated_base.update(dict(visible_gpus=[gpus[gpu % len(gpus)]]))

            path = os.path.join(recipe_dir, 'compiled_scripts')
            path = os.path.join(path, '_'.join(f'{var[0]}_{os.path.basename(val)}' for var, val in combination_dict.items()
                                               if var != 'pattern_ids')) + '.json'

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, 'w') as fp:
                json.dump(updated_base, fp, indent=2)
            if verbose: print(f'Saved {path}')

            window_name = os.path.basename(updated_base["output_dir"]).format(**updated_base)
            command = f'petcli {path}'

            bash_script += [f'tmux new-window -n {window_name} "{command} ; exec bash"']

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
