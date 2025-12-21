# Copyright 2025 The VLA-Arena Authors.
#
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

import os

import yaml


# This is a default path for localizing all the benchmark related files
vla_arena_config_path = os.environ.get('VLA_ARENA_CONFIG_PATH', os.path.expanduser('~/.vla_arena'))
config_file = os.path.join(vla_arena_config_path, 'config.yaml')


def get_default_path_dict(custom_location=None):
    if custom_location is None:
        benchmark_root_path = os.path.dirname(os.path.abspath(__file__))
    else:
        benchmark_root_path = custom_location

    # This is a default path for localizing all the default bddl files
    bddl_files_default_path = os.path.join(benchmark_root_path, './bddl_files')

    # This is a default path for localizing all the default bddl files
    init_states_default_path = os.path.join(benchmark_root_path, './init_files')

    # This is a default path for localizing all the default assets
    assets_default_path = os.path.join(benchmark_root_path, './assets')

    return {
        'benchmark_root': benchmark_root_path,
        'bddl_files': bddl_files_default_path,
        'init_states': init_states_default_path,
        'assets': assets_default_path,
    }


def get_vla_arena_path(query_key):
    with open(config_file) as f:
        config = dict(yaml.load(f.read(), Loader=yaml.FullLoader))

    # Give warnings in case the user needs to access the paths
    for key in config:
        if not os.path.exists(config[key]):
            print(f'[Warning]: {key} path {config[key]} does not exist!')

    assert (
        query_key in config
    ), f'Key {query_key} not found in config file {config_file}. You need to modify it. Available keys are: {config.keys()}'
    return config[query_key]


def set_vla_arena_default_path(custom_location=os.path.dirname(os.path.abspath(__file__))):
    print(
        f'[Warning] You are changing the default path for vla_arena config. This will affect all the paths in the config file.'
    )
    new_config = get_default_path_dict(custom_location)
    with open(config_file, 'w') as f:
        yaml.dump(new_config, f)


if not os.path.exists(vla_arena_config_path):
    os.makedirs(vla_arena_config_path)

if not os.path.exists(config_file):
    # Create a default config file

    default_path_dict = get_default_path_dict()
    print('Initializing the default config file...')
    print(f'The following information is stored in the config file: {config_file}')
    # write all the paths into a yaml file
    with open(config_file, 'w') as f:
        yaml.dump(default_path_dict, f)
    for key, value in default_path_dict.items():
        print(f'{key}: {value}')
