# Copyright (c) 2024-2025 VLA-Arena Team. All Rights Reserved.
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
# ==============================================================================

import argparse
import os
import time

import numpy as np
import torch

from vla_arena.vla_arena.envs.env_wrapper import OffScreenRenderEnv


# import debugpy
# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

parser = argparse.ArgumentParser()
parser.add_argument('--bddl_file', type=str, required=True, help='BDDL文件路径或目录')
parser.add_argument('--resolution', type=int, default=256, help='分辨率')
parser.add_argument(
    '--output_path',
    type=str,
    default='./vla_arena/vla_arena/init_files',
    help='输出路径',
)
args = parser.parse_args()


def process_single_file_with_retry(bddl_file, relative_path='', max_retries=4):
    """
    处理单个BDDL文件，带重试机制

    Args:
        bddl_file: BDDL文件的完整路径
        relative_path: 相对于输入根目录的路径，用于保持目录结构
        max_retries: 最大重试次数
    """
    for attempt in range(max_retries + 1):  # +1 因为包括第一次尝试
        try:
            print(f'Processing file: {bddl_file} (Attempt {attempt + 1}/{max_retries + 1})')
            process_single_file(bddl_file, relative_path)
            return  # 成功处理，直接返回

        except Exception as e:
            error_name = e.__class__.__name__

            # 检查是否是RandomizationError
            if 'RandomizationError' in error_name or 'randomization' in str(e).lower():
                if attempt < max_retries:
                    print(f'Encountered RandomizationError: {e}')
                    print(f'Retrying... ({attempt + 1}/{max_retries} retries used)')
                    time.sleep(0.5)  # 短暂等待后重试
                    continue
                print(f'Failed after {max_retries} retries due to RandomizationError')
                print(f'Error details: {e}')
                raise e
            # 如果不是RandomizationError，直接抛出异常
            print(f'Encountered non-RandomizationError: {error_name}')
            raise e


def process_single_file(bddl_file, relative_path=''):
    """
    处理单个BDDL文件

    Args:
        bddl_file: BDDL文件的完整路径
        relative_path: 相对于输入根目录的路径，用于保持目录结构
    """
    resolution = args.resolution

    """初始化并返回LIBERO环境"""
    env_args = {
        'bddl_file_name': bddl_file,
        'camera_heights': resolution,
        'camera_widths': resolution,
    }
    env = None

    try:
        env = OffScreenRenderEnv(**env_args)

        # 1. 加载环境
        obs = env.reset()
        print('ok')
        # 2. 保存当前初始状态
        init_states = []
        flattened_state = env.get_sim_state()
        print(flattened_state.shape, type(flattened_state))
        if isinstance(flattened_state, np.ndarray) and flattened_state.ndim == 1:
            init_states.append(flattened_state)

        # 3. 构建输出路径，保持原有目录结构
        task_name = os.path.basename(bddl_file)
        task_name = task_name.replace('.bddl', '')

        # 如果有相对路径，创建相应的目录结构
        if relative_path:
            output_dir = os.path.join(args.output_path, relative_path)
        else:
            output_dir = args.output_path

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f'{task_name}.pruned_init')

        print(init_states)
        # 4. torch.save the init_state
        torch.save(init_states, output_file)

        print(f'Init file saved to {output_file}')

    finally:
        # 5. close the environment
        if env is not None:
            env.close()


def process_directory_recursive(directory, root_dir=None):
    """
    递归处理目录中的所有BDDL文件

    Args:
        directory: 当前处理的目录
        root_dir: 根目录，用于计算相对路径
    """
    if root_dir is None:
        root_dir = directory

    # 遍历目录中的所有文件和子目录
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path) and item.endswith('.bddl'):
            # 计算相对于根目录的路径
            relative_dir = os.path.relpath(directory, root_dir)
            if relative_dir == '.':
                relative_dir = ''

            # 处理BDDL文件，使用重试机制
            try:
                process_single_file_with_retry(item_path, relative_dir)
            except Exception as e:
                print(f'Error processing {item_path}: {e}')
                print('Skipping this file and continuing with others...')
                continue

        elif os.path.isdir(item_path):
            # 递归处理子目录
            process_directory_recursive(item_path, root_dir)


def main():
    bddl_path = args.bddl_file

    if os.path.isfile(bddl_path):
        # 如果是单个文件，直接处理（带重试）
        process_single_file_with_retry(bddl_path)
    elif os.path.isdir(bddl_path):
        # 如果是目录，递归遍历所有.bddl文件
        print(f'Recursively processing all .bddl files in {bddl_path}')
        process_directory_recursive(bddl_path)
    else:
        print(f'错误: {bddl_path} 既不是文件也不是目录')


if __name__ == '__main__':
    main()
