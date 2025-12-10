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

import random

from vla_arena.evaluation.policy.base import Policy, PolicyRegistry


@PolicyRegistry.register('random')
class RandomPolicy(Policy):

    def predict(self, obs, **kwargs):

        return [random.uniform(-0.1, 0.1) for _ in range(7)]

    @property
    def name(self):
        return 'random'
