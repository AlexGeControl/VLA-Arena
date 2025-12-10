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

from vla_arena.evaluation.policy.base import Policy, PolicyRegistry
from vla_arena.evaluation.policy.openpi import OpenPI
from vla_arena.evaluation.policy.openvla import OpenVLA
from vla_arena.evaluation.policy.openvla_oft import OpenVLAOFT
from vla_arena.evaluation.policy.random import RandomPolicy
from vla_arena.evaluation.policy.smolvla import SmolVLA
from vla_arena.evaluation.policy.univla import UniVLA
