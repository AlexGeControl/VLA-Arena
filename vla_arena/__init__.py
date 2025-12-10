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

"""VLA-Arena: A Comprehensive Benchmark for Vision-Language-Action Models."""

from vla_arena.__version__ import __version__


__all__ = ['__version__']


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies during package build."""
    if name in globals():
        return globals()[name]

    # Lazy import from vla_arena.vla_arena
    try:
        from vla_arena import vla_arena as _vla_arena

        attr = getattr(_vla_arena, name)
        globals()[name] = attr
        return attr
    except (ImportError, AttributeError):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
