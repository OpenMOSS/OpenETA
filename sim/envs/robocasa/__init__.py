# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboCasa environment wrappers.

The benchmark-native direct wrapper must remain importable without RLinf's
OmegaConf training dependencies, so the legacy vector wrapper is lazy.
"""

from typing import Any

__all__ = ["RobocasaEnv", "RoboCasaDirectEnv"]


def __getattr__(name: str) -> Any:
    if name == "RobocasaEnv":
        from sim.envs.robocasa.robocasa_env import RobocasaEnv

        return RobocasaEnv
    if name == "RoboCasaDirectEnv":
        from sim.envs.robocasa.direct_env import RoboCasaDirectEnv

        return RoboCasaDirectEnv
    raise AttributeError(name)
