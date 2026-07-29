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

import copy
import gc
import inspect
import json
import os
import random
import time
from types import SimpleNamespace
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sim.envs.behavior.instance_loader import ActivityInstanceLoader
from sim.envs.behavior.seeding import (
    derive_behavior_seed,
    seed_behavior_reset_rngs,
)
from sim.envs.behavior.utils import (
    apply_env_wrapper,
    apply_runtime_renderer_settings,
    convert_uint8_rgb,
    setup_omni_cfg,
)
from sim.envs._logging import get_logger
from sim.envs.utils import to_tensor

try:
    import ray
except ImportError:  # pragma: no cover - exercised in lightweight unit tests
    ray = None

__all__ = ["BehaviorEnv"]


def _preload_numba_llvmlite() -> None:
    # Isaac Sim's ``omni.isaac.core_archive`` ships an older numba in its
    # ``pip_prebundle`` and loads a few submodules during Kit startup,
    # which then mix with the venv's newer ``llvmlite`` and fail with
    # ``unknown attr 'nocapture'``. Preload the venv copies of just those
    # submodules so they win the ``sys.modules`` cache.
    import importlib

    for name in (
        "llvmlite",
        "numba",
        "numba.np.arrayobj",
        "numba.core.runtime.context",
    ):
        try:
            importlib.import_module(name)
        except Exception:
            pass


def _reset_vector_rows(
    vector_env,
    *,
    reset_indices: list[int] | None,
    reset_seeds: list[int] | None,
    get_obs: bool,
    supports_env_indices: bool,
):
    """Reset all vector rows or an explicitly selected subset."""

    if reset_indices is None and reset_seeds is None:
        return vector_env.reset(get_obs=get_obs)
    child_envs = getattr(vector_env, "envs", None)
    if reset_indices is None:
        if not isinstance(child_envs, list):
            raise RuntimeError(
                "Per-row BEHAVIOR reset seeds require a child env list."
            )
        reset_indices = list(range(len(child_envs)))
    if reset_seeds is not None and len(reset_seeds) != len(reset_indices):
        raise ValueError(
            "reset_seeds must have one entry per selected environment, got "
            f"{len(reset_seeds)} seeds for {len(reset_indices)} rows."
        )
    if supports_env_indices and reset_seeds is None:
        return vector_env.reset(
            get_obs=get_obs,
            env_indices=reset_indices,
        )

    # Pinned OmniGibson's VectorEnvironment.reset accepts **kwargs but
    # forwards them to every child; ``env_indices`` is not an indexing
    # contract there. Reset selected child environments directly instead.
    if not isinstance(child_envs, list):
        raise TypeError(
            "OmniGibson vector environment does not expose indexed reset "
            "or a child env list; partial reset is unavailable."
        )
    results = []
    for position, idx in enumerate(reset_indices):
        if reset_seeds is not None:
            seed_behavior_reset_rngs(reset_seeds[position])
        results.append(child_envs[idx].reset(get_obs=get_obs))
    if not get_obs:
        return None
    if any(
        not isinstance(result, tuple) or len(result) != 2
        for result in results
    ):
        raise RuntimeError(
            "OmniGibson child reset must return (observation, info)."
        )
    return (
        [result[0] for result in results],
        [result[1] for result in results],
    )


def _validate_reset_isolation(
    reset_indices: list[int] | None,
    *,
    num_envs: int,
) -> list[int]:
    """Reject partial resets that would step sibling scenes in pinned OG."""

    selected = (
        list(range(num_envs))
        if reset_indices is None
        else [int(idx) for idx in reset_indices]
    )
    if len(set(selected)) != len(selected):
        raise ValueError(f"reset_indices must be unique, got {selected}.")
    if any(idx < 0 or idx >= num_envs for idx in selected):
        raise IndexError(
            f"reset_indices must be within [0, {num_envs}), got {selected}."
        )
    if num_envs > 1 and set(selected) != set(range(num_envs)):
        raise RuntimeError(
            "Pinned OmniGibson resets perform a simulator-global physics step, "
            "so a partial reset cannot be isolated from sibling scenes in the "
            "same actor. Configure one child environment per actor (the subprocess "
            "count must equal the per-worker environment count), or reset the "
            "complete actor shard."
        )
    return selected


def _physical_worker_index(
    seed_offset: int,
    *,
    pipeline_stage_num: int,
) -> int:
    """Recover the physical worker index from RLinf's stage-aware offset."""

    if pipeline_stage_num <= 0:
        raise ValueError("pipeline_stage_num must be positive.")
    return int(seed_offset) // int(pipeline_stage_num)


class BehaviorProcess:
    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        pipeline_stage_num: int,
        process_seed: int = 0,
    ):
        _preload_numba_llvmlite()
        from omnigibson.envs import VectorEnvironment

        self.logger = get_logger()
        self.num_envs = int(num_envs)
        self.pipeline_stage_num = pipeline_stage_num
        self.process_seed = int(process_seed)
        random.seed(self.process_seed)
        np.random.seed(self.process_seed % (2**32))
        torch.manual_seed(self.process_seed)
        omni_cfg = setup_omni_cfg(cfg)
        self.instance_loader = ActivityInstanceLoader.from_omni_cfg(
            omni_cfg,
            seed=self.process_seed,
        )

        # create env and apply env wrapper if enabled
        omni_cfg_dict = OmegaConf.to_container(
            omni_cfg,
            resolve=True,
            throw_on_missing=True,
        )
        # When pipeline stages > 1, each stage independently advances the
        # global physics per chunk step.  Divide physics_frequency so the
        # total physics rate stays at the configured value.
        if pipeline_stage_num > 1:
            omni_cfg_dict["env"]["physics_frequency"] = (
                omni_cfg_dict["env"]["physics_frequency"] / pipeline_stage_num
            )
        self.env = VectorEnvironment(num_envs, omni_cfg_dict)
        apply_runtime_renderer_settings()
        wrapper_name = OmegaConf.select(omni_cfg, "env.env_wrapper")
        self.env = apply_env_wrapper(self.env, wrapper_name)

        # Isaac Sim's `omni.kit.app` calls ``gc.disable()`` at startup.
        # OmniGibson has self-referential cycles and leaks memory when
        # cyclic GC is disabled. Since we do not need real-time performance,
        # enable cyclic GC here so that we do not encounter OOMs in long runs.
        gc.enable()

        step_signature = inspect.signature(self.env.step)
        step_params = step_signature.parameters.values()
        step_supports_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in step_params
        )
        self.step_supports_get_obs = (
            step_supports_kwargs or "get_obs" in step_signature.parameters
        )
        self.step_supports_render = (
            step_supports_kwargs or "render" in step_signature.parameters
        )
        self.step_supports_env_indices = "env_indices" in step_signature.parameters
        reset_signature = inspect.signature(self.env.reset)
        self.reset_supports_env_indices = (
            "env_indices" in reset_signature.parameters
        )
        self.skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )

        if self.skip_intermediate_obs_in_chunk and not self.step_supports_get_obs:
            self.logger.warning(
                "skip_intermediate_obs_in_chunk is True but OmniGibson env step does not "
                "support get_obs; this config will be ignored."
            )

        if (
            self.pipeline_stage_num > 1
            and self.num_envs > 1
            and not self.step_supports_env_indices
        ):
            raise RuntimeError(
                "pipeline_stage_num > 1 but OmniGibson env step does not support env_indices; "
                "advancing inactive rows with zero actions would corrupt their episode "
                "state. Use an OmniGibson build with indexed vector stepping or set "
                "pipeline_stage_num=1."
            )
        if (
            bool(OmegaConf.select(cfg, "auto_reset", default=True))
            and self.num_envs > 1
        ):
            raise RuntimeError(
                "BEHAVIOR auto_reset requires one child environment per actor "
                "because pinned OmniGibson reset steps all scenes globally. Set "
                "num_env_subprocess to the per-worker environment count."
            )

    def get_activity_name(self):
        return self.instance_loader.activity_name

    def _call_step(self, actions, env_indices=None, get_obs=True, render=True):
        """Call ``self.env.step`` forwarding only the kwargs it supports."""
        kwargs = {}
        if self.step_supports_get_obs:
            kwargs["get_obs"] = get_obs
        if self.step_supports_render:
            kwargs["render"] = render
        if env_indices is not None:
            kwargs["env_indices"] = env_indices
        return self.env.step(actions, **kwargs)

    def _call_reset(
        self,
        reset_indices=None,
        get_obs=True,
        reset_seeds=None,
    ):
        """Call ``self.env.reset`` through one normalized code path."""
        return _reset_vector_rows(
            self.env,
            reset_indices=reset_indices,
            reset_seeds=reset_seeds,
            get_obs=get_obs,
            supports_env_indices=self.reset_supports_env_indices,
        )

    def _step_shard(
        self,
        actions: torch.Tensor,
        env_indices: list[int],
        need_obs: bool,
    ):
        """Step one shard for a single chunk timestep.

        ``actions`` is the zero-padded ``[num_shard, action_dim]`` action
        tensor (inactive rows already carry zero actions). ``env_indices``
        is the ascending list of local rows that should advance.

        Returns outputs only for ``env_indices``, in that same order.
        """
        if self.step_supports_env_indices:
            raw_obs, rewards, terminates, truncates, infos = self._call_step(
                [actions[i] for i in env_indices],
                env_indices=env_indices,
                get_obs=need_obs,
                render=need_obs,
            )
        else:
            if env_indices != list(range(actions.shape[0])):
                raise RuntimeError(
                    "OmniGibson env step has no indexed stepping support, but "
                    f"this shard requested only rows {env_indices}. Refusing to "
                    "advance inactive episodes with zero actions."
                )
            raw_obs, rewards, terminates, truncates, infos = self._call_step(
                actions,
                get_obs=need_obs,
                render=need_obs,
            )
            if need_obs:
                raw_obs = [raw_obs[i] for i in env_indices]
            rewards = [rewards[i] for i in env_indices]
            terminates = [terminates[i] for i in env_indices]
            truncates = [truncates[i] for i in env_indices]
            infos = [infos[i] for i in env_indices]

        return (
            list(raw_obs) if need_obs else None,
            to_tensor(rewards),
            to_tensor(terminates),
            to_tensor(truncates),
            list(infos),
        )

    def chunk_step(self, actions, env_indices):
        """Step a full chunk for one shard.

        Args:
            actions: Zero-padded ``[num_shard, chunk, action_dim]`` action
                matrix for this VectorEnvironment.
            env_indices: Ascending local rows that should advance every
                chunk step.
        """
        _, chunk_size, _ = actions.shape

        results: list[tuple] = []
        for t in range(chunk_size):
            is_last = t == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            results.append(
                self._step_shard(actions[:, t], env_indices, need_obs=need_obs)
            )
        return tuple(zip(*results))

    def reset(self, reset_indices=None, get_obs=True, reset_seeds=None):
        reset_indices = _validate_reset_isolation(
            reset_indices,
            num_envs=self.num_envs,
        )
        self.instance_loader.prepare_reset(
            self.env,
            env_indices=reset_indices,
            seeds=reset_seeds,
        )
        result = self._call_reset(
            reset_indices=reset_indices,
            get_obs=get_obs,
            reset_seeds=reset_seeds,
        )
        if not get_obs:
            return None, None

        raw_obs, infos = result
        return list(raw_obs), list(infos)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


if ray is not None:
    BehaviorProcess = ray.remote(num_cpus=1)(BehaviorProcess)


class BehaviorProcessPool:
    """Singleton OmniGibson subprocess pool manager.

    Use :meth:`acquire_shared` to obtain the singleton pool; use :meth:`release_shared` when done.
    """

    _shared_pool: ClassVar["BehaviorProcessPool | None"] = None
    _shared_refcount: ClassVar[int] = 0
    _shared_leases: ClassVar[dict[int, int]] = {}
    _shared_key: ClassVar[tuple[str, int, int, int] | None] = None

    @staticmethod
    def _config_key(
        cfg: DictConfig,
        group_world_size: int,
        pipeline_stage_num: int,
        physical_worker_index: int,
    ) -> tuple[str, int, int, int]:
        config = (
            OmegaConf.to_container(cfg, resolve=True)
            if OmegaConf.is_config(cfg)
            else cfg
        )
        return (
            json.dumps(config, sort_keys=True, default=str),
            int(group_world_size),
            int(pipeline_stage_num),
            int(physical_worker_index),
        )

    @staticmethod
    def _find_lease_offset(
        total_num_envs: int,
        leases: dict[int, int],
        requested_num_envs: int,
    ) -> int | None:
        cursor = 0
        for offset, length in sorted(leases.items()):
            if cursor + requested_num_envs <= offset:
                return cursor
            cursor = max(cursor, offset + length)
        if cursor + requested_num_envs <= total_num_envs:
            return cursor
        return None

    @classmethod
    def acquire_shared(
        cls,
        cfg: DictConfig,
        worker_info,
        pipeline_stage_num: int,
        num_envs: int,
        worker_seed_offset: int,
    ) -> tuple["BehaviorProcessPool", int]:
        """Attach to the shared pool and return ``(pool, pool_offset)``."""
        group_world_size = int(worker_info.group_world_size)
        if group_world_size <= 0:
            raise ValueError(
                "worker_info.group_world_size must be positive, got "
                f"{group_world_size}."
            )
        if pipeline_stage_num <= 0:
            raise ValueError(
                f"pipeline_stage_num must be positive, got {pipeline_stage_num}."
            )
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")

        physical_worker_index = _physical_worker_index(
            worker_seed_offset,
            pipeline_stage_num=pipeline_stage_num,
        )
        shared_key = cls._config_key(
            cfg,
            group_world_size,
            pipeline_stage_num,
            physical_worker_index,
        )
        if cls._shared_pool is not None and cls._shared_key != shared_key:
            raise RuntimeError(
                "A BEHAVIOR process pool is already active with a different "
                "configuration or worker topology. Close all attached BehaviorEnv "
                "instances before constructing an incompatible one."
            )

        if cls._shared_pool is None:  # pool init
            configured_total_envs = OmegaConf.select(
                cfg,
                "total_num_envs",
                default=None,
            )
            if configured_total_envs is None:
                # One contiguous slice is leased per pipeline stage on each worker.
                total_envs_per_worker = num_envs * pipeline_stage_num
            else:
                total_envs = int(configured_total_envs)
                if total_envs <= 0:
                    raise ValueError(
                        f"total_num_envs must be positive, got {total_envs}."
                    )
                if total_envs % group_world_size != 0:
                    raise ValueError(
                        f"total_num_envs ({total_envs}) must be divisible by "
                        f"worker_info.group_world_size ({group_world_size})."
                    )
                total_envs_per_worker = total_envs // group_world_size
            configured_subprocesses = OmegaConf.select(
                cfg,
                "num_env_subprocess",
                default=None,
            )
            if configured_subprocesses is None:
                # Pinned OmniGibson performs a simulator-global physics step
                # during reset. Auto-reset therefore needs one isolated actor
                # per child unless the caller explicitly disables it.
                num_env_subprocess = (
                    total_envs_per_worker
                    if bool(OmegaConf.select(cfg, "auto_reset", default=True))
                    else 1
                )
            else:
                num_env_subprocess = int(configured_subprocesses)
            pool = cls(
                cfg,
                total_envs_per_worker,
                num_env_subprocess,
                pipeline_stage_num,
                physical_worker_index,
            )
            cls._shared_pool = pool
            cls._shared_key = shared_key
            cls._shared_leases = {}

        pool = cls._shared_pool
        global_offset = cls._find_lease_offset(
            pool.total_num_envs,
            cls._shared_leases,
            num_envs,
        )
        if global_offset is None:
            raise ValueError(
                f"BehaviorEnv cannot lease {num_envs} rows from pool "
                f"total_num_envs={pool.total_num_envs}; active leases are "
                f"{sorted(cls._shared_leases.items())}."
            )
        cls._shared_leases[global_offset] = num_envs
        cls._shared_refcount = len(cls._shared_leases)
        return pool, global_offset

    @classmethod
    def release_shared(cls, pool_offset: int | None = None) -> None:
        """Release one leased slice and close the pool after the final lease."""
        if cls._shared_pool is None:
            return
        if pool_offset is None:
            if len(cls._shared_leases) != 1:
                raise RuntimeError(
                    "pool_offset is required when multiple BEHAVIOR slices are active."
                )
            pool_offset = next(iter(cls._shared_leases))
        if int(pool_offset) not in cls._shared_leases:
            raise RuntimeError(
                f"Unknown BEHAVIOR pool lease offset {pool_offset}."
            )
        cls._shared_leases.pop(int(pool_offset))
        cls._shared_refcount = len(cls._shared_leases)
        if cls._shared_refcount <= 0:
            cls._shared_pool.close()
            cls._shared_pool = None
            cls._shared_leases = {}
            cls._shared_key = None

    def __init__(
        self,
        cfg: DictConfig,
        total_num_envs: int,
        num_env_subprocess: int,
        pipeline_stage_num: int,
        physical_worker_index: int,
    ):
        if ray is None:
            raise ImportError(
                "BehaviorEnv's distributed process pool requires Ray. Install "
                "the BEHAVIOR / RLinf environment dependencies before constructing it."
            )
        if total_num_envs <= 0:
            raise ValueError(
                f"total_num_envs must be positive, got {total_num_envs}."
            )
        if num_env_subprocess <= 0:
            raise ValueError(
                f"num_env_subprocess must be positive, got {num_env_subprocess}."
            )
        if total_num_envs % num_env_subprocess != 0:
            raise ValueError(
                f"total_num_envs({total_num_envs}) must be divisible by num_env_subprocess({num_env_subprocess})"
            )

        self.logger = get_logger()
        self.cfg = cfg
        self.total_num_envs = total_num_envs
        self.num_env_subprocess = num_env_subprocess
        self.num_env_shard = total_num_envs // num_env_subprocess
        self.skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )
        base_seed = int(OmegaConf.select(cfg, "seed", default=0))

        # Create subprocess actors with a retry/backoff loop. Actor startup
        # can fail (e.g. simulator plugin errors); retry a few times to handle
        # transient failures. Configurable via `behavior.init_retry_*` keys.
        max_attempts = int(
            OmegaConf.select(cfg, "behavior.init_retry_count", default=3)
        )
        retry_delay = float(
            OmegaConf.select(cfg, "behavior.init_retry_delay", default=5.0)
        )
        backoff = float(
            OmegaConf.select(cfg, "behavior.init_retry_backoff", default=2.0)
        )

        for attempt in range(1, max_attempts + 1):
            try:
                self.env_processes = [
                    BehaviorProcess.remote(
                        self.cfg,
                        self.num_env_shard,
                        pipeline_stage_num,
                        derive_behavior_seed(
                            base_seed,
                            worker_seed_offset=physical_worker_index,
                            row_index=process_idx,
                            reset_count=0,
                            stream=1,
                        ),
                    )
                    for process_idx in range(self.num_env_subprocess)
                ]

                # Wait for all instances to initialize and fetch their activity name
                activity_names_refs = [
                    proc.get_activity_name.remote() for proc in self.env_processes
                ]
                activity_names = ray.get(activity_names_refs)
                break
            except Exception as e:  # noqa: BLE001 - we want to catch any Ray/OmniGibson init error
                # Best-effort cleanup of any partially-created actors
                for proc in getattr(self, "env_processes", []):
                    try:
                        ray.kill(proc)
                    except Exception:
                        pass
                self.env_processes = []

                if attempt >= max_attempts:
                    self.logger.error(
                        "Failed to start BehaviorProcess actors after %d attempts: %s",
                        attempt,
                        e,
                    )
                    raise

                self.logger.warning(
                    "BehaviorProcess creation failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay *= backoff

        if len(set(activity_names)) != 1:
            raise RuntimeError(
                f"Behavior env subprocesses reported different activity_name: "
                f"{activity_names}"
            )
        self.activity_name = activity_names[0]

    def _slice_plan(
        self, global_start: int, num_envs: int
    ) -> list[tuple[int, list[int], list[int]]]:
        """Build the per-subprocess plan for a contiguous global slice.

        Returns entries of ``(subproc_idx, slice_positions, local_rows)``.
        ``slice_positions`` are indices inside the caller's slice and
        ``local_rows`` are the matching rows owned by that subprocess.
        """
        slice_positions_by_proc = [[] for _ in range(self.num_env_subprocess)]
        local_rows_by_proc = [[] for _ in range(self.num_env_subprocess)]
        for pos in range(num_envs):
            global_idx = global_start + pos
            sp = global_idx % self.num_env_subprocess
            slice_positions_by_proc[sp].append(pos)
            local_rows_by_proc[sp].append(global_idx // self.num_env_subprocess)

        return [
            (sp, slice_positions_by_proc[sp], local_rows_by_proc[sp])
            for sp in range(self.num_env_subprocess)
            if slice_positions_by_proc[sp]
        ]

    def env_reset_slice(
        self,
        global_start: int,
        num_envs: int,
        env_indices: list[int] | None = None,
        reset_seeds: list[int] | None = None,
    ):
        """Reset selected rows from one contiguous caller slice.

        Returned observations and infos follow ``env_indices`` order. Omitting
        ``env_indices`` preserves the historical full-slice behavior.
        """
        selected = (
            list(range(num_envs))
            if env_indices is None
            else [int(idx) for idx in env_indices]
        )
        if not selected:
            return [], []
        if len(set(selected)) != len(selected):
            raise ValueError(f"env_indices must be unique, got {selected}.")
        if any(idx < 0 or idx >= num_envs for idx in selected):
            raise IndexError(
                f"env_indices must be within [0, {num_envs}), got {selected}."
            )
        if reset_seeds is not None and len(reset_seeds) != len(selected):
            raise ValueError(
                "reset_seeds must have one entry per selected environment, got "
                f"{len(reset_seeds)} seeds for {len(selected)} environments."
            )

        positions_by_proc = [[] for _ in range(self.num_env_subprocess)]
        local_rows_by_proc = [[] for _ in range(self.num_env_subprocess)]
        seeds_by_proc = [[] for _ in range(self.num_env_subprocess)]
        for output_pos, slice_pos in enumerate(selected):
            global_idx = global_start + slice_pos
            sp = global_idx % self.num_env_subprocess
            positions_by_proc[sp].append(output_pos)
            local_rows_by_proc[sp].append(global_idx // self.num_env_subprocess)
            if reset_seeds is not None:
                seeds_by_proc[sp].append(int(reset_seeds[output_pos]))
        plan = [
            (
                sp,
                positions_by_proc[sp],
                local_rows_by_proc[sp],
                seeds_by_proc[sp] if reset_seeds is not None else None,
            )
            for sp in range(self.num_env_subprocess)
            if positions_by_proc[sp]
        ]
        refs = [
            self.env_processes[sp].reset.remote(local_rows, True, process_seeds)
            for sp, _positions, local_rows, process_seeds in plan
        ]

        shard_results = ray.get(refs)
        all_raw_obs: list = [None] * len(selected)
        all_infos: list = [None] * len(selected)
        for (raw_obs, infos), (
            _sp,
            positions,
            _local_rows,
            _process_seeds,
        ) in zip(shard_results, plan):
            for pos, obs, info in zip(positions, raw_obs, infos):
                all_raw_obs[pos] = obs
                all_infos[pos] = info
        return all_raw_obs, all_infos

    def env_chunk_step_slice(
        self,
        global_start: int,
        slice_num_envs: int,
        chunk_actions: torch.Tensor,
    ):
        """Run chunk_step on shards; pool handles all sharding/merging.
        ``chunk_actions`` must be ``[slice_num_envs, chunk, action_dim]``.
        """
        chunk_size = chunk_actions.shape[1]
        action_dim = chunk_actions.shape[-1]
        plan = self._slice_plan(global_start, slice_num_envs)

        refs = []
        for sp, positions, local_rows in plan:
            actions_j = torch.zeros(
                self.num_env_shard,
                chunk_size,
                action_dim,
                dtype=chunk_actions.dtype,
            )
            actions_j[local_rows] = chunk_actions[positions]
            refs.append(self.env_processes[sp].chunk_step.remote(actions_j, local_rows))

        shard_results = ray.get(refs)
        return self._merge_shards(shard_results, plan, slice_num_envs, chunk_size)

    def _merge_shards(
        self,
        shard_results: list,
        plan: list[tuple[int, list[int], list[int]]],
        slice_num_envs: int,
        chunk_size: int,
    ):
        """Gather per-subprocess shard outputs into ``[chunk][slice]`` order."""
        merged_obs: list = []
        merged_rewards: list = []
        merged_terms: list = []
        merged_trunc: list = []
        merged_infos: list = []
        for t in range(chunk_size):
            is_last = t == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            obs_t: list | None = [None] * slice_num_envs if need_obs else None
            reward_t = torch.zeros(slice_num_envs, dtype=torch.float32)
            term_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            trunc_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            info_t: list = [{} for _ in range(slice_num_envs)]
            for (obs_per_t, rewards_per_t, terms_per_t, truncs_per_t, infos_per_t), (
                _sp,
                positions,
                _local_rows,
            ) in zip(shard_results, plan):
                obs_at_t = obs_per_t[t]
                rewards_at_t = rewards_per_t[t]
                terms_at_t = terms_per_t[t]
                truncs_at_t = truncs_per_t[t]
                infos_at_t = infos_per_t[t]
                for i, pos in enumerate(positions):
                    if need_obs:
                        obs_t[pos] = obs_at_t[i]
                    reward_t[pos] = float(rewards_at_t[i])
                    term_t[pos] = bool(terms_at_t[i])
                    trunc_t[pos] = bool(truncs_at_t[i])
                    info_t[pos] = infos_at_t[i]
            merged_obs.append(obs_t)
            merged_rewards.append(reward_t)
            merged_terms.append(term_t)
            merged_trunc.append(trunc_t)
            merged_infos.append(info_t)
        return merged_obs, merged_rewards, merged_terms, merged_trunc, merged_infos

    def close(self) -> None:
        refs = [proc.close.remote() for proc in self.env_processes]
        ray.get(refs)

        # Kill the procs to free up resources immediately
        for proc in self.env_processes:
            ray.kill(proc)

        self.env_processes = []


class BehaviorEnv(gym.Env):
    """RLinf-compatible vector adapter around the Ray OmniGibson pool."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info=None,
        record_metrics=True,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")

        self.reward_coef = float(cfg.get("reward_coef", 1.0))
        self.ignore_terminations = bool(cfg.get("ignore_terminations", False))
        self.use_rel_reward = bool(cfg.get("use_rel_reward", False))
        self.seed_offset = int(seed_offset)
        self.seed = int(cfg.get("seed", 0)) + self.seed_offset
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info or SimpleNamespace(group_world_size=1)
        group_world_size = int(
            getattr(self.worker_info, "group_world_size", 1)
        )
        if group_world_size <= 0:
            raise ValueError(
                "worker_info.group_world_size must be positive, got "
                f"{group_world_size}."
            )
        if self.total_num_processes <= 0:
            raise ValueError(
                "total_num_processes must be positive, got "
                f"{self.total_num_processes}."
            )
        if self.total_num_processes % group_world_size != 0:
            raise ValueError(
                f"total_num_processes ({self.total_num_processes}) must be divisible by "
                f"worker_info.group_world_size ({group_world_size}) to infer "
                "pipeline_stage_num."
            )
        self.pipeline_stage_num = self.total_num_processes // group_world_size

        self.record_metrics = bool(record_metrics)
        self._is_start = True
        self.enable_offload = bool(cfg.get("enable_offload", False))
        self.enable_init_offload = bool(cfg.get("enable_init_offload", True))
        self.auto_reset = bool(cfg.get("auto_reset", True))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 1000))
        if self.max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be positive, got "
                f"{self.max_episode_steps}."
            )
        self.use_fixed_reset_state_ids = bool(
            cfg.get("use_fixed_reset_state_ids", False)
        )
        if self.use_fixed_reset_state_ids:
            raise ValueError(
                "BehaviorEnv does not support fixed reset_state_ids. Use BEHAVIOR "
                "task.instance_resample_mode with a fixed activity_instance_id instead."
            )

        self.group_size = int(cfg.get("group_size", 1))
        self.num_group = max(1, self.num_envs // max(1, self.group_size))
        self.video_cfg = cfg.get("video_cfg", None)
        self.pool = None
        self.pool_offset = None
        self.task_description = None
        self.current_raw_obs: list[Any] | None = None
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int64)
        self._reset_counts = torch.zeros(self.num_envs, dtype=torch.int64)
        self._reset_seed_bases = torch.full(
            (self.num_envs,),
            int(cfg.get("seed", 0)),
            dtype=torch.int64,
        )
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32)
        self._init_metrics()

        if not (self.enable_offload and not self.enable_init_offload):
            self._init_env()

    def _ensure_pool(self):
        if self.pool is None:
            self.pool, self.pool_offset = BehaviorProcessPool.acquire_shared(
                self.cfg,
                self.worker_info,
                self.pipeline_stage_num,
                self.num_envs,
                self.seed_offset,
            )

    def _load_tasks_cfg(self, activity_name: str):
        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r", encoding="utf-8") as f:
            task_description = [
                json.loads(line) for line in f.read().splitlines() if line.strip()
            ]
        task_description_map = {
            item["task_name"]: item["task"] for item in task_description
        }
        self.task_description = task_description_map.get(
            activity_name,
            activity_name.replace("_", " "),
        )

    def _init_env(self):
        self._ensure_pool()
        self._load_tasks_cfg(self.pool.activity_name)

    def env_reset(
        self,
        env_idx: list[int] | None = None,
        reset_seeds: list[int] | None = None,
    ):
        self._ensure_pool()
        return self.pool.env_reset_slice(
            self.pool_offset,
            self.num_envs,
            env_indices=env_idx,
            reset_seeds=reset_seeds,
        )

    def env_chunk_step(self, chunk_actions: torch.Tensor):
        self._ensure_pool()
        return self.pool.env_chunk_step_slice(
            self.pool_offset,
            self.num_envs,
            chunk_actions,
        )

    def _extract_obs_image(self, raw_obs):
        state = None
        left_image = None
        right_image = None
        zed_image = None
        for sensor_data in raw_obs.values():
            assert isinstance(sensor_data, dict)
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_image = convert_uint8_rgb(v["rgb"])
                elif "right_realsense_link:Camera:0" in k:
                    right_image = convert_uint8_rgb(v["rgb"])
                elif "zed_link:Camera:0" in k:
                    zed_image = convert_uint8_rgb(v["rgb"])
                elif "proprio" in k:
                    state = v
        missing = [
            name
            for name, value in (
                ("state", state),
                ("left wrist camera", left_image),
                ("right wrist camera", right_image),
                ("head camera", zed_image),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "BEHAVIOR observation is missing required entries: "
                + ", ".join(missing)
            )

        return {
            "main_images": zed_image,
            "wrist_images": torch.stack([left_image, right_image], dim=0),
            "state": to_tensor(state),
        }

    def _wrap_obs(self, obs_list):
        extracted_obs_list = [self._extract_obs_image(obs) for obs in obs_list]
        return {
            "main_images": torch.stack(
                [obs["main_images"] for obs in extracted_obs_list],
                dim=0,
            ).cpu(),
            "wrist_images": torch.stack(
                [obs["wrist_images"] for obs in extracted_obs_list],
                dim=0,
            ).cpu(),
            "task_descriptions": [
                self.task_description for _ in range(self.num_envs)
            ],
            "states": torch.stack(
                [obs["state"] for obs in extracted_obs_list],
                dim=0,
            ).cpu(),
        }

    @staticmethod
    def _info_list_to_dict(infos: list[dict[str, Any]]) -> dict[str, list[Any]]:
        keys: list[str] = []
        for info in infos:
            for key in info:
                if key not in keys:
                    keys.append(key)
        return {key: [info.get(key) for info in infos] for key in keys}

    def _scatter_partial_infos(
        self,
        infos: dict[str, list[Any]],
        env_idx: list[int],
    ) -> dict[str, Any]:
        """Expand selected-row reset info to the full vector shape."""

        if len(env_idx) == self.num_envs:
            return infos
        scattered: dict[str, Any] = {}
        for key, values in infos.items():
            if not isinstance(values, list) or len(values) != len(env_idx):
                scattered[key] = values
                continue
            full_values: list[Any] = [None] * self.num_envs
            mask = torch.zeros(self.num_envs, dtype=torch.bool)
            for position, idx in enumerate(env_idx):
                full_values[idx] = values[position]
                mask[idx] = True
            scattered[key] = full_values
            scattered[f"_{key}"] = mask
        return scattered

    @staticmethod
    def _to_cpu_vector(value, *, dtype: torch.dtype, num_envs: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=dtype).detach().cpu().reshape(-1)
        if tensor.numel() != num_envs:
            raise ValueError(
                f"Expected {num_envs} values from BEHAVIOR, got {tensor.numel()}."
            )
        return tensor

    @staticmethod
    def _extract_info_status(info: dict[str, Any]) -> tuple[bool, bool, bool]:
        """Return ``(terminated, truncated, success)`` from OG info variants."""
        if not isinstance(info, dict):
            return False, False, False

        done = info.get("done", {})
        terminated = bool(info.get("terminated", False))
        truncated = bool(
            info.get("truncated", False) or info.get("TimeLimit.truncated", False)
        )
        success = bool(info.get("success", False))

        if isinstance(done, bool):
            terminated |= done
        elif isinstance(done, dict):
            success |= bool(done.get("success", False))
            terminated |= bool(done.get("terminated", False))
            truncated |= bool(done.get("truncated", False))
            termination_conditions = done.get("termination_conditions", {})
            if isinstance(termination_conditions, dict):
                for name, condition in termination_conditions.items():
                    condition_done = (
                        bool(condition.get("done", False))
                        if isinstance(condition, dict)
                        else bool(condition)
                    )
                    if not condition_done:
                        continue
                    normalized_name = str(name).lower()
                    if any(
                        token in normalized_name
                        for token in ("timeout", "time_limit", "max_step", "maxstep")
                    ):
                        truncated = True
                    else:
                        terminated = True
        success |= bool(info.get("task_success", False))
        terminated |= success
        return terminated, truncated, success

    @classmethod
    def _extract_info_done(cls, info: dict[str, Any]) -> bool:
        terminated, truncated, _success = cls._extract_info_status(info)
        return terminated or truncated

    def _calc_step_reward(self, reward: torch.Tensor) -> torch.Tensor:
        scaled_reward = self.reward_coef * reward.to(torch.float32).cpu()
        if not self.use_rel_reward:
            return scaled_reward
        relative_reward = scaled_reward - self.prev_step_reward
        self.prev_step_reward = scaled_reward.clone()
        return relative_reward

    def _normalize_env_indices(self, env_idx=None) -> list[int]:
        if env_idx is None:
            return list(range(self.num_envs))
        if isinstance(env_idx, (int, np.integer)):
            indices = [int(env_idx)]
        elif isinstance(env_idx, torch.Tensor):
            indices = [int(idx) for idx in env_idx.detach().cpu().reshape(-1)]
        else:
            indices = [int(idx) for idx in np.asarray(env_idx).reshape(-1)]
        if len(set(indices)) != len(indices):
            raise ValueError(f"env_idx must contain unique indices, got {indices}.")
        if any(idx < 0 or idx >= self.num_envs for idx in indices):
            raise IndexError(
                f"env_idx must be within [0, {self.num_envs}), got {indices}."
            )
        return indices

    def _next_reset_seeds(
        self,
        env_idx: list[int],
        seed: int | None = None,
    ) -> list[int]:
        if seed is not None:
            self._reset_seed_bases[env_idx] = torch.as_tensor(
                [int(seed)] * len(env_idx),
                dtype=torch.int64,
            )
            self._reset_counts[env_idx] = 0
        pool_offset = int(self.pool_offset or 0)
        seeds = [
            derive_behavior_seed(
                int(self._reset_seed_bases[idx].item()),
                worker_seed_offset=self.seed_offset,
                row_index=pool_offset + idx,
                reset_count=int(self._reset_counts[idx].item()),
                stream=2,
            )
            for idx in env_idx
        ]
        self._reset_counts[env_idx] += 1
        return seeds

    def reset(
        self,
        env_idx=None,
        reset_state_ids=None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if options:
            if env_idx is None:
                env_idx = options.get("env_idx")
            if reset_state_ids is None:
                reset_state_ids = options.get(
                    "reset_state_ids",
                    options.get("episode_id"),
                )
        if reset_state_ids is not None:
            raise ValueError(
                "BehaviorEnv reset_state_ids are unsupported; configure "
                "task.activity_instance_id / task.instance_resample_mode instead."
            )
        if self.enable_offload and self.pool is None:
            self._init_env()

        indices = self._normalize_env_indices(env_idx)
        if not indices:
            if self.current_raw_obs is None:
                raise RuntimeError(
                    "Cannot perform an empty reset before the first full reset."
                )
            return self._wrap_obs(self.current_raw_obs), {}
        if self.current_raw_obs is None and len(indices) != self.num_envs:
            raise RuntimeError(
                "A full BehaviorEnv reset is required before a partial reset."
            )

        reset_seeds = self._next_reset_seeds(indices, seed=seed)
        raw_obs, raw_infos = self.env_reset(indices, reset_seeds)
        if len(raw_obs) != len(indices) or len(raw_infos) != len(indices):
            raise RuntimeError(
                "BEHAVIOR pool returned an invalid partial-reset batch: "
                f"{len(raw_obs)} observations and {len(raw_infos)} infos for "
                f"{len(indices)} requested environments."
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for idx, obs in zip(indices, raw_obs):
            self.current_raw_obs[idx] = obs

        self._reset_metrics(indices)
        self._is_start = False
        obs = self._wrap_obs(self.current_raw_obs)
        infos = self._info_list_to_dict(
            [info if isinstance(info, dict) else {} for info in raw_infos]
        )
        infos = self._scatter_partial_infos(infos, indices)
        return obs, infos

    def _process_step_result(
        self,
        raw_obs,
        raw_rewards,
        raw_terminations,
        raw_truncations,
        raw_infos,
        *,
        auto_reset: bool,
    ):
        rewards = self._to_cpu_vector(
            raw_rewards,
            dtype=torch.float32,
            num_envs=self.num_envs,
        )
        terminations = self._to_cpu_vector(
            raw_terminations,
            dtype=torch.bool,
            num_envs=self.num_envs,
        )
        truncations = self._to_cpu_vector(
            raw_truncations,
            dtype=torch.bool,
            num_envs=self.num_envs,
        )
        info_list = [
            info if isinstance(info, dict) else {} for info in list(raw_infos)
        ]
        if len(info_list) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} BEHAVIOR info dicts, got "
                f"{len(info_list)}."
            )

        self._elapsed_steps += 1
        info_status = [self._extract_info_status(info) for info in info_list]
        info_terminations = torch.tensor(
            [status[0] for status in info_status],
            dtype=torch.bool,
        )
        info_truncations = torch.tensor(
            [status[1] for status in info_status],
            dtype=torch.bool,
        )
        successes = torch.tensor(
            [status[2] for status in info_status],
            dtype=torch.bool,
        )
        terminations |= info_terminations
        truncations |= info_truncations
        truncations |= self._elapsed_steps >= self.max_episode_steps

        step_rewards = self._calc_step_reward(rewards)
        infos = self._info_list_to_dict(info_list)
        infos = self._record_metrics(step_rewards, successes, infos)
        if self.ignore_terminations:
            if "episode" in infos:
                infos["episode"]["success_at_end"] = successes.clone()
            else:
                infos["success_at_end"] = successes.clone()
            terminations.zero_()

        if raw_obs is None:
            obs = None
        else:
            self.current_raw_obs = list(raw_obs)
            obs = self._wrap_obs(self.current_raw_obs)

        dones = terminations | truncations
        if bool(dones.any()) and auto_reset and self.auto_reset:
            if obs is None:
                raise RuntimeError(
                    "BEHAVIOR auto-reset requires the final step observation."
                )
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return obs, step_rewards, terminations, truncations, infos

    def step(self, actions=None, auto_reset=True):
        """Advance every vector row by one action."""
        if actions is None:
            raise ValueError("BehaviorEnv.step requires an action batch.")
        action_tensor = torch.as_tensor(actions).detach().cpu()
        if action_tensor.ndim == 1 and self.num_envs == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.ndim != 2 or action_tensor.shape[0] != self.num_envs:
            raise ValueError(
                "BehaviorEnv.step actions must have shape "
                f"[{self.num_envs}, action_dim], got {tuple(action_tensor.shape)}."
            )
        results = self.env_chunk_step(action_tensor.unsqueeze(1))
        if any(len(component) != 1 for component in results):
            raise RuntimeError(
                "BEHAVIOR pool returned an invalid singleton chunk for step()."
            )
        return self._process_step_result(
            *(component[0] for component in results),
            auto_reset=bool(auto_reset),
        )

    def chunk_step(self, chunk_actions):
        """Advance an action chunk while retaining LIBERO-compatible semantics."""
        action_tensor = torch.as_tensor(chunk_actions).detach().cpu()
        if action_tensor.ndim != 3 or action_tensor.shape[0] != self.num_envs:
            raise ValueError(
                "BehaviorEnv.chunk_step actions must have shape "
                f"[{self.num_envs}, chunk, action_dim], got "
                f"{tuple(action_tensor.shape)}."
            )
        chunk_size = int(action_tensor.shape[1])
        if chunk_size <= 0:
            raise ValueError("BehaviorEnv.chunk_step requires a non-empty chunk.")

        raw_results = self.env_chunk_step(action_tensor)
        if any(len(component) != chunk_size for component in raw_results):
            raise RuntimeError(
                "BEHAVIOR pool returned a chunk with inconsistent timestep counts."
            )

        obs_list = []
        infos_list = []
        reward_list = []
        termination_list = []
        truncation_list = []
        for step_result in zip(*raw_results):
            obs, reward, termination, truncation, infos = (
                self._process_step_result(*step_result, auto_reset=False)
            )
            obs_list.append(obs)
            reward_list.append(reward)
            termination_list.append(termination)
            truncation_list.append(truncation)
            infos_list.append(infos)

        chunk_rewards = torch.stack(reward_list, dim=1)
        raw_chunk_terminations = torch.stack(termination_list, dim=1)
        raw_chunk_truncations = torch.stack(truncation_list, dim=1)
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = past_terminations | past_truncations

        if bool(past_dones.any()) and self.auto_reset:
            if obs_list[-1] is None:
                raise RuntimeError(
                    "BEHAVIOR auto-reset requires the final chunk observation."
                )
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones,
                obs_list[-1],
                infos_list[-1],
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations
            chunk_truncations = raw_chunk_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    @property
    def device(self):
        return "cpu"

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = bool(value)

    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)
        self.success_episode_len = torch.zeros(
            self.num_envs,
            dtype=torch.int64,
        )

    def _reset_metrics(self, env_idx=None):
        indices = self._normalize_env_indices(env_idx)
        if not indices:
            return
        self.prev_step_reward[indices] = 0.0
        self._elapsed_steps[indices] = 0
        self.success_once[indices] = False
        self.returns[indices] = 0.0
        self.success_episode_len[indices] = 0

    def _record_metrics(self, rewards, successes, infos):
        if not self.record_metrics:
            return infos
        pending_success = ~self.success_once
        self.returns += rewards * pending_success.to(rewards.dtype)
        new_successes = successes & pending_success
        self.success_episode_len[new_successes] = self.elapsed_steps[
            new_successes
        ]
        self.success_once |= successes
        episode_len_for_reward = torch.where(
            self.success_once,
            self.success_episode_len,
            self.elapsed_steps,
        )
        episode_info = {
            "success": successes.clone(),
            "success_once": self.success_once.clone(),
            "return": self.returns.clone(),
            "episode_len": self.elapsed_steps.clone(),
            "reward": self.returns
            / torch.clamp(episode_len_for_reward, min=1).to(torch.float32),
        }
        infos["episode"] = episode_info
        return infos

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        dones = torch.as_tensor(dones, dtype=torch.bool).detach().cpu()
        final_obs = copy.deepcopy(extracted_obs)
        final_info = copy.deepcopy(infos)
        env_idx = torch.nonzero(dones, as_tuple=False).flatten().tolist()
        extracted_obs, reset_infos = self.reset(env_idx=env_idx)
        reset_infos["final_observation"] = final_obs
        reset_infos["final_info"] = final_info
        reset_infos["_final_info"] = dones.clone()
        reset_infos["_final_observation"] = dones.clone()
        reset_infos["_elapsed_steps"] = dones.clone()
        return extracted_obs, reset_infos

    def update_reset_state_ids(self):
        raise NotImplementedError(
            "BehaviorEnv uses task.instance_resample_mode instead of reset_state_ids."
        )

    def offload(self):
        self.close()

    def close(self):
        if self.pool is not None:
            BehaviorProcessPool.release_shared(self.pool_offset)
            self.pool = None
            self.pool_offset = None
