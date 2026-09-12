"""Operator-only LIBERO snapshots and raw safety traces; never returned by tools.

Enabled by the server process, not by Agent arguments. No pickle or environment
dump. Binary MuJoCo model + INTEGRATION state preserve geometry and solver state.
The restore helper targets a compatible existing environment for local diagnosis.
"""
from pathlib import Path
import hashlib
import json
import os
import random
import time
import uuid
import numpy as np


def _numeric_fields(obj):
    fields, omitted = {}, []
    for key, value in vars(obj).items():
        if isinstance(value, np.ndarray) and value.dtype.kind in 'bifu' and value.size < 100000:
            fields[key] = {'array': value.tolist(), 'dtype': str(value.dtype)}
        elif value is None or type(value) in (str, bool, int, float):
            fields[key] = value
        else:
            omitted.append(key)
    return fields, omitted


def _runtime_objects(raw, robot):
    objects = {'environment':raw.env,'robot':robot,'gripper':robot.gripper,'controller':robot.controller}
    for prefix, obj in list(objects.items()):
        for name,value in vars(obj).items():
            # Derivative and recent-motion buffers affect the velocity PID.
            if type(value).__module__ == 'robosuite.utils.buffers':
                objects[prefix+'.'+name] = value
    return objects


def capture(env, phase, request=None, *, root=None):
    root = root or os.environ.get('OPENETA_PRIVATE_STATE_DIR')
    if not root:
        return None
    from sim.controllers.mink_goal import _libero_runtime
    try:
        raw, robot = _libero_runtime(env.unwrapped)
    except RuntimeError:
        return None
    import mujoco
    model, data = raw.sim.model._model, raw.sim.data._data
    directory = Path(root).resolve() / f'{time.time_ns()}-{uuid.uuid4().hex[:8]}-{phase}'
    directory.mkdir(parents=True, mode=0o700)
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    digest = hashlib.sha256(binary.tobytes()).hexdigest()
    model_path = directory.parent / f'model-{digest}.mjb'
    if not model_path.exists():
        model_path.write_bytes(binary.tobytes())
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, state, state_spec)
    # Store mutable model geometry separately for restore into a compatible env.
    np.savez_compressed(directory/'state.npz', integration=state, qpos=data.qpos,
        qvel=data.qvel, ctrl=data.ctrl, qacc_warmstart=data.qacc_warmstart,
        body_pos=model.body_pos, body_quat=model.body_quat)
    runtime = {}; omitted = {}
    for name, obj in _runtime_objects(raw,robot).items():
        runtime[name], omitted[name] = _numeric_fields(obj)
    rng = env.unwrapped._env._reset_rng
    ns = rng.numpy
    sources = {}
    for relative in ['sim/env_registry.py','sim/controllers/mink_goal.py','sim/controllers/gripper_guard.py','sim/bench_worker.py']:
        path = Path(__file__).resolve().parents[1]/relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {'schema_version':'openeta.private_sim_state.v1','phase':phase,
        'env_id':getattr(env,'_env_id',None),'task':getattr(env.unwrapped._env,'_task_description',''),
        'controller_profile':getattr(env.unwrapped._env,'_controller_profile',None),
        'model_file':model_path.name,'model_sha256':digest,'mujoco_version':mujoco.__version__,
        'state_spec':int(state_spec),'request':request,'runtime':runtime,'omitted_object_fields':omitted,
        'reset_rng':{'seed':rng.seed,'numpy':[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]],'python':rng.python},
        'global_numpy_state':_numpy_json_state(np.random.get_state()),'global_python_state':random.getstate(),
        'source_sha256':sources,
        'restore_scope':'compatible_worker_environment; unsupported object fields listed explicitly'}
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return directory


def _numpy_json_state(state):
    return [state[0],state[1].tolist(),state[2],state[3],state[4]]


def _numpy_state(state):
    return (state[0],np.asarray(state[1],dtype=np.uint32),state[2],state[3],state[4])


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value,list) else value


def restore(env, directory):
    """Restore our own trusted snapshot into a compatible LIBERO instance.

    Not an Agent tool. Does not assert arbitrary cross-version bitwise replay.
    """
    import mujoco
    from sim.controllers.mink_goal import _libero_runtime
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    if meta['mujoco_version'] != mujoco.__version__:
        raise ValueError('snapshot MuJoCo version mismatch')
    binary = (directory.parent/meta['model_file']).read_bytes()
    if hashlib.sha256(binary).hexdigest() != meta['model_sha256']:
        raise ValueError('snapshot model checksum mismatch')
    raw, robot = _libero_runtime(env.unwrapped)
    model,data=raw.sim.model._model,raw.sim.data._data
    saved_model = mujoco.MjModel.from_binary_path(str(directory.parent/meta['model_file']))
    # Copy every writable model array, including randomized fixture placement,
    # dynamics and contact parameters. References in robosuite remain valid.
    if (model.nq,model.nv,model.nbody,model.ngeom)!=(saved_model.nq,saved_model.nv,saved_model.nbody,saved_model.ngeom):
        raise ValueError('incompatible snapshot model topology')
    for name in dir(saved_model):
        source,target=getattr(saved_model,name),getattr(model,name)
        if isinstance(source,np.ndarray) and isinstance(target,np.ndarray) and target.flags.writeable and source.shape==target.shape:
            target[...] = source
    with np.load(directory/'state.npz',allow_pickle=False) as arrays:
        mujoco.mj_setState(model,data,arrays['integration'],meta['state_spec'])
        mujoco.mj_forward(model,data)
        # mj_forward changes warmstart/derived solver values; integration state
        # must be restored again after refreshing kinematics.
        mujoco.mj_setState(model,data,arrays['integration'],meta['state_spec'])
    for name,obj in _runtime_objects(raw,robot).items():
        for key,value in meta['runtime'][name].items():
            setattr(obj,key,np.asarray(value['array'],dtype=value['dtype']) if isinstance(value,dict) and 'array' in value else value)
    rng=env.unwrapped._env._reset_rng
    rng.seed=meta['reset_rng']['seed'];rng.numpy=_numpy_state(meta['reset_rng']['numpy']);rng.python=_tuples(meta['reset_rng']['python'])
    np.random.set_state(_numpy_state(meta['global_numpy_state']));random.setstate(_tuples(meta['global_python_state']))


def complete(directory, result):
    if directory is not None:
        # Deliberately private: retain exact geometry and controller failure
        # diagnostics without duplicating large images or public observations.
        payload={k:v for k,v in result.items() if k not in ('observation','cameras')}
        (directory/'outcome.json').write_text(json.dumps(payload,indent=2))


def recorded(env, phase, request, function):
    before = capture(env, phase+'-start', request)
    result = {}
    try:
        result = function()
        return result
    except Exception as exc:
        result = {'error_type':type(exc).__name__,'error':str(exc)}
        raise
    finally:
        try:
            complete(before, result)
            after = capture(env, phase+'-end', request)
            complete(after, result)
        except Exception as exc:
            # Physics may already have executed. A disk failure must not
            # replace a known motion receipt with a fictitious zero-step error.
            import logging
            logging.getLogger(__name__).exception('Operator post-state recording failed')
            result['operator_recording_status'] = 'post_state_failed'
