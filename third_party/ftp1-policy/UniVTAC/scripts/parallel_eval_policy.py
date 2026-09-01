import os
import sys
import time
import json
import yaml
import argparse
import importlib
import traceback
import io
from tqdm import tqdm
from queue import Empty
from pathlib import Path
from typing import Literal, TYPE_CHECKING
from multiprocessing import Process, Queue, Manager, Event
from eval_result_utils import build_run_id, load_worker_metadata, summarize_worker_metadata, worker_name, write_json

if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg
    from policy._base_policy import BasePolicy


# Avoid importing isaacsim before process start in main; workers will import.
def get_config(file, default_root: Path, type: Literal['yaml', 'json']):
    if type == 'yaml':
        file = Path(file) if (file.endswith('.yml') or file.endswith('.yaml')) else (default_root / f'{file}.yml')
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        file = Path(file) if file.endswith('.json') else (default_root / f'{file}.json')
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file


def split_devices(cuda_visible_devices: str, workers: int):
    devices = [d.strip() for d in cuda_visible_devices.split(',') if d.strip() != '']
    if not devices:
        return [['']] * workers
    assignment = []
    for i in range(workers):
        assignment.append([devices[i % len(devices)]])
    return assignment


def write_run_summary(
    run_root: Path,
    *,
    policy_name: str,
    task_name: str,
    run_id: str,
    deploy_config: dict,
    task_config_file: Path,
    deploy_config_file: Path,
    total_tests: int,
    workers: int,
) -> dict:
    worker_payloads = load_worker_metadata(run_root / "metadata")
    summary = summarize_worker_metadata(worker_payloads)
    summary.update({
        "policy_name": policy_name,
        "task_name": task_name,
        "run_id": run_id,
        "run_root": str(run_root),
        "task_config_file": str(task_config_file),
        "deploy_config_file": str(deploy_config_file),
        "total_tests_target": total_tests,
        "workers_target": workers,
        "worker_metadata_files": [f"metadata/{name}.json" for name in sorted(worker_payloads)],
        "deploy_config": deploy_config,
    })
    write_json(run_root / "metadata.json", summary)
    return summary


def worker_run(worker_idx, args, deploy_config, task_config, task_file_name, policy_name,
               instructions, base_save_dir: Path, seed_q: Queue, progress, stop_event: Event,
               log_file: Path, device_list, status_dict, result_q: Queue):
    # Per-process env: set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_list) if device_list else ''

    # Redirect stdout/stderr to unified log (append, line-buffered)
    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    sys.path.insert(0, '.')
    sys.path.insert(0, './policy')

    # Import isaac app and launch inside process init
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    # Build minimal args for app
    app_args = parser.parse_args([])  # empty to use defaults
    app_args.enable_cameras = True
    app_args.livestream = 2
    app_args.num_envs = 1

    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    try:
        # Dynamic imports after app launch
        task_module = importlib.import_module(f"envs.{task_file_name}")
        policy_module = importlib.import_module(f"policy.{policy_name}")

        env_cfg:'BaseTaskCfg' = task_module.TaskCfg()
        worker_id = worker_name(worker_idx)
        env_cfg.save_dir = base_save_dir
        env_cfg.worker_name = worker_id
        env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
        env_cfg.obs_data_type = task_config.get("observations", {})
        env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
        env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
        env_cfg.scene.num_envs = 1

        # Device stays default; CUDA env controls GPU routing
        init_start = time.perf_counter()
        policy:'BasePolicy' = policy_module.Policy(deploy_config)
        policy_init_cost = time.perf_counter() - init_start

        init_start = time.perf_counter()
        task:'BaseTask' = task_module.Task(env_cfg, mode='eval')

        # Override _step_callback to update status_dict
        original_step_callback = task._step_callback
        def step_callback(status):
            nonlocal status_dict
            status_dict['steps'] = status['step_count']
            status_dict['action'] = status['take_action_cnt']
            status_dict['atom'] = f"{status['atom_id']}:{status['atom_tag']}"
            return original_step_callback(status)
        task._step_callback = step_callback

        task_init_cost = time.perf_counter() - init_start

        print(f"[Worker {worker_id}] Task init in {task_init_cost:.2f}s, Policy init in {policy_init_cost:.2f}s")

        # Initialize worker status
        status_dict['state'] = 'idle'
        status_dict['current_seed'] = None
        status_dict['last_result'] = '-'
        status_dict['last_cost'] = None
        status_dict['steps'] = None
        status_dict['actions'] = None
        status_dict['done'] = 0
        status_dict['succ'] = 0
        status_dict['errors'] = 0

        # Eval loop: consume seeds
        while not stop_event.is_set():
            try:
                seed = seed_q.get(timeout=1.0)
            except Exception:
                continue
            if seed is None:  # sentinel
                break

            succ = False
            status_dict['state'] = 'running'
            status_dict['current_seed'] = seed
            eval_start = time.perf_counter()
            task.mode = 'eval'
            try:
                task.reset(seed=seed, instructions=instructions[deploy_config.get("instruction_type", "seen")])
                task.mean_steps = task.cfg.step_lim
                policy.reset()
                while task.take_action_cnt < task.cfg.step_lim:
                    observation = task._get_observations()
                    policy.eval(task, observation)
                    if task.eval_success:
                        succ = True
                        break
                    if task.check_early_stop():
                        break
            except Exception as e:
                print(f"[Worker {worker_id}] Seed {seed} exception: {e}\n{traceback.format_exc()}")
                succ_status = 'error'
                task.clean_cache(result=succ_status)
                progress['errors'] += 1
                status_dict['errors'] += 1
                status_dict['last_result'] = 'error'
                status_dict['last_cost'] = None
                status_dict['steps'] = None
                status_dict['actions'] = None
                status_dict['current_seed'] = None
                status_dict['state'] = 'idle'
                result_q.put({
                    'worker': worker_id,
                    'seed': seed,
                    'result': 'error',
                    'cost': None,
                    'steps': None,
                    'actions': None,
                    'traceback': traceback.format_exc(),
                })
            else:
                eval_cost = time.perf_counter() - eval_start
                succ_status = 'success' if succ else 'failed'
                task.clean_cache(result=succ_status)
                if succ:
                    progress['succ'] += 1
                    status_dict['succ'] += 1
                progress['done'] += 1
                status_dict['done'] += 1
                status_dict['last_result'] = succ_status
                status_dict['last_cost'] = eval_cost
                status_dict['steps'] = task.step_count
                status_dict['actions'] = task.take_action_cnt
                status_dict['current_seed'] = None
                status_dict['state'] = 'idle'
                print(f"[Worker {worker_id}] Seed {seed} {succ_status} after {eval_cost:.2f}s; "
                      f"steps {task.step_count}, actions {task.take_action_cnt}")
                result_q.put({
                    'worker': worker_id,
                    'seed': seed,
                    'result': succ_status,
                    'cost': eval_cost,
                    'steps': task.step_count,
                    'actions': task.take_action_cnt,
                    'traceback': None,
                })
        try:
            task.close()
            policy.close()
        finally:
            simulation_app.close()
            os.close(log_fd)
    except Exception:
        # Ensure app closes on fatal errors
        try:
            simulation_app.close()
        except Exception:
            pass
        raise

def main():
    parser = argparse.ArgumentParser(description="Eval Policy (multiprocessing)")
    parser.add_argument("task_name", type=str, help="Task name")
    parser.add_argument("task_config", type=str, help="Task config")
    parser.add_argument("deploy_config", type=str, help="Deploy file name")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker processes")
    parser.add_argument("--total_num", type=int, default=100, help="Total tests across all workers")
    parser.add_argument("--gpu", type=str, default=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                        help="CUDA_VISIBLE_DEVICES list to split among workers")
    args = parser.parse_args()
    
    train_config = os.environ.get('TRAIN_CONFIG', 'Unknown')

    # Load configs
    task_config, task_config_file = get_config(
        args.task_config, default_root=Path(__file__).parent.parent / 'task_config', type='yaml'
    )
    deploy_config, deploy_config_file = get_config(
        args.deploy_config, default_root=Path(__file__).parent.parent / 'policy', type='yaml'
    )
    policy_name = deploy_config['policy_name']
    deploy_config['task_name'] = args.task_name
    deploy_config['task_config'] = task_config_file.stem
    deploy_config['instruction_file'] = deploy_config.get('instruction_file', args.task_name)
    try:
        instructions, _ = get_config(
            deploy_config['instruction_file'], default_root=Path(__file__).parent.parent / 'instructions', type='json'
        )
        # Fallback if instructions missing keys
        if not isinstance(instructions, dict) or 'seen' not in instructions or 'unseen' not in instructions:
            instructions = {'seen': ['Empty'], 'unseen': ['Empty']}
    except Exception:
        instructions = {'seen': ['Empty'], 'unseen': ['Empty']}

    # Base save dir with unified date
    run_id = build_run_id()
    base_save_dir = Path('eval_result') / policy_name / args.task_name / run_id
    base_save_dir.mkdir(parents=True, exist_ok=True)
    out_log = base_save_dir / 'out.log'
    clean_log = base_save_dir / 'log.log'
    out_log.touch()
    clean_log.touch()

    # Seed producer-consumer
    manager = Manager()
    seed_q = Queue()
    progress = manager.dict(done=0, succ=0, errors=0)
    stop_event = Event()
    result_q = Queue()

    def write_out(msg: str):
        with open(out_log, 'a') as f:
            f.write(msg + '\n')

    def write_clean(msg: str):
        stamped = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
        with open(clean_log, 'a') as f:
            f.write(stamped + '\n')

    # Dynamic seed generation so errors don't count toward total_num
    # Start seed aligned with original logic: base 1,000,000 offset if needed
    start_seed = 1000000 * (1 + deploy_config.get("seed", 0))
    next_seed = start_seed

    # GPU allocation
    assignments = split_devices(args.gpu, args.workers)

    # Write initial run parameters
    write_clean("Run parameters:")
    write_out("Run parameters:")
    params_json = json.dumps({
        'task': args.task_name,
        'task_config_file': str(task_config_file),
        'deploy_config_file': str(deploy_config_file),
        'train_config': train_config,
        'total_tests': args.total_num,
        'workers': args.workers,
        'cuda_assignments': assignments,
        'save_dir': str(base_save_dir),
        'start_seed': start_seed,
    }, ensure_ascii=False, indent=4)
    write_clean(params_json)
    write_out(params_json)
    write_clean("Deploy config:")
    write_out("Deploy config:")
    deploy_json = json.dumps(deploy_config, ensure_ascii=False, indent=4)
    write_clean(deploy_json)
    write_out(deploy_json)

    workers = []
    worker_status = []
    for w in range(args.workers):
        status_proxy = manager.dict()
        worker_status.append(status_proxy)
        p = Process(
            target=worker_run,
            name=worker_name(w),
            args=(w, args, deploy_config, task_config, args.task_name, policy_name, instructions,
                  base_save_dir, seed_q, progress, stop_event, out_log, assignments[w], status_proxy, result_q)
        )
        p.start()
        workers.append(p)

    pbar_io = io.StringIO()
    pbar = tqdm(total=args.total_num, file=pbar_io, ncols=100)

    # Status board and producer loop
    try:
        last_update = 0
        last_render = 0
        last_block = ""
        # Prime the queue with up to workers seeds to start
        for _ in range(args.workers):
            seed_q.put(next_seed)
            next_seed += 1

        while any(p.is_alive() for p in workers):
            # Drain results queue and log clean summaries
            while True:
                try:
                    event = result_q.get_nowait()
                except Empty:
                    break
                else:
                    prefix = f"{event['worker']} Seed {event['seed']}"
                    if event['result'] == 'success':
                        write_clean(f"{prefix} success in {event['cost']:.2f}s; steps {event['steps']}, actions {event['actions']}")
                    elif event['result'] == 'failed':
                        write_clean(f"{prefix} failed in {event['cost']:.2f}s; steps {event['steps']}, actions {event['actions']}")
                    else:
                        write_clean(f"{prefix} error; see out.log for traceback")
                    write_out(f"{prefix} result={event['result']} cost={event['cost']} steps={event['steps']} actions={event['actions']}")

                    done = progress.get('done', 0)
                    if done < args.total_num:
                        # Feed more seeds; try to keep queue non-empty
                        # Put a few seeds per iteration to avoid starvation
                        for _ in range(args.workers):
                            seed_q.put(next_seed)
                            next_seed += 1
                    else:
                        # Signal workers to stop once target reached
                        for _ in range(args.workers):
                            seed_q.put(None)
                        # Wait for workers to drain and exit
                        stop_event.set()

            # Keep feeding seeds until reaching target completed (done)
            done = progress.get('done', 0)
            pbar.n = done
            pbar.refresh()

            now = time.time()
            if now - last_update > 0.2:
                last_update = now
                succ = progress.get('succ', 0)
                errors = progress.get('errors', 0)
                rate = (succ / done * 100) if done > 0 else 0
                
                bar_str = pbar_io.getvalue()
                pbar_io.seek(0)
                pbar_io.truncate(0)

                status_lines = [
                    f"[Task: {args.task_name} (Config: {task_config_file.stem})] [Policy: {policy_name} (Config: {train_config})] [Deploy Config {deploy_config_file.stem}]",
                    f"[Status] {done}/{args.total_num} done | succ {succ} | errors {errors} | success_rate {rate:.2f}%"
                ]
                for idx, st in enumerate(worker_status):
                    seed_str = str(st.get('current_seed', '-')) if st.get('current_seed') is not None else '-'
                    cost_val = st.get('last_cost')
                    cost_str = f"{cost_val:.2f}" if cost_val is not None else '-'
                    step_str = str(st.get('steps', '-')) if st.get('steps') is not None else '-'
                    action_str = str(st.get('actions', '-')) if st.get('actions') is not None else '-'
                    atom_str = str(st.get('atom', '-'))
                    status_lines.append(
                        f"  {worker_name(idx)}: {st.get('state', 'init'):<7} seed={seed_str:<8} "
                        f"last={st.get('last_result', '-'):<7} cost={cost_str:<7} steps={step_str:<6} actions={action_str:<6} atom={atom_str}"
                    )
                block = '\n'.join(status_lines)
                if block != last_block or (now - last_render > 2.0):
                    # Clear screen then print the latest block to keep console clean
                    print("\033[2J\033[H" + bar_str + "\n" + block, end='', flush=True)
                    last_block = block
                    last_render = now
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        # Send stop sentinels
        for _ in range(args.workers):
            seed_q.put(None)
    finally:
        pbar.close()
        pbar_io.close()
        for p in workers:
            p.join()
        print("\n[Status] All workers finished.")
        # Final summary
        done = progress.get('done', 0)
        succ = progress.get('succ', 0)
        errors = progress.get('errors', 0)
        rate = (succ / done * 100) if done > 0 else 0
        summary = write_run_summary(
            base_save_dir,
            policy_name=policy_name,
            task_name=args.task_name,
            run_id=run_id,
            deploy_config=deploy_config,
            task_config_file=task_config_file,
            deploy_config_file=deploy_config_file,
            total_tests=args.total_num,
            workers=args.workers,
        )
        final_msg = (
            f"[Final] {succ}/{done} ({rate:.2f}%) success; errors {errors}; "
            f"out_log={out_log}; clean_log={clean_log}; summary={base_save_dir / 'metadata.json'}; "
            f"episodes={summary['total_episodes']}"
        )
        print(final_msg)
        write_clean(final_msg)
        write_out(final_msg)


if __name__ == "__main__":
    main()
