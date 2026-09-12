#!/usr/bin/env python3
"""Frozen shot-selection manifest, resumable rollout queue and independent media."""
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import io
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.univtac.run_autonomous_insert_hole import codex_command, operator_prompt
from scripts.univtac.run_eight_task_coverage import (
    retryable_initialization,
    reviewed_initialization_issue,
    run_cell,
    startup_crashes_since_ready,
)
from scripts.univtac.run_fourway_capacity import Coordinator, resources
from sim.envs.univtac.feedback_protocol import PROTOCOL, project_query
from sim.envs.univtac.tactile_history import export_review_video
from tools.embodied_mcp_server import build_live_backend_server

PROJECTIONS = {'B': 'visual_action_icl', 'C': 'tactile_action_icl'}
TERMINAL = {'completed', 'initialization_unavailable'}


def load(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    pending.replace(path)


def ordered_cells(tasks, seeds):
    combinations = [(shot, condition) for shot in (1, 2, 4) for condition in ('B', 'C')]
    for i, seed in enumerate(seeds):
        for j in range(6):
            for k in range(len(tasks)):
                task = tasks[(k+i+j) % len(tasks)]
                shot, condition = combinations[(j+i+tasks.index(task)) % 6]
                ids = [i % 2] if shot == 1 else list(range(shot))
                yield {'task':task, 'seed':seed, 'query_index':i, 'shot':shot,
                       'comparison_condition':condition, 'condition':f'{condition}_{shot}shot',
                       'expert_ids':ids, 'demonstration_set_id':'official_'+'_'.join(map(str, ids))}


def fetch_additional(args, settings):
    """Only the next two successful official IDs per task; direct downloads."""
    source = REPO/settings['source_coverage']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    selections = {}
    for task in settings['tasks']:
        folder = args.output_root/'data'/task
        folder.mkdir(parents=True, exist_ok=True)
        metadata = load(source/'data'/task/'metadata.json')
        ids = [int(k) for k in sorted(metadata, key=int) if int(k)>1 and metadata[k]['result']=='success'][:2]
        if len(ids)!=2:
            raise ValueError(f'{task}: fewer than two additional official successes')
        selections[task] = [0, 1, *ids]
        atomic_json(folder/'metadata.json', metadata)
        atomic_json(folder/'selection.json', {'rule':'Next two metadata successes after unchanged official episodes 0/1; no query selection', 'additional_ids':ids})
        for episode in ids:
            target = folder/f'{episode}.hdf5'
            if target.exists():
                continue
            url = 'https://modelscope.cn/api/v1/datasets/byml2024/UniVTAC/repo?' + urllib.parse.urlencode(
                {'Revision':'master', 'FilePath':f'isaac51/{task}/hdf5/{episode}.hdf5'})
            start = time.time()
            try:
                with opener.open(url, timeout=60) as response, target.with_suffix('.hdf5.part').open('wb') as f:
                    while block := response.read(1024*1024):
                        f.write(block)
            except OSError as exc:
                atomic_json(target.with_suffix('.download_error.json'), {'error':str(exc), 'url':url})
                subprocess.run(['curl','--noproxy','*','--fail','--location','--retry','2',
                    '--connect-timeout','30','--max-time','600','--output',str(target.with_suffix('.hdf5.part')),url], check=True)
            target.with_suffix('.hdf5.part').replace(target)
            atomic_json(target.with_suffix('.download.json'), {'url':url, 'bytes':target.stat().st_size,
                'elapsed_seconds':time.time()-start, 'direct_connection':True})
            print('downloaded', task, episode, target.stat().st_size, flush=True)
    atomic_json(args.output_root/'expert_selection.json', selections)


def matched_subset(sources, ids, target):
    """Select already-processed episodes without altering their content or pixels."""
    target.mkdir(parents=True, exist_ok=True)
    base = copy.deepcopy(sources[0][1]['text'])
    base.pop('historical_touch')
    base['examples'] = []
    touch = []
    vision_images, touch_images = [], []
    for episode in ids:
        name = f'official_episode_{episode}'
        package, full = next((p, f) for p, f in sources if any(e['example_id']==name for e in f['text']['examples']))
        example = next(e for e in full['text']['examples'] if e['example_id']==name)
        base['examples'].append(copy.deepcopy(example))
        history = [x for x in full['text']['historical_touch'] if x['example_id']==name]
        touch.extend(copy.deepcopy(history))
        visual_labels = {v for s in example['segments'] for v in s['vision_labels']}
        tactile_labels = {v for h in history for v in h['image_labels']}
        for im in full['images']:
            descriptor = {**im, 'path':str((package/im['path']).resolve())}
            if im['label'] in visual_labels:
                vision_images.append(descriptor)
            elif im['label'] in tactile_labels:
                touch_images.append(descriptor)
    b = {'text':base, 'images':vision_images}
    c = {'text':{**copy.deepcopy(base), 'historical_touch':touch}, 'images':vision_images+touch_images}
    atomic_json(target/'visual_action_icl.json', b)
    atomic_json(target/'tactile_action_icl.json', c)
    return b, c


def check_mcp(package):
    results = {}
    b, c = [load(package/f'{name}.json') for name in PROJECTIONS.values()]
    common = copy.deepcopy(c['text']); common.pop('historical_touch')
    assert common == b['text'] and c['images'][:len(b['images'])] == b['images']
    for condition, name in PROJECTIONS.items():
        projection = load(package/f'{name}.json')
        payload = {'ok':True, 'text':{'demonstrations':projection['text'], 'terminal':None,
            'image_labels':[im['label'] for im in projection['images']]}, 'images':projection['images']}
        server = build_live_backend_server(root=package, worker_url='http://unused', demonstrations=True, feedback_protocol=PROTOCOL)
        assert 'check_task' not in server._tool_manager._tools
        with patch('urllib.request.urlopen', side_effect=lambda *_a, data=payload, **_k: io.BytesIO(json.dumps(data).encode())):
            blocks = asyncio.run(server.call_tool('review_demonstrations', {}))
        images = [block for block in blocks if block.type=='image']
        assert len(images)==len(projection['images'])
        for block, descriptor in zip(images, projection['images']):
            assert np.array_equal(np.asarray(Image.open(io.BytesIO(base64.b64decode(block.data)))),
                                  np.asarray(Image.open(descriptor['path'])))
        results[condition] = {'expert_ids':[e['example_id'] for e in projection['text']['examples']],
                              'images':len(images), 'native_mcp_pixels_match':True}
    (package/'operator_context.jsonl').replace(package/'offline_mcp_context.jsonl')
    atomic_json(package/'offline_mcp_validation.json', results)
    return results


def prepare(args, settings):
    root = args.output_root
    if (root/'manifest.json').exists():
        raise FileExistsError('Frozen manifest exists; use --phase run to resume, never overwrite it')
    seeds = load(REPO/settings['seed_list'])
    assert len(seeds)==100 and len(set(seeds))==100
    selections = load(root/'expert_selection.json')
    source = REPO/settings['source_coverage']
    package_checks = {}
    for task in settings['tasks']:
        config = yaml.safe_load((source/'configs'/f'{task}_C.yaml').read_text())
        config['seeds'] = seeds
        config_file = root/'preparation_configs'/f'{task}.yaml'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        old = source/'demonstrations'/task
        sources = [(old, load(old/'tactile_action_icl.json'))]
        for episode in selections[task][2:]:
            out = root/'processed_experts'/task/str(episode)
            if not (out/'provenance.json').exists():
                if out.exists():
                    raise RuntimeError(f'Incomplete expert export preserved at {out}')
                subprocess.run([str(args.runtime_python), '-m', 'scripts.univtac.prepare_official_demonstrations',
                    '--raw', str(root/'data'/task), '--output', str(out), '--config', str(config_file),
                    '--episodes', str(episode)], cwd=REPO, check=True)
            sources.append((out, load(out/'tactile_action_icl.json')))
        for ids in ([0], [1], [0,1], selections[task]):
            name = 'official_'+'_'.join(map(str, ids))
            package = root/'demonstrations'/task/name
            matched_subset(sources, ids, package)
            package_checks[task, name] = check_mcp(package)
        print('prepared demonstrations', task, flush=True)
    cells = []
    for row in ordered_cells(settings['tasks'], seeds):
        task = row['task']
        # Slot indices map to frozen official IDs; source_seed is independent.
        row['expert_ids'] = [selections[task][i] for i in row['expert_ids']]
        row['demonstration_set_id'] = 'official_'+'_'.join(map(str, row['expert_ids']))
        metadata = load(root/'data'/task/'metadata.json')
        row['expert_source_seeds'] = [metadata[str(i)].get('source_seed', metadata[str(i)]['seed']) for i in row['expert_ids']]
        assert not set(seeds).intersection(row['expert_source_seeds'])
        cfg = yaml.safe_load((source/'configs'/f'{task}_C.yaml').read_text())
        key = [task,row['seed'],row['condition'],row['shot'],row['demonstration_set_id'],PROTOCOL]
        cfg.update({**row, 'round':settings['round'], 'seeds':[row['seed']], 'cell_key':key,
            'demonstration_package':str(root/'demonstrations'/task/row['demonstration_set_id']),
            'demonstration_condition':PROJECTIONS[row['comparison_condition']], 'shared_demonstration_media':True,
            'defer_review_video':True, 'scored':True, 'native_reset_time_limit_seconds':120})
        config_path = root/'configs'/task/str(row['seed'])/f'{row["condition"]}.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        row.update(cell_key=key, feedback_protocol=PROTOCOL, config=str(config_path), effective_config=cfg,
            mcp_expectation=package_checks[task,row['demonstration_set_id']][row['comparison_condition']], status='not_run')
        assert 'check_task' not in ' '.join(codex_command(root/'offline', 'http://unused', cfg))
        cells.append(row)
    assert len(cells)==2400 and len({tuple(c['cell_key']) for c in cells})==2400
    for task in settings['tasks']:
        task_cells = [c for c in cells if c['task']==task]
        assert len({operator_prompt(c['effective_config']) for c in task_cells})==1
        for shot in (1,2,4):
            for condition in ('B','C'):
                subset = [c for c in task_cells if c['shot']==shot and c['comparison_condition']==condition]
                assert sorted(c['seed'] for c in subset)==sorted(seeds)
                if shot==1:
                    assert sum(c['expert_ids']==[0] for c in subset)==50
                    assert sum(c['expert_ids']==[1] for c in subset)==50
    atomic_json(root/'manifest.json', {'feedback_protocol':PROTOCOL, 'settings':settings, 'seeds':seeds,
        'source_head':subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO,text=True).strip(),
        'cells':cells, 'status':'frozen_prepared'})
    print('frozen',len(cells),'cells',flush=True)


def validate_frozen(manifest):
    expected={'model':'gpt-6-astra','reasoning_effort':'low','feedback_protocol':PROTOCOL,
        'current_tactile':True,'max_move_requests':30,'max_tool_calls':100,
        'max_control_steps_per_move':80,'codex_timeout_seconds':3600,'terminal_grace_seconds':300,
        'native_reset_time_limit_seconds':120,'startup_timeout_seconds':900,'shutdown_timeout_seconds':300,
        'shared_demonstration_media':True,'defer_review_video':True}
    for cell in manifest['cells']:
        cfg=cell['effective_config']
        assert all(cfg.get(k)==v for k,v in expected.items()), (cell['cell_key'],'effective budget/model mismatch')
        assert cfg['native_control_step_limit']==(500 if cell['task']=='lift_bottle' else 300)
        assert not cfg.get('motion_pacing'), 'Original controller required'
        assert cfg['cell_key']==cell['cell_key'] and cfg['expert_ids']==cell['expert_ids']
    return {'cells_checked':len(manifest['cells']),'effective_settings_match':True}


def recover_cell(root, cell, attempt_limit=3):
    """Read accepted outcomes before dispatch, including interruption after cleanup."""
    base = root/'cells'/cell['task']/str(cell['seed'])/cell['condition']
    for number in range(1,4):
        folder = base/f'attempt_{number}'
        if not folder.exists():
            if number>attempt_limit:
                return {**cell,'status':'initialization_unavailable','attempts':number-1,
                        'episode_path':str(base/f'attempt_{number-1}')}
            return None
        reviewed=reviewed_initialization_issue(folder,cell)
        if reviewed:
            return reviewed
        episode = load(folder/'episode.json') if (folder/'episode.json').exists() else {}
        life = load(folder/'worker_lifecycle.json') if (folder/'worker_lifecycle.json').exists() else {}
        accepted = (folder/'ready.json').exists() or (folder/'codex_command.json').exists()
        marker = folder/'reviewed_operator_issue.json'
        if marker.exists():
            review = load(marker)
            assert accepted and life.get('cleanup_complete'), 'Reviewed operator must be cleaned'
            assert review['cell_key'] == cell['cell_key'] and review['no_retry'] is True
            assert review['status'] in ('infrastructure_issue', 'cancelled')
            return {**cell, 'status':review['status'], 'episode_path':str(folder),
                    'attempts':number, 'reviewed_skip':True, 'review_reason':review['reason']}
        if accepted:
            if episode.get('status')!='completed' and (folder/'final_result.json').exists() and (folder/'codex_lifecycle.json').exists():
                model = load(folder/'codex_lifecycle.json')
                final = load(folder/'final_result.json')
                if life.get('cleanup_complete') and life.get('returncode')==0 and model.get('exit_mode')=='natural_exit' and not episode.get('infrastructure_error') and not final.get('infrastructure_error'):
                    episode.update(final)
                    episode['evaluable'] = bool(final.get('reset_valid') and final.get('native_success_available'))
                    if episode['evaluable']:
                        episode.update(status='completed', recovered_after_cleanup=True)
                        atomic_json(folder/'episode.json', episode)
            if episode.get('status')=='completed' and life.get('cleanup_complete'):
                result={**cell, 'status':'completed', 'episode':episode, 'episode_path':str(folder), 'attempts':number}
                if (folder/'protocol_delivery_error.json').exists():
                    result['delivery_or_runner_error']=load(folder/'protocol_delivery_error.json')['error']
                return result
            return {**cell, 'status':'unresolved_previous_attempt', 'episode_path':str(folder)}
        if not life.get('cleanup_complete') or not retryable_initialization(folder):
            return {**cell, 'status':'unresolved_previous_attempt', 'episode_path':str(folder)}
    return {**cell, 'status':'initialization_unavailable', 'attempts':3, 'episode_path':str(folder)}


def costs(folder):
    attempts=[]
    for attempt in sorted(folder.parent.glob('attempt_*')):
        life=load(attempt/'worker_lifecycle.json') if (attempt/'worker_lifecycle.json').exists() else {}
        episode=load(attempt/'episode.json') if (attempt/'episode.json').exists() else {}
        attempts.append({'attempt':attempt.name,'ready':(attempt/'ready.json').exists(),
            'worker_seconds_including_cleanup':life.get('elapsed_seconds'),
            'initialization_wall_seconds':episode.get('initialization_wall_seconds'),
            'startup_failure_wall_seconds':episode.get('startup_failure_wall_seconds')})
    model=load(folder/'codex_lifecycle.json') if (folder/'codex_lifecycle.json').exists() else {}
    usage=load(folder/'codex_trace_summary.json').get('usage') if (folder/'codex_trace_summary.json').exists() else None
    return {'initialization_attempts':attempts,'codex_seconds':model.get('elapsed_seconds'),
            'model_exit_mode':model.get('exit_mode'),'usage':usage}


def verify_delivery(folder, cell):
    contexts = jsonl(folder/'operator_context.jsonl')
    hosts = jsonl(folder/'host_tool_trace.jsonl')
    assert len(contexts)==len(hosts), 'MCP context/host count mismatch'
    demos = []
    for actual, host in zip(contexts, hosts):
        assert actual['tool']==host['tool']
        payload = host['result']
        expected = project_query(payload, tool=host['tool'], ended=bool(payload['text'].get('terminal') or payload['text'].get('finished')))
        if 'observation' in expected['text']:
            expected['text']['observation']['images'] = [{'label':im['label']} for im in expected['text']['observation']['images']]
        assert json.loads(actual['response_text_blocks'][0])==expected['text'], 'Actual query projection mismatch'
        assert actual['response_image_paths']==[im['path'] for im in expected['images']]
        if host['tool']=='review_demonstrations':
            demos.append(actual)
    assert len(demos)==1, 'Historical demonstration not delivered exactly once'
    content = json.loads(demos[0]['response_text_blocks'][0])['demonstrations']
    assert [e['example_id'] for e in content['examples']]==cell['mcp_expectation']['expert_ids'], 'Expert count/identity mismatch'
    assert len(demos[0]['response_image_paths'])==cell['mcp_expectation']['images'], 'Historical image count mismatch'
    assert 'check_task' not in load(folder/'mcp_tools.json')['tools']
    trace = jsonl(folder/'tool_trace.jsonl')
    stops = [i for i,x in enumerate(trace) if x['result']['text'].get('terminal') or x['result']['text'].get('finished')]
    if stops:
        assert len({x['counts']['physics_steps'] for x in trace[stops[0]:]})==1
    atomic_json(folder/'protocol_delivery_check.json', {'passed':True, 'responses':len(contexts),
        'expert_count':len(content['examples']), 'historical_images':len(demos[0]['response_image_paths'])})


def render_media(folder, cell, rate):
    import imageio_ffmpeg
    if not (folder/'samples.jsonl').exists():
        return
    if (folder/'media_check.json').exists() and load(folder/'media_check.json').get('passed'):
        return
    label = f'{cell["task"]} · {cell["comparison_condition"]} · {cell["shot"]}-shot · seed {cell["seed"]} · GPT-6 low自主操作'
    video = export_review_video(folder, playback_rate=rate, review_label=label)
    proc = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v','error','-threads','2','-i',str(video),'-map','0:v:0',
        '-fps_mode','passthrough','-f','null','-','-progress','pipe:1'],capture_output=True,text=True,check=True)
    counts = [int(line.split('=')[1]) for line in proc.stdout.splitlines() if line.startswith('frame=')]
    expected = len(jsonl(folder/'samples.jsonl'))
    assert counts and counts[-1]==expected, f'Video frame count {counts[-1:]}, expected {expected}'
    atomic_json(folder/'media_check.json', {'passed':True,'decoded_frame_count':counts[-1], 'recorded_sample_count':expected,
        'playback_rate':rate,'browser_watched':False})
    from scripts.univtac.summarize_shot_scaling import episode_page
    episode_page(folder,cell)


def scoped_cells(cells, conditions):
    """Keep the original order and records; annotate excluded pending cells."""
    result = []
    for cell in cells:
        row = dict(cell)
        row['in_current_scope'] = row['comparison_condition'] in conditions
        if not row['in_current_scope']:
            row['scope_disposition'] = (
                'deferred_to_main_table' if row['status'] == 'not_run'
                else 'executed_before_scope_revision'
            )
        result.append(row)
    return result


def revise_scope(args, settings):
    """Offline scope overlay; never prepare again or release a paused queue."""
    root = args.output_root
    manifest = load(root/'manifest.json')
    frozen_settings = {k:v for k,v in settings.items() if k not in ('dispatch_conditions','dispatch_max_initialization_attempts')}
    assert frozen_settings == manifest['settings']
    validate_frozen(manifest)
    conditions = settings['dispatch_conditions']
    assert conditions == ['C']
    data = load(root/'results.json')
    # Preserve the original pause views once; raw episodes and manifest are untouched.
    for name in ('results.json', 'run_manifest.json', 'statistics.json', 'report.html'):
        source = root/name
        target = root/'pre_c_only_scope'/name
        if source.exists() and not target.exists():
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
    for cell in manifest['cells']:
        assert yaml.safe_load(Path(cell['config']).read_text()) == cell['effective_config']
        old = recover_cell(root, cell)
        saved = next(c for c in data['cells'] if c['cell_key'] == cell['cell_key'])
        assert (old['status'] if old else 'not_run') == saved['status']
    data['cells'] = scoped_cells(data['cells'], conditions)
    active = [c for c in data['cells'] if c['in_current_scope']]
    assert len(active) == 1200
    revision = {'dispatch_conditions':conditions, 'planned':len(active),
                'original_planned':len(manifest['cells']), 'pause_released':False,
                'ordered_cell_keys':[c['cell_key'] for c in active]}
    atomic_json(root/'scope_revision.json', revision)
    data.update(planned=len(active), original_planned=len(manifest['cells']),
                dispatch_conditions=conditions)
    atomic_json(root/'results.json', data)
    from scripts.univtac.summarize_shot_scaling import report
    report(root)


def run(args, settings):
    root = args.output_root
    manifest = load(root/'manifest.json')
    assert manifest['settings']=={k:v for k,v in settings.items() if k not in ('dispatch_conditions','dispatch_max_initialization_attempts')}, 'Use frozen settings to resume'
    conditions=settings.get('dispatch_conditions',['B','C'])
    args.max_initialization_attempts=settings.get('dispatch_max_initialization_attempts',3)
    if 'seed_list' in settings:
        validate_frozen(manifest)
        assert load(REPO/settings['seed_list'])==manifest['seeds'], 'Shared seed list differs from frozen manifest'
    cells = [{k:v for k,v in c.items() if k!='effective_config'} for c in manifest['cells']]
    coordinator = Coordinator(root,[tuple(c['cell_key']) for c in cells],protocol_smoke=True)
    state = {tuple(c['cell_key']):c for c in cells}
    assignment = getattr(args, 'assignment', None)
    host = getattr(args, 'host', None)
    assigned = None
    if assignment:
        groups = load(assignment)['assignments']
        assert host in groups, 'Unknown assignment host'
        keys = [tuple(k) for values in groups.values() for k in values]
        assert len(keys) == len(set(keys)), 'Overlapping host assignments'
        assert set(keys) <= {tuple(c['cell_key']) for c in cells}, 'Unknown assigned cells'
        assigned = {tuple(k) for k in groups[host]}
    pending = []
    if startup_crashes_since_ready(root)>=3:
        coordinator.abort('Three preserved omniClient startup crashes without an intervening ready')
    for cell in cells:
        old = recover_cell(root,cell,args.max_initialization_attempts)
        if old:
            if old['status']=='completed' and not old.get('delivery_or_runner_error'):
                folder=Path(old['episode_path'])
                if not (folder/'protocol_delivery_check.json').exists():
                    try:
                        verify_delivery(folder,cell)
                    except Exception as exc:  # noqa: BLE001 -- preserve interrupted/invalid delivery
                        old['delivery_or_runner_error']=str(exc)
                        atomic_json(folder/'protocol_delivery_error.json',{'error':str(exc)})
            if old.get('delivery_or_runner_error'):
                coordinator.abort('Preserved delivery failure: '+old['episode_path'])
            if old.get('episode_path'):
                old['costs']=costs(Path(old['episode_path']))
            state[tuple(cell['cell_key'])] = old
            if old['status']=='unresolved_previous_attempt':
                coordinator.abort('Unresolved prior accepted/in-flight attempt: '+old['episode_path'])
        elif cell['comparison_condition'] in conditions and (assigned is None or tuple(cell['cell_key']) in assigned):
            pending.append(cell)
    def save():
        atomic_json(root/'results.json',{'feedback_protocol':PROTOCOL,'planned':sum(c['comparison_condition'] in conditions for c in cells),
            'original_planned':len(cells),'dispatch_conditions':conditions,
            'dispatch_max_initialization_attempts':args.max_initialization_attempts,
            'cells':scoped_cells(list(state.values()),conditions),'abort_reason':coordinator.abort_reason,'updated_s':time.time()})
    save()
    from scripts.univtac.summarize_shot_scaling import report
    report(root)
    previous = {}; done = threading.Event()
    if not (root/'resources_before.json').exists():
        atomic_json(root/'resources_before.json',resources(coordinator,previous))
    def monitor():
        while not done.wait(1):
            try:
                row=resources(coordinator,previous)
                with coordinator.lock:
                    row['phases']={str(k):coordinator.phases[k] for k in coordinator.active_roots}
                row['disk_free_bytes']=shutil.disk_usage(root).free
                with (root/'resources.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                if row['disk_free_bytes']<settings['minimum_disk_free_bytes']:
                    coordinator.abort('Insufficient disk space for preserving further records')
                with coordinator.lock:
                    for key,folder in coordinator.active_roots.items():
                        error=folder/'worker_error.json'
                        if error.exists():
                            coordinator.abort(error.read_text(),key)
                        log=folder/'launcher/stdout_stderr.log'
                        if log.exists():
                            with log.open('rb') as f:
                                f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace').lower()
                            if any(w in tail for w in ('out of gpu memory','out of memory','failed to allocate memory')):
                                coordinator.abort('Native OOM: '+str(folder))
            except Exception as exc:  # noqa: BLE001 -- record read-only monitoring failures
                coordinator.event(None,'resource_sampling_error',error=str(exc))
    thread=threading.Thread(target=monitor);thread.start()
    if shutil.disk_usage(root).free < settings['minimum_disk_free_bytes']:
        coordinator.abort('Insufficient disk space before dispatch')
    session={'started_s':time.time(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    args.execution_head=session['head']
    runtime=load(root/'run_manifest.json') if (root/'run_manifest.json').exists() else {'feedback_protocol':PROTOCOL,'started_s':session['started_s'],'planned':2400}
    runtime['dispatch_max_initialization_attempts']=args.max_initialization_attempts
    runtime.update(planned=sum(c['comparison_condition'] in conditions for c in cells), original_planned=len(cells), dispatch_conditions=conditions)
    runtime.setdefault('sessions',[]).append(session);runtime['status']='running';atomic_json(root/'run_manifest.json',runtime)
    media=[]
    def media_job(folder,cell):
        if shutil.disk_usage(root).free < settings['minimum_disk_free_bytes']:
            atomic_json(folder/'media_pending.json', {'reason':'insufficient_disk_space; raw frames preserved'})
            return
        try:
            render_media(folder,cell,settings['video_playback_rate'])
        except Exception as exc:  # noqa: BLE001 -- preserve raw evidence, rebuild offline
            atomic_json(folder/'media_error.json',{'error':str(exc)})
    try:
        with ThreadPoolExecutor(max_workers=settings['media_workers']) as media_pool, ThreadPoolExecutor(max_workers=2) as operators:
            for value in state.values():
                if value['status']=='completed':
                    media.append(media_pool.submit(media_job,Path(value['episode_path']),value))
            queue=iter(pending);active={}
            while True:
                while len(active)<2 and not coordinator.cancel.is_set() and not coordinator.dispatch_paused.is_set():
                    cell=next(queue,None)
                    if cell is None:
                        break
                    assert yaml.safe_load(Path(cell['config']).read_text()) == next(c['effective_config'] for c in manifest['cells'] if c['cell_key']==cell['cell_key']), 'Effective config differs from frozen manifest'
                    state[tuple(cell['cell_key'])]={**cell,'status':'in_progress'}
                    active[operators.submit(run_cell,args,cell,coordinator)]=cell
                save()
                if not active:
                    break
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    cell=active.pop(future);key=tuple(cell['cell_key'])
                    try:
                        result=future.result()
                        if result.get('episode_path'):
                            result['costs']=costs(Path(result['episode_path']))
                        state[key]=result
                        if result['status']=='completed':
                            verify_delivery(Path(result['episode_path']),cell)
                            media.append(media_pool.submit(media_job,Path(result['episode_path']),cell))
                    except Exception as exc:  # noqa: BLE001 -- never replace an accepted episode
                        coordinator.abort(str(exc));state[key]={**state[key],'delivery_or_runner_error':str(exc)}
                        if state[key].get('episode_path'):
                            atomic_json(Path(state[key]['episode_path'])/'protocol_delivery_error.json',{'error':str(exc)})
                    with coordinator.lock:
                        coordinator.active_roots.pop(key,None)
                        coordinator.ready.pop(key,None)
                        for kind in ('worker','codex'):
                            coordinator.processes.pop((key,kind),None)
                    save()
                    report(root)
    except BaseException as exc:
        coordinator.abort('Queue interrupted: '+str(exc))
        raise
    finally:
        done.set();thread.join()
        atomic_json(root/'resources_after.json',resources(coordinator,previous))
        runtime.update(status='paused' if coordinator.cancel.is_set() or coordinator.dispatch_paused.is_set() else 'completed',ended_s=time.time())
        atomic_json(root/'run_manifest.json',runtime);save();report(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', type=Path)
    parser.add_argument('--host', choices=['local', 'hzz-server'])
    parser.add_argument('--phase',choices=('download','prepare','revise-scope','run'),required=True)
    parser.add_argument('--config',type=Path,default=REPO/'configs/univtac/shot_scaling.yaml')
    parser.add_argument('--output-root',type=Path,default=REPO/'outputs/univtac-shot-scaling')
    parser.add_argument('--runtime-python',type=Path,default=Path('/home/ubuntu/anaconda3/envs/UniVTAC-isaac51-sm120-r09/bin/python3.11'))
    parser.add_argument('--source-root',type=Path,default=Path('/home/ubuntu/wybcode/.worktrees/univtac-isaac51-r081'))
    args=parser.parse_args();args.output_root=args.output_root.resolve();args.output_root.mkdir(parents=True,exist_ok=True)
    settings=yaml.safe_load(args.config.read_text())
    if args.phase=='download':
        fetch_additional(args,settings)
    elif args.phase=='prepare':
        prepare(args,settings)
    elif args.phase=='revise-scope':
        revise_scope(args,settings)
    else:
        run(args,settings)


if __name__=='__main__':
    main()
