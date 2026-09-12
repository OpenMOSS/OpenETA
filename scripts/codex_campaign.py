"""Sequential, resumable LIBERO Object/Goal campaign with isolated owned services."""
import ast, collections, datetime, hashlib, json, os, pathlib, signal, socket, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/codex-libero20-20260910'
PORT=18786
STOP=False
STATE={}
def now():return datetime.datetime.now().astimezone().isoformat(timespec='milliseconds')
def save(path,d):
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');t.replace(path)
def state():
 STATE['updated']=now();save(BASE/'batch-status.json',STATE)
def stop(*args):
 global STOP
 STOP=True
def procs():
 result={}
 for p in pathlib.Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    s=(p/'stat').read_text().rsplit(')',1)[1].split()
    result[int(p.name)]={'ppid':int(s[1]),'start':s[19],'state':s[0]}
   except (OSError,ValueError,IndexError):pass
 return result
def track(roots,owned):
 ps=procs(); selected=set(roots)
 for pid,birth in list(owned.items()):
  if pid in ps and ps[pid]['start']==birth:selected.add(pid)
 changed=True
 while changed:
  changed=False
  for pid,d in ps.items():
   if d['ppid'] in selected and pid not in selected:selected.add(pid);changed=True
 for pid in selected:
  if pid in ps:owned[pid]=ps[pid]['start']
def living(owned):
 ps=procs();return [pid for pid,birth in owned.items() if pid in ps and ps[pid]['start']==birth and ps[pid]['state']!='Z']
def cleanup(processes,owned):
 track([p.pid for p in processes],owned)
 for p in processes:
  if p.poll() is None:
   try:os.killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 for pid in living(owned):
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 end=time.monotonic()+12
 while living(owned) and time.monotonic()<end:time.sleep(.2)
 for pid in living(owned):
  try:os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass
 for p in processes:
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:pass
 return living(owned)
def port_free():
 with socket.socket() as s:return s.connect_ex(('127.0.0.1',PORT))!=0
class Timeline:
 def __init__(self,out):self.out=out;self.pos=0;self.buffer=b'';self.errors=[];self.calls=0;self.latest=None
 def poll(self):
  p=self.out/'codex-events.jsonl'
  if not p.exists():return
  with p.open('rb') as f:f.seek(self.pos);b=f.read();self.pos=f.tell()
  lines=(self.buffer+b).split(b'\n');self.buffer=lines.pop()
  with (self.out/'event-timeline.jsonl').open('a') as log:
   for line in lines:
    try:d=json.loads(line)
    except ValueError:continue
    item=d.get('item') or {};r={'observed_at':now(),'event':d.get('type')}
    if item.get('type')=='mcp_tool_call':
     r.update(tool=item.get('tool'),id=item.get('id'),status=item.get('status'),arguments=item.get('arguments'),is_error=(item.get('result') or {}).get('isError'))
     self.latest={k:v for k,v in r.items() if k!='arguments'}
     if d.get('type')=='item.completed':self.calls+=1
    elif d.get('type') in ('error','turn.failed'):
     r['message']=d.get('message') or d.get('error');self.errors.append(r)
    elif d.get('type') in ('thread.started','turn.started','turn.completed'):
     r['usage']=d.get('usage')
    else:continue
    log.write(json.dumps(r,ensure_ascii=False)+'\n')
def readjson(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return {}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def analyze(out,timeline):
 a=[];p=out/'host/atomic-commands.jsonl'
 if p.exists():
  for l in p.read_text().splitlines():
   try:a.append(json.loads(l))
   except ValueError:pass
 gaps=[{'previous':x['tool'],'next':y['tool'],'gap_s':round(y['started_episode_s']-x['ended_episode_s'],3)} for x,y in zip(a,a[1:])]
 motions=[]
 for x in a:
  m=(x.get('feedback') or {}).get('motion')
  if isinstance(m,dict):motions.append({'reason':m.get('reason_code'),'dispatched':m.get('motion_dispatched'),'summary':m.get('motion_summary')})
 s=readjson(out/'summary.json');err=out/'codex-stderr.log'
 result={'summary':s,'stream_errors':timeline.errors,'native_completed_calls':timeline.calls,'atomic_calls':dict(collections.Counter(x['tool'] for x in a)),
 'atomic_errors':[{'tool':x['tool'],'error':x.get('error'),'is_error':x.get('is_error')} for x in a if x.get('error') or x.get('is_error')],
 'atomic_tool_duration_s':round(sum(x['ended_episode_s']-x['started_episode_s'] for x in a),3),'longest_inter_atomic_gap':max(gaps,key=lambda x:x['gap_s']) if gaps else None,'motions':motions,
 'catalog_refresh_errors':err.read_text(errors='replace').count('failed to refresh available models') if err.exists() else 0}
 save(out/'analysis.json',result);return result


def successful_attempt(entry):
 return (entry.get('task_success') is True and entry.get('integration_passed') is True
         and entry.get('port_released') is True and entry.get('private_auth_removed') is True
         and entry.get('remaining_owned_pids') == [] and entry.get('source_changed') == [])


def attempt_boundary_error(entry):
 if (entry.get('remaining_owned_pids') != [] or entry.get('port_released') is not True
     or entry.get('private_auth_removed') is not True or entry.get('source_changed') != []):
  return 'Cleanup incomplete or source changed; paused before next task'
 if entry.get('status') != 'interrupted' and entry.get('integration_passed') is not True:
  return 'Codex/Host integration failed; verify tool access before starting another task'
 return None


def final_observation(summary, timeline):
 """Use final Host and drained Codex evidence, not the last live snapshot."""
 host=summary.get('host') or {}
 return {'native_completed_calls':timeline.calls,'stream_errors':len(timeline.errors),
  'latest_native':timeline.latest,
  'host':{k:host.get(k) for k in ('requests','tool_calls','elapsed_s','closed','failure_reason','official_task_success')}}


def source_snapshot():
 paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z',
  '--','agent','adapter','sim','tools','scripts','plugins'],cwd=ROOT).decode().split('\0')
 snapshot={p:digest(ROOT/p) for p in sorted(set(paths)) if p and (ROOT/p).is_file()}
 if STATE.get('model_catalog_json'):
  snapshot['@model_catalog_json']=digest(pathlib.Path(STATE['model_catalog_json']))
 return snapshot


def main():
 import argparse, fcntl
 global BASE, PORT, STATE
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=pathlib.Path,default=BASE)
 parser.add_argument('--tasks',nargs='+',default=['goal:3']+
  [f'object:{i}' for i in range(10)]+[f'goal:{i}' for i in range(10) if i!=3])
 parser.add_argument('--port',type=int,default=PORT)
 parser.add_argument('--model-catalog-json',type=pathlib.Path,default=None)
 parser.add_argument('--effort',choices=('low','medium','high','xhigh','max'),default=None)
 args=parser.parse_args();BASE=args.output.resolve();PORT=args.port;BASE.mkdir(parents=True,exist_ok=True)
 lock=(BASE/'campaign.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 mapping=ast.literal_eval(ast.parse(pathlib.Path('/tmp/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text()).body[0].value)
 manifest={f'{suite}:{i}':{'suite':suite,'task_id':i,'task':mapping[f'libero_{suite}'][i].replace('_',' '),
  'env_id':f'openeta/libero_libero_{suite}_task{i}-v0','attempts':[]}
  for suite in ('object','goal') for i in range(10)}
 for key in args.tasks:
  if key not in manifest:parser.error(f'Unknown task {key}')
 STATE=readjson(BASE/'batch-status.json') or {'created':now(),'model':'gpt-6-astra','effort':'medium',
  'controller':'mink_joint_velocity','tool_profile':'atomic','seed':0,'seed_policy':'procedural_reset',
  'episode_timeout_s':2400,'max_requests':160,'tasks':manifest}
 if set(STATE['tasks'])!=set(manifest):raise RuntimeError('Existing campaign task manifest differs')
 if args.model_catalog_json:
  STATE['model_catalog_json']=str(args.model_catalog_json.resolve())
 if args.effort is not None:STATE['effort']=args.effort
 STATE.update(status='running',pid=os.getpid(),started=now(),requested_tasks=args.tasks)
 STATE.pop('ended',None)
 STATE.pop('error',None)
 (BASE/'batch.pid').write_text(str(os.getpid())+'\n');state()
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),LIBERO_DIR='/tmp/LIBERO',
  OPENETA_WORKER_GPUS='0',OPENETA_WORKER_POOL_MAX='1')
 try:
  for key in args.tasks:
   if STOP or (BASE/'pause-after-current').exists():break
   task=STATE['tasks'][key]
   if any(successful_attempt(a) for a in task['attempts']):continue
   assert port_free(),'Dedicated simulator port occupied; not touching its owner'
   snapshot=source_snapshot()
   attempt_no=len(task['attempts'])+1
   run=BASE/f"{task['suite']}-{task['task_id']}"/f'attempt-{attempt_no:03d}'
   run.mkdir(parents=True)
   setup=run/'setup';setup.mkdir();out=run/'run'
   save(run/'source-snapshot.json',snapshot)
   # Each attempt has its own monitor, authentication copy, service, and logs.
   net=run/'network';net.mkdir()
   import shutil
   shutil.copy2(ROOT/'tmp/codex-goal3-symmetry-20260910/network/monitor.py',net/'monitor.py')
   env['OPENETA_WORKER_LOG_DIR']=str(setup/'worker-logs')
   entry={'attempt':attempt_no,'started':now(),'status':'starting','output':str(run),
    'model':STATE['model'],'reasoning_effort':STATE['effort']}
   task['attempts'].append(entry);STATE['current_task']=key;state()
   processes=[];owned={};timeline=Timeline(out)
   with (setup/'sim.log').open('w') as sl,(setup/'launcher.log').open('w') as ll,(net/'monitor.log').open('w') as nl:
    try:
     monitor=subprocess.Popen(['/usr/bin/python3',str(net/'monitor.py')],env=env,stdout=nl,stderr=subprocess.STDOUT,start_new_session=True)
     processes.append(monitor)
     server=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.codex_sim_server',
      '--host','127.0.0.1','--port',str(PORT),'--private-state-dir',str(setup/'private-state')],
      cwd=ROOT,env=env,stdout=sl,stderr=subprocess.STDOUT,start_new_session=True)
     processes.append(server);entry['server_pid']=server.pid
     ready=False
     for _ in range(150):
      if STOP:raise RuntimeError('Campaign interrupted')
      if server.poll() is not None:raise RuntimeError('Simulator exited during startup')
      if not port_free():ready=True;break
      time.sleep(.2)
     if not ready:raise RuntimeError('Simulator startup exceeded 30 seconds')
     cmd=[str(ROOT/'.venv/bin/python'),'-u','-m','scripts.codex_plugin_smoke',
      '--sim-url',f'http://127.0.0.1:{PORT}/sse','--env-id',task['env_id'],'--task',task['task']+'.',
      '--seed',str(STATE['seed']),'--output',str(out),'--model',STATE['model'],'--effort',STATE['effort'],
      '--timeout',str(STATE['episode_timeout_s']),'--max-requests',str(STATE['max_requests']),'--tool-profile','atomic']
     if STATE.get('model_catalog_json'):cmd.extend(['--model-catalog-json',STATE['model_catalog_json']])
     save(setup/'command.json',cmd)
     launcher=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=ll,stderr=subprocess.STDOUT,start_new_session=True)
     processes.append(launcher);entry.update(launcher_pid=launcher.pid,status='running');state()
     end=time.monotonic()+STATE['episode_timeout_s']+450;last=0
     while launcher.poll() is None and not STOP and time.monotonic()<end:
      track([p.pid for p in processes],owned);timeline.poll()
      if time.monotonic()-last>5:
       h=readjson(out/'host/host-status.json');entry.update(native_completed_calls=timeline.calls,
        stream_errors=len(timeline.errors),latest_native=timeline.latest,
        host={k:h.get(k) for k in ('requests','tool_calls','elapsed_s','closed','failure_reason','official_task_success')})
       state();last=time.monotonic()
      time.sleep(1)
     entry['launcher_returncode']=launcher.poll()
     entry['status']='interrupted' if STOP else 'watchdog_timeout' if launcher.poll() is None else 'finished'
    except Exception as exc:entry.update(status='setup_or_supervisor_error',error=str(exc))
    finally:
     entry['remaining_owned_pids']=cleanup(processes,owned)
     (out/'codex-home/auth.json').unlink(missing_ok=True)
     timeline.poll();entry.update(ended=now(),port_released=port_free(),
      private_auth_removed=not (out/'codex-home/auth.json').exists())
     current_snapshot=source_snapshot()
     entry['source_changed']=[p for p in sorted(set(snapshot)|set(current_snapshot)) if snapshot.get(p)!=current_snapshot.get(p)]
     if out.exists():
      a=analyze(out,timeline);s=a['summary'];entry.update(result_status=s.get('status'),
       task_success=s.get('task_success') is True,integration_passed=s.get('integration_passed') is True,
       elapsed_s=s.get('elapsed_s'),stream_errors=len(timeline.errors))
      entry['last_monitor_host']=entry.get('host')
      entry.update(final_observation(s,timeline))
     save(run/'result.json',entry)
     task['completed']=any(successful_attempt(a) for a in task['attempts'])
     STATE['completed_count']=sum(any(successful_attempt(a) for a in t['attempts']) for t in STATE['tasks'].values())
     state();print(json.dumps({'task':key,**entry},ensure_ascii=False),flush=True)
   issue=attempt_boundary_error(entry)
   if issue:raise RuntimeError(issue)
  STATE['status']='interrupted' if STOP else 'paused_for_maintenance' if (BASE/'pause-after-current').exists() else 'pass_finished'
 except Exception as exc:STATE.update(status='error',error=str(exc));raise
 finally:
  STATE['ended']=now();state();lock.close()


if __name__=='__main__':main()
