import random
from pathlib import Path
import sys
import pytest
from agent.runtime.observation_packets import find_packet_references_for_paths as references_for_paths


def reference_lookup(entries, path):
    # Frozen semantic oracle: canonical aliases must all agree on the frame;
    # only a unique owner at the newest observation index is accepted.
    if not path:return {}
    canonical=Path(path).resolve()
    owners={}
    for entry in entries:
        for a in entry['artifacts']:
            if Path(a['path']).resolve()==canonical:
                owners[(entry['packet_id'],a['frame_id'])]=entry['observation_index']
    if not owners or len({frame for _,frame in owners})!=1:return {}
    indices={owner:(index if type(index) is int else -1) for owner,index in owners.items()}
    newest=[owner for owner,index in indices.items() if index==max(indices.values())]
    return dict(zip(('source_packet_id','camera_frame_id'),newest[0])) if len(newest)==1 else {}


def test_bulk_keeps_alias_collision_newest_and_invalid_path_semantics(tmp_path):
 files=[tmp_path/f'image{i}.png' for i in range(4)]
 for p in files:p.write_text('fixture')
 alias=tmp_path/'alias.png';alias.symlink_to(files[0])
 paths=[str(p) for p in files]+[str(alias),str(tmp_path/'missing.png'),'']
 rng=random.Random(2718)
 for _ in range(120):
  entries=[{'packet_id':f'obs-{i}','observation_index':rng.choice([None,False,1,2,3]),'artifacts':[
    {'path':rng.choice(paths),'frame_id':rng.choice(['wrist','agentview'])} for _ in range(rng.randrange(1,5))]}
    for i in range(rng.randrange(1,12))]
  expected={p:reference_lookup(entries,p) for p in paths if p}
  assert references_for_paths(iter(entries),paths)==expected
  assert references_for_paths(entries,[None,0,False,''])=={}


def test_symlink_retarget_between_calls_is_not_cached(tmp_path):
 a,b=tmp_path/'a',tmp_path/'b';a.write_text('a');b.write_text('b')
 alias=tmp_path/'current';alias.symlink_to(a)
 entries=[{'packet_id':'a','observation_index':1,'artifacts':[{'path':str(a),'frame_id':'wrist'}]},
          {'packet_id':'b','observation_index':2,'artifacts':[{'path':str(b),'frame_id':'agentview'}]}]
 assert references_for_paths(entries,[str(alias)])[str(alias)]['source_packet_id']=='a'
 alias.unlink();alias.symlink_to(b)
 assert references_for_paths(entries,[str(alias)])[str(alias)]['source_packet_id']=='b'


def test_bulk_resolves_duplicate_history_paths_once_per_call(tmp_path,monkeypatch):
 p=str(tmp_path/'image');paths=[p,p];entries=[{'packet_id':str(i),'observation_index':i,
   'artifacts':[{'path':p,'frame_id':'wrist'}]} for i in range(100)]
 original=Path.resolve;calls=[]
 def resolve(self,*args,**kwargs):calls.append(str(self));return original(self,*args,**kwargs)
 monkeypatch.setattr(Path,'resolve',resolve)
 assert references_for_paths(entries,paths)[p]['source_packet_id']=='99'
 assert calls==[p]
