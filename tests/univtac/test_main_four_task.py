import json
from collections import Counter
from scripts.univtac.main_four_task import fixed_cells, exact_pair, TASKS
from scripts.univtac.run_fourway_capacity import Coordinator


def test_fixed_pairs_are_disjoint_and_counterbalanced():
    cells=fixed_cells(list(range(1000000,1000100)))
    assert len(cells)==len({tuple(c['cell_key']) for c in cells})==800
    assert Counter(c['host'] for c in cells)=={'local':400,'hzz-server':400}
    assert Counter((c['task'],c['condition']) for c in cells)=={(t,c):100 for t in TASKS for c in ['A','B_2shot']}
    for first,second in zip(cells[::2],cells[1::2]):
        assert (first['task'],first['seed'],first['host'])==(second['task'],second['seed'],second['host'])
        t=TASKS.index(first['task']);i=first['seed']-1000000
        assert first['condition']==('A' if (i//2+t)%2==0 else 'B_2shot')
        assert first['host']==('local' if (i+t)%2==0 else 'hzz-server')
    assert all(c['expert_ids']==([] if c['condition']=='A' else [0,1]) for c in cells)
    assert fixed_cells(list(range(1000000,1000100)))==cells


def test_capacity_pause_does_not_cancel_sibling(tmp_path):
    co=Coordinator(tmp_path,[('task',0,'A'),('task',1,'A')],protocol_smoke=True)
    co.pause_dispatch('provider_model_capacity',('task',0,'A'))
    assert co.dispatch_paused.is_set()
    assert not co.cancel.is_set()
    assert all(not event.is_set() for event in co.lane_cancel.values())


def test_exact_pairs_keep_failure_and_unpaired():
    result=exact_pair({1:True,2:False,3:False},{1:False,2:True,3:False,4:True})
    assert result['left_only_success']==result['right_only_success']==1
    assert result['both_failure']==1 and result['valid_pairs']==3
    assert result['unpaired']==97 and result['p_exact']==1
