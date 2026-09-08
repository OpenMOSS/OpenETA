"""Goal-pad interventions stay local and geometry distances are not origin gaps."""
from types import SimpleNamespace

import numpy as np
import pytest

from sim.envs.univtac.pad_scene_diagnostic import pad_arguments, transformed_surface


@pytest.mark.parametrize('name',['green_pad','orange_pad'])
def test_one_variable_only(name):
    original=SimpleNamespace(p=np.array([.4,.08,.002]),q=np.array([1.,0,0,0]))
    o,m=pad_arguments('O',name,original,'dynamic');assert o is original and m=='dynamic'
    k,m=pad_arguments('K',name,original,'dynamic');assert k is original and m=='kinematic'
    z,m=pad_arguments('Z',name,original,'dynamic');assert z is not original and m=='dynamic'
    np.testing.assert_array_equal(original.p,[.4,.08,.002]);np.testing.assert_array_equal(z.p,[.4,.08,.010])
    np.testing.assert_array_equal(z.q,original.q)
    actor,m=pad_arguments('K','rough_prism',original,'dynamic');assert actor is original and m=='dynamic'
    with pytest.raises(ValueError):pad_arguments('O',name,original,'kinematic')


def test_surface_signed_gap_uses_vertices_and_complete_transform():
    points=np.array([[-.02,0,-.003],[.02,0,.007],[0,.02,0]])
    matrix=np.eye(4);matrix[:3,3]=[.4,.08,.002]
    row=transformed_surface(points,matrix,.001,[0,0,1])
    assert row['min_signed_ground_gap_m']==pytest.approx(-.002)
    # Origin gap is positive but this collision surface extends below the plane.
    assert matrix[2,3]-.001>0
    matrix[2,3]=.010
    assert transformed_surface(points,matrix,.001,[0,0,1])['min_signed_ground_gap_m']==pytest.approx(.006)
