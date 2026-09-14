from types import SimpleNamespace as NS
import numpy as np
from sim.controllers.stall_feedback import stalled_motion_context, public_stall_context


def test_binding_clearance_and_actual_palm_contact_are_distinct_evidence():
    # First row binds; second has slack; third involves only fixed non-robot DoFs.
    limit=NS(geom_id_pairs=[(2,11),(3,10),(3,11)],compute_qp_inequalities=lambda c,dt:NS(
        G=np.array([[1.,0.],[1.,0.],[0.,1.]]),h=np.array([.001,.02,0.])))
    policy={'limit':limit,'robot_geom_ids':{1,2,3},'gripper_geom_ids':{2,3},'finger_geom_ids':{3},
            'authorized_target_geom_ids':{10},'authorized_target_geom_count':1}
    v=np.array([.02,0.]);original=v.copy()
    result=stalled_motion_context(None,policy,v,v/2,np.array([0]),.05,
        NS(contact=[NS(geom1=2,geom2=11,dist=-.00001),NS(geom1=2,geom2=3,dist=0),
                    NS(geom1=3,geom2=10,dist=.001),NS(geom1=3,geom2=10,dist=float('nan'))]),collision_qp_used=True)
    assert np.array_equal(v,original)
    assert result['qp_evidence']=='available' and result['contact_evidence']=='available'
    assert result['active_clearance_constraints']==result['measured_robot_contacts']==[
        {'robot_part':'gripper','gripper_part':'base_or_palm','obstacle_relation':'outside_contact_target'}]
    assert result['causal_attribution']=='not_established'


def test_diagnostic_failure_preserves_unknown_without_fabricating_clearance():
    result=stalled_motion_context(None,{},np.array([0.]),np.array([0.]),np.array([0]),.05,
                                  None,collision_qp_used=True)
    assert result['qp_evidence']==result['contact_evidence']=='unavailable'
    assert 'active_clearance_constraints' not in result
    assert 'measured_robot_contacts' not in result


def test_stall_projection_rejects_invalid_values_and_caps_rows():
    result=public_stall_context({'qp_evidence':{},'requested_joint_speed_max_rad_s':float('nan'),
        'commanded_joint_speed_max_rad_s':True,'active_clearance_constraints':[{'robot_part':'arm','geom_id':20}]*100})
    assert result['active_clearance_constraints']==[{'robot_part':'arm'}]
    assert 'qp_evidence' not in result and 'requested_joint_speed_max_rad_s' not in result
    assert 'commanded_joint_speed_max_rad_s' not in result
