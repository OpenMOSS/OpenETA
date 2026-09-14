import math
from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation
from sim.controllers.cartesian_segment import CartesianSegment, checked_segment_velocity
from sim.controllers.candidate_feedback import public_candidate_trace
I=[0,0,0,1]


def test_candidate_diagnostics_distinguish_tracking_and_geometry_rejection():
    class Config:
        q=np.zeros(3)
        def integrate(self,v,dt):return self.q+v*dt
    trace=[]
    segment=CartesianSegment([0,0,0],[0,0,.1],I,I)
    assert checked_segment_velocity(Config(),np.array([0,0,.24]),np.arange(3),.05,
        segment,lambda q:(q,I),lambda *args:False,candidate_trace=trace,
        geometry_feedback=lambda:{'geometry_rejections':['robot_collision']}) is None
    assert trace[0]['geometry_status']=='not_checked'
    assert all(r['geometry_status']=='rejected' for r in trace[1:])
    assert all(r['tracking_passed'] for r in trace[1:])
    assert not any(r['checks_passed'] for r in trace)
    assert len(public_candidate_trace(trace*20))==36


def test_recovery_has_bounded_extra_corridor_but_same_step_guard():
    strict=CartesianSegment([0,0,0],[0,0,.1],I,I)
    recovery=CartesianSegment([0,0,0],[0,0,.1],I,I,recovery=True)
    now=np.array([.014,0,.04]);later=[.014,0,.044]
    assert not strict.candidate_allowed(now,I,later,I)
    assert recovery.candidate_allowed(now,I,later,I)
    assert not recovery.candidate_allowed(now,I,[.014,0,.06],I)
    assert recovery.observe([.021,0,.05],I,.05,0)=='cartesian_path_deviation'
    q=Rotation.from_euler('z',13,degrees=True).as_quat()
    assert recovery.observe([0,0,.05],q,.05,.2)=='cartesian_path_deviation'


def test_tracking_limit_constrains_world_jacobian_and_redundancy_remains_available():
    import mujoco,mink
    from sim.controllers.recovery_solver import TrackingLimit, alternative_velocities
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/>
      <joint name="z" type="slide" axis="0 0 1"/><geom type="sphere" size="0.02" mass="1"/>
      <site name="tip"/></body></worldbody></mujoco>''')
    config=mink.Configuration(model);segment=CartesianSegment([0,0,0],[0,0,.1],I,I)
    constraint=TrackingLimit(0,segment).compute_qp_inequalities(config,.05)
    assert np.all(constraint.G @ np.array([0,0,.003]) <= constraint.h+1e-9)
    assert np.any(constraint.G @ np.array([0,0,.012]) > constraint.h)
    task=mink.FrameTask('tip','site',position_cost=1,orientation_cost=0)
    task.set_target(mink.SE3.from_translation(np.array([0,0,.1])))
    variants=list(alternative_velocities(config,[task],[],.05,segment,0,np.arange(3),recovery=True))
    assert len(variants)==5
    assert all(np.max(np.abs(v*.05))<=.004+1e-8 for _,v in variants)
