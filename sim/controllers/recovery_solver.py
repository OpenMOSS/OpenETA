"""Local re-solves within declared tracking bounds; no task or scene planner."""
import math
import numpy as np


class TrackingLimit:
    """Linearized step/corridor bounds; nonlinear checks remain authoritative."""
    def __init__(self, site_id, segment):
        self.site_id, self.segment = site_id, segment

    def compute_qp_inequalities(self, configuration, dt):
        import mujoco
        from mink.limits import Constraint
        model, data = configuration.model, configuration.data
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, self.site_id)
        # Inscribed boxes leave headroom for nonlinear integration/physics.
        G = [jp, -jp, jr, -jr]
        h = [np.full(3, .004), np.full(3, .004),
             np.full(3, math.radians(2)), np.full(3, math.radians(2))]
        segment = self.segment
        direction = segment.delta / segment.length if segment.length > .002 else np.zeros(3)
        projection = np.eye(3) - np.outer(direction, direction)
        lateral = projection @ (data.site_xpos[self.site_id] - segment.start)
        radius = segment.candidate_cross_track_m / math.sqrt(3)
        # An already out-of-box state may only improve, never be forced farther out.
        bounds = np.maximum(radius, np.abs(lateral))
        G.extend([projection @ jp, -projection @ jp])
        h.extend([bounds-lateral, bounds+lateral])
        return Constraint(G=np.vstack(G), h=np.concatenate(h))


def alternative_velocities(configuration, tasks, limits, dt, segment, site_id,
                           arm_indices, *, recovery):
    """Yield at most five checked-solve candidates; never execute a candidate."""
    import mink
    bounded_limits = [*limits, TrackingLimit(site_id, segment)]
    alternatives = [('track_constrained', tasks)]
    if recovery:
        # Replace the endpoint posture attraction, while retaining the frame task.
        # Opposite joint-center biases explore local redundancy, not object geometry.
        for name, shift in [('hold_posture', 0.), ('posture_positive', .15), ('posture_negative', -.15)]:
            posture = mink.PostureTask(configuration.model, cost=1.)
            q = configuration.q.copy()
            # Arm qpos and tangent indices coincide for this Panda hinge chain.
            # Map indices explicitly instead of assuming this for other models.
            model = configuration.model
            for joint in range(model.njnt):
                if model.jnt_dofadr[joint] in arm_indices and model.jnt_type[joint] == 3:
                    address = model.jnt_qposadr[joint]
                    q[address] += shift
                    if model.jnt_limited[joint]:
                        q[address] = np.clip(q[address], *model.jnt_range[joint])
            posture.set_target(q)
            alternatives.append((name, [tasks[0], posture]))
        alternatives.append(('seed_relaxed', [tasks[0]]))
    for name, candidate_tasks in alternatives:
        try:
            velocity = mink.solve_ik(configuration, candidate_tasks, dt, solver='quadprog',
                damping=1e-6, safety_break=False, limits=bounded_limits)
        except AssertionError:
            continue
        if np.all(np.isfinite(velocity)):
            yield name, velocity
