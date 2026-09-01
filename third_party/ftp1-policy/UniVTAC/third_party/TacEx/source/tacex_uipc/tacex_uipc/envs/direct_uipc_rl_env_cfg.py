from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass

from tacex_uipc import UipcSimCfg


@configclass
class UipcEnvCfg(DirectRLEnvCfg):
    """Configuration for an RL environment defined with the direct workflow.

    Please refer to the :class:`isaaclab.envs.direct_rl_env.DirectRLEnv` class for more details.
    """

    # UIPC simulation settings

    uipc_sim: UipcSimCfg = None
    """ Config for UIPC simulation.

    None means, no uipc simulation and just the PhysX simulation from Isaac.
    """
