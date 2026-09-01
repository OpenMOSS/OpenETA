# Deploy Your Policy

To deploy and evaluate your own policy in UniVTAC, you need to create three files under `policy/YourPolicy/`:

**1. `deploy_policy.py`** — Implements the policy interface. The following components must be defined:

```python
# policy/YourPolicy/deploy_policy.py
import torch
from policy._base_policy import BasePolicy

class Policy(BasePolicy):
    def __init__(self, args):
        """
        Load your model. `args` is a dict containing all fields from
        deploy.yml, plus runtime fields:
            - args['task_name']   : str — the current task name
            - args['task_config'] : str — the task config file stem
        """
        super().__init__(args)
        self.model = load_your_model(args)

    def encode_obs(self, observation):
        """
        Post-process raw observation into your model's input format.
        """
        return your_processed_obs

    def eval(self, task, observation):
        """
        Run one inference step. Called in a loop until the task
        reaches step_lim or reports success.
        """
        obs = self.encode_obs(observation)
        action = self.model.get_action(obs).reshape(-1)
        action = torch.from_numpy(action).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type='qpos')
        # action_type options:
        #   'qpos'     — joint positions: Tensor([8])  (7 arm + 1 gripper)
        #   'ee'       — end-effector pose: Tensor([8]) (position(3) + quaternion(4) + gripper)
        #   'delta_ee' — delta end-effector: Tensor([7]) (delta_position(3) + delta_rotation(3) + delta_gripper)

    def reset(self):
        """Reset internal state (e.g., temporal buffers) at the start of each episode."""
        if hasattr(self.model, 'reset'):
            self.model.reset()
```

**2. `deploy.yml`** — Specifies deployment parameters, which are used to configuate the deployment and passed to `Policy.__init__()` as `args`. Only the `policy_name` field is mandatory and it must match the directory under `policy/`; you can add any other custom fields needed by your policy.

**3. Run evaluation:**

```bash
bash eval_policy.sh ${task_name} ${task_config} ${policy_name}/${policy_config_name} ${gpu_id}
# Example: bash eval_policy.sh lift_bottle demo YourPolicy/deploy 0
```

or 

```bash
bash parallel_eval.sh ${task_name} ${task_config} ${policy_config} ${gpu_id} [num_processes] [total_num]
# Example: bash parallel_eval.sh lift_bottle demo YourPolicy/deploy 3 0,1,2
```