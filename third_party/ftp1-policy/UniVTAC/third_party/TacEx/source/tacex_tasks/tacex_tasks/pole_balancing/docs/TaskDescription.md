# Ball Rolling Tactile
**Task**: The robot should move/roll a ball to a goal position.

As a robot we use a Franka panda arm with the Single Adapter and a GelSight Mini as endeffector.

<!--todo need to update for tactile : -->

## base_env
This env is basically the ball_rolling_privileged extended for mixed observation (height map x proprioception). Its also using privileged information (i.e. the current ball position) for making sure that the env works.

- Actions (=5): Task space with relative IK Controller (dx, dy, dz, droll, dpitch)
    - dyaw is omitted, since the z rotation of the ee is irrelevant for our task (a fixed value is passed over during pre_physics step)
- Observations contain vision and proprioception data (=14):
    - proprioception: ee pos (=3), ee orientation (=2 -> roll, pitch), goal pos (=2 -> x,y) and **current** obj pos (=2 -> x,y), as well as the actions (=5)
    - vision: depth map of the cameras inside the scene (for testing purposes)

- Reward:
    - reaching part (distance obj and ee)
    - bonus rew, when ee is close enough to obj
    - target tracking part (distance obj and goal -> applied when ee is close enough to obj)
    - success reward -> applied when ball is close enough to target pos
    - penalty if ee is too close to the ground
    - penalty if ee roll and pitch orientation is too big
    - action_rate and joint_vel penalties

An [IK-solver](https://github.com/UM-ARM-Lab/pytorch_kinematics]) is used to place the ee at the ball after every reset.
This is necessary to make the training with the tactile readings work.
Otherwise, we would have a lot useless tactile images if the agent has to learn the reaching part from scratch (like in the ball_rolling_privileged case)


>[!Note]
>The other ball-rolling-tactile environments inherit from this env. This simplifies the setup and prevents redundancy, such as having the same "setup_scene" method.



## depth_map
Inherits from the base env.
Difference: uses depth map from camera, instead of height map.
And the privileged data is omitted. Instead of having the current ball position as an observation, the initial ball position is used.

- Actions (=5): Task space with relative IK Controller (dx, dy, dz, droll, dpitch)
    - dyaw is omitted, since the z rotation of the ee is irrelevant for our task (a fixed value is passed over during pre_physics step)
- Observations contain vision and proprioception data (=14):
    - proprioception: ee pos (=3), ee orientation (=2 -> roll, pitch), goal pos (=2 -> x,y) and **initial** obj pos (=2 -> x,y), as well as the actions (=5)
    - vision: depth map of the cameras inside the scene

- Reward:
    - reaching part (distance obj and ee)
    - bonus rew, when ee is close enough to obj
    - target tracking part (distance obj and goal -> applied when ee is close enough to obj)
    - success reward -> applied when ball is close enough to target pos
    - penalty if ee is too close to the ground
    - penalty if ee roll and pitch orientation is too big
    - action_rate and joint_vel penalties
