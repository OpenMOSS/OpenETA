# Ball Rolling Privileged
**Task**: The robot should move/roll a ball to a goal position.

As a robot we use a Franka panda arm with the Single Adapter and a GelSight Mini as endeffector.
The environments do not use tactile readings, therefore tactile simulation is omitted. This increases the simulation performance by a lot.
Instead of tactile readings, privileged simulation data is used such as the current ball position.

## base_env
- Actions (=5): Task space with relative IK Controller (dx, dy, dz, droll, dpitch)
    - dyaw is omitted, since the z rotation of the ee is irrelevant for our task
- Observations (=14): ee pos (=3), ee orientation (=2 -> roll, pitch), goal pos (=2 -> x,y) and current obj pos (=2 -> x,y), as well as the actions (=5)
- Reward:
    - reaching part (distance obj and ee)
    - bonus rew, when ee is close enough to obj
    - target tracking part (distance obj and goal -> applied when ee is close enough to obj)
    - success reward -> applied when ball is close enough to target pos
    - penalty if ee is too close to the ground
    - penalty if ee roll and pitch orientation is too big
    - action_rate and joint_vel penalties

>[!Note]
>The other ball-rolling-privileged env inherit from this env. This simplifies the setup and prevents redundancy, such as having the same "setup_scene" method.

## reset_with_IK_solver
**Idea**:
Make the task simpler by omitting that the robot needs to learn to reach for the ball.

This env is like the base env, but instead of resetting the robot at a fixed position we
set the ee to be close the ball.
For this we compute a target position (slightly above the ball) and use an [IK-solver](https://github.com/UM-ARM-Lab/pytorch_kinematics]) to compute the required joint values
These are directly written into the joint state of the robots.
In case the solver does not converge, we take the solution with the smallest pos_error.

Due to the IK computation we lose quite a lot of simulation performance.
(We can tweak some IK solver variables, mainly `max_iterations` and `num_retries` to find a good trade off between solution error and performance).

While we lose simulation performance, the reaching part of the task becomes a lot easier.
So the question is: What is better - More sim performance or making the task easier?

If we just use the same reward function (i.e. also with the reaching part) as the base_env,
then **not using the IK solver is a lot better**.
Here are some training metrics with 1024 robots:

**Task Performance**:
![task_perf](image-1.png)
> - green = base_env after about 1.5h
> - orange = reset_with_IK_solver after about 1.8h
As you can see, the base_env trains a lot faster with better/comparable performance
w.r.t pushing the obj to the goal.

**FPS**:
![sim_perf](image-2.png)
> - jumps in performance are due to turning on/off the GUI rendering (which was done to observe the agents)

## without_reaching
**Idea**: Omit reaching part also from the rew function, to make rew function simpler
and easier to tune

A big problem with the reaching part in the reward function, is that it becomes quite difficult to tune the reward function.

Some of our experiences we got while trying to make the ball rolling work:
We found that our reward function is quite sensitive w.r.t. to its weights.
We need to find an appropriate balance between the rewards terms, which turned out be quite difficult.
If the reaching rew is too big, the robot does not try to move the ball towards the goal. Instead it is happy enough to just be at the ball.
If the goal-tracking rew is too big, then the reaching will not work anymore.
The other reward terms behave similarly.

We also need to make sure that the goal-tracking rew is always positive.
Otherwise it could be that the robot learns to move the ee close to the ball while avoiding moving the ball.
Imagine the scenario: robot moves around randomly -> hits the ball far away from the goal -> big negative reward -> robot: "lets not move the ball".
> The base_env is designed in such a way where the goal-tracking rew is never negative.

This environment omits the reaching term of the reward function completely and also uses parameters for the other reward terms (especially the goal-tracking part, which can get negative now).
