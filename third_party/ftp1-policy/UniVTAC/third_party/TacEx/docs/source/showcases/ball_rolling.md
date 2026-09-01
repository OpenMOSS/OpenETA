For measuring the performance of the tactile simulations we have a ball rolling environment.
Here a robot with a single GelSight Mini sensor is used to roll a ball around in a certain pattern (backward, forward, left, right...).

We currently have three different environments, which can be found in `./scripts/benchmarking/tactile_sim_performance/envs`:
- one in which the gelpad is simulated as a (compliant) rigid body with PhysX
- two in which the gelpad is simulated with UIPC

Currently, each env configuration uses GPU Taxim simulation for tactile RGB.
You can of course change the sensor config and use e.g. a different tactile image resolution.

To run the script call
```bash
python ./scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py
```

Arguments are:
- `--env`, which defines which env is used. Current Options are:
  - `physx_rigid`: physx rigid gelpad
  - `uipc`: soft gelpad simulated via UIPC
  - `uipc_textured`: soft gelpad simulated via UIPC.
    - The gelpad also has a marker texture and a camera is used to get the RGB image of this texture.
- `--num_envs`, which defines how many environments should be simulated.
  - **Currently the UIPC based envs can only be run with `--num_envs=1`**
- `--debug_vis`, which toggles rendering of the sensor output in the gui on
  - is off by default

Example call:
```bash
# runs the ball rolling experiment with the uipc_textured env, 1 env and with showing the sensor outputs in the GUI
isaaclab -p ./scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py --env uipc_textured --num_envs 1 --debug_vis
```

To visualize the tactile output:
- select a `gelsight_mini_case` Xform prim.
- scroll down to `Raw USD Properties/Extra Properties` at the bottom
- toggle `debug_tactile_rgb` on
  - should be directly under `Extra Properties` (see image below)

![](image.png)
