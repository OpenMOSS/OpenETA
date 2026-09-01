# TacEx: Assets for Robots and Objects
> ref.: https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_assets

This extension contains configurations for various assets and sensors. The configuration instances are used to spawn and configure the instances in the simulation. They are passed to their corresponding classes during construction.

# Overview

For TacEx, we use multiple robots and sensor models:
- Franka
- GelSight Mini sensor

The directories `robots` and `sensors` contain python files with the configuration classes for the assets.

The `data` directory contains the USD models of the assets.
The directory structure follows the recommended structure (kinda) and is as follows:
- `Robots/<Company-Name>/<Robot-Name>/<Sensor-Type>/<Robot-Type>`: The USD files should be inside `<Robot-Type>` directory with the name of the robot. E.g. `Robots/Franka/GelSight_Mini/Gripper` contains USD files for Franka Panda arms with a gripper as endeffector and GelSight Mini's attached to it (here we omitted the robot name for now).
>I know, that are a lot of subdir's, but I hope this makes stuff structured. Ofc happy to take improvement suggestions!

- `Props/<Prop-Type>/<Prop-Name>`: The USD files should be inside `<Prop-Name>` directory with the name of the prop. This includes mounts, objects and markers.

- `Policies/<Task-Name>`: The policy should be JIT/ONNX compiled with the name `policy.pt`. It should also contain the parameters used for training the checkpoint. This is to ensure reproducibility.

- `Sensor/<Sensor-Type>/<Sensor-Name>`: Contains models of Tactile Sensors. For example sensors from the `GelSight Mini`-Type (with rigid gelpad or with softbody gelpad).

- `Test/<Test-Name>`: The asset used for unit testing purposes.

# Referring to the assets in your code

You can use the following snippet to refer to the assets:

```python
from tacex_assets import TACEX_ASSETS_DATA_DIR
# ANYmal-C
ball = f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd"
```


# How to create your own sensor/robot asset
Example for creating an asset for Tactile Simulation - Creating an asset for the GelSight Mini
