This tutorial showcases how you can create an USD asset with GelSight Sensors.


# Set up the USD asset
First, open the Isaac GUI.

- create new file in `source/tacex_assets/Robots/your_dir`
- import your robot asset
- set it as default prim
- import the GelSight Mini case into the scene (drag and drop the file `source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/Case.usd` into the scene)
-> Asset Path should be relative (is done automatically by Isaac), so that it works across different systems.
![alt text](../_static/tutorials/creating_robot_asset_with_sensors/rel_asset_path.png)

- move sensor case to desired position and attach with fixed joint to the articulation
- import gelpad asset `source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/Gelpad_low_res.usd`
- move gelpad asset into desired position
- select the gelpad meshes and use the same visual material as `gelsight_mini_case/plate/mesh`, i.e. the transculent one. This will prevent that the camera includes the gelpad mesh in its rendering.

>[!NOTE]
> We use different bodies for gelpad and sensor case to set different physics properties.

## Create Robot with PhysX rigid body gelpad
- apply rigid body api with colliders to the gelpad
  - select gelpad XForm
  - press `Add -> Physics -> Rigid Body with Colliders Preset`
  - to adjust the collider select the mesh under the Xform and scroll down to collider properties
- attach gelpad to sensor case via fixed joint (select Xform of gelpad_left and then of gelsight_mini_case-> right click -> Create -> Physics -> Joint -> Fixed Joint )
- rename the joint to something like `FixedJointCaseLeft` to make it unique

Press play to check that it works as intended, i.e. Robot moves a little bit and the gelpads move accordingly.

## Create robot with UIPC gelpads
Instead of applying the rigid body api, we dont apply any Isaac Simulation property.
Instead we create tet mesh data and let UIPC simulate it.

Press play to check that it works as intended, i.e. Robot moves a little bit while the gelpads stay in place.


### Create precomputed data for uipc sim
You don't need to precompute data. In the standalone scripts you can decide if you compute the tet data (and attachment data) or use the precomputed ones to save some time.

To do this, first enable tacex_uipc ui extension (see [Guide](tacex_uipc_ui_extension.md)).
Then
- select gelpad mesh
- press `compute tet mesh` button
- select the sensor case first and then the gelpad mesh (order is important)
  - then press `compute attachment data`
- save the updated usd file

![alt text](../_static/tutorials/creating_robot_asset_with_sensors/uipc_gelpad.png)

# Create Config File
- go to `source/tacex_assets/tacex_assets/robots/franka` and create a new python file
- you can copy-paste the contents of another config file
- just adjust the `usd_path`in the spawn config, e.g.
`usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Single_Adapter/physx_rigid_gelpad.usd"`
