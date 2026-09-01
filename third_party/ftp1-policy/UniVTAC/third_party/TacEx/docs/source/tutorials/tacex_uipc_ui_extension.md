# Omniverse Extension - tacex_uipc

The `tacex_uipc` package also comes with a UI extension which is defined in `source/tacex_uipc/ui_extension.py`.

You can us it inside the "normal" ( = not standalone) Isaac Sim GUI.
It enables you to
- precompute a Tetrahedra mesh for a selected prim (via [wildmeshing](https://github.com/wildmeshing/wildmeshing-python))
- update the surface mesh based on the tetrahedra mesh
- create precomputed data for the `UIPC x Isaac` Attachments
- extract the current uv map or set it via a numpy array (This is just experimental for the uipc_texture based approach, which will be explained in another tutorial)


# Setup the Extension

To enable the extension, follow these steps:

1. **Add the search path of the repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, press the plus symbol and enter the absolute path to the`source` directory (e.g. `/home/user/Projects/TacEx/source`)
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find the `tacex_uipc` extension under the `Third Party` category.
    - Toggle it to enable your extension.
    - Now you should see the `tacex_uipc` UI

![alt text](../_static/tutorials/tacex_uipc_extension/tacex_uipc_ui_extension.png)

# Precompute Tet Mesh
- Open the scene with the prim
- Select the mesh of the prim (not just the xForm!)
- press `Compute Tet Mesh`

Now you should see the tet mesh drawn as lines.
Here is an example:
![alt text](../_static/tutorials/tacex_uipc_extension/compute_tet_mesh.png)

Now save the USD file to use the precomputed tet data for standalone simulations.

# Update Surface Mesh
This is just for visualization purposes of the tet mesh.
If you press the `Update Surface Mesh` button, the surface of the mesh is colored based on the tetrahedra surfaces.
This is normally done during the simulation setup by the `tacex_uipc` simulation automatically.

Example:
![alt text](../_static/tutorials/tacex_uipc_extension/update_mesh_surface.png)

It is recommend to **not save the USD file** after updating the surface mesh.

Reason is, that the current surface mesh is used to compute the tet mesh when the precomputed data isn't used (imagine you use a different tet-mesh configuration during the scene setup).

If we use the surface mesh of the tet mesh, we lose details and the tet-mesh generation takes a lot longer.

# Create Attachment
- select the rigid body (here you can select the xForm)
- press `shift` and select the tet mesh next (the selection order is important here!)
- then press the `Create Attachment` Button

Now you should see the attachment points, the rigid body center and the connection lines.
Example:
![alt text](../_static/tutorials/tacex_uipc_extension/compute_attachment.png)

![alt text](![alt text](../_static/tutorials/tacex_uipc_extension/compute_attachment_wireframe.png)

Save the updated USD file to keep the data.

# Tips
- If you changed the extension code, turn the extension off and on to update the extension code.
- You currently cannot set the tet-configuration from the UI. Right now, you need to go to the source code and adjust it there via the `TetMeshCfg` class
