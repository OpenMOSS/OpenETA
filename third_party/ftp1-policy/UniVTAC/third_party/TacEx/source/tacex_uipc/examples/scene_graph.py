import numpy as np

import omni.usd
import warp as wp
from usdrt import Sdf, Usd, Vt

wp.init()


@wp.kernel
def deform(positions: wp.array(dtype=wp.vec3), t: float):
    tid = wp.tid()

    x = positions[tid]
    offset = -wp.sin(x[0])
    scale = wp.sin(t) * 10.0

    x = x + wp.vec3(0.0, offset * scale, 0.0)

    positions[tid] = x


def deform_mesh_with_warp(stage_id, path, time):
    """Use Warp to deform a Mesh prim"""

    if path is None:
        return "Nothing selected"

    stage = Usd.Stage.Attach(stage_id)
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if not prim:
        return f"Prim at path {path} is not in Fabric"

    if not prim.HasAttribute("points"):
        return f"Prim at path {path} does not have points attribute"

    # Tell OmniHydra to render points from Fabric
    if not prim.HasAttribute("Deformable"):
        prim.CreateAttribute("Deformable", Sdf.ValueTypeNames.PrimTypeTag, True)

    points = prim.GetAttribute("points")
    pointsarray = np.array(points.Get())
    # print("pointsarray ", pointsarray)
    warparray = wp.array(pointsarray, dtype=wp.vec3, device="cuda")
    # print("points ", warparray)

    wp.launch(kernel=deform, dim=len(pointsarray), inputs=[warparray, time], device="cuda")

    points.Set(Vt.Vec3fArray(warparray.numpy()))

    return f"Deformed points on prim {path}"


stage_id = omni.usd.get_context().get_stage_id()
path = "/World/Torus"
for t in range(100000):
    print(deform_mesh_with_warp(stage_id, path, t))
