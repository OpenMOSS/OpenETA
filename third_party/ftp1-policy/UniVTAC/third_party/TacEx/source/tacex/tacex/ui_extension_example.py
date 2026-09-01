# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import math
import numpy as np
import random

import carb
import carb.events
import omni.ext
import omni.ui as ui
import omni.usd
from usdrt import Gf, Rt, Sdf, Usd, Vt

try:
    import warp as wp

    wp.init()

    @wp.kernel
    def deform(positions: wp.array(dtype=wp.vec3), t: float):
        tid = wp.tid()

        x = positions[tid]
        offset = -wp.sin(x[0])
        scale = wp.sin(5.0 * t) * 0.05

        x = x + wp.vec3(0.0, offset * scale, 0.0)

        positions[tid] = x

except ImportError:
    pass


def get_selected_prim_path():
    """Return the path of the first selected prim"""
    context = omni.usd.get_context()
    selection = context.get_selection()
    paths = selection.get_selected_prim_paths()

    return None if not paths else paths[0]


def get_stage_id():
    """Return the stage Id of the current stage"""
    context = omni.usd.get_context()
    return context.get_stage_id()


def is_vtarray(obj):
    """Check if this is a VtArray type

    In Python, each data type gets its own
    VtArray class i.e. Vt.Float3Array etc.
    so this helper identifies any of them.
    """
    return hasattr(obj, "IsFabricData")


def condensed_vtarray_str(data):
    """Return a string representing VtArray data

    Include at most 6 values, and the total items
    in the array
    """
    size = len(data)
    if size > 6:
        datastr = "[{}, {}, {}, .. {}, {}, {}] (size: {})".format(
            data[0], data[1], data[2], data[-3], data[-2], data[-1], size
        )
    else:
        datastr = "["
        for i in range(size - 1):
            datastr += str(data[i]) + ", "
        datastr += str(data[-1]) + "]"

    return datastr


def get_fabric_data_for_prim(stage_id, path):
    """Get the Fabric data for a path as a string"""
    if path is None:
        return "Nothing selected"

    stage = Usd.Stage.Attach(stage_id)

    # If a prim does not already exist in Fabric,
    # it will be fetched from USD by simply creating the
    # Usd.Prim object. At this time, only the attributes that have
    # authored opinions will be fetch into Fabric.
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if not prim:
        return f"Prim at path {path} is not in Fabric"

    # This diverges a bit from USD - only attributes
    # that exist in Fabric are returned by this API
    attrs = prim.GetAttributes()

    result = f"Fabric data for prim at path {path}\n\n\n"
    for attr in attrs:
        try:
            data = attr.Get()
            datastr = str(data)
            if data is None:
                datastr = "<no value>"
            elif is_vtarray(data):
                datastr = condensed_vtarray_str(data)

        except TypeError:
            # Some data types not yet supported in Python
            datastr = "<no Python conversion>"

        result += f"{attr.GetName()} ({str(attr.GetTypeName().GetAsToken())}): {datastr}\n"

    return result


def apply_random_rotation(stage_id, path):
    """Apply a random world space rotation to a prim in Fabric"""
    if path is None:
        return "Nothing selected"

    stage = Usd.Stage.Attach(stage_id)
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if not prim:
        return f"Prim at path {path} is not in Fabric"

    rtxformable = Rt.Xformable(prim)
    if not rtxformable.HasWorldXform():
        rtxformable.SetWorldXformFromUsd()

    angle = random.random() * math.pi * 2
    axis = Gf.Vec3f(random.random(), random.random(), random.random()).GetNormalized()
    halfangle = angle / 2.0
    shalfangle = math.sin(halfangle)
    rotation = Gf.Quatf(math.cos(halfangle), axis[0] * shalfangle, axis[1] * shalfangle, axis[2] * shalfangle)

    rtxformable.GetWorldOrientationAttr().Set(rotation)

    return f"Set new world orientation on {path} to {rotation}"


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

    if not wp:
        return "Warp failed to initialize. Install/Load the warp extension."

    # Tell OmniHydra to render points from Fabric
    if not prim.HasAttribute("Deformable"):
        prim.CreateAttribute("Deformable", Sdf.ValueTypeNames.PrimTypeTag, True)

    points = prim.GetAttribute("points")
    pointsarray = np.array(points.Get())
    warparray = wp.array(pointsarray, dtype=wp.vec3, device="cuda")

    wp.launch(kernel=deform, dim=len(pointsarray), inputs=[warparray, time], device="cuda")

    points.Set(Vt.Vec3fArray(warparray.numpy()))

    return f"Deformed points on prim {path}"


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class UsdrtExamplePythonExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[omni.example.python.usdrt] startup")

        self._window = ui.Window(
            "What's in Fabric?", width=300, height=300, dockPreference=ui.DockPreference.RIGHT_BOTTOM
        )
        self._t = 0
        self._dt = 0.01
        self.sub = None
        self.playing = False

        with self._window.frame:
            with ui.VStack():
                frame = ui.ScrollingFrame()
                with frame:
                    label = ui.Label("Select a prim and push a button", alignment=ui.Alignment.LEFT_TOP)

                def get_fabric_data():
                    label.text = get_fabric_data_for_prim(get_stage_id(), get_selected_prim_path())

                def rotate_prim():
                    label.text = apply_random_rotation(get_stage_id(), get_selected_prim_path())

                def toggle_deform_prim():
                    self.playing = not self.playing
                    if not self.sub:
                        self.init_on_update()

                ui.Button("What's in Fabric?", clicked_fn=get_fabric_data, height=0)
                ui.Button("Rotate it in Fabric!", clicked_fn=rotate_prim, height=0)
                ui.Button("Toggle: Deform it with Warp!", clicked_fn=toggle_deform_prim, height=0)

    def init_on_update(self):
        @carb.profiler.profile(zone_name="omni.example.python.usdrt.on_update")
        def on_update(e: carb.events.IEvent):
            if not self.playing:
                return
            try:
                deform_mesh_with_warp(get_stage_id(), get_selected_prim_path(), self._t)
                self._t += self._dt
            except Exception as error:
                carb.log_error(error)
            return

        update_stream = omni.kit.app.get_app().get_update_event_stream()
        self.sub = update_stream.create_subscription_to_pop(on_update, name="omni.example.python.usdrt.on_update")
        return

    def on_shutdown(self):
        print("[omni.example.python.usdrt] shutdown")
