import random

import omni.usd
from pxr import Gf, UsdGeom


class Example:
    def create_tet(self):
        # Create Point List
        N = 500
        scale = 0.05
        self.point_list = [
            (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0)) for _ in range(N)
        ]
        self.colors = [(1, 1, 1, 1) for _ in range(N)]
        self.sizes = [(random.uniform(0.1, 2.0), random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)) for _ in range(N)]
        self.face_count = 4

        # Set up Geometry to be Instanced
        tet_path = "/World/Tetrahedra"
        stage = omni.usd.get_context().get_stage()

        tet = UsdGeom.Mesh.Define(stage, tet_path)
        tet.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        tet.CreateFaceVertexCountsAttr([3] * self.face_count)
        tet.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3])
        tet.CreateDisplayColorPrimvar().Set([(0, 1, 1)])
        tet.CreateDisplayOpacityPrimvar().Set([0.15] * self.face_count)
        tet.AddScaleOp().Set(Gf.Vec3d(1, 1, 1) * scale)

        # Set up Point Instancer

        instance_path = "/World/PointInstancer"
        self.point_instancer = UsdGeom.PointInstancer(stage.DefinePrim(instance_path, "PointInstancer"))
        # Create & Set the Positions Attribute
        self.positions_attr = self.point_instancer.CreatePositionsAttr()
        self.positions_attr.Set(self.point_list)
        self.scale_attr = self.point_instancer.CreateScalesAttr()
        self.scale_attr.Set(self.sizes)
        # Set the Instanced Geometry
        self.point_instancer.CreatePrototypesRel().SetTargets([tet.GetPath()])

        self.proto_indices_attr = self.point_instancer.CreateProtoIndicesAttr()
        self.proto_indices_attr.Set([0] * len(self.point_list))

    def create(self):
        # Create Point List
        N = 500
        scale = 0.05
        self.point_list = [
            (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0)) for _ in range(N)
        ]
        self.colors = [(1, 1, 1, 1) for _ in range(N)]
        self.sizes = [(random.uniform(0.1, 2.0), random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)) for _ in range(N)]

        # Set up Geometry to be Instanced
        tet_path = "/World/Tetrahedra"
        stage = omni.usd.get_context().get_stage()
        cube = UsdGeom.Cube(stage.DefinePrim(tet_path, "Cube"))
        cube.AddScaleOp().Set(Gf.Vec3d(1, 1, 1) * scale)
        cube.CreateDisplayColorPrimvar().Set([(0, 1, 1)])
        # Set up Point Instancer

        instance_path = "/World/PointInstancer"
        self.point_instancer = UsdGeom.PointInstancer(stage.DefinePrim(instance_path, "PointInstancer"))
        # Create & Set the Positions Attribute
        self.positions_attr = self.point_instancer.CreatePositionsAttr()
        self.positions_attr.Set(self.point_list)
        self.scale_attr = self.point_instancer.CreateScalesAttr()
        self.scale_attr.Set(self.sizes)
        # Set the Instanced Geometry
        self.point_instancer.CreatePrototypesRel().SetTargets([cube.GetPath()])

        self.proto_indices_attr = self.point_instancer.CreateProtoIndicesAttr()
        self.proto_indices_attr.Set([0] * len(self.point_list))

    def update(self):
        # modify the point list
        for i in range(len(self.point_list)):
            self.point_list[i] = (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0))
        # update the points
        self.positions_attr.Set(self.point_list)


import omni

example = Example()
example.create_tet()
