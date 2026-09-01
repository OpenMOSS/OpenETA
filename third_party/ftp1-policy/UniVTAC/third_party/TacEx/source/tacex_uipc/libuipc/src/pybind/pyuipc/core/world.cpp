#include "uipc/common/type_define.h"
#include <pyuipc/core/world.h>
#include <uipc/core/world.h>
#include <uipc/core/engine.h>
#include <pyuipc/as_numpy.h>

namespace pyuipc::core
{
using namespace uipc::core;

PyWorld::PyWorld(py::module& m)
{
    auto class_World = py::class_<World>(m, "World");

    class_World.def(py::init<Engine&>())
        .def("init", &World::init, py::arg("scene"))
        .def("advance", &World::advance)
        .def("sync", &World::sync)
        .def("retrieve", &World::retrieve)
        .def("dump", &World::dump)
        .def("recover", &World::recover, py::arg("dst_frame") = ~0ull)
        .def("backward", &World::backward)
        .def("frame", &World::frame)
        .def("features", &World::features, py::return_value_policy::reference_internal)
        .def("is_valid", &World::is_valid)
        .def("write_vertex_pos_to_sim", 
            [](World& self, py::array_t<Float> positions, IndexT global_vertex_offset, IndexT local_vertex_offset, SizeT vertex_count, string system_name)
            {return self.write_vertex_pos_to_sim(as_span_of<Vector3>(positions), IndexT(global_vertex_offset), IndexT(local_vertex_offset), SizeT(vertex_count), system_name); },
            py::arg("positions"),
            py::arg("global_vertex_offset"),
            py::arg("local_vertex_offset"),
            py::arg("vertex_count"),
            py::arg("system_name") = string{""}
        );
}

}  // namespace pyuipc::core
