// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "FixDLExt.h"
#include "PyDLExt.h"

#include "modify.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

#include <limits>

using namespace LAMMPS_NS;
using namespace LAMMPS_NS::dlext;

namespace py = pybind11;

LAMMPS_NS::LAMMPS* to_lammps_ptr(py::object lmp)
{
    auto pyptr = lmp.attr("lmp");
    auto pyaddr = pyptr.attr("value");
    auto lmp_addr = pyaddr.cast<uintptr_t>();
    return reinterpret_cast<LAMMPS*>(lmp_addr);
}

void export_LAMMPSView(py::module& m)
{
    py::class_<LAMMPSView>(m, "LAMMPSView")
        .def(py::init([](py::object lmp) {
            return cxx11::make_unique<LAMMPSView>(to_lammps_ptr(lmp));
        }))
        .def("local_particle_number", &LAMMPSView::local_particle_number)
        .def("global_particle_number", &LAMMPSView::global_particle_number)
        .def("synchronize", &LAMMPSView::synchronize, py::arg("space") = kOnDevice)
        ;
}

void export_FixDLExt(py::module& m)
{
    // Fixes are managed by each lammps' instance `modify` member,
    // so we should prevent python from trying to delete them.
    using FixDLExtHolder = std::unique_ptr<FixDLExt, py::nodelete>;

    py::class_<FixDLExt, FixDLExtHolder>(m, "FixDLExt")
        .def(py::init([](py::object lmp, std::string& args) {
            auto lmp_ptr = to_lammps_ptr(lmp);
            register_FixDLExt(lmp_ptr);
            auto fix = lmp_ptr->modify->add_fix(args);
            return static_cast<FixDLExt*>(fix);
        }))
        .def("set_callback", &FixDLExt::set_callback)
        .def("set_virial_callback", &FixDLExt::set_virial_callback)
        ;
}

PYBIND11_MODULE(_api, m)
{
    // We want to display the members of the module as `lammps.dlext.x`
    // instead of `lammps.dlext._api.x`.
    py::str module_name = m.attr("__name__");
    m.attr("__name__") = "lammps.dlext";

    // Enums
    py::enum_<ExecutionSpace>(m, "ExecutionSpace")
        .value("kOnDevice", kOnDevice)
        .value("kOnHost", kOnHost)
        .export_values();

    // Classes
    export_LAMMPSView(m);
    export_FixDLExt(m);

    // Methods
    m.def("has_kokkos_cuda_enabled", [](py::object lmp) {
        return has_kokkos_cuda_enabled(to_lammps_ptr(lmp));
    });
    m.def("positions", enpycapsulate<&positions>);
    m.def("velocities", enpycapsulate<&velocities>);
    m.def("masses", enpycapsulate<&masses>);
    m.def("forces", enpycapsulate<&forces>);
    m.def("images", enpycapsulate<&images>);
    m.def("tags", enpycapsulate<&tags>);
    m.def("tags_map", enpycapsulate<&tags_map>);
    m.def("types", enpycapsulate<&types>);

    // Other attributes
    m.attr("kImgMask") = IMGMASK;
    m.attr("kImgMax") = IMGMAX;
    m.attr("kImgBits") = IMGBITS;
    m.attr("kImg2Bits") = IMG2BITS;
    m.attr("kImgBitSize") = sizeof(imageint) * CHAR_BIT;

    // Set back the module_name to its original value
    m.attr("__name__") = module_name;
}
