// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef PY_LAMMPS_DLPACK_EXTENSION_H_
#define PY_LAMMPS_DLPACK_EXTENSION_H_

#include "DLExt.h"
#include "pybind11/pybind11.h"

namespace LAMMPS_NS
{
namespace dlext
{

using PropertyGetter = DLManagedTensor* (*)(const LAMMPSView&, ExecutionSpace);
using PyCapsule = pybind11::capsule;

const char* const kDLTensorCapsuleName = "dltensor";

// TODO: Write and export a class that follows the Python Specification for DLPack
// instead of directly exporting a `PyCapsule`.
// See the DLPack Documentation https://dmlc.github.io/dlpack/latest/python_spec.html

template <PropertyGetter property>
inline PyCapsule enpycapsulate(const LAMMPS& view, ExecutionSpace space)
{
    auto dl_managed_tensor = property(view, space);
    return PyCapsule(
        dl_managed_tensor,     // PyCapsule pointer
        kDLTensorCapsuleName,  // PyCapsule name
        [](PyObject* obj) {    // PyCapsule destructor
            auto dlmt = static_cast<DLManagedTensor*>(
                PyCapsule_GetPointer(obj, kDLTensorCapsuleName)
            );
            if (dlmt && dlmt->deleter) {
                dlmt->deleter(dlmt);
            } else {
                PyErr_Clear();
            }
        }
    );
}

}  // namespace dlext
}  // namspace LAMMPS_NS

#endif  // PY_LAMMPS_DLPACK_EXTENSION_H_
