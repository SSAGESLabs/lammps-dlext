// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef LAMMPS_DLPACK_EXTENSION_H_
#define LAMMPS_DLPACK_EXTENSION_H_

#include "atom.h"
#include "fix.h"

#ifdef LMP_KOKKOS
#include "atom_kokkos.h"
#endif

#include <type_traits>
#include <vector>

namespace LAMMPS_NS
{
namespace dlext
{

#ifndef LMP_KOKKOS
using LMP_FLOAT = double;
using X_FLOAT = double;
using V_FLOAT = double;
using F_FLOAT = double;
#endif

static struct Positions { } kPositions;
static struct Velocities { } kVelocities;
static struct Masses { } kMasses;
static struct Forces { } kForces;
static struct Images { } kImages;
static struct Tags { } kTags;
static struct TagsMap { } kTagsMap;
static struct Types { } kTypes;
static struct Virial { } kVirial;

static struct SecondDim { } kSecondDim;

struct DLDataBridge {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLManagedTensor tensor;
};

void delete_bridge(DLManagedTensor* tensor)
{
    if (tensor)
        delete static_cast<DLDataBridge*>(tensor->manager_ctx);
}

template <typename T>
inline void* opaque(T* data)
{
    return static_cast<void*>(data);
}

template <typename T>
inline void* opaque(const T* data)
{
    return const_cast<void*>(data);
}

// if LAMMPS is built with KOKKOS, bind the PROPERTY struct to the corresponding ACCESSOR
#ifdef LMP_KOKKOS
#define DLEXT_OPAQUE_ATOM_KOKKOS(PROPERTY, ACCESSOR)       \
    inline void* opaque(const AtomKokkos* atom, PROPERTY)  \
    {                                                      \
        return opaque(atom->ACCESSOR.d_view.data());       \
    }

DLEXT_OPAQUE_ATOM_KOKKOS(Positions, k_x)
DLEXT_OPAQUE_ATOM_KOKKOS(Velocities, k_v)
DLEXT_OPAQUE_ATOM_KOKKOS(Masses, k_mass)
DLEXT_OPAQUE_ATOM_KOKKOS(Forces, k_f)
DLEXT_OPAQUE_ATOM_KOKKOS(Images, k_image)
DLEXT_OPAQUE_ATOM_KOKKOS(Tags, k_tag)
DLEXT_OPAQUE_ATOM_KOKKOS(TagsMap, k_map_array)
DLEXT_OPAQUE_ATOM_KOKKOS(Types, k_type)

#undef DLEXT_OPAQUE_ATOM_KOKKOS
#endif

// return the underlying pointers in LAMMPS (Property can be used as a tag, or actually bound to the KOKKOS accessor as above)
inline void* opaque(const Atom* atom, Positions) { return opaque(atom->x[0]); }
inline void* opaque(const Atom* atom, Velocities) { return opaque(atom->v[0]); }
inline void* opaque(const Atom* atom, Masses) { return opaque(atom->mass); }
inline void* opaque(const Atom* atom, Forces) { return opaque(atom->f[0]); }
inline void* opaque(const Atom* atom, Images) { return opaque(atom->image); }
inline void* opaque(const Atom* atom, Tags) { return opaque(atom->tag); }
inline void* opaque(const Atom* atom, Types) { return opaque(atom->type); }
inline void* opaque(const Atom* atom, TagsMap)
{
    return opaque(const_cast<Atom*>(atom)->get_map_array());
}
inline void* opaque(const Fix* fix, Virial) { return opaque(fix->virial); }

template <typename Property>
inline void* opaque(const Fix* fix, DLDeviceType device_type, Property p)
{
#ifdef LMP_KOKKOS
    if (device_type == kDLCUDA)
        return opaque(fix->view.atom_kokkos_ptr(), p);
#endif
    return opaque(fix->view.atom_ptr(), p);
}

// get the device info (id) from fix and device_type, return a DLDevice struct
inline DLDevice device_info(const Fix* fix, DLDeviceType device_type)
{
    return DLDevice { device_type, fix->view.device_id() };
}

// return the DLDataType code corresonding to the actual data type of the "Tag"
constexpr DLDataTypeCode dtype_code(Positions) { return kDLFloat; }
constexpr DLDataTypeCode dtype_code(Velocities) { return kDLFloat; }
constexpr DLDataTypeCode dtype_code(Masses) { return kDLFloat; }
constexpr DLDataTypeCode dtype_code(Forces) { return kDLFloat; }
constexpr DLDataTypeCode dtype_code(Images) { return kDLInt; }
constexpr DLDataTypeCode dtype_code(Tags) { return kDLInt; }
constexpr DLDataTypeCode dtype_code(TagsMap) { return kDLInt; }
constexpr DLDataTypeCode dtype_code(Types) { return kDLInt; }
constexpr DLDataTypeCode dtype_code(Virial) { return kDLFloat; }

// return the number of bits of the data type of a given PROPERTY
#define DLEXT_BITS_FLOAT_ARRAY(PROPERTY, TYPE)                                          \
    inline uint8_t bits(DLDeviceType device_type, PROPERTY)                             \
    {                                                                                   \
        return (device_type == kDLCPU || std::is_same<TYPE, double>::value) ? 64 : 32;  \
    }

DLEXT_BITS_FLOAT_ARRAY(Positions, X_FLOAT)
DLEXT_BITS_FLOAT_ARRAY(Velocities, V_FLOAT)
DLEXT_BITS_FLOAT_ARRAY(Masses, LMP_FLOAT)
DLEXT_BITS_FLOAT_ARRAY(Forces, F_FLOAT)
DLEXT_BITS_FLOAT_ARRAY(Virial, F_FLOAT)

#undef DLEXT_BITS_FLOAT_ARRAY

#define DLEXT_BITS_INT_ARRAY(PROPERTY, TYPE)                  \
    inline uint8_t bits(DLDeviceType device_type, PROPERTY)   \
    {                                                         \
        cxx11::maybe_unused(device_type);                     \
        return std::is_same<TYPE, int64_t>::value ? 64 : 32;  \
    }

DLEXT_BITS_INT_ARRAY(Images, imageint)
DLEXT_BITS_INT_ARRAY(Tags, tagint)

#undef DLEXT_BITS_INT_ARRAY

inline uint8_t bits(DLDeviceType device_type, TagsMap) { return 32; }
inline uint8_t bits(DLDeviceType device_type, Types) { return 32; }

template <typename Property>
inline DLDataType dtype(DLDeviceType device_type, Property p)
{
    return DLDataType { dtype_code(p), bits(device_type, p), 1 };
}

template <typename Property>
inline int64_t size(const Fix* fix, Property)
{
    return fix->view.local_particle_number();
}
inline int64_t size(const Fix* fix, Masses) { return fix->view.atom_ptr()->ntypes + 1; }
inline int64_t size(const Fix* fix, TagsMap) { return fix->view.atom_ptr()->get_map_size(); }
inline int64_t size(const Fix* fix, Virial) { return 6; }

template <typename Property>
inline int64_t size(const Fix* fix, Property, SecondDim)
{
    return 1;
}
inline int64_t size(const Fix* fix, Positions, SecondDim) { return 3; }
inline int64_t size(const Fix* fix, Velocities, SecondDim) { return 3; }
inline int64_t size(const Fix* fix, Forces, SecondDim) { return 3; }

template <typename Property>
constexpr uint64_t offset(const Fix* fix, Property p)
{
    return 0;
}

// a templated function for wrapping a C array given its data type and dimensions
// and returning a pointer to a DLPack tensor 
template <typename Property>
DLManagedTensor* wrap(const Fix* fix, Property property, ExecutionSpace exec_space)
{
    // get the device type of the fix (host or device)
    auto device_type = fix->view.device_type(exec_space);

    auto bridge = std::make_unique<DLDataBridge>();
    bridge->tensor.manager_ctx = bridge.get();
    bridge->tensor.deleter = delete_bridge;

    // acquire the actual dltensor pointer
    auto& dltensor = bridge->tensor.dl_tensor;

    // fill in the dltensor struct
    // get the underlying array/accessor of the given property and assign it to data (as void*)
    dltensor.data = opaque(fix, device_type, property);
    // get the device info from fix and device_type and assign it to device (as DLDevice)
    dltensor.device = device_info(fix, device_type);
    // get the data type of the underlying array (DLDataType) given the data type code and number of bits
    dltensor.dtype = dtype(device_type, property);

    // fill in the tensor shape (dimensions), strides and byte offsets
    auto& shape = bridge->shape;
    auto size2 = size(fix, property, kSecondDim);
    shape.push_back(size(fix, property));
    // if the array is 2D
    if (size2 > 1)
        shape.push_back(size2);
    // strides between consecutive elements in each dim
    auto& strides = bridge->strides;
    strides.push_back(size2);
    if (size2 > 1)
        strides.push_back(1);

    dltensor.ndim = shape.size();
    dltensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
    dltensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
    dltensor.byte_offset = offset(fix, property);

    return &(bridge.release()->tensor);
}

// macro that returns a DLManagedTensor from fix for a given SELECTOR (Property)
#define DLEXT_PROPERTY_FROM_FIX(FN, SELECTOR)                                \
    inline DLManagedTensor* FN(const Fix* fix, ExecutionSpace space)  \
    {                                                                         \
        return wrap(fix, SELECTOR, space);                                   \
    }

// finally, all the function instances to pack arrays into DLManagedTensor structs
DLEXT_PROPERTY_FROM_FIX(positions, kPositions)
DLEXT_PROPERTY_FROM_FIX(velocities, kVelocities)
DLEXT_PROPERTY_FROM_FIX(masses, kMasses)
DLEXT_PROPERTY_FROM_FIX(forces, kForces)
DLEXT_PROPERTY_FROM_FIX(images, kImages)
DLEXT_PROPERTY_FROM_FIX(tags, kTags)
DLEXT_PROPERTY_FROM_FIX(tags_map, kTagsMap)
DLEXT_PROPERTY_FROM_FIX(types, kTypes)
DLEXT_PROPERTY_FROM_FIX(virial, kVirial)

#undef DLEXT_PROPERTY

}  // namespace dlext
}  // namespace LAMMPS_NS

#endif  // LAMMPS_DLPACK_EXTENSION_H_
