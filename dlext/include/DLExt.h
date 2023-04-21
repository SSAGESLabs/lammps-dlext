// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef LAMMPS_DLPACK_EXTENSION_H_
#define LAMMPS_DLPACK_EXTENSION_H_

#include "cxx11utils.h"
#include "dlpack/dlpack.h"
#include "lammps.h"
#include "KOKKOS/kokkos_type.h"

#include <type_traits>
#include <vector>

// DLPack: https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h

namespace LAMMPS_NS
{
namespace dlext
{

using namespace Kokkos;

// { // Aliases

using DLManagedTensorPtr = DLManagedTensor*;
using DLManagedTensorDeleter = void (*)(DLManagedTensorPtr);

// see kokkos_type.h for example of tdual_array
template <typename T>
using tdual_array = Kokkos::DualView<T, Kokkos::LayoutRight, Kokkos::CudaSpace>;

//template <typename T>
//using ArrayHandleUPtr = std::unique_ptr<tdual_array<T>>;
template <typename T>
using ArrayHandleUPtr = std::unique_ptr<T*>;

// } // Aliases

// { // Constants

constexpr uint8_t kBits = 64; // std::is_same<Scalar, float>::value ? 32 : 64;

constexpr DLManagedTensor kInvalidDLManagedTensor {
    DLTensor {
        nullptr,  // data
        DLDevice { kDLExtDev, -1 },  // device
        -1,  // ndim
        DLDataType { 0, 0, 0 },  // dtype
        nullptr,  // shape
        nullptr,  // stride
        0  // byte_offset
    },
    nullptr,
    nullptr
};

// } // Constants

template <typename T>
struct DLDataBridge {
    ArrayHandleUPtr<T> handle;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLManagedTensor tensor;

    DLDataBridge(void* _handle)
        : handle { std::move(handle) }
    { }
};

template <typename T>
using DLDataBridgeUPtr = std::unique_ptr<DLDataBridge<T>>;

template <typename T>
void delete_bridge(DLManagedTensorPtr tensor)
{
    if (tensor)
        delete static_cast<DLDataBridge<T>*>(tensor->manager_ctx);
}

//void do_not_delete(DLManagedTensorPtr tensor) { }

typedef double Scalar;
struct Scalar3 { Scalar x, y, z; };
struct Scalar4 { Scalar x, y, z, w; };

// cast T* into to void* 
template <typename T>
inline void* opaque(T* data) { return static_cast<void*>(data); }

template <typename T>
inline void* opaque(const T* data) { return (void*)(data); }

template <typename>
constexpr DLDataType dtype();
template <>
constexpr DLDataType dtype<Scalar4>() { return DLDataType { kDLFloat, kBits, 1 }; }
template <>
constexpr DLDataType dtype<Scalar3>() { return DLDataType { kDLFloat, kBits, 1 }; }
template <>
constexpr DLDataType dtype<Scalar>() { return DLDataType { kDLFloat, kBits, 1 }; }
template <>
constexpr DLDataType dtype<int3>() { return DLDataType { kDLInt, 32, 1 }; }
template <>
constexpr DLDataType dtype<unsigned int>() { return DLDataType { kDLUInt, 32, 1 }; }

template <typename>
constexpr int64_t stride1();
template <>
constexpr int64_t stride1<Scalar4>() { return 4; }
template <>
constexpr int64_t stride1<Scalar3>() { return 3; }
template <>
constexpr int64_t stride1<Scalar>() { return 1; }
template <>
constexpr int64_t stride1<int3>() { return 3; }
template <>
constexpr int64_t stride1<unsigned int>() { return 1; }

}  // namespace dlext

} // namespace LAMMPS_NS

#endif  // LAMMPS_DLPACK_EXTENSION_H_
