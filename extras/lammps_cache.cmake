# preset that enables KOKKOS and selects CUDA compilation with OpenMP
# enabled as well. This preselects CC 5.0 as default GPU arch, since
# that is compatible with all higher CC, but not the default CC 3.5
set(PKG_KOKKOS ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_CUDA   ON CACHE BOOL "" FORCE)
set(Kokkos_ARCH_PASCAL60 ON CACHE BOOL "" FORCE)
set(BUILD_MPI            ON CACHE BOOL "" FORCE)
set(BUILD_OMP            OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_LIBDL  OFF CACHE BOOL "" FORCE) 
#get_filename_component(NVCC_WRAPPER_CMD ${CMAKE_CURRENT_SOURCE_DIR}/../lib/kokkos/bin/nvcc_wrapper ABSOLUTE)
get_filename_component(NVCC_WRAPPER_CMD /home/ndtrung/Codes/lammps-git/lib/kokkos/bin/nvcc_wrapper ABSOLUTE)
set(CMAKE_CXX_COMPILER ${NVCC_WRAPPER_CMD} CACHE FILEPATH "" FORCE)

set(PKG_ASPHERE    ON CACHE BOOL "" FORCE) 
set(PKG_MOLECULE   ON CACHE BOOL "" FORCE) 
set(PKG_KSPACE     ON CACHE BOOL "" FORCE) 
set(PKG_REPLICA    ON CACHE BOOL "" FORCE) 
set(PKG_RIGID      ON CACHE BOOL "" FORCE) 

set(PKG_PYTHON     ON CACHE BOOL "" FORCE) 
set(BUILD_SHARED_LIBS    ON CACHE BOOL "" FORCE) 