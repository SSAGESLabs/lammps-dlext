// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#include "FixDLExt.h"

#include "accelerator_kokkos.h"
#include "error.h"
#include "update.h"

namespace LAMMPS_NS
{
namespace dlext
{

FixDLExt::FixDLExt(LAMMPS* lmp, int narg, char** arg)
    : Fix(lmp, narg, arg)
{
    auto on_host = true;
    auto bad_args = narg != 3 && narg != 5;
    if (narg == 5) {
        auto on_device = strcmp(arg[4], "device") == 0;
        on_host = strcmp(arg[4], "host") == 0;
        bad_args |= strcmp(arg[3], "space") != 0;
        bad_args |= !on_host && !on_device;
    }
    if (bad_args)
        error->all(FLERR, "Illegal fix dlext command");

    if (!atom->tag_enable)
        error->all(FLERR, "Fix dlext requires atoms to have IDs");

    if (atom->map_style != Atom::MAP_ARRAY)
        error->all(FLERR, "Fix dlext requires to map atoms as arrays");

    // signal that this fix contributes to the global virial
    virial_global_flag = 1;
    thermo_virial = 1;

    kokkosable = has_kokkos_cuda_enabled(lmp);
    atomKK = dynamic_cast<AtomKokkos*>(atom);
    execution_space = (on_host || !kokkosable) ? kOnHost : kOnDevice;
    datamask_read = EMPTY_MASK;
    datamask_modify = EMPTY_MASK;
}

int FixDLExt::setmask() { return FixConst::POST_FORCE; }
void FixDLExt::post_force(int vflag) 
{
    // virial setup
    v_init(vflag);

    // invoke callback
    callback(update->ntimestep);

    // put the virial from the bias into this fix's member variable virial[6] (see fix.h)
    setVirial(virial);
}
void FixDLExt::set_callback(DLExtCallback& cb) { callback = cb; }

void FixDLExt::set_virial_callback(DLExtSetVirial& cb) { setVirial = cb; }

void register_FixDLExt(LAMMPS* lmp)
{
    auto fix_map = lmp->modify->fix_map;
    (*fix_map)[std::string("dlext")] = [](LAMMPS* lmp, int narg, char** arg) -> Fix* {
        return new FixDLExt(lmp, narg, arg);
    };
}

} // dlext
} // LAMMPS_NS
