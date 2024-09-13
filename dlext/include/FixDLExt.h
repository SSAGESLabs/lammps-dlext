// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef DLEXT_SAMPLER_H_
#define DLEXT_SAMPLER_H_

#include "dlpack/dlpack.h"
#include "fix.h"
#include <functional>
#include "LAMMPSView.h"

namespace LAMMPS_NS
{
namespace dlext
{

// { // Aliases

using TimeStep = bigint;  // bigint depends on how LAMMPS was built
using DLExtCallback = std::function<void(TimeStep)>;
using DLExtSetVirial = std::function<void(double*)>;

// } // Aliases

//!
//!  FixDLExt is essentially a LAMMPS fix that allows an external callback
//!  to access and or modify the atom information. The callback interface
//!  allows for complete flexibility on what code to execute during its call,
//!  so it's advised to use it with caution.
//!
//!  NOTE: A closely related example is the existing FixExternal in LAMMPS
//!    (docs.lammps.org/fix_external.html)
//!
class DEFAULT_VISIBILITY FixDLExt : public Fix {
public:
    //! Constructor
    FixDLExt(LAMMPS* lmp, int narg, char** arg);

    int setmask() override;
    void post_force(int) override;
    void set_callback(DLExtCallback& cb);
    void set_virial_callback(DLExtSetVirial& cb);
    void set_virial_global(int flag) { virial_global_flag = flag; }
    void set_view(LAMMPSView _view);
    LAMMPSView get_view() const;

protected:
    DLExtCallback callback = [](TimeStep) { };
    DLExtSetVirial setVirial = [](double*) { };
    LAMMPSView view;
};

void register_FixDLExt(LAMMPS* lmp);

}  // namespace dlext
}  // namespace LAMMPS_NS

#endif  // DLEXT_SAMPLER_H_
