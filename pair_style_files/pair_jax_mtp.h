/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   JAX-MTP Implementation
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(jax/mtp, PairJaxMTP)
#else
#ifndef LMP_PAIR_JAX_MTP_H
#define LMP_PAIR_JAX_MTP_H

#include "pair.h"
#include <memory>
#include <vector>

namespace ZeroOverhead {
  class ZeroOverheadContext;
}

namespace LAMMPS_NS {

class PairJaxMTP : public Pair {
 public:
  PairJaxMTP(class LAMMPS *);
  ~PairJaxMTP() override;
  
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

private:
  char *jax_function_path;
  double cutoff;
  int max_atoms, max_neighbors;
  int debug_level;
  ZeroOverhead::ZeroOverheadContext* zero_overhead_context;
  long long total_calls;
  bool python_initialized;
  
  double(**persistent_forces_array);
  std::vector<std::vector<double>> persistent_position_data;
  std::vector<std::vector<int>> persistent_neighbor_data;
  std::vector<std::vector<int>> persistent_neighbor_type_data;

  void allocate();
  void initialize_persistent_arrays();
  void init_python_direct();
  void cleanup_python();
};

} // namespace LAMMPS_NS
#endif
#endif
