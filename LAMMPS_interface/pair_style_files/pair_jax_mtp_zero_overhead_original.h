/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Zero Overhead JAX-MTP Implementation - Refactored Version 5.1 (Final)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(jax/mtp_zero_overhead_original, PairJaxMTPZeroOverheadOriginal)
#else
#ifndef LMP_PAIR_JAX_MTP_ZERO_OVERHEAD_ORIGINAL_H
#define LMP_PAIR_JAX_MTP_ZERO_OVERHEAD_ORIGINAL_H

#include "pair.h"
#include <memory>
#include <vector>

namespace ZeroOverheadOriginal {
  class ZeroOverheadOriginalContext;
}

namespace LAMMPS_NS {

class PairJaxMTPZeroOverheadOriginal : public Pair {
 public:
  PairJaxMTPZeroOverheadOriginal(class LAMMPS *);
  ~PairJaxMTPZeroOverheadOriginal() override;
  
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
  ZeroOverheadOriginal::ZeroOverheadOriginalContext* zero_overhead_context;
  long long total_calls;
  bool python_initialized;
  
  double(**persistent_forces_array);
  std::vector<std::vector<double>> persistent_position_data;
  // --- Reverted to 32-bit integers ---
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