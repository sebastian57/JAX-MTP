/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Zero Overhead JAX Implementation Header
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(jax/mtp_zero_overhead,PairJaxMTPZeroOverhead);
// clang-format on
#else

#ifndef LMP_PAIR_JAX_MTP_ZERO_OVERHEAD_H
#define LMP_PAIR_JAX_MTP_ZERO_OVERHEAD_H

#include "pair.h"
#include <memory>

// Forward declarations
namespace ZeroOverhead {
  class ZeroOverheadContext;
  class OverheadProfiler;
}

namespace LAMMPS_NS {

class PairJaxMTPZeroOverhead : public Pair {
 public:
  PairJaxMTPZeroOverhead(class LAMMPS *);
  ~PairJaxMTPZeroOverhead() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

 protected:
  void allocate();
  
  // JAX function management
  char *jax_function_path;
  double cutoff;
  int max_atoms, max_neighbors;
  
  // Zero overhead system
  ZeroOverhead::ZeroOverheadContext* zero_overhead_context;
  std::unique_ptr<ZeroOverhead::OverheadProfiler> overhead_profiler;
  
  // Performance tracking
  int total_calls;
  double total_computation_time;
  double total_overhead_time;
  
  // Python integration
  bool python_initialized;
  
  // Pre-allocated persistent data structures (eliminates allocation overhead)
  double** persistent_forces_array;
  std::vector<std::vector<double>> persistent_position_data;
  std::vector<std::vector<int>> persistent_neighbor_data; 
  std::vector<std::vector<int>> persistent_neighbor_type_data;
  
  void init_python_direct();
  void cleanup_python();
  void initialize_persistent_arrays();
};

}

#endif
#endif