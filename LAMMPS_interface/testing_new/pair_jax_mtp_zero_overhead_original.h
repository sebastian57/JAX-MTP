/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Zero Overhead JAX Implementation Header
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(jax/mtp_zero_overhead_original,PairJaxMTPZeroOverheadOriginal);
// clang-format on
#else

#ifndef LMP_PAIR_JAX_MTP_ZERO_OVERHEAD_ORIGINAL_H
#define LMP_PAIR_JAX_MTP_ZERO_OVERHEAD_ORIGINAL_H

#include "pair.h"
#include <memory>

// Forward declarations
namespace ZeroOverheadOriginal {
  class ZeroOverheadOriginalContext;
  class OverheadOriginalProfiler;
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

 protected:
  void allocate();
  
  // JAX function management
  char *jax_function_path;
  double cutoff;
  int max_atoms, max_neighbors;
  
  // Debug and profiling control
  int debug_level;
  enum DebugLevel {
    DEBUG_NONE = 0,
    DEBUG_BASIC = 1,
    DEBUG_TIMING = 2,
    DEBUG_DATA = 3,
    DEBUG_FULL = 4
  };
  
  // Zero overhead system
  ZeroOverheadOriginal::ZeroOverheadOriginalContext* zero_overhead_context;
  std::unique_ptr<ZeroOverheadOriginal::OverheadOriginalProfiler> overhead_profiler;
  
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
