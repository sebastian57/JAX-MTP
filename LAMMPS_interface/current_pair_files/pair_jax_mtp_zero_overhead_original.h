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
#include <map>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <array>
#include <set>
#include <algorithm>

#include "mlip2_neighbor_builder.hpp"

// Forward declarations
namespace ZeroOverheadOriginal {
  class ZeroOverheadOriginalContext;
  class OverheadOriginalProfiler;
  class ZeroOverheadOriginalManager;
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
    // MLIP2 neighbor builder
    MLIP2Utils::MLIP2NeighborBuilder mlip2_builder;

 protected:
  void allocate();
  
  // FIXED: Add missing function declarations
  void initialize_persistent_arrays();
  bool process_multi_batch_system(
    ZeroOverheadOriginal::ZeroOverheadOriginalManager* manager,
    const char* jax_function_path,
    double cutoff,
    const std::vector<int>& atom_types_vec,
    const std::vector<int>& neighbor_counts, 
    const std::vector<const double*>& atom_positions,
    const std::vector<int>& atom_types,
    const std::vector<const int*>& neighbor_lists,
    const std::vector<int>& neighbor_counts_ref,
    const std::vector<const int*>& neighbor_types_lists,
    double& total_energy,
    double** forces,
    double* virial
  );
  
  // Force validation function for debugging
  void validate_force_corrections(int actual_atoms, double** forces, 
                                 const std::vector<int>& neighbor_counts);

  // Fast LAMMPS neighbor processing (replaces slow ASE algorithm)
  void build_jax_data_from_lammps_neighbors(
    class NeighList* list,
    double** x,
    int* type,
    double cutoff_distance,  // ADD CUTOFF PARAMETER
    std::vector<const double*>& atom_positions,
    std::vector<int>& atom_types_vec,
    std::vector<const int*>& neighbor_lists,
    std::vector<int>& neighbor_counts,
    std::vector<const int*>& neighbor_types_lists
  );
  
  // Helper function to map ghost atoms to local equivalents
  void build_ghost_to_local_mapping(
    double** x,
    int* ilist,
    int inum,
    std::map<int, int>& ghost_to_local
  );
  
  // ASE-compatible neighbor finder (ensures exact same neighbors as ASE)
  void find_ase_compatible_neighbors(
    int ii, int i, double** x, int* type, int inum, int* ilist, double cutoff_distance,
    int& neighbor_count, std::vector<double>& pos_storage, 
    std::vector<int>& neighbor_storage, std::vector<int>& type_storage
  );
  
  // PHASE 1: Smart boundary detection and optimization
  bool is_near_periodic_boundary(int ii, int i, double** x, double cutoff_distance);
  void determine_relevant_periodic_images(int ii, int i, double** x, double cutoff_distance, 
                                         std::vector<std::array<int,3>>& relevant_images);
  
  // PHASE 2: Hybrid LAMMPS+ASE neighbor processing (linear scaling)
  void build_jax_data_hybrid_optimized(
    NeighList* list, double** x, int* type, double cutoff_distance,
    std::vector<const double*>& atom_positions, std::vector<int>& atom_types_vec,
    std::vector<const int*>& neighbor_lists, std::vector<int>& neighbor_counts,
    std::vector<const int*>& neighbor_types_lists
  );
  
  int process_lammps_neighbors_with_cutoff(int ii, int i, NeighList* list, double** x, 
                                          int* type, double cutoff_distance,
                                          std::vector<double>& pos_storage, 
                                          std::vector<int>& neighbor_storage,
                                          std::vector<int>& type_storage);
  
  void add_missing_periodic_neighbors_targeted(int ii, int i, double** x, int* type, 
                                              int inum, int* ilist, double cutoff_distance,
                                              std::vector<double>& pos_storage, 
                                              std::vector<int>& neighbor_storage,
                                              std::vector<int>& type_storage,
                                              int& neighbor_count);
  
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
  
  // FIXED: Add missing member variables for persistent arrays
  double(**persistent_forces_array);
  std::vector<std::vector<double>> persistent_position_data;
  std::vector<std::vector<int>> persistent_neighbor_data;
  std::vector<std::vector<int>> persistent_neighbor_type_data;
  
  // Timing variables
  std::chrono::high_resolution_clock::time_point compute_start;
  
  // Initialization and cleanup
  void init_python_direct();
  void cleanup_python();
};

}

#endif
#endif