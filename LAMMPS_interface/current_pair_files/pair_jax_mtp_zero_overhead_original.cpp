/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Zero Overhead JAX Implementation
   
   Eliminates 70% CPU overhead through:
   - Persistent memory pools (no allocation per timestep)
   - Zero-copy data paths (direct memory mapping)  
   - Optimized JAX integration (cached functions, batched calls)
   - Performance profiling and monitoring
   
   Based on: GROMACS GPU optimizations, HOOMD-blue GPU architecture
------------------------------------------------------------------------- */

#include "pair_jax_mtp_zero_overhead_original.h"
#include "zero_overhead_buffer_manager_original.hpp"

#include <map>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>

// Add missing includes for error formatting
#include "fmt/format.h"

using namespace LAMMPS_NS;
using namespace ZeroOverheadOriginal;

/* ---------------------------------------------------------------------- */

PairJaxMTPZeroOverheadOriginal::PairJaxMTPZeroOverheadOriginal(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  
  jax_function_path = nullptr;
  cutoff = 0.0;  
  max_atoms = 0;
  max_neighbors = 0;
  debug_level = DEBUG_NONE;
  
  // Zero overhead system
  zero_overhead_context = nullptr;
  overhead_profiler = std::make_unique<ZeroOverheadOriginal::OverheadOriginalProfiler>();
  
  // Performance tracking
  total_calls = 0;
  total_computation_time = 0.0;
  total_overhead_time = 0.0;
  
  // Python integration
  python_initialized = false;
  
  // FIXED: Initialize new member variables
  persistent_forces_array = nullptr;
  persistent_position_data.clear();
  persistent_neighbor_data.clear();
  persistent_neighbor_type_data.clear();
  
  if (comm->me == 0) {
    utils::logmesg(lmp, "✅ Zero Overhead JAX-MTP initialized\n");
    utils::logmesg(lmp, "   Target: 80%+ GPU utilization, <20% CPU overhead\n");
  }
}

/* ---------------------------------------------------------------------- */

PairJaxMTPZeroOverheadOriginal::~PairJaxMTPZeroOverheadOriginal()
{
  cleanup_python();
  delete[] jax_function_path;
  
  // Clean up persistent arrays
  if (persistent_forces_array) {
    memory->destroy(persistent_forces_array);
    persistent_forces_array = nullptr;
  }
  
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  
  // Print performance summary using comprehensive timing
  if (total_calls > 0 && comm->me == 0) {
    // Get comprehensive timing statistics from zero overhead manager
    auto* manager = zero_overhead_context ? zero_overhead_context->get_manager() : nullptr;
    if (manager) {
      auto stats = manager->get_comprehensive_timing_stats();
      
      // Calculate accurate breakdown
      double jax_computation_time_ms = stats.avg_jax_call_ms; // Actual JAX computation time
      double overhead_time_ms = stats.avg_total_time_ms - jax_computation_time_ms; // Everything else is overhead
      
      utils::logmesg(lmp, "\\n=== COMPREHENSIVE ZERO OVERHEAD JAX-MTP PERFORMANCE SUMMARY ===\\n");
      utils::logmesg(lmp, "Total JAX calls: {}\\n", stats.total_calls);
      utils::logmesg(lmp, "Average total time per call: {:.3f} ms\\n", stats.avg_total_time_ms);
      utils::logmesg(lmp, "Average JAX computation: {:.3f} ms\\n", jax_computation_time_ms);
      utils::logmesg(lmp, "Average overhead: {:.3f} ms\\n", overhead_time_ms);
      utils::logmesg(lmp, "\\nDetailed Breakdown:\\n");
      utils::logmesg(lmp, "  Data preparation: {:.3f} ms\\n", stats.avg_data_prep_ms);
      utils::logmesg(lmp, "  JAX initialization: {:.3f} ms\\n", stats.avg_jax_init_ms);
      utils::logmesg(lmp, "  JAX computation: {:.3f} ms\\n", stats.avg_jax_call_ms);
      utils::logmesg(lmp, "  Result processing: {:.3f} ms\\n", stats.avg_result_processing_ms);
      utils::logmesg(lmp, "\\nTotal times (for {} calls):\\n", stats.total_calls);
      utils::logmesg(lmp, "  Total JAX computation: {:.1f} ms\\n", stats.avg_jax_call_ms * stats.total_calls);
      utils::logmesg(lmp, "  Total overhead: {:.1f} ms\\n", overhead_time_ms * stats.total_calls);
      utils::logmesg(lmp, "  Total function time: {:.1f} ms\\n", stats.total_time_ms);
      utils::logmesg(lmp, "Target achieved: {}\\n", 
                    overhead_time_ms < 25.0 ? "YES ✅" : "NO ❌ (needs optimization)");
    } else {
      utils::logmesg(lmp, "\\n=== ZERO OVERHEAD JAX-MTP PERFORMANCE SUMMARY ===\\n");
      utils::logmesg(lmp, "No comprehensive timing data available\\n");
    }
  }
  
  // Cleanup zero overhead system
  if (zero_overhead_context) {
    delete zero_overhead_context;
    zero_overhead_context = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

void PairJaxMTPZeroOverheadOriginal::compute(int eflag, int vflag)
{
  // Start timing
  compute_start = std::chrono::high_resolution_clock::now();
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  NeighList *neighlist = neighbor->lists[0];
  if (!neighlist) error->all(FLERR, "PairJaxMTPZeroOverheadOriginal: neighbor->lists[0] is null");
  int inum = neighlist->inum;
  int *ilist = neighlist->ilist;
  int *numneigh = neighlist->numneigh;
  int **firstneigh = neighlist->firstneigh;
  
  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr = 0;
  
  // Initialize persistent arrays if needed
  initialize_persistent_arrays();
  
  // FIXED: Declare all variables at function scope so they're accessible throughout
  int natoms_actual = std::min(inum, max_atoms);
  int nneigh_actual = 0;
  
  // Data preparation arrays - declare at function scope
  std::vector<const double*> atom_positions;
  std::vector<int> atom_types_vec;
  std::vector<const int*> neighbor_lists;
  std::vector<int> neighbor_counts;
  std::vector<const int*> neighbor_types_lists;
  
  // Data preparation with zero-copy optimization
  {
    // Calculate lattice matrix
    double lattice[3][3];
    if (domain->triclinic) {
      lattice[0][0] = domain->h[0];  lattice[0][1] = domain->h[5];  lattice[0][2] = domain->h[4];
      lattice[1][0] = 0.0;           lattice[1][1] = domain->h[1];  lattice[1][2] = domain->h[3];
      lattice[2][0] = 0.0;           lattice[2][1] = 0.0;           lattice[2][2] = domain->h[2];
    } else {
      lattice[0][0] = domain->xprd;  lattice[0][1] = 0.0;           lattice[0][2] = 0.0;
      lattice[1][0] = 0.0;           lattice[1][1] = domain->yprd;  lattice[1][2] = 0.0;
      lattice[2][0] = 0.0;           lattice[2][1] = 0.0;           lattice[2][2] = domain->zprd;
    }
    
    // Debug atom indexing if needed
    if (debug_level >= DEBUG_BASIC && comm->me == 0) {
      utils::logmesg(lmp, "🔍 DEBUGGING ATOM INDEXING:\n");
      utils::logmesg(lmp, "  natoms_actual = {}\n", natoms_actual);
      utils::logmesg(lmp, "  First 10 ilist values: ");
      for (int ii = 0; ii < std::min(natoms_actual, 10); ii++) {
        utils::logmesg(lmp, "{} ", ilist[ii]);
      }
      utils::logmesg(lmp, "\n  Last 10 ilist values: ");
      for (int ii = std::max(0, natoms_actual-10); ii < natoms_actual; ii++) {
        utils::logmesg(lmp, "{} ", ilist[ii]);
      }
      utils::logmesg(lmp, "\n");
    }
    
    // ALGORITHM SELECTION: Pure ASE approach for optimization testing
    const bool use_hybrid_algorithm = false;  // DISABLED: Switch to pure ASE for optimization
    const bool use_lammps_neighbors = false;  // Legacy LAMMPS-only approach  
    const bool compare_both_methods = false;  // No comparison needed
    
    if (compare_both_methods) {
      // COMPARISON MODE: Run both ASE and LAMMPS methods
      if (comm->me == 0) {
        utils::logmesg(lmp, "🔍 COMPARISON MODE: Running both ASE and LAMMPS neighbor processing\n");
      }
      
      // First run ASE algorithm
      std::vector<const double*> ase_atom_positions;
      std::vector<int> ase_atom_types_vec;
      std::vector<const int*> ase_neighbor_lists;
      std::vector<int> ase_neighbor_counts;
      std::vector<const int*> ase_neighbor_types_lists;
      
      std::vector<double*> lammps_positions(natoms_actual);
      std::vector<int> lammps_types(natoms_actual);
      
      for (int ii = 0; ii < natoms_actual; ii++) {
        int i = ilist[ii];
        lammps_positions[ii] = x[i];
        lammps_types[ii] = type[i];
      }
      
      mlip2_builder.build_neighbor_lists(
        lammps_positions.data(), lammps_types.data(), natoms_actual,
        lattice, cutoff, ilist, numneigh, firstneigh,
        ase_atom_positions, ase_atom_types_vec, ase_neighbor_lists, 
        ase_neighbor_counts, ase_neighbor_types_lists, debug_level
      );
      
      // Then run LAMMPS algorithm  
      build_jax_data_from_lammps_neighbors(
        neighlist, x, type, cutoff, atom_positions, atom_types_vec, 
        neighbor_lists, neighbor_counts, neighbor_types_lists
      );
      
      // Compare results
      if (comm->me == 0) {
        utils::logmesg(lmp, "📊 COMPARISON RESULTS:\n");
        utils::logmesg(lmp, "  ASE: {} atoms, {} neighbors (atom 0: {})\n", 
                      ase_atom_positions.size(), ase_neighbor_counts[0], 
                      ase_neighbor_counts.size() > 0 ? ase_neighbor_counts[0] : 0);
        utils::logmesg(lmp, "  LAMMPS: {} atoms, {} neighbors (atom 0: {})\n", 
                      atom_positions.size(), neighbor_counts[0],
                      neighbor_counts.size() > 0 ? neighbor_counts[0] : 0);
        
        // Compare first few neighbor indices
        for (int ii = 0; ii < std::min(2, (int)atom_positions.size()); ii++) {
          utils::logmesg(lmp, "  Atom[{}] neighbor indices:\n", ii);
          utils::logmesg(lmp, "    ASE: ");
          for (int jj = 0; jj < std::min(5, ase_neighbor_counts[ii]); jj++) {
            utils::logmesg(lmp, "{} ", ase_neighbor_lists[ii][jj]);
          }
          utils::logmesg(lmp, "\n    LAMMPS: ");
          for (int jj = 0; jj < std::min(5, neighbor_counts[ii]); jj++) {
            utils::logmesg(lmp, "{} ", neighbor_lists[ii][jj]);
          }
          utils::logmesg(lmp, "\n");
        }
      }
    } else if (use_hybrid_algorithm) {
      // NEW HYBRID LAMMPS+ASE algorithm (linear scaling with ASE compatibility)
      build_jax_data_hybrid_optimized(
        neighlist, x, type, cutoff, atom_positions, atom_types_vec, 
        neighbor_lists, neighbor_counts, neighbor_types_lists
      );
    } else if (use_lammps_neighbors) {
      // LEGACY LAMMPS neighbor processing
      build_jax_data_from_lammps_neighbors(
        neighlist, x, type, cutoff, atom_positions, atom_types_vec, 
        neighbor_lists, neighbor_counts, neighbor_types_lists
      );
    } else {
      // LEGACY ASE algorithm (O(N²×27) scaling)
      std::vector<double*> lammps_positions(natoms_actual);
      std::vector<int> lammps_types(natoms_actual);
      
      for (int ii = 0; ii < natoms_actual; ii++) {
        int i = ilist[ii];
        lammps_positions[ii] = x[i];
        lammps_types[ii] = type[i];
      }
      
      mlip2_builder.build_neighbor_lists(
        lammps_positions.data(), lammps_types.data(), natoms_actual,
        lattice, cutoff, ilist, numneigh, firstneigh,
        atom_positions, atom_types_vec, neighbor_lists, 
        neighbor_counts, neighbor_types_lists, debug_level
      );
    }
    
    // Calculate total neighbors for JAX call (using actual processed atoms)
    nneigh_actual = 0;
    int actual_processed_atoms = atom_positions.size();  // May be < natoms_actual if truncated
    for (int i = 0; i < actual_processed_atoms; i++) {
      nneigh_actual += neighbor_counts[i];
    }
    natoms_actual = actual_processed_atoms;  // Update to actual processed atoms
    
    // Debug output
    if (debug_level >= DEBUG_BASIC && comm->me == 0 && total_calls < 3) {
      utils::logmesg(lmp, "JAX-compatible neighbor data prepared (FAST LAMMPS processing):\n");
      utils::logmesg(lmp, "  Atoms processed: {}\n", natoms_actual);
      utils::logmesg(lmp, "  Total neighbors: {} (LAMMPS neighbor list)\n", nneigh_actual);
      utils::logmesg(lmp, "  Cutoff used: {:.3f} Å\n", cutoff);
    }
  }
    
  // Zero-copy JAX computation
  {
    double total_energy = 0.0;
    double virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    // Zero forces in persistent array
    for (int i = 0; i < natoms_actual; i++) {
      persistent_forces_array[i][0] = persistent_forces_array[i][1] = persistent_forces_array[i][2] = 0.0;
    }
    
    if (debug_level >= DEBUG_BASIC && comm->me == 0) {
      utils::logmesg(lmp, "🔍 Pre-JAX: Forces array zeroed for {} atoms\n", natoms_actual);
    }
    
    // CRITICAL FIX: Add null pointer check for zero_overhead_context
    if (zero_overhead_context == nullptr) {
      error->all(FLERR, "ZeroOverheadOriginalContext not initialized - call init_style() first");
    }
    
    ZeroOverheadOriginal::ZeroOverheadOriginalManager* manager = zero_overhead_context->get_manager();
    if (manager == nullptr) {
      error->all(FLERR, "ZeroOverheadOriginalManager is null - context initialization failed");
    }
    
    bool success = false;
    
    // Detect neighbor overflow for multi-batch processing
    std::vector<int> main_batch_atoms;
    std::vector<int> overflow_batch_atoms;
    
    for (int i = 0; i < natoms_actual; i++) {
      if (neighbor_counts[i] <= max_neighbors) {
        main_batch_atoms.push_back(i);
      } else {
        overflow_batch_atoms.push_back(i);
        if (comm->me == 0) {
          utils::logmesg(lmp, "⚠️ MULTI-BATCH: Atom {} has {} neighbors (max_neighbors={})\n", 
                        i, neighbor_counts[i], max_neighbors);
        }
      }
    }
    
    // Comprehensive interaction counting and validation
    int total_processed_neighbors = 0;
    int atoms_with_overflow = 0;
    int max_neighbors_per_atom = 0;
    
    for (int i = 0; i < natoms_actual; i++) {
      total_processed_neighbors += neighbor_counts[i];
      if (neighbor_counts[i] > max_neighbors) {
        atoms_with_overflow++;
      }
      max_neighbors_per_atom = std::max(max_neighbors_per_atom, neighbor_counts[i]);
    }
    
    if (debug_level >= DEBUG_BASIC && comm->me == 0 && total_calls < 5) {
      utils::logmesg(lmp, "   Atoms with overflow: {} out of {}\n", atoms_with_overflow, natoms_actual);
    }
    
    // Calculate volume for JAX call
    double volume;
    if (domain->triclinic) {
      volume = domain->h[0] * (domain->h[1] * domain->h[2] - domain->h[3] * domain->h[4]) -
               domain->h[5] * (domain->h[6] * domain->h[2] - domain->h[8] * domain->h[4]) +
               domain->h[7] * (domain->h[6] * domain->h[3] - domain->h[8] * domain->h[1]);
      volume = fabs(volume);
    } else {
      volume = domain->xprd * domain->yprd * domain->zprd;
    }
    
    // Process system using multi-batch if needed
    if (!overflow_batch_atoms.empty()) {
      success = process_multi_batch_system(
        manager,
        jax_function_path,
        volume,
        main_batch_atoms,
        overflow_batch_atoms,
        atom_positions,
        atom_types_vec,
        neighbor_lists,
        neighbor_counts,
        neighbor_types_lists,
        total_energy,
        persistent_forces_array,
        virial
      );
    } else {
      // Single batch processing - call JAX function directly
      
      success = manager->call_jax_ultra_optimized(
        jax_function_path,
        natoms_actual,
        nneigh_actual,  // FIX: Use total neighbors, not atom count!
        atom_positions.data(),
        atom_types_vec.data(),
        neighbor_lists.data(),
        neighbor_counts.data(),
        neighbor_types_lists.data(),
        volume,
        total_energy,
        persistent_forces_array,
        virial
      );
      
      // CRITICAL DEBUG: Show data being sent to JAX vs MLIP2 format
      if (debug_level >= DEBUG_BASIC && comm->me == 0 && total_calls < 2) {
        utils::logmesg(lmp, "🔍 JAX INPUT DATA DEBUG:\n");
        utils::logmesg(lmp, "  natoms_actual: {}\n", natoms_actual);
        utils::logmesg(lmp, "  nneigh_actual: {}\n", nneigh_actual);
        utils::logmesg(lmp, "  volume: {:.6f}\n", volume);
        
        // Debug first 3 atoms' data sent to JAX
        utils::logmesg(lmp, "  First 3 atoms sent to JAX:\n");
        for (int i = 0; i < std::min(natoms_actual, 3); i++) {
          utils::logmesg(lmp, "    Atom[{}]: type={}, neighbors={}\n", 
                        i, atom_types_vec[i], neighbor_counts[i]);
          
          // Show first 2 neighbors
          if (neighbor_counts[i] > 0) {
            utils::logmesg(lmp, "      First 2 neighbors: ");
            for (int n = 0; n < std::min(2, neighbor_counts[i]); n++) {
              int nidx = neighbor_lists[i][n];
              int ntype = neighbor_types_lists[i][n];
              double dx = atom_positions[i][n*3 + 0];
              double dy = atom_positions[i][n*3 + 1];
              double dz = atom_positions[i][n*3 + 2];
              double dist = sqrt(dx*dx + dy*dy + dz*dz);
              utils::logmesg(lmp, "[idx={},type={},dist={:.3f}] ", nidx, ntype, dist);
            }
            utils::logmesg(lmp, "\n");
          }
        }
      }
      
      if (debug_level >= DEBUG_BASIC && comm->me == 0) {
        utils::logmesg(lmp, "Single batch JAX call completed: success={}\n", success);
        utils::logmesg(lmp, "Energy returned by JAX: {:.6f} eV\n", total_energy);
        
        // DEBUG: Check forces returned by JAX
        double max_force = 0.0, total_force_norm = 0.0;
        int nonzero_forces = 0;
        for (int ii = 0; ii < std::min(natoms_actual, 5); ii++) {  // Check first 5 atoms
          double fx = persistent_forces_array[ii][0];
          double fy = persistent_forces_array[ii][1]; 
          double fz = persistent_forces_array[ii][2];
          double f_mag = sqrt(fx*fx + fy*fy + fz*fz);
          
          if (f_mag > 1e-12) nonzero_forces++;
          max_force = std::max(max_force, f_mag);
          total_force_norm += f_mag;
          
          utils::logmesg(lmp, "  JAX Force[{}]: ({:.6f}, {:.6f}, {:.6f}) |F|={:.6f}\n", 
                        ii, fx, fy, fz, f_mag);
        }
        utils::logmesg(lmp, "Force summary: max_force={:.6f}, nonzero_forces={}/{}, avg_norm={:.6f}\n",
                      max_force, nonzero_forces, std::min(natoms_actual, 5), total_force_norm/std::min(natoms_actual, 5));
      }
    }
    
    // Apply forces to LAMMPS atoms
    if (success) {
      double applied_max_force = 0.0;
      int applied_nonzero = 0;
      
      for (int ii = 0; ii < natoms_actual; ii++) {
        int i = ilist[ii];
        double old_fx = f[i][0], old_fy = f[i][1], old_fz = f[i][2];
        
        f[i][0] += persistent_forces_array[ii][0];
        f[i][1] += persistent_forces_array[ii][1];
        f[i][2] += persistent_forces_array[ii][2];
        
        // Track applied forces for debugging  
        if (ii < 5 && debug_level >= DEBUG_BASIC && comm->me == 0) {
          utils::logmesg(lmp, "  LAMMPS Force[{}]: before=({:.6f},{:.6f},{:.6f}) after=({:.6f},{:.6f},{:.6f})\n",
                        ii, old_fx, old_fy, old_fz, f[i][0], f[i][1], f[i][2]);
        }
        
        double f_mag = sqrt(f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2]);
        if (f_mag > 1e-12) applied_nonzero++;
        applied_max_force = std::max(applied_max_force, f_mag);
      }
      
      if (debug_level >= DEBUG_BASIC && comm->me == 0) {
        utils::logmesg(lmp, "Applied forces: max={:.6f}, nonzero_atoms={}/{}\n", 
                      applied_max_force, applied_nonzero, natoms_actual);
      }
    } else {
      if (comm->me == 0) {
        utils::logmesg(lmp, "❌ JAX call failed - no forces applied!\n");
      }
    }
    
    // Store energy
    if (eflag_global) eng_vdwl += total_energy;
    
    // JAX stress computation - use JAX-computed virial for accurate stress
    const bool use_jax_stress = true;
    if (vflag_global) { // was if  (use_jax_stress && vflag_global) {
      // JAX stress values are already in virial[] array from call_jax_ultra_optimized
      // No need for virial_fdotr_compute() since JAX computed the stress directly
      if (debug_level >= DEBUG_BASIC && comm->me == 0) {
        utils::logmesg(lmp, "🎯 Using JAX-computed stress (virial): [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
                      virial[0], virial[1], virial[2], virial[3], virial[4], virial[5]);
      }
    } else {
      // Use LAMMPS virial computation from forces (backup method)
      if (vflag_fdotr) virial_fdotr_compute();
      
      // Manually negate virial to match MLIP2 sign convention
      // LAMMPS computes positive virial, MLIP2 expects negative (thermodynamic stress)
      if (vflag_global) {
        for (int i = 0; i < 6; i++) {
          virial[i] = -virial[i];
        }
      }
      
      if (debug_level >= DEBUG_BASIC && comm->me == 0) {
        utils::logmesg(lmp, "⚙️ Using LAMMPS-computed stress (virial, sign-corrected): [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
                      virial[0], virial[1], virial[2], virial[3], virial[4], virial[5]);
      }
    }
  }

  // Update performance tracking
  auto compute_end = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration<double>(compute_end - compute_start).count();
  total_computation_time += total_time;
  total_calls++;

  // Periodic performance logging
  if (total_calls % 100 == 0 && debug_level >= DEBUG_TIMING) {
    if (comm->me == 0) {
      utils::logmesg(lmp, "Performance after {} calls:\n", total_calls);
      utils::logmesg(lmp, "  Average computation time: {:.6f} ms\n", 
                    (total_computation_time * 1000.0) / total_calls);
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairJaxMTPZeroOverheadOriginal::initialize_persistent_arrays()
{
  // Initialize persistent arrays if not already done
  if (persistent_forces_array == nullptr) {
    memory->create(persistent_forces_array, max_atoms, 3, "pair:persistent_forces");
  }
  
  // Resize persistent data vectors
  persistent_position_data.resize(max_atoms);
  persistent_neighbor_data.resize(max_atoms);
  persistent_neighbor_type_data.resize(max_atoms);
  
  // Reserve capacity to avoid frequent reallocations
  for (int i = 0; i < max_atoms; i++) {
    persistent_position_data[i].reserve(max_neighbors * 3);  // 3D positions
    persistent_neighbor_data[i].reserve(max_neighbors);
    persistent_neighbor_type_data[i].reserve(max_neighbors);
  }
}

/* ---------------------------------------------------------------------- */

void PairJaxMTPZeroOverheadOriginal::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal pair_style command - usage: pair_style jax/mtp_zero_overhead <bin_file> [max_atoms] [max_neighbors] [cutoff] [debug_level]");

  // Parse arguments
  int n = strlen(arg[0]) + 1;
  jax_function_path = new char[n];
  strcpy(jax_function_path, arg[0]);
  
  // Set default or user-specified system sizes
  if (narg >= 3) {
    max_atoms = utils::inumeric(FLERR, arg[1], false, lmp);
    max_neighbors = utils::inumeric(FLERR, arg[2], false, lmp);
  } else {
    // Auto-detect system size for optimal memory allocation
    max_atoms = 65536;  // Default large enough for most systems
    max_neighbors = 200;
  }
  
  // Parse cutoff (optional 4th parameter)
  if (narg >= 4) {
    cutoff = utils::numeric(FLERR, arg[3], false, lmp);
    if (cutoff <= 0.0) error->all(FLERR, "Cutoff distance must be positive");
  } else {
    cutoff = 10.0;  // Default cutoff
  }
  
  // Parse debug level (optional 5th parameter)
  if (narg >= 5) {
    debug_level = utils::inumeric(FLERR, arg[4], false, lmp);
    if (debug_level < DEBUG_NONE || debug_level > DEBUG_FULL) {
      error->all(FLERR, "Debug level must be 0-4 (0=none, 1=basic, 2=timing, 3=data, 4=full)");
    }
  } else {
    debug_level = DEBUG_NONE;
  }

  if (max_atoms <= 0) error->all(FLERR, "Maximum atoms must be positive");
  if (max_neighbors <= 0) error->all(FLERR, "Maximum neighbors must be positive");

  if (comm->me == 0) {
    utils::logmesg(lmp, "Zero Overhead JAX/MTP settings:\\n");
    utils::logmesg(lmp, "  JAX function: {}\\n", jax_function_path);
    utils::logmesg(lmp, "  System capacity: {} atoms × {} neighbors\\n", max_atoms, max_neighbors);
    utils::logmesg(lmp, "  Cutoff distance: {:.3f} Å (user-specified)\\n", cutoff);
    utils::logmesg(lmp, "  Debug level: {} ", debug_level);
    
    const char* debug_names[] = {"NONE", "BASIC", "TIMING", "DATA", "FULL"};
    if (debug_level >= 0 && debug_level <= 4) {
      utils::logmesg(lmp, "({})\\n", debug_names[debug_level]);
    } else {
      utils::logmesg(lmp, "(INVALID)\\n");
    }
    
    utils::logmesg(lmp, "  Optimization: Zero overhead persistent memory\\n");
    
    if (debug_level >= DEBUG_BASIC) {
      utils::logmesg(lmp, "  Enhanced profiling and debugging enabled\\n");
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairJaxMTPZeroOverheadOriginal::coeff(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = 1; j <= atom->ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutoff * cutoff;
    }
  }
}

double PairJaxMTPZeroOverheadOriginal::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  return cutoff;
}

void PairJaxMTPZeroOverheadOriginal::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style jax/mtp_zero_overhead requires atom IDs");

  if (max_atoms <= 0 || max_neighbors <= 0) {
    error->all(FLERR, "Array sizes not properly set - check pair_style arguments");
  }

  neighbor->add_request(this, NeighConst::REQ_FULL);
  init_python_direct();
  
  // CRITICAL FIX: Initialize zero_overhead_context
  if (zero_overhead_context == nullptr) {
    try {
      zero_overhead_context = new ZeroOverheadOriginal::ZeroOverheadOriginalContext(max_atoms, max_neighbors, 1);
      if (!zero_overhead_context->is_ready()) {
        delete zero_overhead_context;
        zero_overhead_context = nullptr;
        error->all(FLERR, "ZeroOverheadOriginalContext failed to initialize properly");
      }
      if (comm->me == 0) {
        utils::logmesg(lmp, "✅ ZeroOverheadOriginalContext initialized (max_atoms={}, max_neighbors={})\n", max_atoms, max_neighbors);
      }
    } catch (const std::exception& e) {
      error->all(FLERR, fmt::format("Failed to initialize ZeroOverheadOriginalContext: {}", e.what()));
    }
  }
  
  if (comm->me == 0) {
    utils::logmesg(lmp, "✅ Zero Overhead JAX-MTP initialization complete\n");
    utils::logmesg(lmp, "   Ready for high-performance GPU computation\n");
  }
}

void PairJaxMTPZeroOverheadOriginal::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* ---------------------------------------------------------------------- */

void PairJaxMTPZeroOverheadOriginal::init_python_direct()
{
  if (python_initialized) return;
  
  if (comm->me == 0) {
    utils::logmesg(lmp, "Initializing Python environment for zero overhead...\\n");
  }

  try {
    if (!Py_IsInitialized()) {
      Py_Initialize();
      
      // Import numpy through Python (avoid C API macro issues)
      PyRun_SimpleString("import numpy as np");
      if (PyErr_Occurred()) {
        PyErr_Print();
        error->all(FLERR, "Failed to import NumPy");
        return;
      }
    }
    
    python_initialized = true;
    
    if (comm->me == 0) {
      utils::logmesg(lmp, "✅ Python environment ready for zero overhead\\n");
    }
    
  } catch (const std::exception& e) {
    error->all(FLERR, fmt::format("Python initialization failed: {}", e.what()));
  }
}

void PairJaxMTPZeroOverheadOriginal::cleanup_python()
{
  if (python_initialized) {
    // Trigger explicit cleanup of zero overhead manager before Python shuts down
    try {
      auto* manager = ZeroOverheadOriginal::ZeroOverheadOriginalManager::get_instance();
      if (manager) {
        manager->cleanup();
      }
    } catch (...) {
      // Ignore cleanup exceptions to prevent segfaults
    }
    python_initialized = false;
  }
}

/* ---------------------------------------------------------------------- */

void PairJaxMTPZeroOverheadOriginal::validate_force_corrections(
    int actual_atoms, double** forces, const std::vector<int>& neighbor_counts) {
    
    // Calculate force statistics
    double total_force[3] = {0.0, 0.0, 0.0};
    double max_force = 0.0;
    double force_magnitudes_sum = 0.0;
    
    for (int i = 0; i < actual_atoms; i++) {
        double fx = forces[i][0];
        double fy = forces[i][1]; 
        double fz = forces[i][2];
        double fmag = std::sqrt(fx*fx + fy*fy + fz*fz);
        
        total_force[0] += fx;
        total_force[1] += fy;
        total_force[2] += fz;
        
        max_force = std::max(max_force, fmag);
        force_magnitudes_sum += fmag;
    }
    
    double total_force_magnitude = std::sqrt(total_force[0]*total_force[0] + 
                                           total_force[1]*total_force[1] + 
                                           total_force[2]*total_force[2]);
    
    if (comm->me == 0) {  // Only master process prints
        utils::logmesg(lmp, "🔧 FORCE VALIDATION:\n");
        utils::logmesg(lmp, "   Total force magnitude: {:.6f} (should be ~0 for isolated systems)\n", 
                      total_force_magnitude);
        utils::logmesg(lmp, "   Max individual force: {:.3f} eV/Å\n", max_force);
        utils::logmesg(lmp, "   Average force magnitude: {:.3f} eV/Å\n", force_magnitudes_sum/actual_atoms);
        
        // Check for the "atom 0 gets all forces" bug
        if (actual_atoms > 0) {
            double atom0_force_mag = std::sqrt(forces[0][0]*forces[0][0] + 
                                              forces[0][1]*forces[0][1] + 
                                              forces[0][2]*forces[0][2]);
            double avg_force = force_magnitudes_sum / actual_atoms;
            double force_concentration = (avg_force > 0) ? atom0_force_mag / avg_force : 0.0;
            
            if (force_concentration > 5.0) {
                utils::logmesg(lmp, "   ⚠️  WARNING: Atom 0 force concentration = {:.1f}x average (indicates neighbor index bug)\n", 
                              force_concentration);
            } else {
                utils::logmesg(lmp, "   ✅ Force distribution looks normal\n");
            }
        }
        
        // Validate neighbor counts
        int total_neighbors = 0;
        for (int i = 0; i < actual_atoms; i++) {
            total_neighbors += neighbor_counts[i];
        }
        utils::logmesg(lmp, "   Total neighbors processed: {}\n", total_neighbors);
    }
}

/* ---------------------------------------------------------------------- */

// Helper function to build ghost atom to local atom mapping using periodic imaging
void PairJaxMTPZeroOverheadOriginal::build_ghost_to_local_mapping(
    double** x, 
    int* ilist, 
    int inum, 
    std::map<int, int>& ghost_to_local
) {
    // Build mapping from ghost atoms to their local equivalents
    // For each atom in the system (including ghosts), find which local atom it represents
    
    double xprd = domain->xprd;
    double yprd = domain->yprd; 
    double zprd = domain->zprd;
    double xprd_half = domain->xprd_half;
    double yprd_half = domain->yprd_half;
    double zprd_half = domain->zprd_half;
    
    // For all atoms in the system (including ghosts)
    for (int global_j = 0; global_j < atom->nlocal + atom->nghost; global_j++) {
        // Skip if this is already a local atom
        bool is_local = false;
        for (int ii = 0; ii < inum; ii++) {
            if (ilist[ii] == global_j) {
                is_local = true;
                break;
            }
        }
        if (is_local) continue;
        
        // This is a ghost atom - find which local atom it corresponds to
        double ghost_x = x[global_j][0];
        double ghost_y = x[global_j][1]; 
        double ghost_z = x[global_j][2];
        
        // Try to match with each local atom using periodic imaging
        for (int ii = 0; ii < inum; ii++) {
            int local_i = ilist[ii];
            double local_x = x[local_i][0];
            double local_y = x[local_i][1];
            double local_z = x[local_i][2];
            
            // Check if ghost is a periodic image of local atom
            double dx = ghost_x - local_x;
            double dy = ghost_y - local_y;
            double dz = ghost_z - local_z;
            
            // Apply periodic boundary conditions to find minimum image
            if (domain->xperiodic) {
                if (dx > xprd_half) dx -= xprd;
                else if (dx < -xprd_half) dx += xprd;
            }
            if (domain->yperiodic) {
                if (dy > yprd_half) dy -= yprd;
                else if (dy < -yprd_half) dy += yprd;
            }
            if (domain->zperiodic) {
                if (dz > zprd_half) dz -= zprd;
                else if (dz < -zprd_half) dz += zprd;
            }
            
            // If the minimum image distance is very small, they're the same atom
            double dist_sq = dx*dx + dy*dy + dz*dz;
            if (dist_sq < 1e-12) {  // Tolerance for numerical precision
                ghost_to_local[global_j] = ii;  // Map ghost to local index
                break;
            }
        }
    }
}

/* ---------------------------------------------------------------------- */

// ASE-compatible neighbor finder - ensures exact same neighbors as ASE method
void PairJaxMTPZeroOverheadOriginal::find_ase_compatible_neighbors(
    int ii, int i, double** x, int* type, int inum, int* ilist, double cutoff_distance,
    int& neighbor_count, std::vector<double>& pos_storage, 
    std::vector<int>& neighbor_storage, std::vector<int>& type_storage
) {
    neighbor_count = 0;
    
    double pos_i[3] = {x[i][0], x[i][1], x[i][2]};
    
    // Get domain lattice vectors
    double xprd = domain->xprd;
    double yprd = domain->yprd; 
    double zprd = domain->zprd;
    
    // ASE-style comprehensive search: check every local atom through all periodic images
    for (int jj = 0; jj < inum; jj++) {
        int j = ilist[jj];  // Local atom j
        if (i == j) continue;  // Skip self
        
        double pos_j[3] = {x[j][0], x[j][1], x[j][2]};
        
        // Check all 27 periodic images (like ASE does)
        for (int nx = -1; nx <= 1; nx++) {
            for (int ny = -1; ny <= 1; ny++) {
                for (int nz = -1; nz <= 1; nz++) {
                    // Calculate periodic image position
                    double image_pos[3];
                    image_pos[0] = pos_j[0] + nx * xprd;
                    image_pos[1] = pos_j[1] + ny * yprd;
                    image_pos[2] = pos_j[2] + nz * zprd;
                    
                    // Calculate distance vector (same as ASE)
                    double dx = image_pos[0] - pos_i[0];
                    double dy = image_pos[1] - pos_i[1];
                    double dz = image_pos[2] - pos_i[2];
                    double dist = sqrt(dx*dx + dy*dy + dz*dz);
                    
                    // Apply cutoff (same as ASE)
                    if (dist <= cutoff_distance && dist > 1e-10) {
                        // Store this neighbor (using local atom index j)
                        pos_storage[neighbor_count * 3 + 0] = dx;
                        pos_storage[neighbor_count * 3 + 1] = dy;
                        pos_storage[neighbor_count * 3 + 2] = dz;
                        neighbor_storage[neighbor_count] = jj;  // Use LOCAL index jj
                        type_storage[neighbor_count] = type[j] - 1;  // Convert to 0-based
                        
                        neighbor_count++;
                        
                        // Debug first few neighbors for atom 0
                        if (ii == 0 && neighbor_count <= 5 && debug_level >= DEBUG_DATA && comm->me == 0) {
                            utils::logmesg(lmp, "    ASE-style neighbor[{}]: j={}, dist={:.3f}, rij=({:.3f},{:.3f},{:.3f})\n",
                                         neighbor_count-1, jj, dist, dx, dy, dz);
                        }
                        
                        if (neighbor_count >= max_neighbors) goto next_atom;  // Avoid overflow
                        
                        break; // Found this atom, don't check more periodic images
                    }
                }
            }
        }
        next_atom:;
    }
    
    // Pad remaining slots (like ASE does)
    for (int pad = neighbor_count; pad < max_neighbors; pad++) {
        pos_storage[pad * 3 + 0] = 20.0;
        pos_storage[pad * 3 + 1] = 20.0;
        pos_storage[pad * 3 + 2] = 20.0;
        neighbor_storage[pad] = -1;
        type_storage[pad] = -1;
    }
}

/* ---------------------------------------------------------------------- */
// PHASE 1: Smart boundary detection - inspired by MLIP2's efficient approach
bool PairJaxMTPZeroOverheadOriginal::is_near_periodic_boundary(int ii, int i, double** x, double cutoff_distance) {
    double pos_i[3] = {x[i][0], x[i][1], x[i][2]};
    
    // Get domain boundaries
    double xlo = domain->boxlo[0];
    double xhi = domain->boxhi[0]; 
    double ylo = domain->boxlo[1];
    double yhi = domain->boxhi[1];
    double zlo = domain->boxlo[2];
    double zhi = domain->boxhi[2];
    
    // Check if atom is within cutoff distance of any periodic boundary
    bool near_x_boundary = (domain->xperiodic) && 
                          ((pos_i[0] - xlo) < cutoff_distance || (xhi - pos_i[0]) < cutoff_distance);
    bool near_y_boundary = (domain->yperiodic) && 
                          ((pos_i[1] - ylo) < cutoff_distance || (yhi - pos_i[1]) < cutoff_distance);
    bool near_z_boundary = (domain->zperiodic) && 
                          ((pos_i[2] - zlo) < cutoff_distance || (zhi - pos_i[2]) < cutoff_distance);
    
    return near_x_boundary || near_y_boundary || near_z_boundary;
}

/* ---------------------------------------------------------------------- */
// PHASE 1: Determine relevant periodic images - MLIP2-inspired smart selection
void PairJaxMTPZeroOverheadOriginal::determine_relevant_periodic_images(
    int ii, int i, double** x, double cutoff_distance, 
    std::vector<std::array<int,3>>& relevant_images
) {
    relevant_images.clear();
    
    double pos_i[3] = {x[i][0], x[i][1], x[i][2]};
    
    // Get domain boundaries and sizes
    double xlo = domain->boxlo[0]; double xhi = domain->boxhi[0]; double xprd = domain->xprd;
    double ylo = domain->boxlo[1]; double yhi = domain->boxhi[1]; double yprd = domain->yprd; 
    double zlo = domain->boxlo[2]; double zhi = domain->boxhi[2]; double zprd = domain->zprd;
    
    // Smart image selection: only check images that could actually contain neighbors
    for (int nx = -1; nx <= 1; nx++) {
        for (int ny = -1; ny <= 1; ny++) {
            for (int nz = -1; nz <= 1; nz++) {
                if (nx == 0 && ny == 0 && nz == 0) continue; // Skip original cell
                
                // Check if this periodic image could contain neighbors within cutoff
                bool x_relevant = !domain->xperiodic || (nx == 0) ||
                                 (nx == -1 && (pos_i[0] - xlo) < cutoff_distance) ||
                                 (nx == +1 && (xhi - pos_i[0]) < cutoff_distance);
                                 
                bool y_relevant = !domain->yperiodic || (ny == 0) ||
                                 (ny == -1 && (pos_i[1] - ylo) < cutoff_distance) ||
                                 (ny == +1 && (yhi - pos_i[1]) < cutoff_distance);
                                 
                bool z_relevant = !domain->zperiodic || (nz == 0) ||
                                 (nz == -1 && (pos_i[2] - zlo) < cutoff_distance) ||
                                 (nz == +1 && (zhi - pos_i[2]) < cutoff_distance);
                
                if (x_relevant && y_relevant && z_relevant) {
                    relevant_images.push_back({{nx, ny, nz}});
                }
            }
        }
    }
    
    // Debug output for first few atoms
    if (debug_level >= DEBUG_DATA && comm->me == 0 && ii < 3) {
        utils::logmesg(lmp, "    Smart periodic images for atom[{}]: {} relevant (vs 26 total)\n", 
                      ii, relevant_images.size());
        if (debug_level >= DEBUG_FULL) {
            utils::logmesg(lmp, "      Images: ");
            for (const auto& img : relevant_images) {
                utils::logmesg(lmp, "[{},{},{}] ", img[0], img[1], img[2]);
            }
            utils::logmesg(lmp, "\n");
        }
    }
}

/* ---------------------------------------------------------------------- */
// PHASE 2: Hybrid LAMMPS+ASE neighbor processing (linear scaling algorithm)
void PairJaxMTPZeroOverheadOriginal::build_jax_data_hybrid_optimized(
    NeighList* list, 
    double** x, 
    int* type, 
    double cutoff_distance,
    std::vector<const double*>& atom_positions,
    std::vector<int>& atom_types_vec,
    std::vector<const int*>& neighbor_lists,
    std::vector<int>& neighbor_counts,
    std::vector<const int*>& neighbor_types_lists
) {
    int inum = list->inum;
    int* ilist = list->ilist;
    
    if (debug_level >= DEBUG_BASIC && comm->me == 0) {
        utils::logmesg(lmp, "🚀 HYBRID ALGORITHM: LAMMPS+ASE neighbor processing (linear scaling)\n");
        utils::logmesg(lmp, "  inum (local atoms): {}\n", inum);
        utils::logmesg(lmp, "  max_atoms: {}, max_neighbors: {}\n", max_atoms, max_neighbors);
        utils::logmesg(lmp, "  🎯 CUTOFF_DISTANCE parameter: {:.3f} Å\n", cutoff_distance);
    }
    
    // Resize output arrays
    atom_positions.resize(inum);
    atom_types_vec.resize(inum);
    neighbor_lists.resize(inum);
    neighbor_counts.resize(inum);
    neighbor_types_lists.resize(inum);
    
    int total_neighbors_debug = 0;
    int boundary_atoms = 0;
    int smart_periodic_neighbors = 0;
    
    // HYBRID PROCESSING: Process each local atom efficiently 
    for (int ii = 0; ii < inum; ii++) {
        int i = ilist[ii];  // LAMMPS global atom index
        
        // Store atom type (convert to 0-based)
        atom_types_vec[ii] = type[i] - 1;
        
        // Reserve space for this atom's data (zero-overhead memory management)
        persistent_position_data[ii].resize(max_neighbors * 3);
        persistent_neighbor_data[ii].resize(max_neighbors);
        persistent_neighbor_type_data[ii].resize(max_neighbors);
        
        // PHASE 1: Process LAMMPS neighbors efficiently with cutoff filtering
        std::vector<double> pos_storage(max_neighbors * 3);
        std::vector<int> neighbor_storage(max_neighbors);
        std::vector<int> type_storage(max_neighbors);
        
        int neighbor_count = process_lammps_neighbors_with_cutoff(
            ii, i, list, x, type, cutoff_distance, 
            pos_storage, neighbor_storage, type_storage
        );
        
        // PHASE 2: Smart periodic boundary processing (only where needed)
        bool is_boundary = is_near_periodic_boundary(ii, i, x, cutoff_distance);
        if (is_boundary) {
            boundary_atoms++;
            int initial_count = neighbor_count;
            
            add_missing_periodic_neighbors_targeted(
                ii, i, x, type, inum, ilist, cutoff_distance,
                pos_storage, neighbor_storage, type_storage, neighbor_count
            );
            
            smart_periodic_neighbors += (neighbor_count - initial_count);
            
            if (debug_level >= DEBUG_DATA && comm->me == 0 && ii < 3) {
                utils::logmesg(lmp, "    Boundary atom[{}]: LAMMPS={}, periodic={}, total={}\n", 
                              ii, initial_count, neighbor_count - initial_count, neighbor_count);
                              
                // DEBUG: Comprehensive periodic addition failure analysis
                if (neighbor_count == initial_count && ii == 0) {  // Focus on atom 0 for detailed debug
                    utils::logmesg(lmp, "      🔍 PERIODIC ADDITION FAILURE ANALYSIS:\n");
                    std::vector<std::array<int,3>> test_images;
                    determine_relevant_periodic_images(ii, i, x, cutoff_distance, test_images);
                    utils::logmesg(lmp, "        Smart images found: {} (should be >0 for boundary atoms)\n", test_images.size());
                    
                    // DEBUG: Try manual ASE-style search to see if we can find atoms 1,2,3,7,8
                    utils::logmesg(lmp, "        Manual ASE-style search for missing neighbors:\n");
                    std::vector<int> missing_ase = {1, 2, 3, 7, 8};
                    double pos_i[3] = {x[i][0], x[i][1], x[i][2]};
                    
                    for (int target_idx : missing_ase) {
                        if (target_idx < list->inum) {
                            int target_global = list->ilist[target_idx];
                            double pos_j[3] = {x[target_global][0], x[target_global][1], x[target_global][2]};
                            
                            utils::logmesg(lmp, "          Searching for atom {} (global {}):\n", target_idx, target_global);
                            
                            // Check all 27 periodic images manually
                            double closest_dist = 999.0;
                            int best_nx = 0, best_ny = 0, best_nz = 0;
                            
                            for (int nx = -1; nx <= 1; nx++) {
                                for (int ny = -1; ny <= 1; ny++) {
                                    for (int nz = -1; nz <= 1; nz++) {
                                        double image_x = pos_j[0] + nx * domain->xprd;
                                        double image_y = pos_j[1] + ny * domain->yprd;
                                        double image_z = pos_j[2] + nz * domain->zprd;
                                        
                                        double dx = image_x - pos_i[0];
                                        double dy = image_y - pos_i[1];
                                        double dz = image_z - pos_i[2];
                                        double dist = sqrt(dx*dx + dy*dy + dz*dz);
                                        
                                        if (dist < closest_dist && dist > 1e-10) {
                                            closest_dist = dist;
                                            best_nx = nx; best_ny = ny; best_nz = nz;
                                        }
                                    }
                                }
                            }
                            
                            utils::logmesg(lmp, "            Closest distance: {:.3f} Å (image [{},{},{}])\n", 
                                          closest_dist, best_nx, best_ny, best_nz);
                            utils::logmesg(lmp, "            Within cutoff {:.3f}? {}\n", 
                                          cutoff_distance, (closest_dist <= cutoff_distance) ? "YES" : "NO");
                        }
                    }
                }
            }
        } else if (debug_level >= DEBUG_DATA && comm->me == 0 && ii < 3) {
            utils::logmesg(lmp, "    Interior atom[{}]: LAMMPS={} neighbors (no periodic check needed)\n", 
                          ii, neighbor_count);
        }
        
        // PHASE 3: Zero-copy data finalization
        int actual_neighbors = std::min(neighbor_count, max_neighbors);
        for (int n = 0; n < actual_neighbors; n++) {
            persistent_position_data[ii][n * 3 + 0] = pos_storage[n * 3 + 0];
            persistent_position_data[ii][n * 3 + 1] = pos_storage[n * 3 + 1];
            persistent_position_data[ii][n * 3 + 2] = pos_storage[n * 3 + 2];
            persistent_neighbor_data[ii][n] = neighbor_storage[n];
            persistent_neighbor_type_data[ii][n] = type_storage[n];
        }
        
        // Pad remaining slots
        for (int n = actual_neighbors; n < max_neighbors; n++) {
            persistent_position_data[ii][n * 3 + 0] = 20.0;
            persistent_position_data[ii][n * 3 + 1] = 20.0;
            persistent_position_data[ii][n * 3 + 2] = 20.0;
            persistent_neighbor_data[ii][n] = -1;
            persistent_neighbor_type_data[ii][n] = -1;
        }
        
        // Set output pointers
        neighbor_counts[ii] = actual_neighbors;
        atom_positions[ii] = persistent_position_data[ii].data();
        neighbor_lists[ii] = persistent_neighbor_data[ii].data();
        neighbor_types_lists[ii] = persistent_neighbor_type_data[ii].data();
        
        total_neighbors_debug += actual_neighbors;
    }
    
    if (debug_level >= DEBUG_BASIC && comm->me == 0) {
        utils::logmesg(lmp, "🚀 HYBRID ALGORITHM RESULTS:\n");
        utils::logmesg(lmp, "  Local atoms processed: {}\n", inum);
        utils::logmesg(lmp, "  Boundary atoms: {} ({:.1f}%)\n", boundary_atoms, 100.0 * boundary_atoms / inum);
        utils::logmesg(lmp, "  Total neighbors: {} (LAMMPS + {} periodic)\n", 
                      total_neighbors_debug, smart_periodic_neighbors);
        utils::logmesg(lmp, "  Average neighbors/atom: {:.1f}\n", (double)total_neighbors_debug / inum);
        utils::logmesg(lmp, "  Performance: O(neighbors) vs O(N²×27) = {}x speedup estimate\n", 
                      (inum * inum * 27) / std::max(total_neighbors_debug, 1));
    }
}

/* ---------------------------------------------------------------------- */

void PairJaxMTPZeroOverheadOriginal::build_jax_data_from_lammps_neighbors(
    NeighList* list,
    double** x,
    int* type,
    double cutoff_distance,  // ADD CUTOFF PARAMETER
    std::vector<const double*>& atom_positions,
    std::vector<int>& atom_types_vec,
    std::vector<const int*>& neighbor_lists,
    std::vector<int>& neighbor_counts,
    std::vector<const int*>& neighbor_types_lists
) {
    int inum = list->inum;
    int* ilist = list->ilist;
    int* numneigh = list->numneigh;
    int** firstneigh = list->firstneigh;
    
    if (debug_level >= DEBUG_BASIC && comm->me == 0) {
        utils::logmesg(lmp, "🔍 DEBUGGING LAMMPS neighbor processing:\n");
        utils::logmesg(lmp, "  inum (local atoms): {}\n", inum);
        utils::logmesg(lmp, "  max_atoms: {}, max_neighbors: {}\n", max_atoms, max_neighbors);
        utils::logmesg(lmp, "  🎯 CUTOFF_DISTANCE parameter: {:.3f} Å\n", cutoff_distance);
    }
    
    // Resize output vectors
    atom_positions.resize(inum);
    atom_types_vec.resize(inum);
    neighbor_lists.resize(inum);
    neighbor_counts.resize(inum);
    neighbor_types_lists.resize(inum);
    
    // Resize persistent storage
    persistent_position_data.resize(inum);
    persistent_neighbor_data.resize(inum);
    persistent_neighbor_type_data.resize(inum);
    
    // CRITICAL FIX: Create mapping from global LAMMPS index to local index
    std::map<int, int> global_to_local;
    for (int ii = 0; ii < inum; ii++) {
        int i = ilist[ii];
        global_to_local[i] = ii;  // Map global index i to local index ii
    }
    
    // NEW: Create ghost-to-local mapping using periodic imaging
    std::map<int, int> ghost_to_local;
    build_ghost_to_local_mapping(x, ilist, inum, ghost_to_local);
    
    if (debug_level >= DEBUG_BASIC && comm->me == 0) {
        utils::logmesg(lmp, "  Global to local mapping: first 5 entries:\n");
        int count = 0;
        for (auto& pair : global_to_local) {
            if (count < 5) {
                utils::logmesg(lmp, "    global[{}] -> local[{}]\n", pair.first, pair.second);
                count++;
            }
        }
        
        utils::logmesg(lmp, "  Ghost to local mapping: {} entries found\n", ghost_to_local.size());
        if (!ghost_to_local.empty()) {
            utils::logmesg(lmp, "  First 5 ghost mappings:\n");
            count = 0;
            for (auto& pair : ghost_to_local) {
                if (count < 5) {
                    utils::logmesg(lmp, "    ghost[{}] -> local[{}]\n", pair.first, pair.second);
                    count++;
                }
            }
        }
    }
    
    int total_neighbors_debug = 0;
    int local_neighbors = 0, ghost_neighbors = 0;
    
    // Process each local atom using ASE-compatible comprehensive periodic search
    for (int ii = 0; ii < inum; ii++) {
        int i = ilist[ii];  // LAMMPS global atom index
        
        // Store atom type (convert to 0-based)
        atom_types_vec[ii] = type[i] - 1;
        
        if (debug_level >= DEBUG_BASIC && comm->me == 0 && ii < 3) {
            utils::logmesg(lmp, "🔍 ASE-COMPATIBLE: Processing atom[{}] with comprehensive periodic search\n", ii);
        }
        
        // Reserve space for this atom's data  
        persistent_position_data[ii].resize(max_neighbors * 3);
        persistent_neighbor_data[ii].resize(max_neighbors);
        persistent_neighbor_type_data[ii].resize(max_neighbors);
        
        // Use ASE-compatible neighbor finder (comprehensive 3x3x3 periodic image search)
        int neighbor_count = 0;
        std::vector<double> pos_storage(max_neighbors * 3);
        std::vector<int> neighbor_storage(max_neighbors);
        std::vector<int> type_storage(max_neighbors);
        
        find_ase_compatible_neighbors(
            ii, i, x, type, inum, ilist, cutoff_distance,
            neighbor_count, pos_storage, neighbor_storage, type_storage
        );
        
        if (debug_level >= DEBUG_BASIC && comm->me == 0 && ii < 3) {
            utils::logmesg(lmp, "    Hybrid found {} neighbors for atom[{}]\n", neighbor_count, ii);
            if (neighbor_count > 0) {
                utils::logmesg(lmp, "      First few neighbors: ");
                for (int n = 0; n < std::min(5, neighbor_count); n++) {
                    utils::logmesg(lmp, "{} ", neighbor_storage[n]);
                }
                utils::logmesg(lmp, "\n");
                
                // CRITICAL: Deep diagnostic analysis for first atom
                if (ii == 0) {
                    std::vector<int> expected_ase = {1, 2, 3, 7, 8}; // Known ASE neighbors for atom 0
                    std::vector<int> found_neighbors;
                    std::vector<double> found_distances;
                    
                    for (int n = 0; n < std::min(neighbor_count, 10); n++) {
                        found_neighbors.push_back(neighbor_storage[n]);
                        // Calculate distance from stored position data
                        double dx = pos_storage[n * 3 + 0];
                        double dy = pos_storage[n * 3 + 1]; 
                        double dz = pos_storage[n * 3 + 2];
                        double dist = sqrt(dx*dx + dy*dy + dz*dz);
                        found_distances.push_back(dist);
                    }
                    
                    utils::logmesg(lmp, "      🔍 DEEP NEIGHBOR ANALYSIS:\n");
                    utils::logmesg(lmp, "        Expected ASE neighbors: ");
                    for (int idx : expected_ase) utils::logmesg(lmp, "{} ", idx);
                    utils::logmesg(lmp, "\n        Found LAMMPS neighbors: ");
                    for (int idx : found_neighbors) utils::logmesg(lmp, "{} ", idx);
                    
                    // Check if we found the expected ASE neighbors
                    int ase_matches = 0;
                    for (int expected : expected_ase) {
                        if (std::find(found_neighbors.begin(), found_neighbors.end(), expected) != found_neighbors.end()) {
                            ase_matches++;
                        }
                    }
                    utils::logmesg(lmp, "\n        ASE compatibility: {}/{} expected neighbors found ({}%)\n", 
                                  ase_matches, expected_ase.size(), (100 * ase_matches) / expected_ase.size());
                    
                    // DISTANCE COMPARISON: Compare distances from atom 0 to both neighbor sets
                    utils::logmesg(lmp, "        DISTANCE COMPARISON:\n");
                    utils::logmesg(lmp, "          LAMMPS neighbors (first 5): ");
                    for (int n = 0; n < std::min(5, neighbor_count); n++) {
                        utils::logmesg(lmp, "[{}:{:.3f}] ", found_neighbors[n], found_distances[n]);
                    }
                    
                    // Calculate distances to expected ASE neighbors for comparison
                    utils::logmesg(lmp, "\n          ASE expected distances:   ");
                    for (int ase_idx : expected_ase) {
                        if (ase_idx < list->inum) {  // Make sure it's a valid local atom
                            int global_ase = list->ilist[ase_idx];
                            double dx = x[global_ase][0] - x[i][0];
                            double dy = x[global_ase][1] - x[i][1];
                            double dz = x[global_ase][2] - x[i][2];
                            
                            // Apply minimum image convention 
                            if (domain->xperiodic) {
                                if (dx > domain->xprd_half) dx -= domain->xprd;
                                else if (dx < -domain->xprd_half) dx += domain->xprd;
                            }
                            if (domain->yperiodic) {
                                if (dy > domain->yprd_half) dy -= domain->yprd;
                                else if (dy < -domain->yprd_half) dy += domain->yprd;
                            }
                            if (domain->zperiodic) {
                                if (dz > domain->zprd_half) dz -= domain->zprd;
                                else if (dz < -domain->zprd_half) dz += domain->zprd;
                            }
                            
                            double ase_dist = sqrt(dx*dx + dy*dy + dz*dz);
                            utils::logmesg(lmp, "[{}:{:.3f}] ", ase_idx, ase_dist);
                        }
                    }
                    
                    utils::logmesg(lmp, "\n        HYPOTHESIS: Are LAMMPS neighbors periodic images of ASE neighbors?\n");
                }
            }
        }
        
        // Copy data to persistent arrays
        int actual_neighbors = std::min(neighbor_count, max_neighbors);
        for (int n = 0; n < actual_neighbors; n++) {
            persistent_position_data[ii][n * 3 + 0] = pos_storage[n * 3 + 0];
            persistent_position_data[ii][n * 3 + 1] = pos_storage[n * 3 + 1];
            persistent_position_data[ii][n * 3 + 2] = pos_storage[n * 3 + 2];
            persistent_neighbor_data[ii][n] = neighbor_storage[n];
            persistent_neighbor_type_data[ii][n] = type_storage[n];
            
            local_neighbors++; // All are effectively local after ASE processing
        }
        
        // Pad remaining slots
        for (int n = actual_neighbors; n < max_neighbors; n++) {
            persistent_position_data[ii][n * 3 + 0] = 20.0;
            persistent_position_data[ii][n * 3 + 1] = 20.0;
            persistent_position_data[ii][n * 3 + 2] = 20.0;
            persistent_neighbor_data[ii][n] = -1;
            persistent_neighbor_type_data[ii][n] = -1;
        }
        
        // Set output pointers
        neighbor_counts[ii] = actual_neighbors;
        atom_positions[ii] = persistent_position_data[ii].data();
        neighbor_lists[ii] = persistent_neighbor_data[ii].data();
        neighbor_types_lists[ii] = persistent_neighbor_type_data[ii].data();
        
        total_neighbors_debug += actual_neighbors;
    }
    
    if (debug_level >= DEBUG_BASIC && comm->me == 0) {
        utils::logmesg(lmp, "🚀 LAMMPS neighbor processing complete:\n");
        utils::logmesg(lmp, "  Local atoms processed: {}\n", inum);
        utils::logmesg(lmp, "  Total neighbors: {} (local: {}, ghost: {})\n", 
                      total_neighbors_debug, local_neighbors, ghost_neighbors);
        utils::logmesg(lmp, "  Ghost neighbor ratio: {:.2f}%\n", 
                      100.0 * ghost_neighbors / (local_neighbors + ghost_neighbors));
        utils::logmesg(lmp, "  Array format: ({}, {}, 3) padded to ({}, {}, 3)\n", 
                      inum, neighbor_counts[0], max_atoms, max_neighbors);
    }
}

/* ---------------------------------------------------------------------- */
// PHASE 2: Efficient LAMMPS neighbor processing with cutoff filtering
int PairJaxMTPZeroOverheadOriginal::process_lammps_neighbors_with_cutoff(
    int ii, int i, NeighList* list, double** x, int* type, double cutoff_distance,
    std::vector<double>& pos_storage, std::vector<int>& neighbor_storage, 
    std::vector<int>& type_storage
) {
    int* numneigh = list->numneigh;
    int** firstneigh = list->firstneigh;
    
    int neighbor_count = 0;
    int jnum = numneigh[i];
    int* jlist = firstneigh[i];
    
    double pos_i[3] = {x[i][0], x[i][1], x[i][2]};
    
    // Process LAMMPS neighbors with cutoff filtering (much more efficient than full periodic search)
    for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj] & NEIGHMASK;  // Remove LAMMPS mask
        
        // Calculate minimum image distance
        double dx = x[j][0] - pos_i[0];
        double dy = x[j][1] - pos_i[1];
        double dz = x[j][2] - pos_i[2];
        
        // Apply minimum image convention
        if (domain->xperiodic) {
            if (dx > domain->xprd_half) dx -= domain->xprd;
            else if (dx < -domain->xprd_half) dx += domain->xprd;
        }
        if (domain->yperiodic) {
            if (dy > domain->yprd_half) dy -= domain->yprd;
            else if (dy < -domain->yprd_half) dy += domain->yprd;
        }
        if (domain->zperiodic) {
            if (dz > domain->zprd_half) dz -= domain->zprd;
            else if (dz < -domain->zprd_half) dz += domain->zprd;
        }
        
        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        
        // Apply JAX cutoff filter (essential for accuracy)
        if (dist <= cutoff_distance && dist > 1e-10) {
            // CRITICAL FIX: Map to local indices for ASE compatibility
            int local_j = -1;
            
            if (j < list->inum) {
                // Local atom - find its local index in ilist
                for (int kk = 0; kk < list->inum; kk++) {
                    if (list->ilist[kk] == j) {
                        local_j = kk;
                        break;
                    }
                }
                
                if (debug_level >= DEBUG_FULL && comm->me == 0 && ii < 2 && neighbor_count < 3) {
                    utils::logmesg(lmp, "      Local atom: global_j={} -> local_j={}\n", j, local_j);
                }
            } else {
                // Ghost atom - find local equivalent using position matching
                double ghost_pos[3] = {x[j][0], x[j][1], x[j][2]};
                
                for (int kk = 0; kk < list->inum; kk++) {
                    int local_atom = list->ilist[kk];
                    double lx = x[local_atom][0], ly = x[local_atom][1], lz = x[local_atom][2];
                    
                    // Check if ghost j is a periodic image of local atom kk
                    for (int nx = -1; nx <= 1; nx++) {
                        for (int ny = -1; ny <= 1; ny++) {
                            for (int nz = -1; nz <= 1; nz++) {
                                double image_x = lx + nx * domain->xprd;
                                double image_y = ly + ny * domain->yprd;
                                double image_z = lz + nz * domain->zprd;
                                
                                double ghost_dist = sqrt((ghost_pos[0] - image_x) * (ghost_pos[0] - image_x) +
                                                        (ghost_pos[1] - image_y) * (ghost_pos[1] - image_y) +
                                                        (ghost_pos[2] - image_z) * (ghost_pos[2] - image_z));
                                
                                if (ghost_dist < 1e-8) {  // Found the local equivalent
                                    local_j = kk;  // Use LOCAL index kk
                                    
                                    if (debug_level >= DEBUG_FULL && comm->me == 0 && ii < 2 && neighbor_count < 3) {
                                        utils::logmesg(lmp, "      Ghost atom: global_j={} -> local_j={} (dist={:.2e})\n", 
                                                      j, local_j, ghost_dist);
                                    }
                                    goto found_local_equiv;
                                }
                            }
                        }
                    }
                    found_local_equiv:;
                    if (local_j != -1) break;
                }
                
                if (local_j == -1) {
                    if (debug_level >= DEBUG_FULL && comm->me == 0 && ii < 2) {
                        utils::logmesg(lmp, "      WARNING: Ghost atom {} unmapped, skipping\n", j);
                    }
                    continue; // Skip unmapped ghost atoms
                }
            }
            
            if (local_j == -1) continue; // Safety check
            
            // Store neighbor with local index
            pos_storage[neighbor_count * 3 + 0] = dx;
            pos_storage[neighbor_count * 3 + 1] = dy; 
            pos_storage[neighbor_count * 3 + 2] = dz;
            neighbor_storage[neighbor_count] = local_j;
            type_storage[neighbor_count] = type[j] - 1;  // 0-based
            neighbor_count++;
            
            if (neighbor_count >= max_neighbors) break;  // Avoid overflow
        }
    }
    
    return neighbor_count;
}

/* ---------------------------------------------------------------------- */
// PHASE 2: Add missing periodic neighbors using smart image selection
void PairJaxMTPZeroOverheadOriginal::add_missing_periodic_neighbors_targeted(
    int ii, int i, double** x, int* type, int inum, int* ilist, double cutoff_distance,
    std::vector<double>& pos_storage, std::vector<int>& neighbor_storage,
    std::vector<int>& type_storage, int& neighbor_count
) {
    // Get relevant periodic images (much fewer than 26!)
    std::vector<std::array<int,3>> relevant_images;
    determine_relevant_periodic_images(ii, i, x, cutoff_distance, relevant_images);
    
    double pos_i[3] = {x[i][0], x[i][1], x[i][2]};
    std::set<int> existing_neighbors;
    
    // Track existing neighbors to avoid duplicates
    for (int n = 0; n < neighbor_count; n++) {
        existing_neighbors.insert(neighbor_storage[n]);
    }
    
    // Search through relevant periodic images only
    for (const auto& img : relevant_images) {
        int nx = img[0], ny = img[1], nz = img[2];
        
        for (int jj = 0; jj < inum; jj++) {
            int j = ilist[jj];
            if (i == j) continue;
            
            // Calculate periodic image position  
            double image_pos[3];
            image_pos[0] = x[j][0] + nx * domain->xprd;
            image_pos[1] = x[j][1] + ny * domain->yprd;
            image_pos[2] = x[j][2] + nz * domain->zprd;
            
            double dx = image_pos[0] - pos_i[0];
            double dy = image_pos[1] - pos_i[1];
            double dz = image_pos[2] - pos_i[2];
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist <= cutoff_distance && dist > 1e-10) {
                // Check if this neighbor is already included
                if (existing_neighbors.find(jj) == existing_neighbors.end()) {
                    pos_storage[neighbor_count * 3 + 0] = dx;
                    pos_storage[neighbor_count * 3 + 1] = dy;
                    pos_storage[neighbor_count * 3 + 2] = dz;
                    neighbor_storage[neighbor_count] = jj;  // Local index
                    type_storage[neighbor_count] = type[j] - 1;
                    
                    existing_neighbors.insert(jj);
                    neighbor_count++;
                    
                    if (neighbor_count >= max_neighbors) return;  // Avoid overflow
                }
                break; // Found this atom in this image, don't check other images
            }
        }
    }
}

/* ---------------------------------------------------------------------- */

bool PairJaxMTPZeroOverheadOriginal::process_multi_batch_system(
    ZeroOverheadOriginal::ZeroOverheadOriginalManager* manager, 
    const char* jax_function_path, double volume,
    const std::vector<int>& main_batch_atoms, 
    const std::vector<int>& overflow_batch_atoms,
    const std::vector<const double*>& atom_positions,
    const std::vector<int>& atom_types_vec,
    const std::vector<const int*>& neighbor_lists,
    const std::vector<int>& neighbor_counts,
    const std::vector<const int*>& neighbor_types_lists,
    double& total_energy, double** forces, double* stress
) {
    // PHASE 2.3: Simple multi-batch processing - main batch + overflow batch
    total_energy = 0.0;
    for (int i = 0; i < 6; i++) stress[i] = 0.0;
    
    bool main_success = true;
    bool overflow_success = true;
    
    // Process main batch (atoms that fit within max_neighbors limit)
    if (!main_batch_atoms.empty()) {
        if (comm->me == 0 && debug_level >= DEBUG_BASIC) {
            utils::logmesg(lmp, "🔧 Processing main batch: {} atoms\\n", main_batch_atoms.size());
        }
        
        // Create data arrays for main batch
        std::vector<const double*> main_positions;
        std::vector<int> main_types;
        std::vector<const int*> main_neighbor_lists;
        std::vector<int> main_neighbor_counts;
        std::vector<const int*> main_neighbor_types;
        
        for (int idx : main_batch_atoms) {
            main_positions.push_back(atom_positions[idx]);
            main_types.push_back(atom_types_vec[idx]);
            main_neighbor_lists.push_back(neighbor_lists[idx]);
            main_neighbor_counts.push_back(neighbor_counts[idx]);
            main_neighbor_types.push_back(neighbor_types_lists[idx]);
        }
        
        // Calculate main batch neighbors
        int main_total_neighbors = 0;
        for (int count : main_neighbor_counts) {
            main_total_neighbors += count;
        }
        
        double main_energy = 0.0;
        double main_stress[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        
        main_success = manager->call_jax_ultra_optimized(
            jax_function_path,
            (int)main_batch_atoms.size(),
            main_total_neighbors,  // FIX: Use total neighbors for this batch!
            main_positions.data(),
            main_types.data(),
            main_neighbor_lists.data(),
            main_neighbor_counts.data(),
            main_neighbor_types.data(),
            volume,
            main_energy,
            persistent_forces_array,  // Forces written directly to main array
            main_stress
        );
        
        if (main_success) {
            total_energy += main_energy;
            // Accumulate JAX stress values (with sign correction for LAMMPS virial convention)
            // MLIP2 uses negative virial, LAMMPS expects positive virial
            for (int s = 0; s < 6; s++) {
                stress[s] -= main_stress[s];
            }
        }
    }
    
    // Process overflow batch (atoms with too many neighbors)
    if (!overflow_batch_atoms.empty() && main_success) {
        if (comm->me == 0 && debug_level >= DEBUG_BASIC) {
            utils::logmesg(lmp, "🔧 Processing overflow batch: {} atoms\\n", overflow_batch_atoms.size());
        }
        
        // CRITICAL FIX: Process overflow atoms with neighbor chunking
        for (size_t i = 0; i < overflow_batch_atoms.size(); i++) {
            int atom_idx = overflow_batch_atoms[i];
            int total_neighbors = neighbor_counts[atom_idx];
            
            if (comm->me == 0 && debug_level >= DEBUG_BASIC) {
                utils::logmesg(lmp, "🔧 Processing overflow atom {} with {} neighbors\\n", atom_idx, total_neighbors);
            }
            
            // Process overflow atom in chunks that fit max_neighbors
            int neighbors_processed = 0;
            double atom_energy = 0.0;
            double atom_stress[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            
            // Zero forces for this atom initially
            for (int k = 0; k < 3; k++) {
                persistent_forces_array[main_batch_atoms.size() + i][k] = 0.0;
            }
            
            while (neighbors_processed < total_neighbors) {
                int neighbors_in_chunk = std::min(max_neighbors, total_neighbors - neighbors_processed);
                
                // Create chunked data for this atom
                std::vector<double> chunk_positions;
                std::vector<int> chunk_neighbors;
                std::vector<int> chunk_neighbor_types;
                
                for (int n = 0; n < neighbors_in_chunk; n++) {
                    int neighbor_idx = neighbors_processed + n;
                    chunk_positions.push_back(atom_positions[atom_idx][neighbor_idx * 3 + 0]);
                    chunk_positions.push_back(atom_positions[atom_idx][neighbor_idx * 3 + 1]);
                    chunk_positions.push_back(atom_positions[atom_idx][neighbor_idx * 3 + 2]);
                    chunk_neighbors.push_back(neighbor_lists[atom_idx][neighbor_idx]);
                    chunk_neighbor_types.push_back(neighbor_types_lists[atom_idx][neighbor_idx]);
                }
                
                // Process this chunk
                std::vector<const double*> chunk_position_ptr = {chunk_positions.data()};
                std::vector<int> chunk_type = {atom_types_vec[atom_idx]};
                std::vector<const int*> chunk_neighbors_ptr = {chunk_neighbors.data()};
                std::vector<int> chunk_count = {neighbors_in_chunk};
                std::vector<const int*> chunk_neighbor_types_ptr = {chunk_neighbor_types.data()};
                
                double chunk_energy = 0.0;
                double chunk_stress[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                double** chunk_forces = &persistent_forces_array[main_batch_atoms.size() + i];
                
                bool chunk_success = manager->call_jax_ultra_optimized(
                    jax_function_path,
                    1,  // Single atom for energy calculation
                    neighbors_in_chunk,  // FIX: Use actual neighbor count for this chunk!
                    chunk_position_ptr.data(),
                    chunk_type.data(),
                    chunk_neighbors_ptr.data(),
                    chunk_count.data(),
                    chunk_neighbor_types_ptr.data(),
                    volume,
                    chunk_energy,
                    chunk_forces,
                    chunk_stress
                );
                
                if (!chunk_success) {
                    overflow_success = false;
                    break;
                }
                
                // Accumulate results
                atom_energy += chunk_energy;
                // Accumulate JAX stress values (with sign correction for LAMMPS virial convention)
                for (int s = 0; s < 6; s++) {
                    atom_stress[s] -= chunk_stress[s];
                }
                
                neighbors_processed += neighbors_in_chunk;
                
                if (comm->me == 0 && debug_level >= DEBUG_BASIC) {
                    utils::logmesg(lmp, "   Processed chunk: {} neighbors, energy contribution: {:.6f}\\n", 
                                  neighbors_in_chunk, chunk_energy);
                }
            }
            
            if (overflow_success) {
                total_energy += atom_energy;
                // Accumulate JAX stress values (atom_stress already sign-corrected)
                for (int s = 0; s < 6; s++) {
                    stress[s] += atom_stress[s];
                }
            } else {
                break;
            }
        }
    }
    
    if (comm->me == 0 && debug_level >= DEBUG_BASIC) {
        utils::logmesg(lmp, "✅ Multi-batch processing complete: Energy={:.6f}\\n", total_energy);
    }
    
    return main_success && overflow_success;
}
