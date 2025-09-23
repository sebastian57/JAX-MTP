/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Zero Overhead JAX-MTP Implementation - Final Corrected Version
   - REVISED neighbor processing to correctly handle all local and ghost neighbors.
   - SIMPLIFIED indexing to use local atom indices directly.
   - ENSURED JAX operates on all local atoms for correct environment description.
------------------------------------------------------------------------- */
#include "pair_jax_mtp_zero_overhead_original.h"
#include "zero_overhead_buffer_manager_original.hpp"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
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
  debug_level = 0;
  zero_overhead_context = nullptr;
  total_calls = 0;
  python_initialized = false;
  persistent_forces_array = nullptr;
  persistent_position_data.clear();
  persistent_neighbor_data.clear();
  persistent_neighbor_type_data.clear();

  if (comm->me == 0) {
    utils::logmesg(lmp, "✅ Zero Overhead JAX-MTP (LAMMPS neighbors) initialized\n");
  }
}

/* ---------------------------------------------------------------------- */

PairJaxMTPZeroOverheadOriginal::~PairJaxMTPZeroOverheadOriginal()
{
  cleanup_python();
  delete[] jax_function_path;

  if (persistent_forces_array) {
    memory->destroy(persistent_forces_array);
  }
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  delete zero_overhead_context;

  if (total_calls > 0 && comm->me == 0) {
    utils::logmesg(lmp, "\n=== LAMMPS-NEIGHBOR JAX-MTP PERFORMANCE SUMMARY ===\n");
    utils::logmesg(lmp, "Total JAX calls: {}\n", total_calls);
  }
}

void PairJaxMTPZeroOverheadOriginal::compute(int eflag, int vflag)
{
  // 1. SETUP
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  
  NeighList *neighlist = neighbor->lists[0];
  int inum = neighlist->inum;
  int *ilist = neighlist->ilist;
  int *numneigh = neighlist->numneigh;
  int **firstneigh = neighlist->firstneigh;
  
  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr = 0;
  
  initialize_persistent_arrays();
  
  // 2. DATA PREPARATION (Corrected Logic)
  int natoms_total_local = atom->nlocal;
  int nneigh_total = 0;
  const double cutoff_sq = cutoff * cutoff;

  // These vectors hold the pointers that will be passed to the buffer manager.
  // We size them to max_atoms, which is assumed to be >= natoms_total_local.
  std::vector<const double*> atom_positions(max_atoms);
  std::vector<int> atom_types_vec(max_atoms, -1);
  std::vector<const int*> neighbor_lists(max_atoms);
  std::vector<int> neighbor_counts(max_atoms, 0);
  std::vector<const int*> neighbor_types_lists(max_atoms);
  
  // Populate types for all local atoms. The JAX model needs a complete type list.
  for (int i = 0; i < natoms_total_local; i++) {
    atom_types_vec[i] = type[i] - 1;
  }

  // Build a map: LAMMPS tag -> dense local index (0..natoms_total_local-1)
  // This maps any atom tag owned locally to the dense index we use for JAX.
  // We'll use this to convert ghost entries (which are stored as array indices)
  // into the dense indices JAX expects.
  int max_tag = 0;
  for (int i = 0; i < natoms_total_local; ++i)
    if (atom->tag[i] > max_tag) max_tag = atom->tag[i];

  std::vector<int> tag_to_dense(max_tag + 1, -1);
  for (int i = 0; i < natoms_total_local; ++i) {
    int tg = atom->tag[i];
    if (tg >= 0 && tg <= max_tag) tag_to_dense[tg] = i;
  }

  // Iterate through the atoms for which we have neighbor lists (the 'owned' atoms).
  for (int ii = 0; ii < inum; ii++) {
    int i_local_idx = ilist[ii]; // This is the central atom index in atom->x arrays
    
    int jnum = numneigh[i_local_idx];
    int* jlist = firstneigh[i_local_idx];

    double xi = x[i_local_idx][0], yi = x[i_local_idx][1], zi = x[i_local_idx][2];
    int neighbor_count_for_atom = 0;

    // Process all neighbors: local and ghost.
    for (int jj = 0; jj < jnum; jj++) {
      // Mask neighbor entry to get the true atom index (strip flag bits).
      int j_index = jlist[jj] & NEIGHMASK;

      if (i_local_idx == j_index) continue; // skip self

      // Use atom->x[...] even for ghost indices — LAMMPS stores ghost positions there
      double delx = x[j_index][0] - xi;
      double dely = x[j_index][1] - yi;
      double delz = x[j_index][2] - zi;

      domain->minimum_image(__FILE__, __LINE__, delx, dely, delz);

      double r_sq = delx*delx + dely*dely + delz*delz;

      if (r_sq < cutoff_sq && neighbor_count_for_atom < max_neighbors) {
        // Always store the relative vector (this ensures all_rijs gets local+ghost entries)
        persistent_position_data[i_local_idx][neighbor_count_for_atom * 3 + 0] = delx;
        persistent_position_data[i_local_idx][neighbor_count_for_atom * 3 + 1] = dely;
        persistent_position_data[i_local_idx][neighbor_count_for_atom * 3 + 2] = delz;

        // --- FIX: map neighbor (which may be a ghost array index) to the dense index
        // We look up the neighbor's tag and use tag_to_dense to retrieve the index
        // used in the JAX arrays (0..natoms_total_local-1). If not present, write sentinel.
        int neighbor_dense_idx = natoms_total_local; // sentinel (padded value)
        int j_tag = atom->tag[j_index];
        if (j_tag >= 0 && j_tag <= max_tag) {
          int mapped = tag_to_dense[j_tag];
          if (mapped != -1) {
            neighbor_dense_idx = mapped;
          }
        }
        // Write dense index (local atom index) into neighbor data for JAX.
        persistent_neighbor_data[i_local_idx][neighbor_count_for_atom] = neighbor_dense_idx;

        // Store the neighbor's type (works for ghost entries too). Use -1 for invalid.
        if (neighbor_dense_idx != natoms_total_local) {
          // If mapped to a real local atom, store its type.
          persistent_neighbor_type_data[i_local_idx][neighbor_count_for_atom] = type[j_index] - 1;
        } else {
          // Remote owner / cannot map -> mark as invalid
          persistent_neighbor_type_data[i_local_idx][neighbor_count_for_atom] = -1;
          // Optionally you could also choose to skip this neighbor entirely rather than
          // including it mapped to sentinel; keep consistent with padding strategy.
        }

        neighbor_count_for_atom++;
      }
    }

    // Update counts and total after finishing this atom's neighbors
    neighbor_counts[i_local_idx] = neighbor_count_for_atom;
    nneigh_total += neighbor_count_for_atom;

    // Pad the remaining neighbor slots for this atom (keep your previous sentinel pos)
    for (int jj = neighbor_count_for_atom; jj < max_neighbors; ++jj) {
      persistent_position_data[i_local_idx][jj*3 + 0] = 20.0;
      persistent_position_data[i_local_idx][jj*3 + 1] = 20.0;
      persistent_position_data[i_local_idx][jj*3 + 2] = 20.0;
      persistent_neighbor_data[i_local_idx][jj] = natoms_total_local; 
      persistent_neighbor_type_data[i_local_idx][jj] = -1;
    }
  }
  
  // Set up the pointers for all local atoms for the buffer manager.
  // For atoms not in ilist, their data will be empty and counts will be 0.
  for(int i = 0; i < natoms_total_local; i++) {
      atom_positions[i] = persistent_position_data[i].data();
      neighbor_lists[i] = persistent_neighbor_data[i].data();
      neighbor_types_lists[i] = persistent_neighbor_type_data[i].data();
  }

  // 3. JAX KERNEL CALL
  double total_energy = 0.0;
  double virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  ZeroOverheadOriginalManager* manager = zero_overhead_context->get_manager();

  double volume = (domain->triclinic) ? 
      (domain->h[0]*(domain->h[4]*domain->h[8] - domain->h[5]*domain->h[7]) - domain->h[1]*(domain->h[3]*domain->h[8] - domain->h[5]*domain->h[6]) + domain->h[2]*(domain->h[3]*domain->h[7] - domain->h[4]*domain->h[6])) : 
      (domain->xprd * domain->yprd * domain->zprd);

  // We pass data for ALL local atoms to JAX.
  bool success = manager->execute_potential(
    jax_function_path, natoms_total_local, nneigh_total,
    atom_positions.data(), atom_types_vec.data(),
    neighbor_lists.data(), neighbor_counts.data(),
    neighbor_types_lists.data(),
    volume, total_energy, persistent_forces_array, virial
  );

  if (!success) {
      error->all(FLERR, "JAX function execution failed. Check terminal output for Python errors.");
  }

  // 4. PROCESS RESULTS
  // JAX returns forces for all nlocal atoms. We apply them only to the ones in ilist.
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    f[i][0] += persistent_forces_array[i][0];
    f[i][1] += persistent_forces_array[i][1];
    f[i][2] += persistent_forces_array[i][2];
  }
  
  if (eflag_global) eng_vdwl += total_energy;
  if (vflag_global) {
      for(int k=0; k<6; k++) this->virial[k] += virial[k];
  }
  total_calls++;
}

void PairJaxMTPZeroOverheadOriginal::initialize_persistent_arrays()
{
  if (!persistent_forces_array) {
    memory->create(persistent_forces_array, max_atoms, 3, "pair:persistent_forces");
  }
  
  if (persistent_position_data.empty()) {
    persistent_position_data.resize(max_atoms);
    persistent_neighbor_data.resize(max_atoms);
    persistent_neighbor_type_data.resize(max_atoms);
    
    for (int i = 0; i < max_atoms; i++) {
      persistent_position_data[i].resize(max_neighbors * 3, 0.0);
      persistent_neighbor_data[i].resize(max_neighbors, 0);
      persistent_neighbor_type_data[i].resize(max_neighbors, 0);
    }
  }
}

void PairJaxMTPZeroOverheadOriginal::settings(int narg, char **arg)
{
  if (narg < 4) error->all(FLERR, "Illegal pair_style command - usage: pair_style jax/mtp_zero_overhead <bin_file> <max_atoms> <max_neighbors> <cutoff> [debug_level]");

  int n = strlen(arg[0]) + 1;
  jax_function_path = new char[n];
  strcpy(jax_function_path, arg[0]);
  
  max_atoms = utils::inumeric(FLERR, arg[1], false, lmp);
  max_neighbors = utils::inumeric(FLERR, arg[2], false, lmp);
  cutoff = utils::numeric(FLERR, arg[3], false, lmp);
  
  if (narg >= 5) debug_level = utils::inumeric(FLERR, arg[4], false, lmp);
  
  if (max_atoms <= 0 || max_neighbors <= 0 || cutoff <= 0.0) {
      error->all(FLERR, "max_atoms, max_neighbors, and cutoff must be positive");
  }
}

void PairJaxMTPZeroOverheadOriginal::coeff(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = 1; j <= atom->ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutoff * cutoff;
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
  if (max_atoms <= 0 || max_neighbors <= 1)
    error->all(FLERR, "System capacity not set - check pair_style arguments");

  // Request a FULL neighbor list, required for many-body potentials.
  neighbor->add_request(this, NeighConst::REQ_FULL);
  
  init_python_direct();
  
  if (!zero_overhead_context) {
    try {
      zero_overhead_context = new ZeroOverheadOriginal::ZeroOverheadOriginalContext(max_atoms, max_neighbors);
    } catch (const std::exception& e) {
      error->all(FLERR, fmt::format("Failed to initialize ZeroOverheadContext: {}", e.what()));
    }
  }
}

void PairJaxMTPZeroOverheadOriginal::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

void PairJaxMTPZeroOverheadOriginal::init_python_direct()
{
  if (python_initialized) return;
  try {
    if (!Py_IsInitialized()) Py_Initialize();
    python_initialized = true;
  } catch (...) {
    error->all(FLERR, "Python initialization failed");
  }
}

void PairJaxMTPZeroOverheadOriginal::cleanup_python()
{
  if (python_initialized) {
    python_initialized = false;
  }
}

