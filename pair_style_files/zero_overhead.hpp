/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   JAX-MTP Zero Overhead Buffer Manager
------------------------------------------------------------------------- */

#ifndef ZERO_OVERHEAD_HPP
#define ZERO_OVERHEAD_HPP

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

namespace ZeroOverhead {

class PersistentMemoryPool {
public:
    PersistentMemoryPool() = default;
    ~PersistentMemoryPool();
    bool initialize(int max_atoms, int max_neighbors);
    void update_atom_data_zero_copy(
        int natoms_actual,
        const double* const* atom_positions, const int* atom_types,
        const int* const* neighbor_lists, const int* neighbor_counts,
        const int* const* neighbor_types_lists
    );
    bool is_initialized() const { return pool_initialized; }

    PyObject* get_types_buffer() { return itypes_array; }
    PyObject* get_neighbors_buffer() { return all_js_array; }
    PyObject* get_positions_buffer() { return all_rijs_array; }
    PyObject* get_neighbor_types_buffer() { return all_jtypes_array; }
    PyObject* get_cell_rank_obj() { return cell_rank_obj; }
    PyObject* get_volume_obj() { return volume_obj; }
    PyObject* get_natoms_actual_obj() { return natoms_actual_obj; }
    PyObject* get_nneigh_total_obj() { return nneigh_total_obj; }

private:
    bool pool_initialized = false;
    int config_max_atoms = 0, config_max_neighbors = 0;
    PyObject *itypes_array = nullptr, *all_js_array = nullptr, *all_rijs_array = nullptr, *all_jtypes_array = nullptr;
    PyObject *cell_rank_obj = nullptr, *volume_obj = nullptr;
    PyObject *natoms_actual_obj = nullptr, *nneigh_total_obj = nullptr;
    mutable std::mutex pool_mutex;
    void cleanup();
    static bool initialize_numpy_api();
};

class ZeroOverheadManager {
public:
    static ZeroOverheadManager* get_instance();
    ~ZeroOverheadManager();
    bool initialize_for_system(int max_atoms, int max_neighbors);
    bool execute_potential(
        const std::string& function_path, int natoms_actual, int nneigh_total,
        const double* const* atom_positions, const int* atom_types,
        const int* const* neighbor_lists, const int* neighbor_counts,
        const int* const* neighbor_types_lists, double volume,
        double& energy, double** forces, double* stress
    );
    void cleanup();

private:
    ZeroOverheadManager() = default;
    static std::unique_ptr<ZeroOverheadManager> instance;
    static std::mutex instance_mutex;
    std::map<std::string, std::unique_ptr<PersistentMemoryPool>> memory_pools;
    mutable std::mutex pools_mutex;
    PersistentMemoryPool* active_pool = nullptr;
    PyObject* cached_jax_function = nullptr, *cached_jax_call_method = nullptr;
    std::string cached_function_path;
    bool load_and_cache_jax_function(const std::string& function_path);
};

class ZeroOverheadContext {
public:
    ZeroOverheadContext(int max_atoms, int max_neighbors);
    ~ZeroOverheadContext() = default;
    bool is_ready() const { return initialized; }
    ZeroOverheadManager* get_manager() { return manager; }
private:
    ZeroOverheadManager* manager;
    bool initialized = false;
};

} // namespace ZeroOverhead
#endif
