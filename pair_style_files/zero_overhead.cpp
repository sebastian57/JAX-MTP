/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   JAX-MTP Zero Overhead Buffer Manager
------------------------------------------------------------------------- */
#include "zero_overhead.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cmath>

using namespace ZeroOverhead;

std::unique_ptr<ZeroOverheadManager> ZeroOverheadManager::instance = nullptr;
std::mutex ZeroOverheadManager::instance_mutex;
static bool numpy_api_initialized = false;
static std::mutex numpy_init_mutex;

bool PersistentMemoryPool::initialize_numpy_api() {
    std::lock_guard<std::mutex> lock(numpy_init_mutex);
    if (numpy_api_initialized) return true;
    if (!Py_IsInitialized()) Py_Initialize();
    import_array1(false);
    if (PyErr_Occurred()) { PyErr_Print(); return false; }
    numpy_api_initialized = true;
    return true;
}

PersistentMemoryPool::~PersistentMemoryPool() { cleanup(); }

bool PersistentMemoryPool::initialize(int max_atoms, int max_neighbors) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (pool_initialized) return true;
    if (!initialize_numpy_api()) return false;

    npy_intp dims1[] = {(npy_intp)max_atoms};
    npy_intp dims2[] = {(npy_intp)max_atoms, (npy_intp)max_neighbors};
    npy_intp dims3[] = {(npy_intp)max_atoms, (npy_intp)max_neighbors, 3};

    itypes_array = PyArray_ZEROS(1, dims1, NPY_INT32, 0);
    all_js_array = PyArray_ZEROS(2, dims2, NPY_INT32, 0);
    all_rijs_array = PyArray_ZEROS(3, dims3, NPY_FLOAT32, 0);
    all_jtypes_array = PyArray_ZEROS(2, dims2, NPY_INT32, 0);
    cell_rank_obj = PyArray_ZEROS(0, nullptr, NPY_INT32, 0);
    volume_obj = PyArray_ZEROS(0, nullptr, NPY_FLOAT32, 0);
    natoms_actual_obj = PyArray_ZEROS(0, nullptr, NPY_INT32, 0);
    nneigh_total_obj = PyArray_ZEROS(0, nullptr, NPY_INT32, 0);
    
    if (!itypes_array || !all_js_array || !all_rijs_array || !all_jtypes_array ||
        !cell_rank_obj || !volume_obj || !natoms_actual_obj || !nneigh_total_obj) {
        cleanup();
        return false;
    }
    config_max_atoms = max_atoms;
    config_max_neighbors = max_neighbors;
    pool_initialized = true;
    return true;
}

void PersistentMemoryPool::update_atom_data_zero_copy(
    int natoms_actual,
    const double* const* relative_positions,
    const int* atom_types,
    const int* const* neighbor_indices,
    const int* neighbor_counts,
    const int* const* neighbor_types)
{
    if (!pool_initialized) return;

    int32_t* itypes_data = (int32_t*)PyArray_DATA((PyArrayObject*)itypes_array);
    int32_t* js_data = (int32_t*)PyArray_DATA((PyArrayObject*)all_js_array);
    float* rijs_data = (float*)PyArray_DATA((PyArrayObject*)all_rijs_array);
    int32_t* jtypes_data = (int32_t*)PyArray_DATA((PyArrayObject*)all_jtypes_array);

    int32_t index_sentinel = natoms_actual; 
    int32_t type_sentinel = -1;
    const float SAFE_DISTANCE = 2000.0f;

    for (int i = 0; i < config_max_atoms * config_max_neighbors; i++) {
        js_data[i] = index_sentinel;
        jtypes_data[i] = type_sentinel;
    }
    
    size_t total_rij_elements = (size_t)config_max_atoms * config_max_neighbors * 3;
    for (size_t idx = 0; idx < total_rij_elements; ++idx) {
        rijs_data[idx] = SAFE_DISTANCE;
    }
    
    for (int i = 0; i < config_max_atoms; ++i) {
        itypes_data[i] = type_sentinel;
    }
    
    if (natoms_actual > 0) {
      std::memcpy(itypes_data, atom_types, natoms_actual * sizeof(int32_t));
    }

    for (int i = 0; i < natoms_actual; ++i) {
        int neighbors_to_copy = neighbor_counts[i];
        
        if (neighbors_to_copy > 0) {
            size_t base_idx = (size_t)i * config_max_neighbors;
            const int* neighbor_idx_src = neighbor_indices[i];
            const int* neighbor_type_src = neighbor_types[i];
            const double* rel_pos_src = relative_positions[i];

            for (int j = 0; j < neighbors_to_copy; j++) {
                js_data[base_idx + j] = (int32_t)neighbor_idx_src[j];
                jtypes_data[base_idx + j] = (int32_t)neighbor_type_src[j];
                size_t pos_base_idx = (base_idx + j) * 3;
                rijs_data[pos_base_idx + 0] = (float)rel_pos_src[j * 3 + 0];
                rijs_data[pos_base_idx + 1] = (float)rel_pos_src[j * 3 + 1];
                rijs_data[pos_base_idx + 2] = (float)rel_pos_src[j * 3 + 2];
            }
        }
    }
}

void PersistentMemoryPool::cleanup() {
    if (!Py_IsInitialized()) {
        itypes_array = nullptr;
        all_js_array = nullptr;
        all_rijs_array = nullptr;
        all_jtypes_array = nullptr;
        cell_rank_obj = nullptr;
        volume_obj = nullptr;
        natoms_actual_obj = nullptr;
        nneigh_total_obj = nullptr;
        pool_initialized = false;
        return;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    Py_XDECREF(itypes_array); itypes_array = nullptr;
    Py_XDECREF(all_js_array); all_js_array = nullptr;
    Py_XDECREF(all_rijs_array); all_rijs_array = nullptr;
    Py_XDECREF(all_jtypes_array); all_jtypes_array = nullptr;
    Py_XDECREF(cell_rank_obj); cell_rank_obj = nullptr;
    Py_XDECREF(volume_obj); volume_obj = nullptr;
    Py_XDECREF(natoms_actual_obj); natoms_actual_obj = nullptr;
    Py_XDECREF(nneigh_total_obj); nneigh_total_obj = nullptr;

    PyGILState_Release(gstate);

    pool_initialized = false;
}

ZeroOverheadManager* ZeroOverheadManager::get_instance() {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (!instance) {
        instance = std::unique_ptr<ZeroOverheadManager>(new ZeroOverheadManager());
    }
    return instance.get();
}

ZeroOverheadManager::~ZeroOverheadManager() { cleanup(); }

bool ZeroOverheadManager::initialize_for_system(int max_atoms, int max_neighbors) {
    std::lock_guard<std::mutex> lock(pools_mutex);
    std::string pool_key = std::to_string(max_atoms) + "x" + std::to_string(max_neighbors);
    if (memory_pools.find(pool_key) == memory_pools.end()) {
        auto new_pool = std::make_unique<PersistentMemoryPool>();
        if (!new_pool->initialize(max_atoms, max_neighbors)) return false;
        memory_pools[pool_key] = std::move(new_pool);
    }
    active_pool = memory_pools[pool_key].get();
    return true;
}

bool ZeroOverheadManager::execute_potential(
    const std::string& function_path, int natoms_actual, int nneigh_total,
    const double* const* atom_positions, const int* atom_types,
    const int* const* neighbor_lists, const int* neighbor_counts,
    const int* const* neighbor_types_lists, double volume,
    double& energy, double** forces, double* stress)
{
    if (!active_pool || !active_pool->is_initialized()) return false;

    active_pool->update_atom_data_zero_copy(
        natoms_actual, atom_positions, atom_types, 
        neighbor_lists, neighbor_counts, neighbor_types_lists
    );

    if (!cached_jax_function || cached_function_path != function_path) {
        if (!load_and_cache_jax_function(function_path)) return false;
    }

    *(int32_t*)PyArray_DATA((PyArrayObject*)active_pool->get_cell_rank_obj()) = 3;
    *(float*)PyArray_DATA((PyArrayObject*)active_pool->get_volume_obj()) = (float)volume;
    *(int32_t*)PyArray_DATA((PyArrayObject*)active_pool->get_natoms_actual_obj()) = natoms_actual;
    *(int32_t*)PyArray_DATA((PyArrayObject*)active_pool->get_nneigh_total_obj()) = nneigh_total;
    
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = PyObject_CallFunctionObjArgs(
        cached_jax_call_method,
        active_pool->get_types_buffer(),
        active_pool->get_neighbors_buffer(),
        active_pool->get_positions_buffer(),
        active_pool->get_neighbor_types_buffer(),
        active_pool->get_cell_rank_obj(),
        active_pool->get_volume_obj(),
        active_pool->get_natoms_actual_obj(),
        active_pool->get_natoms_actual_obj(),
        nullptr
    );
    
    bool success = true;
    if (!result) {
        std::cerr << "FATAL: JAX function call returned NULL" << std::endl;
        PyErr_Print();
        success = false;
    } else if (!PyTuple_Check(result) || PyTuple_Size(result) < 3) {
        std::cerr << "FATAL: JAX function did not return a tuple of size at least 3" << std::endl;
        success = false;
    } else {
        PyObject* energy_obj = PyTuple_GetItem(result, 0);
        PyObject* forces_obj = PyTuple_GetItem(result, 1);
        PyObject* stress_obj = PyTuple_GetItem(result, 2);

        PyObject *forces_np_obj = nullptr, *stress_np_obj = nullptr;
        
        PyObject* numpy_module = PyImport_ImportModule("numpy");
        if (!numpy_module) {
            PyErr_Print();
            std::cerr << "FATAL: Could not import numpy module" << std::endl;
            success = false;
        } else {
            PyObject* asarray_func = PyObject_GetAttrString(numpy_module, "asarray");
            if (asarray_func && PyCallable_Check(asarray_func)) {
                forces_np_obj = PyObject_CallFunctionObjArgs(asarray_func, forces_obj, nullptr);
                stress_np_obj = PyObject_CallFunctionObjArgs(asarray_func, stress_obj, nullptr);
            }
            Py_XDECREF(asarray_func);
            Py_DECREF(numpy_module);
        }

        if (!forces_np_obj || !stress_np_obj) {
            PyErr_Print();
            std::cerr << "FATAL: Failed to convert JAX arrays to NumPy arrays" << std::endl;
            success = false;
        } else if (!PyArray_Check(forces_np_obj) || !PyArray_Check(stress_np_obj)) {
            std::cerr << "FATAL: Returned forces or stress object is not a NumPy array" << std::endl;
            success = false;
        } else {
            energy = PyFloat_AsDouble(energy_obj);
            auto* f_np = (PyArrayObject*)forces_np_obj;
            auto* s_np = (PyArrayObject*)stress_np_obj;

            f_np = (PyArrayObject*)PyArray_Cast(f_np, NPY_DOUBLE);
            s_np = (PyArrayObject*)PyArray_Cast(s_np, NPY_DOUBLE);

            if (f_np && s_np) {
                std::memcpy(forces[0], PyArray_DATA(f_np), natoms_actual * 3 * sizeof(double));
                std::memcpy(stress, PyArray_DATA(s_np), 6 * sizeof(double));
                Py_DECREF(f_np);
                Py_DECREF(s_np);
            } else {
                 std::cerr << "FATAL: Could not cast JAX arrays to double precision" << std::endl;
                 PyErr_Print();
                 success = false;
            }
        }
        Py_XDECREF(forces_np_obj);
        Py_XDECREF(stress_np_obj);
    }
    Py_XDECREF(result);
    
    PyGILState_Release(gstate);
    return success;
}

bool ZeroOverheadManager::load_and_cache_jax_function(const std::string& function_path) {
    Py_XDECREF(cached_jax_function); Py_XDECREF(cached_jax_call_method);
    cached_jax_function = nullptr; cached_jax_call_method = nullptr;

    PyObject *pName = PyUnicode_FromString("jax.export");
    PyObject *pModule = PyImport_Import(pName); Py_DECREF(pName);
    if (!pModule) { PyErr_Print(); return false; }

    PyObject *pFunc = PyObject_GetAttrString(pModule, "deserialize"); Py_DECREF(pModule);
    if (!pFunc || !PyCallable_Check(pFunc)) { PyErr_Print(); Py_XDECREF(pFunc); return false; }
    
    FILE* file = fopen(function_path.c_str(), "rb");
    if (!file) { Py_DECREF(pFunc); return false; }
    fseek(file, 0, SEEK_END); long size = ftell(file); rewind(file);
    char* buffer = new char[size];
    size_t bytes_read = fread(buffer, 1, size, file); fclose(file);

    if (bytes_read != (size_t)size) { delete[] buffer; Py_DECREF(pFunc); return false; }

    PyObject* pBytes = PyBytes_FromStringAndSize(buffer, size); delete[] buffer;
    cached_jax_function = PyObject_CallFunctionObjArgs(pFunc, pBytes, nullptr);
    Py_DECREF(pBytes); Py_DECREF(pFunc);
    if (!cached_jax_function) { PyErr_Print(); return false; }

    cached_jax_call_method = PyObject_GetAttrString(cached_jax_function, "call");
    if (!cached_jax_call_method) { PyErr_Print(); return false; }
    
    return true;
}

void ZeroOverheadManager::cleanup() {
    if (!Py_IsInitialized()) {
        cached_jax_call_method = nullptr;
        cached_jax_function = nullptr;
        return;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    Py_XDECREF(cached_jax_call_method); Py_XDECREF(cached_jax_function);
    cached_jax_call_method = nullptr; cached_jax_function = nullptr;
    PyGILState_Release(gstate);
}

ZeroOverheadContext::ZeroOverheadContext(int max_atoms, int max_neighbors) {
    manager = ZeroOverheadManager::get_instance();
    initialized = manager->initialize_for_system(max_atoms, max_neighbors);
}
