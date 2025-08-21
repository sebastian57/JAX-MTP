#ifndef ZERO_OVERHEAD_BUFFER_MANAGER_ORIGINAL_HPP
#define ZERO_OVERHEAD_BUFFER_MANAGER_ORIGINAL_HPP

// Enable debug output for timing analysis
// #define ZO_DEBUG

// Standard
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Python/NumPy C API
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace ZeroOverheadOriginal {

/* --------------------------------------------------------------------------
 * PersistentMemoryPool
 *
 * Encapsulates persistent NumPy arrays that are reused across timesteps.
 * ----------------------------------------------------------------------- */
class PersistentMemoryPool {
public:
    struct Config {
        int max_atoms = 0;
        int max_neighbors = 0;
        int species_count = 0;
        bool initialized = false;
        size_t estimated_memory_mb = 0;
    };

    struct PerformanceStats {
        int allocation_count = 0;
        int reuse_count = 0;
        double total_allocation_time_ms = 0.0;
        double reuse_ratio = 0.0;
    };

    PersistentMemoryPool() = default;
    ~PersistentMemoryPool();

    bool initialize(int max_atoms, int max_neighbors, int species_count);
    void cleanup();

    bool is_initialized() const { return pool_initialized; }

    // Update pool arrays from C-side buffers (zero-copy where possible)
    void update_atom_data_zero_copy(
        int actual_atoms,
        int actual_neighbors,
        const double* const* atom_positions,
        const int* atom_types,
        const int* const* neighbor_lists,
        const int* neighbor_counts,
        const int* const* neighbor_types_lists
    );
    
    // MLIP2 comparison logging
    void log_for_mlip2_comparison(
        const std::string& filename,
        int actual_atoms,
        const double* const* atom_positions,
        const int* atom_types,
        const int* neighbor_counts,
        double volume,
        double energy = 0.0,
        double** forces = nullptr
    );

    // Accessors to provide PyObject* buffers to caller
    PyObject* get_types_buffer() const { return itypes_array; }
    PyObject* get_neighbors_buffer() const { return all_js_array; }
    PyObject* get_positions_buffer() const { return all_rijs_array; }
    PyObject* get_neighbor_types_buffer() const { return all_jtypes_array; }
    PyObject* get_cell_rank_obj() const { return cell_rank_obj; }
    PyObject* get_volume_obj() const { return volume_obj; }
    PyObject* get_natoms_obj() const { return natoms_actual_obj; }
    PyObject* get_nneigh_obj() const { return nneigh_actual_obj; }

    PerformanceStats get_performance_stats() const;

private:
    // Arrays (owned PyObject* references)
    PyObject* itypes_array = nullptr;     // NPY_INT32 [max_atoms]
    PyObject* all_js_array = nullptr;      // NPY_INT32 [max_atoms, max_neighbors]
    PyObject* all_rijs_array = nullptr;    // NPY_FLOAT32 [max_atoms, max_neighbors, 3]
    PyObject* all_jtypes_array = nullptr;  // NPY_INT32 [max_atoms, max_neighbors]

    // Scalars
    PyObject* cell_rank_obj = nullptr;     // int32 scalar
    PyObject* volume_obj = nullptr;        // float32 scalar
    PyObject* natoms_actual_obj = nullptr; // int32 scalar
    PyObject* nneigh_actual_obj = nullptr; // int32 scalar

    // Result buffers
    PyObject* result_energy = nullptr;     // NPY_FLOAT64 scalar
    PyObject* result_forces = nullptr;     // NPY_FLOAT32 [max_atoms, 3]  (fast path)
    PyObject* result_stress = nullptr;     // NPY_FLOAT64 [6]

    // Pool bookkeeping
    Config config;
    bool pool_initialized = false;
    int allocation_count = 0;
    int reuse_count = 0;
    double total_allocation_time = 0.0;

    // Synchronization for pool internal operations
    std::mutex pool_mutex;

    // Internal helpers
    bool initialize_numpy_api();
};

/* --------------------------------------------------------------------------
 * ZeroOverheadOriginalManager
 *
 * Singleton that manages pools and cached JAX artifacts. This now owns cached
 * Python objects (jax.export module and its deserialize function).
 * ----------------------------------------------------------------------- */
class ZeroOverheadOriginalManager {
public:
    // Singleton access
    static ZeroOverheadOriginalManager* get_instance();

    // Initialize per-system pool (creates PersistentMemoryPool if needed)
    bool initialize_for_system(int max_atoms, int max_neighbors, int species_count);

    // Get a pool by key (or nullptr)
    PersistentMemoryPool* get_memory_pool(int max_atoms, int max_neighbors);

    // Call JAX using the zero-overhead path (fills energy, forces, stress)
    bool call_jax_zero_overhead(
        const std::string& function_path,
        int actual_atoms,
        int actual_neighbors,
        const double* const* atom_positions,
        const int* atom_types,
        const int* const* neighbor_lists,
        const int* neighbor_counts,
        const int* const* neighbor_types_lists,
        double volume,
        double& energy,
        double** forces,
        double* stress
    );
    
    // Ultra-optimized JAX call (eliminates Python overhead from hot path)
    bool call_jax_ultra_optimized(
        const std::string& function_path,
        int actual_atoms,
        int actual_neighbors,
        const double* const* atom_positions,
        const int* atom_types,
        const int* const* neighbor_lists,
        const int* neighbor_counts,
        const int* const* neighbor_types_lists,
        double volume,
        double& energy,
        double** forces,
        double* stress
    );
    
    // CRITICAL FIX: Ultra-optimized JAX call with rectangular arrays and zero overhead
    // Converts rectangular arrays to efficient format while maintaining all optimizations
    bool call_jax_ultra_optimized_rectangular(
        const std::string& function_path,
        int actual_atoms,
        int max_neighbors,
        const std::vector<std::vector<std::vector<double>>>& all_rijs,  // (natoms, max_neighbors, 3)
        const std::vector<int>& atom_types,                             // (natoms,)
        const std::vector<std::vector<int>>& all_js,                    // (natoms, max_neighbors)
        const std::vector<std::vector<int>>& all_jtypes,               // (natoms, max_neighbors)
        const std::vector<int>& neighbor_counts,                       // (natoms,) actual neighbor counts
        double volume,
        double& energy,
        double** forces,
        double* stress
    );

    // Enable / disable batching
    void enable_batching(int batch_size);
    void disable_batching();

    // Cleanup manager/caches
    void cleanup();

    // Get active pool
    PersistentMemoryPool* get_active_pool() { return active_pool; }
    
    // Get comprehensive timing statistics
    struct ComprehensiveTimingStats {
        int total_calls = 0;
        double avg_total_time_ms = 0.0;
        double avg_data_prep_ms = 0.0;
        double avg_jax_init_ms = 0.0;
        double avg_jax_call_ms = 0.0;
        double avg_result_processing_ms = 0.0;
        double total_time_ms = 0.0;
    };
    ComprehensiveTimingStats get_comprehensive_timing_stats();
    
    // Ultra-optimization initialization
    bool initialize_ultra_optimization(const std::string& function_path);
    
    // Initialize cached Python helpers (GIL-safe)
    bool initialize_cached_python_helpers();
    
    // Cached NumPy conversion (eliminates repeated imports)
    PyObject* convert_to_numpy_cached(PyObject* obj);

public:
    // Constructor/destructor public for singleton pattern
    ZeroOverheadOriginalManager() = default;
    ~ZeroOverheadOriginalManager() = default;

    // Non-copyable
    ZeroOverheadOriginalManager(const ZeroOverheadOriginalManager&) = delete;
    ZeroOverheadOriginalManager& operator=(const ZeroOverheadOriginalManager&) = delete;

private:

    // Pools keyed by "NxM"
    std::map<std::string, std::unique_ptr<PersistentMemoryPool>> memory_pools;
    std::mutex pools_mutex;

    // Active pool
    PersistentMemoryPool* active_pool = nullptr;
    std::string active_pool_key;

    // Cached JAX objects for reuse (moved into manager as requested)
    PyObject* cached_jax_function = nullptr;
    PyObject* cached_jax_call_method = nullptr; // Cache the call method (eliminates lookup overhead)
    PyObject* cached_jax_export_module = nullptr;
    PyObject* cached_deserialize_func = nullptr;
    std::string cached_function_path;
    
    // Cached NumPy helpers (eliminates import/lookup overhead)
    PyObject* cached_numpy_module = nullptr;
    PyObject* cached_numpy_asarray = nullptr;
    
    // Pre-bound Python namespace for zero-overhead calls
    PyObject* main_module = nullptr;
    PyObject* main_dict = nullptr;
    
    // Ultra-optimization: Eliminate Python from hot path
    struct UltraOptimization {
        bool enabled = false;
        
        // Pre-compiled Python code strings (execute once at init)
        std::string cached_jax_call_code;
        
        // Pre-allocated result memory (direct C++ access)
        double* result_energy_ptr = nullptr;
        double* result_forces_ptr = nullptr;  
        double* result_stress_ptr = nullptr;
        
        // Persistent Python namespace (avoid repeated lookups)
        PyObject* persistent_globals = nullptr;
        
        // Direct buffer protocol access (zero-copy)
        Py_buffer energy_buffer;
        Py_buffer forces_buffer;  
        Py_buffer stress_buffer;
        bool buffers_initialized = false;
    };
    UltraOptimization ultra_opt;

    // Batching state (simple)
    struct BatchState {
        bool batching_enabled = false;
        int batch_size = 0;
        int current_batch = 0;
        std::vector<double> batched_energies;
        std::vector<std::vector<double>> batched_forces;
        bool has_pending_results = false;
    } batch_state;

    // Comprehensive timing statistics (class members)
    int timing_total_calls = 0;
    double timing_total_time_ms = 0.0;
    double timing_total_data_prep_ms = 0.0;
    double timing_total_jax_init_ms = 0.0;
    double timing_total_jax_call_ms = 0.0;
    double timing_total_result_processing_ms = 0.0;
    
    // For singleton
    static std::unique_ptr<ZeroOverheadOriginalManager> instance;
    static std::mutex instance_mutex;
};

/* --------------------------------------------------------------------------
 * ZeroOverheadOriginalContext - lightweight RAII around manager/pool selection
 * ----------------------------------------------------------------------- */
class ZeroOverheadOriginalContext {
public:
    ZeroOverheadOriginalContext(int max_atoms, int max_neighbors, int species_count = 1);
    ~ZeroOverheadOriginalContext();

    ZeroOverheadOriginalManager* get_manager() { return manager; }
    bool is_ready() const { return initialized; }

private:
    ZeroOverheadOriginalManager* manager = nullptr;
    bool initialized = false;
};

/* --------------------------------------------------------------------------
 * OverheadOriginalProfiler - simple profiler used by the pair-style for breakdowns
 * ----------------------------------------------------------------------- */
class OverheadOriginalProfiler {
public:
    struct ProfilePoint {
        std::string name;
        double start_ms;
        double duration_ms;
    };

    class ScopedTimer {
    public:
        ScopedTimer(OverheadOriginalProfiler* prof, const std::string& name);
        ~ScopedTimer();
    private:
        OverheadOriginalProfiler* profiler = nullptr;
        std::string operation_name;
        double start_time;
    };

    OverheadOriginalProfiler() = default;
    ~OverheadOriginalProfiler() = default;

    OverheadOriginalProfiler(const OverheadOriginalProfiler&) = delete;
    OverheadOriginalProfiler& operator=(const OverheadOriginalProfiler&) = delete;

    struct OverheadBreakdown {
        double total_time_ms = 0.0;
        double allocation_overhead_ms = 0.0;
        double transfer_overhead_ms = 0.0;
        double conversion_overhead_ms = 0.0;
        double jax_call_overhead_ms = 0.0;
        double computation_time_ms = 0.0;
        double overhead_percentage = 0.0;
    };

    OverheadBreakdown analyze_overhead() const;
    void reset_profiling();

private:
    mutable std::mutex profiler_mutex;
    std::vector<ProfilePoint> profile_points;
};

// Convenience macros for automatic timing with debug levels
#define PROFILE_SCOPE(profiler, name) \
    OverheadOriginalProfiler::ScopedTimer _timer(profiler, name)

#define DEBUG_PROFILE_SCOPE(profiler, name, debug_level, required_level) \
    std::unique_ptr<OverheadOriginalProfiler::ScopedTimer> _debug_timer; \
    if (debug_level >= required_level) { \
        _debug_timer = std::make_unique<OverheadOriginalProfiler::ScopedTimer>(profiler, name); \
    }

#define DEBUG_LOG(debug_level, required_level, msg) \
    if (debug_level >= required_level) { \
        std::cout << "[DEBUG] " << msg << std::endl; \
    }

#define DEBUG_LOG_DATA(debug_level, required_level, label, value) \
    if (debug_level >= required_level) { \
        std::cout << "[DEBUG] " << label << ": " << value << std::endl; \
    }

/* -------------------------------------------------------------------------- */
} // namespace ZeroOverheadOriginal

#endif // ZERO_OVERHEAD_BUFFER_MANAGER_ORIGINAL_HPP
