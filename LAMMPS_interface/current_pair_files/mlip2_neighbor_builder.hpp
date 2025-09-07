// mlip2_neighbor_builder.hpp
// High-performance MLIP2-compatible neighbor list builder for LAMMPS integration

#ifndef MLIP2_NEIGHBOR_BUILDER_HPP
#define MLIP2_NEIGHBOR_BUILDER_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstddef>      // For alignas
#include <array>        // For std::array
#include <map>          // For std::map (used in validation functions)
#include <unordered_map> // For spatial hashing
#include <cstdint>      // For int64_t hash keys

// FIXED: Renamed namespace to avoid confusion with class name
namespace MLIP2Utils {

struct Vector3 {
    double x, y, z;
    
    Vector3() : x(0), y(0), z(0) {}
    Vector3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    double& operator[](int i) { return (&x)[i]; }
    const double& operator[](int i) const { return (&x)[i]; }
    
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
};

// PHASE 2: BoxInfo struct for optimized memory access patterns
struct BoxInfo {
    double box_x, box_y, box_z;                    // Box dimensions
    double inv_box_x, inv_box_y, inv_box_z;        // Inverse dimensions for faster boundary checks
    Vector3 lattice_vectors[3];                    // Pre-computed lattice vectors
    
    BoxInfo() : box_x(0), box_y(0), box_z(0), inv_box_x(0), inv_box_y(0), inv_box_z(0) {}
};

// PHASE 5: SIMD-friendly aligned Vector3 for compiler auto-vectorization
struct alignas(16) Vector3Aligned {
    double x, y, z, padding;  // Padding for 16-byte alignment
    
    Vector3Aligned() : x(0), y(0), z(0), padding(0) {}
    Vector3Aligned(double x_, double y_, double z_) : x(x_), y(y_), z(z_), padding(0) {}
    Vector3Aligned(const Vector3& v) : x(v.x), y(v.y), z(v.z), padding(0) {}
    
    // Compiler-friendly distance calculation for auto-vectorization
    inline double distance_squared(const Vector3Aligned& other) const {
        double dx = x - other.x;
        double dy = y - other.y; 
        double dz = z - other.z;
        return dx*dx + dy*dy + dz*dz; // Compiler can auto-vectorize this
    }
    
    // Convert to regular Vector3
    Vector3 to_vector3() const {
        return Vector3(x, y, z);
    }
    
    Vector3Aligned operator+(const Vector3Aligned& other) const {
        return Vector3Aligned(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3Aligned operator-(const Vector3Aligned& other) const {
        return Vector3Aligned(x - other.x, y - other.y, z - other.z);
    }
};

// PHASE 7: Spatial Hash Grid for O(N) neighbor finding
class SpatialHashGrid {
private:
    double cell_size;
    double inv_cell_size;  // Precomputed 1/cell_size for faster division
    std::unordered_map<int64_t, std::vector<int>> cells;
    BoxInfo box_info;  // Store box information for boundary handling
    
    // Hash function for 3D coordinates to 1D hash key
    int64_t hash_position(const Vector3& pos) const {
        // Convert position to cell coordinates
        int32_t cx = static_cast<int32_t>(std::floor(pos.x * inv_cell_size));
        int32_t cy = static_cast<int32_t>(std::floor(pos.y * inv_cell_size));
        int32_t cz = static_cast<int32_t>(std::floor(pos.z * inv_cell_size));
        
        // Combine into 64-bit hash (21 bits per coordinate + sign bit)
        int64_t hash = 0;
        hash |= (static_cast<int64_t>(cx & 0x1FFFFF)) << 42;
        hash |= (static_cast<int64_t>(cy & 0x1FFFFF)) << 21;
        hash |= (static_cast<int64_t>(cz & 0x1FFFFF));
        return hash;
    }
    
    // Get neighboring cell coordinates within cutoff distance
    void get_neighbor_cells(const Vector3& pos, double cutoff, std::vector<int64_t>& neighbor_cell_hashes) const {
        neighbor_cell_hashes.clear();
        
        // Calculate cell range to check
        int cell_range = static_cast<int>(std::ceil(cutoff * inv_cell_size)) + 1;
        
        int32_t center_cx = static_cast<int32_t>(std::floor(pos.x * inv_cell_size));
        int32_t center_cy = static_cast<int32_t>(std::floor(pos.y * inv_cell_size));
        int32_t center_cz = static_cast<int32_t>(std::floor(pos.z * inv_cell_size));
        
        for (int dx = -cell_range; dx <= cell_range; dx++) {
            for (int dy = -cell_range; dy <= cell_range; dy++) {
                for (int dz = -cell_range; dz <= cell_range; dz++) {
                    Vector3 cell_pos;
                    cell_pos.x = (center_cx + dx) * cell_size;
                    cell_pos.y = (center_cy + dy) * cell_size;
                    cell_pos.z = (center_cz + dz) * cell_size;
                    
                    // Quick distance check: only include cells within cutoff sphere
                    double dist = sqrt((cell_pos.x - pos.x) * (cell_pos.x - pos.x) +
                                     (cell_pos.y - pos.y) * (cell_pos.y - pos.y) +
                                     (cell_pos.z - pos.z) * (cell_pos.z - pos.z));
                    
                    if (dist <= cutoff + cell_size * 1.414) { // Add cell diagonal safety margin
                        int64_t hash = 0;
                        int32_t cx = center_cx + dx;
                        int32_t cy = center_cy + dy;
                        int32_t cz = center_cz + dz;
                        
                        hash |= (static_cast<int64_t>(cx & 0x1FFFFF)) << 42;
                        hash |= (static_cast<int64_t>(cy & 0x1FFFFF)) << 21;
                        hash |= (static_cast<int64_t>(cz & 0x1FFFFF));
                        neighbor_cell_hashes.push_back(hash);
                    }
                }
            }
        }
    }

public:
    SpatialHashGrid() : cell_size(1.0), inv_cell_size(1.0) {}
    
    // Initialize the grid with optimal cell size
    void initialize(double cutoff_distance, const BoxInfo& box_info_ref) {
        // Optimal cell size is typically cutoff/2 to cutoff for best performance
        cell_size = cutoff_distance * 0.5;  // Start with cutoff/2
        inv_cell_size = 1.0 / cell_size;
        box_info = box_info_ref;
        cells.clear();
    }
    
    // Populate the grid with atom positions
    void populate_grid(const std::vector<Vector3>& positions) {
        cells.clear();
        for (size_t i = 0; i < positions.size(); i++) {
            int64_t hash = hash_position(positions[i]);
            cells[hash].push_back(static_cast<int>(i));
        }
    }
    
    // Get nearby atoms within cutoff distance using spatial hashing
    void get_nearby_atoms(const Vector3& pos, double cutoff, std::vector<int>& nearby_atoms) const {
        nearby_atoms.clear();
        
        std::vector<int64_t> neighbor_cell_hashes;
        get_neighbor_cells(pos, cutoff, neighbor_cell_hashes);
        
        for (int64_t cell_hash : neighbor_cell_hashes) {
            auto it = cells.find(cell_hash);
            if (it != cells.end()) {
                const std::vector<int>& cell_atoms = it->second;
                nearby_atoms.insert(nearby_atoms.end(), cell_atoms.begin(), cell_atoms.end());
            }
        }
    }
    
    // Get performance statistics
    struct HashStats {
        size_t total_cells = 0;
        size_t occupied_cells = 0;
        double average_atoms_per_cell = 0.0;
        size_t max_atoms_per_cell = 0;
        double load_factor = 0.0;
    };
    
    HashStats get_statistics(int total_atoms) const {
        HashStats stats;
        stats.occupied_cells = cells.size();
        
        size_t total_cell_atoms = 0;
        for (const auto& cell : cells) {
            total_cell_atoms += cell.second.size();
            stats.max_atoms_per_cell = std::max(stats.max_atoms_per_cell, cell.second.size());
        }
        
        if (stats.occupied_cells > 0) {
            stats.average_atoms_per_cell = static_cast<double>(total_cell_atoms) / stats.occupied_cells;
        }
        
        stats.load_factor = static_cast<double>(total_cell_atoms) / total_atoms;
        return stats;
    }
    
    // Optimize cell size based on atom density and cutoff
    void optimize_cell_size(double cutoff_distance, int natoms, const BoxInfo& box_info_ref) {
        double volume = box_info_ref.box_x * box_info_ref.box_y * box_info_ref.box_z;
        double density = natoms / volume;
        
        // Optimal cell size balances cell count vs atoms per cell
        // For high density: smaller cells, for low density: larger cells
        double optimal_atoms_per_cell = 8.0;  // Target 8 atoms per cell on average
        double estimated_cell_volume = optimal_atoms_per_cell / density;
        double estimated_cell_size = std::cbrt(estimated_cell_volume);
        
        // Constrain cell size to reasonable bounds relative to cutoff
        cell_size = std::max(cutoff_distance * 0.3, 
                            std::min(cutoff_distance * 0.8, estimated_cell_size));
        inv_cell_size = 1.0 / cell_size;
        
        box_info = box_info_ref;
    }
};

// PHASE 10: Memory pool for temporary calculations
class NeighborMemoryPool {
private:
    // Pre-allocated vectors for temporary use
    mutable std::vector<Vector3> temp_positions;
    mutable std::vector<double> temp_distances_sq;
    mutable std::vector<int> temp_atom_indices;
    mutable std::vector<std::array<int,3>> temp_images;
    mutable std::vector<int64_t> temp_cell_hashes;
    
    // Pool statistics
    mutable int pool_reuses = 0;
    mutable int pool_resizes = 0;

public:
    NeighborMemoryPool() = default;
    
    // Initialize pool with estimated sizes
    void initialize(int max_atoms, int max_neighbors) {
        temp_positions.reserve(max_neighbors);
        temp_distances_sq.reserve(max_neighbors);
        temp_atom_indices.reserve(max_neighbors);
        temp_images.reserve(27); // Maximum periodic images
        temp_cell_hashes.reserve(64); // Estimated neighbor cells
    }
    
    // Get temporary vectors (reuse existing memory)
    std::vector<Vector3>& get_temp_positions() const {
        temp_positions.clear();
        pool_reuses++;
        return temp_positions;
    }
    
    std::vector<double>& get_temp_distances() const {
        temp_distances_sq.clear();
        pool_reuses++;
        return temp_distances_sq;
    }
    
    std::vector<int>& get_temp_atom_indices() const {
        temp_atom_indices.clear();
        pool_reuses++;
        return temp_atom_indices;
    }
    
    std::vector<std::array<int,3>>& get_temp_images() const {
        temp_images.clear();
        pool_reuses++;
        return temp_images;
    }
    
    std::vector<int64_t>& get_temp_cell_hashes() const {
        temp_cell_hashes.clear();
        pool_reuses++;
        return temp_cell_hashes;
    }
    
    // Pool statistics
    struct PoolStats {
        int reuses = 0;
        int resizes = 0;
        double reuse_efficiency = 0.0;
    };
    
    PoolStats get_statistics() const {
        PoolStats stats;
        stats.reuses = pool_reuses;
        stats.resizes = pool_resizes;
        if (pool_reuses > 0) {
            stats.reuse_efficiency = 100.0 * (1.0 - static_cast<double>(pool_resizes) / pool_reuses);
        }
        return stats;
    }
    
    // Check and potentially resize pool if needed
    void ensure_capacity(size_t required_size) const {
        if (temp_positions.capacity() < required_size) {
            temp_positions.reserve(required_size);
            pool_resizes++;
        }
        if (temp_distances_sq.capacity() < required_size) {
            temp_distances_sq.reserve(required_size);
            pool_resizes++;
        }
        if (temp_atom_indices.capacity() < required_size) {
            temp_atom_indices.reserve(required_size);
            pool_resizes++;
        }
    }
};

// ASE-style padded data structure (not per-atom lists!)
struct ASENeighborData {
    std::vector<std::vector<int>> all_js;           // [natoms][max_neighbors] padded with -1
    std::vector<std::vector<Vector3>> all_rijs;     // [natoms][max_neighbors] relative positions
    std::vector<std::vector<int>> all_jtypes;       // [natoms][max_neighbors] neighbor types, -1 for padding
    int max_neighbors;                              // Maximum neighbors per atom
    int total_neighbors;                            // Total valid neighbors
    
    ASENeighborData() : max_neighbors(0), total_neighbors(0) {}
    
    void clear() {
        all_js.clear();
        all_rijs.clear();
        all_jtypes.clear();
        max_neighbors = 0;
        total_neighbors = 0;
    }
};

class MLIP2NeighborBuilder {
private:
    double cutoff;
    std::vector<Vector3> atom_positions;      // Original atom positions only
    std::vector<int> atom_types;              // Original atom types (0-based)
    ASENeighborData neighbor_data;            // ASE-style padded neighbor data
    
    // PHASE 7: Spatial hash grid for O(N) neighbor finding
    SpatialHashGrid spatial_grid;
    
    // PHASE 10: Memory pool for optimized temporary allocations
    NeighborMemoryPool memory_pool;
    
    // Persistent storage for JAX interface (flattened format)
    std::vector<std::vector<double>> position_storage;
    std::vector<std::vector<int>> neighbor_index_storage;
    std::vector<std::vector<int>> neighbor_type_storage;

public:
    MLIP2NeighborBuilder() : cutoff(0.0) {}
    
    // Main public interface - LAMMPS neighbor list processor  
    void build_neighbor_lists(
        double** atom_positions,
        int* atom_types, 
        int natoms,
        double lattice[3][3],
        double cutoff_distance,
        // LAMMPS neighbor list data
        int* ilist,
        int* numneigh,
        int** firstneigh,
        std::vector<const double*>& output_positions,
        std::vector<int>& output_types,
        std::vector<const int*>& output_neighbor_lists,
        std::vector<int>& output_neighbor_counts,
        std::vector<const int*>& output_neighbor_types,
        int debug_level = 0
    );
    
private:
    // Build ASE-compatible neighbor lists using periodic boundary conditions
    void build_ase_neighbors(int natoms, double lattice[3][3], double cutoff_distance);
    
    // PHASE 2: BoxInfo calculation for optimized access patterns
    BoxInfo calculate_box_info(double lattice[3][3]);
    
    // PHASE 4: Memory pre-allocation optimization
    int estimate_max_neighbors(double cutoff_distance, const BoxInfo& box_info);
    
    // PHASE 7: Spatial hash-based neighbor finding functions
    void build_ase_neighbors_spatial_hash(int natoms, double lattice[3][3], double cutoff_distance, int debug_level = 0);
    void find_neighbors_for_atom_spatial(int atom_i, const Vector3& pos_i, double cutoff_distance,
                                        const std::vector<std::array<int,3>>& relevant_images,
                                        const BoxInfo& box_info, int natoms,
                                        std::vector<int>& js_i, std::vector<Vector3>& rijs_i, 
                                        std::vector<int>& jtypes_i, int& distance_calculations,
                                        int& early_rejections, int& bounding_box_rejections);
    
    // PHASE 8: Early distance rejection functions  
    inline bool quick_distance_check(const Vector3& pos1, const Vector3& pos2, double cutoff) const;
    inline bool bounding_box_check(const Vector3& pos1, const Vector3& pos2, double cutoff) const;
    
    // PHASE 9: Enhanced SIMD vectorization functions
    void calculate_distances_simd(const Vector3& ref_pos, const std::vector<int>& atom_indices, 
                                 const Vector3& image_offset, std::vector<double>& distances_sq,
                                 int& simd_operations) const;
    inline void prefetch_atom_data(int atom_index) const;
    
    // Validation functions for correctness verification
    #ifdef DEBUG_OPTIMIZATION
    void validate_neighbor_lists(const ASENeighborData& reference, const ASENeighborData& optimized, int natoms);
    bool compare_neighbor_data(const ASENeighborData& data1, const ASENeighborData& data2, int natoms, double tolerance = 1e-15);
    void log_neighbor_statistics(const ASENeighborData& data, int natoms, const std::string& label);
    #endif
    
    // Smart periodic image optimization functions (updated to use BoxInfo)
    bool is_near_periodic_boundary_optimized(const Vector3& pos, const BoxInfo& box_info, double cutoff_distance);
    void determine_relevant_periodic_images_optimized(const Vector3& pos, const BoxInfo& box_info, double cutoff_distance, 
                                                     std::vector<std::array<int,3>>& relevant_images);
    
    // Legacy functions (kept for compatibility)
    bool is_near_periodic_boundary(const Vector3& pos, double lattice[3][3], double cutoff_distance);
    void determine_relevant_periodic_images(const Vector3& pos, double lattice[3][3], double cutoff_distance, 
                                           std::vector<std::array<int,3>>& relevant_images);
    
    // Convert to JAX-compatible padded arrays format
    void prepare_jax_data(
        int natoms,
        std::vector<const double*>& output_positions,
        std::vector<int>& output_types,
        std::vector<const int*>& output_neighbor_lists,
        std::vector<int>& output_neighbor_counts,
        std::vector<const int*>& output_neighbor_types
    );
    
    inline double distance(const Vector3& a, const Vector3& b) const {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dz = a.z - b.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    bool is_periodic_dimension(double lattice[3][3], int dim) const {
        return lattice[dim][0] != 0.0
            || lattice[dim][1] != 0.0
            || lattice[dim][2] != 0.0;
    }
};

} // namespace MLIP2Utils

#endif // MLIP2_NEIGHBOR_BUILDER_HPP