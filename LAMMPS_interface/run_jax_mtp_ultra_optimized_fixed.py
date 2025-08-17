import os
import time
import numpy as np
import json
from functools import partial
import jax
import jax.numpy as jnp
from jax import export, device_put, pmap


from motep_original_files.jax_engine.moment_jax import MomentBasis
from motep_original_files.mtp import read_mtp
from motep_original_files.jax_engine.conversion import BasisConverter, get_moment_coeffs_from_mtp
from motep_jax_train_import import *

jax.config.update("jax_enable_x64", True)

def _extract_mtp_parameters(level, mtp_file):
        """Extract MTP parameters (same as testing script)"""
        mtp_data = _initialize_mtp(f'/home/sebastian/master_thesis/motep_jax_git/accuracy_tests/training_data/{mtp_file}.mtp')
        
        moment_basis = MomentBasis(level)
        moment_basis.init_moment_mappings()
        basis_converter = BasisConverter(moment_basis)
        basis_converter.remap_mlip_moment_coeffs(mtp_data)

        moment_coeffs = basis_converter.remapped_coeffs

        basic_moments = moment_basis.basic_moments
        scalar_contractions_str = moment_basis.scalar_contractions
        pair_contractions = moment_basis.pair_contractions
        execution_order_list, computed = _flatten_computation_graph(
            basic_moments, pair_contractions, scalar_contractions_str
        )
        
        print(f"Execution order length: {len(execution_order_list)}")
        print(f"Computed keys: {len(computed)}")
        print("First 10 execution steps:")
        for i, step in enumerate(execution_order_list[:10]):
            print(f"  {i}: {step}")
        if len(execution_order_list) > 10:
            print("  ...")
            print(f"  {len(execution_order_list)-1}: {execution_order_list[-1]}")
        
        execution_order = tuple(execution_order_list)
        scalar_contractions = tuple(scalar_contractions_str)
        species_coeffs = _totuple(mtp_data.species_coeffs)
        moment_coeffs = _totuple(moment_coeffs)
        radial_coeffs = _totuple(mtp_data.radial_coeffs)
        
        # Convert to Python primitives for static arguments (MUST be hashable!)
        def to_hashable_primitive(x):
            if hasattr(x, '__array__'):
                # Convert arrays to tuples (hashable) not lists (not hashable)
                return tuple(x.tolist()) if x.ndim > 0 else float(x)
            elif isinstance(x, list):
                # Convert lists to tuples for hashability 
                return tuple(to_hashable_primitive(item) for item in x)
            elif isinstance(x, tuple):
                return tuple(to_hashable_primitive(item) for item in x)
            else:
                return x

        species = to_hashable_primitive(mtp_data.species)
        scaling = float(mtp_data.scaling)
        min_dist = float(mtp_data.min_dist)  
        max_dist = float(mtp_data.max_dist)
        species_coeffs = to_hashable_primitive(species_coeffs)
        moment_coeffs = to_hashable_primitive(moment_coeffs)
        radial_coeffs = to_hashable_primitive(radial_coeffs)

        return species, scaling, min_dist, max_dist, species_coeffs, moment_coeffs, radial_coeffs, execution_order, scalar_contractions, basic_moments, pair_contractions

def _totuple(x):
    try:
        return tuple(_totuple(y) for y in x)
    except TypeError:
        return x

def _initialize_mtp(mtp_file):
    mtp_data = read_mtp(mtp_file)
    mtp_data.species = np.arange(0, mtp_data.species_count)
    return mtp_data

def _flatten_computation_graph(basic_moments, pair_contractions, scalar_contractions):
    """
    Create a complete topologically sorted execution list that includes:
    1. All basic moments
    2. ALL pair contractions that are needed (recursively) in dependency order
    3. This ensures every intermediate needed by scalar contractions is computed
    
    The key fix: we need to include ALL pair contractions, not just those whose
    immediate dependencies are basic moments. Some pair contractions depend on
    other pair contractions, creating deeper dependency chains.
    """
    execution_order = []
    computed = set()  # Track what's been computed
    needed = set()    # Track what we actually need to compute
    
    # First, find ALL moments that are actually needed
    # Start with basic moments and scalar contractions
    for moment_key in basic_moments:
        needed.add(moment_key)
    
    for scalar_key in scalar_contractions:
        needed.add(scalar_key)
    
    # Now recursively find all pair contractions that are needed
    # by tracing dependencies backwards
    changed = True
    while changed:
        changed = False
        old_needed_size = len(needed)
        
        for contraction_key in pair_contractions:
            # If this contraction is in needed, add its dependencies
            if contraction_key in needed:
                key_left, key_right = contraction_key[0], contraction_key[1]
                if key_left not in needed:
                    needed.add(key_left)
                    changed = True
                if key_right not in needed:
                    needed.add(key_right)
                    changed = True
            
            # Also check if this contraction's output is needed by anything
            # (this catches intermediate contractions)
            key_left, key_right = contraction_key[0], contraction_key[1]
            if key_left in needed or key_right in needed:
                if contraction_key not in needed:
                    needed.add(contraction_key)
                    changed = True
    
    # 1. Basic moments - these have no dependencies
    for moment_key in basic_moments:
        if moment_key in needed:
            execution_order.append(('basic', moment_key))
            computed.add(moment_key)
    
    # 2. Pair contractions - process in dependency order using topological sort
    # But only include those that are actually needed
    remaining_contractions = [c for c in pair_contractions if c in needed]
    max_iterations = len(remaining_contractions) * 2  # Prevent infinite loops
    iteration = 0
    
    while remaining_contractions and iteration < max_iterations:
        made_progress = False
        iteration += 1
        
        for i, contraction_key in enumerate(remaining_contractions):
            key_left, key_right, contraction_type, axes = contraction_key
            
            # Check if both dependencies are satisfied
            if key_left in computed and key_right in computed:
                execution_order.append(('contract', contraction_key))
                computed.add(contraction_key)
                remaining_contractions.pop(i)
                made_progress = True
                break
        
        if not made_progress:
            # Debug: show what's missing
            missing_deps = []
            for contraction_key in remaining_contractions:
                key_left, key_right = contraction_key[0], contraction_key[1]
                if key_left not in computed:
                    missing_deps.append(f"Left: {key_left}")
                if key_right not in computed:
                    missing_deps.append(f"Right: {key_right}")
            raise ValueError(f"Circular dependency in contraction graph. Missing: {missing_deps}")
    
    if remaining_contractions:
        raise ValueError(f"Failed to resolve all contractions. Remaining: {len(remaining_contractions)}")
    
    return execution_order, computed


# Use the same mtp file as the working MOTEP JAX implementation
level = 12
mtp_file = 'Ni3Al-12g'  # This matches the working implementation
species, scaling, min_dist, max_dist, species_coeffs, moment_coeffs, radial_coeffs, execution_order, scalar_contractions, basic_moments, pair_contractions = _extract_mtp_parameters(level, mtp_file)

# Create the same test atoms as the working MOTEP JAX implementation
# 4-atom Al system matching our MLIP2 test
positions = np.array([
    [1.38, 1.38, 1.38],  # Al atom 1
    [4.14, 4.14, 1.38],  # Al atom 2  
    [4.14, 1.38, 4.14],  # Al atom 3
    [1.38, 4.14, 4.14],  # Al atom 4
])

# Create cubic cell
cell = np.array([
    [5.52, 0.0, 0.0],
    [0.0, 5.52, 0.0], 
    [0.0, 0.0, 5.52]
])

# All atoms are Al (type 0 in our MTP)
natoms_actual = 4
itypes = jnp.array([0, 0, 0, 0], dtype=jnp.int32)

# Mock neighbor data - CORRECTED: structure for per-atom calculation
# Each atom has 3 neighbors (simplified for testing)
nneigh_per_atom = 3
nneigh_actual = natoms_actual * nneigh_per_atom

# Create neighbor data structured per-atom (what the ultra-optimized version expects)
# Shape should be [natoms, max_neighbors, 3] for r_ijs

import numpy as np

all_rijs = np.array([
    [
        [ 2.76,  2.76,  0.00],
        [ 2.76,  0.00,  2.76],
        [ 0.00,  2.76,  2.76],
        [ 0.00,  0.00,  5.52],
        [ 0.00,  5.52,  0.00],
        [ 5.52,  0.00,  0.00],
        [ 0.00,  0.00, -5.52],
        [ 0.00, -5.52,  0.00],
        [-5.52,  0.00,  0.00],
        [ 2.76, -2.76,  0.00],
        [-2.76,  2.76,  0.00],
        [-2.76, -2.76,  0.00],
        [ 2.76,  0.00, -2.76],
        [-2.76,  0.00,  2.76],
        [-2.76,  0.00, -2.76],
        [ 0.00,  2.76, -2.76],
        [ 0.00, -2.76,  2.76],
        [ 0.00, -2.76, -2.76]
    ],
    [
        [ 0.00, -2.76,  2.76],
        [-2.76,  0.00,  2.76],
        [ 0.00,  0.00,  5.52],
        [ 0.00,  2.76, -2.76],
        [-2.76,  2.76,  0.00],
        [ 0.00,  5.52,  0.00],
        [ 0.00,  2.76,  2.76],
        [ 2.76,  0.00, -2.76],
        [ 2.76, -2.76,  0.00],
        [ 5.52,  0.00,  0.00],
        [ 2.76,  0.00,  2.76],
        [ 2.76,  2.76,  0.00],
        [-2.76, -2.76,  0.00],
        [ 0.00,  0.00, -5.52],
        [ 0.00, -5.52,  0.00],
        [-5.52,  0.00,  0.00],
        [ 0.00, -2.76, -2.76],
        [-2.76,  0.00, -2.76]
    ],
    [
        [-2.76,  2.76,  0.00],
        [-2.76,  0.00,  2.76],
        [ 0.00,  2.76,  2.76],
        [ 0.00,  0.00,  5.52],
        [ 0.00,  5.52,  0.00],
        [ 2.76, -2.76,  0.00],
        [ 2.76,  0.00, -2.76],
        [ 5.52,  0.00,  0.00],
        [ 2.76,  2.76,  0.00],
        [ 2.76,  0.00,  2.76],
        [-2.76,  0.00, -2.76],
        [ 0.00,  2.76, -2.76],
        [ 0.00, -2.76,  2.76],
        [ 0.00, -2.76, -2.76],
        [ 0.00,  0.00, -5.52],
        [ 0.00, -5.52,  0.00],
        [-5.52,  0.00,  0.00],
        [-2.76, -2.76,  0.00]
    ],
    [
        [ 0.00, -2.76,  2.76],
        [ 2.76,  0.00,  2.76],
        [ 0.00,  0.00,  5.52],
        [ 0.00,  2.76, -2.76],
        [ 2.76,  2.76,  0.00],
        [ 0.00,  5.52,  0.00],
        [ 0.00,  2.76,  2.76],
        [ 5.52,  0.00,  0.00],
        [ 0.00, -2.76, -2.76],
        [ 2.76,  0.00, -2.76],
        [-2.76,  0.00,  2.76],
        [-2.76,  0.00, -2.76],
        [ 2.76, -2.76,  0.00],
        [-2.76,  2.76,  0.00],
        [-2.76, -2.76,  0.00],
        [ 0.00,  0.00, -5.52],
        [ 0.00, -5.52,  0.00],
        [-5.52,  0.00,  0.00]
    ]
])

all_js = np.array([
    [1, 2, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    [2, 3, 1, 2, 0, 1, 2, 3, 0, 1, 3, 0, 0, 1, 1, 1, 2, 3],
    [3, 0, 1, 2, 2, 3, 0, 2, 3, 0, 0, 1, 1, 1, 2, 2, 2, 3],
    [0, 1, 3, 0, 2, 3, 0, 3, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
])

all_jtypes = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])




# Cell and volume information
cell_rank = 3
volume = jnp.linalg.det(cell)

print(f"Test system setup:")
print(f"  natoms_actual: {natoms_actual}")
print(f"  nneigh_actual: {nneigh_actual}")
print(f"  volume: {volume}")
print(f"  mtp_file: {mtp_file}")


print(moment_coeffs)

accurate=False
# Apply JIT with correct static arguments
if accurate==True:
    from jax_mtp_ultra_optimized_accurate import calc_energy_forces_stress_padded_simple_ultra_optimized
    jit_calc = jax.jit(calc_energy_forces_stress_padded_simple_ultra_optimized, 
                    static_argnums=(8, 9, 10, 11, 12, 13, 14, 15, 16, 17))

    print("\nRunning ultra-optimized JAX calculation...")
    energy, forces, stress = jit_calc(
        itypes, all_js, all_rijs, all_jtypes,
        cell_rank, volume, natoms_actual, nneigh_actual,
        species, scaling, min_dist, max_dist,
        species_coeffs, moment_coeffs, radial_coeffs,
        basic_moments, pair_contractions, scalar_contractions
    )
else:
    from jax_mtp_ultra_optimized_fixed_accurate import calc_energy_forces_stress_padded_simple_ultra_optimized
    jit_calc = jax.jit(calc_energy_forces_stress_padded_simple_ultra_optimized, 
                    static_argnums=(8, 9, 10, 11, 12, 13, 14, 15, 16))

    print("\nRunning ultra-optimized JAX calculation...")
    energy, forces, stress = jit_calc(
        itypes, all_js, all_rijs, all_jtypes,
        cell_rank, volume, natoms_actual, nneigh_actual,
        species, scaling, min_dist, max_dist,
        species_coeffs, moment_coeffs, radial_coeffs,
        execution_order, scalar_contractions
    )

print(f"\nResults:")
print(f"Energy: {energy}")
print(f"Forces shape: {forces.shape}")
print(f"Forces: {forces}")
print(f"Stress: {stress}")

# Compare with expected values from working MOTEP JAX (energy: -8.662168636)
expected_energy = -8.662168636
energy_diff = abs(energy - expected_energy)
print(f"\nComparison with working MOTEP JAX:")
print(f"Expected energy: {expected_energy}")
print(f"Actual energy: {energy}")
print(f"Energy difference: {energy_diff}")
print(f"Match (within 1e-6): {energy_diff < 1e-6}")
