import os
import time
import numpy as np
import json
from functools import partial
import jax
import jax.numpy as jnp
from jax import export, device_put, pmap
from jax.export import deserialize


from motep_original_files.jax_engine.moment_jax import MomentBasis
from motep_original_files.mtp import read_mtp
from motep_original_files.jax_engine.conversion import BasisConverter #, get_moment_coeffs_from_mtp
from motep_jax_train_import import *

jax.config.update("jax_enable_x64", True)

def _extract_mtp_parameters(level, mtp_file):
        """Extract MTP parameters (same as testing script)"""
        mtp_data = _initialize_mtp(f'{mtp_file}.mtp')
        
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


level = 12
mtp_file = 'Ni3Al-12g'  
species, scaling, min_dist, max_dist, species_coeffs, moment_coeffs, radial_coeffs, execution_order, scalar_contractions, basic_moments, pair_contractions = _extract_mtp_parameters(level, mtp_file)


import re, os

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

cfgs = sorted(os.listdir("cfgs"), key=natural_sort_key)
print(cfgs)

bin_filenames = sorted(os.listdir("jax_functions_corrected"), key=natural_sort_key)
print(bin_filenames)


avg_times = []
min_times = []
avg_energies = []

avg_times_bin = []
min_times_bin = []
avg_energies_bin = []

natoms = []

from create_cfg import analyze_cfg
import sys
from current_jax_file import calc_energy_forces_stress_padded_simple_ultra_optimized 

for cfg_idx in range(len(cfgs)): 
    
    neighbor_data = analyze_cfg(f'cfgs/{cfgs[cfg_idx]}', cutoff=5.0)

    itypes = np.array(neighbor_data['itypes']).astype(np.int32)
    all_js = np.array(neighbor_data['all_js']).astype(np.int32)
    all_rijs = np.array(neighbor_data['all_rijs']).astype(np.float32)
    all_jtypes = np.array(neighbor_data['all_jtypes']).astype(np.int32)
    cell_rank = neighbor_data['cell_rank'].astype(np.int32)
    volume = neighbor_data['volume'].astype(np.float32)
    natoms_actual = neighbor_data['n_atoms']
    nneigh_actual = neighbor_data['n_neighbors']
    cell = _totuple(neighbor_data['cell'])
    positions = _totuple(neighbor_data['positions'])
    natoms_energy = natoms_actual
 
    max_dist = 5.0  

    @partial(jax.vmap, in_axes=(0, 0, 0, 0))
    def mask_arrays(itype, js, r_ijs, jtypes):
        r_squared = jnp.sum(r_ijs * r_ijs, axis=1)
        r_abs = jnp.sqrt(r_squared)
        valid_mask = r_abs < max_dist

        js_masked = jnp.where(valid_mask, js, -1)           
        rijs_masked = jnp.where(valid_mask[:, None], r_ijs, 20.0)  
        jtypes_masked = jnp.where(valid_mask, jtypes, -1)    
        itype_masked = itype 

        return itype_masked, js_masked, rijs_masked, jtypes_masked

    itypes, all_js, all_rijs, all_jtypes = mask_arrays(itypes, all_js, all_rijs, all_jtypes)

    natoms_actual = all_rijs.shape[0]
    nneigh_actual = all_rijs.shape[1]

    nneigh_actual = np.int32(nneigh_actual)
    natoms_energy = np.int32(natoms_energy)

    natoms.append(natoms_energy)


    jit_calc = jax.jit(calc_energy_forces_stress_padded_simple_ultra_optimized, 
                    static_argnums=(9, 10, 11, 12, 13, 14, 15, 16))

    print("\nWarming up JAX compilation...")
    for i in range(3):
        energy, forces, stress = jit_calc(
            itypes, all_js, all_rijs, all_jtypes,
            cell_rank, volume, natoms_energy, nneigh_actual,
            species, scaling, min_dist, max_dist,
            species_coeffs, moment_coeffs, radial_coeffs,
            execution_order, scalar_contractions
        )
        jax.block_until_ready((energy, forces, stress))

    print("Warmup complete!\n")
    print("MINIMAL OPTIMIZED DETAILED PROFILING RUN")

    import time

    energy_list = []
    times = []
    for i in range(20):
        start = time.perf_counter()
        energy, forces, stress = jit_calc(
            itypes, all_js, all_rijs, all_jtypes,
            cell_rank, volume, natoms_energy, nneigh_actual,
            species, scaling, min_dist, max_dist,
            species_coeffs, moment_coeffs, radial_coeffs,
            execution_order, scalar_contractions
        )

        jax.block_until_ready((energy, forces, stress))
        end = time.perf_counter()
        times.append((end - start) * 1000)
        energy_list.append(energy)


    avg_time = np.mean(times[5:])  
    std_time = np.std(times[5:])
    min_time = np.min(times[5:])
    avg_energy = np.mean(energy_list)

    avg_times.append(avg_time)
    min_times.append(min_time)
    avg_energies.append(avg_energy)

    bin_filename = bin_filenames[cfg_idx]

    with open(f'jax_functions_corrected/{bin_filename}', "rb") as f:
        serialized_data = f.read()

    loaded_func = deserialize(serialized_data)

    energy_bin = []
    times_bin = []
    for i in range(30):
        start = time.perf_counter()

        energy, forces, stress = loaded_func.call(itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, natoms_energy, nneigh_actual)

        jax.block_until_ready((energy, forces, stress))
        end = time.perf_counter()
        times_bin.append((end - start) * 1000)
        energy_bin.append(energy)

    avg_time_bin = np.mean(times_bin[10:])  
    std_time_bin = np.std(times_bin[10:])
    min_time_bin = np.min(times_bin[10:])
    avg_energy_bin = np.mean(energy_bin)

    avg_times_bin.append(avg_time_bin)
    min_times_bin.append(min_time_bin)
    avg_energies_bin.append(avg_energy_bin)


import pandas as pd

avg_times = np.array(avg_times)
min_times = np.array(min_times)
natoms    = np.array(natoms)
avg_energies = np.array(avg_energies)

df = pd.DataFrame({
    "natoms": natoms,
    "avg_time_ms": avg_times,
    "min_time_ms": min_times,
    "energies": avg_energies,
})

df.to_csv("jax_timing_data.csv", index=False)

print("Saved timing data to jax_timing_data.csv")


avg_times_bin = np.array(avg_times_bin)
min_times_bin = np.array(min_times_bin)
natoms    = np.array(natoms)
avg_energies_bin = np.array(avg_energies_bin)

df = pd.DataFrame({
    "natoms": natoms,
    "avg_time_ms": avg_times_bin,
    "min_time_ms": min_times_bin,
    "energies": avg_energies_bin,
})

df.to_csv("jax_timing_data_bin_16.csv", index=False)

print("Saved timing data to jax_timing_data_bin.csv")

