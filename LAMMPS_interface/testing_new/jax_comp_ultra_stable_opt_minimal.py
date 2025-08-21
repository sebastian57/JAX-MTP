#!/usr/bin/env python3
"""
JAX-MTP MINIMAL OPTIMIZED Implementation
EXACT COPY of first optimization + ONLY force accumulation fix

Strategy: Start with EXACT working first optimization (0.84ms)
Apply ONLY the force accumulation fix with zero overhead
NO other changes that could slow things down
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from functools import partial
import string
import time

jax.config.update("jax_enable_x64", False)

# Import ultra-stable precision configuration
import sys
COMPUTE_DTYPE = jnp.float32       # Maintain exact precision  
STABLE_DTYPE = jnp.float32        # Critical calculations
OUTPUT_DTYPE = jnp.float32       # Final outputs

print("🚀 JAX-MTP MINIMAL OPTIMIZED Implementation")
print("   Target: EXACT first optimization (0.84ms) + force fix ONLY")

class ProfileTimer:
    """EXACT copy from first optimization"""
    
    def __init__(self, name, enable_profiling=True):
        self.name = name
        self.enable_profiling = enable_profiling
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        if self.enable_profiling:
            jax.block_until_ready(jnp.array(0.0))
            self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable_profiling and self.start_time is not None:
            jax.block_until_ready(jnp.array(0.0))
            self.end_time = time.perf_counter()
            duration_ms = (self.end_time - self.start_time) * 1000
            print(f"⏱️  {self.name}: {duration_ms:.3f} ms")

def profile_component_timings(func_name, result, enable_profiling=True):
    """EXACT copy from first optimization"""
    if enable_profiling:
        try:
            jax.block_until_ready(result)
        except:
            pass

# ============================================================================
# EXACT COPY: First optimization functions
# ============================================================================

def precompute_unit_vectors_and_powers(r_ijs, r_abs, max_nu=4):
    """EXACT copy from first optimization"""
    r_ijs_unit = jnp.where(r_abs[:, None] > 1e-12, 
                           r_ijs / r_abs[:, None], 
                           jnp.zeros_like(r_ijs))
    
    unit_powers = [jnp.ones_like(r_abs)]
    
    if max_nu >= 1:
        unit_powers.append(r_ijs_unit)
    
    if max_nu >= 2:
        unit_powers.append(r_ijs_unit)
    
    if max_nu >= 3:
        unit_powers.append(r_ijs_unit)
        
    if max_nu >= 4:
        unit_powers.append(r_ijs_unit)
    
    return r_ijs_unit, unit_powers

def optimized_tensor_sum_nu0(rb_values):
    """EXACT copy from first optimization"""
    return jnp.sum(rb_values)

def optimized_tensor_sum_nu1(rb_values, r_ijs_unit):
    """EXACT copy from first optimization"""
    return jnp.dot(rb_values, r_ijs_unit)

def optimized_tensor_sum_nu2(rb_values, r_ijs_unit):
    """EXACT copy from first optimization"""
    return jnp.einsum('i,ij,ik->jk', rb_values, r_ijs_unit, r_ijs_unit, optimize=True)

def optimized_tensor_sum_nu3(rb_values, r_ijs_unit):
    """EXACT copy from first optimization"""
    return jnp.einsum('i,ij,ik,il->jkl', rb_values, r_ijs_unit, r_ijs_unit, r_ijs_unit, optimize=True)

def optimized_tensor_sum_nu4(rb_values, r_ijs_unit):
    """EXACT copy from first optimization"""
    return jnp.einsum('i,ij,ik,il,im->jklm', rb_values, r_ijs_unit, r_ijs_unit, r_ijs_unit, r_ijs_unit, optimize=True)

def optimized_tensor_sum(r_ijs_unit, rb_values, nu):
    """EXACT copy from first optimization"""
    if nu == 0:
        return optimized_tensor_sum_nu0(rb_values)
    elif nu == 1:
        return optimized_tensor_sum_nu1(rb_values, r_ijs_unit)
    elif nu == 2:
        return optimized_tensor_sum_nu2(rb_values, r_ijs_unit)
    elif nu == 3:
        return optimized_tensor_sum_nu3(rb_values, r_ijs_unit)
    elif nu == 4:
        return optimized_tensor_sum_nu4(rb_values, r_ijs_unit)
    else:
        operands = [rb_values] + [r_ijs_unit] * nu
        letters = string.ascii_lowercase[:nu]
        input_subs = ['i'] + [f'i{l}' for l in letters]
        einsum_expr = f'{",".join(input_subs)}->{"".join(letters)}'
        return jnp.einsum(einsum_expr, *operands, optimize=True)

def optimized_chebyshev_polynomials(r, n_terms, min_dist, max_dist):
    """EXACT copy from first optimization - vectorized version"""
    if n_terms == 0:
        return jnp.zeros((r.shape[0], 0), dtype=COMPUTE_DTYPE)
    if n_terms == 1:
        return jnp.ones((r.shape[0], 1), dtype=COMPUTE_DTYPE)
    
    range_inv = 1.0 / (max_dist - min_dist)
    r_scaled = (2 * r - (min_dist + max_dist)) * range_inv
    
    if n_terms == 2:
        T0 = jnp.ones_like(r_scaled)
        T1 = r_scaled
        return jnp.column_stack([T0, T1])
    
    # EXACT copy: vectorized recurrence (this was working!)
    result = jnp.zeros((len(r_scaled), n_terms), dtype=COMPUTE_DTYPE)
    result = result.at[:, 0].set(1.0)  # T0 = 1
    result = result.at[:, 1].set(r_scaled)  # T1 = x
    
    for n in range(2, n_terms):
        result = result.at[:, n].set(2 * r_scaled * result[:, n-1] - result[:, n-2])
    
    return result

def optimized_einsum_radial_basis(scaled_smoothing, coeffs, radial_basis):
    """EXACT copy from first optimization"""
    base = jnp.einsum('jmn,jn->mj', coeffs, radial_basis, optimize=True)
    return (scaled_smoothing[None, :]) * base

def analyze_contractions(execution_order):
    """EXACT copy from first optimization"""
    small_contractions = []
    large_contractions = []
    scalar_contractions = []
    
    for step_type, spec in execution_order:
        if step_type == 'contract':
            left_key, right_key, contraction_type, axes = spec
            if len(axes[0]) == 0 and len(axes[1]) == 0:
                scalar_contractions.append(spec)
            else:
                small_contractions.append(spec)
    
    return small_contractions, large_contractions, scalar_contractions

def optimized_tensor_contractions(calculated, execution_order, enable_profiling=False):
    """EXACT copy from first optimization"""
    small_contractions, large_contractions, scalar_contractions = analyze_contractions(execution_order)
    
    contraction_count = 0
    large_contraction_count = 0
    
    for step_type, spec in execution_order:
        step_type = step_type.decode() if isinstance(step_type, bytes) else step_type
        
        if step_type == 'contract':
            contraction_count += 1
            left_key, right_key, contraction_type, axes = spec
            
            if left_key in calculated and right_key in calculated:
                m1 = calculated[left_key]
                m2 = calculated[right_key]
                
                m1_size = jnp.size(m1) if hasattr(m1, 'size') else 1
                m2_size = jnp.size(m2) if hasattr(m2, 'size') else 1
                total_ops = m1_size * m2_size
                
                if total_ops > 100:
                    large_contraction_count += 1
                    timer_name = f"          opt_LARGE_contraction_{large_contraction_count}_ops{total_ops}"
                else:
                    timer_name = f"          opt_small_contraction_{contraction_count}_ops{total_ops}"
                
                with ProfileTimer(timer_name, enable_profiling):
                    contracted = jnp.tensordot(m1, m2, axes=axes)
                    calculated[spec] = contracted
                    profile_component_timings(f"contraction_{contraction_count}", contracted, enable_profiling)
    
    return calculated

def optimized_basis_computation(r_ijs, r_abs, rb_values, execution_order, scalar_contractions, enable_profiling=False):
    """EXACT copy from first optimization"""
    
    with ProfileTimer("        opt_unit_vector_precompute", enable_profiling):
        r_ijs_unit, unit_powers = precompute_unit_vectors_and_powers(r_ijs, r_abs, max_nu=4)
        profile_component_timings("unit_vectors", r_ijs_unit, enable_profiling)
    
    with ProfileTimer("        opt_data_conversion", enable_profiling):
        r_ijs_compute = r_ijs.astype(COMPUTE_DTYPE)
        rb_values_compute = rb_values.astype(COMPUTE_DTYPE)
        calculated = {}
        profile_component_timings("data_conversion", rb_values_compute, enable_profiling)
    
    basic_count = 0
    basic_counts_by_nu = {}
    
    with ProfileTimer("        opt_basic_moments_total", enable_profiling):
        for step_type, spec in execution_order:
            step_type = step_type.decode() if isinstance(step_type, bytes) else step_type
            
            if step_type == 'basic':
                basic_count += 1
                mu, nu = spec[:2]
                nu_val = int(nu)
                
                if nu_val not in basic_counts_by_nu:
                    basic_counts_by_nu[nu_val] = 0
                basic_counts_by_nu[nu_val] += 1
                
                timer_name = f"          opt_basic_nu{nu_val}_{basic_counts_by_nu[nu_val]}"
                with ProfileTimer(timer_name, enable_profiling):
                    calculated[spec] = optimized_tensor_sum(
                        r_ijs_unit, rb_values_compute[int(mu)], nu_val
                    )
                    profile_component_timings(f"basic_nu{nu_val}", calculated[spec], enable_profiling)
    
    with ProfileTimer("        opt_tensor_contractions", enable_profiling):
        calculated = optimized_tensor_contractions(calculated, execution_order, enable_profiling)
        profile_component_timings("contractions", calculated, enable_profiling)
    
    with ProfileTimer("        opt_basis_extraction", enable_profiling):
        basis_values = []
        for contraction_key in scalar_contractions:
            if contraction_key in calculated:
                basis_values.append(calculated[contraction_key])
            else:
                basis_values.append(jnp.array(0.0, dtype=COMPUTE_DTYPE))
        
        result = jnp.array(basis_values, dtype=STABLE_DTYPE)
        profile_component_timings("basis_extraction", result, enable_profiling)
    
    if enable_profiling:
        print(f"            📊 OPTIMIZED BASIS ANALYSIS: {basic_count} basic moments, {len([s for s in execution_order if s[0] == 'contract'])} contractions")
        for nu, count in basic_counts_by_nu.items():
            print(f"                Nu={nu}: {count} moments")
    
    return result

def optimized_local_energy(
    r_ijs, itype, jtypes, species_coeffs, moment_coeffs, radial_coeffs,
    scaling, min_dist, max_dist, rb_size, execution_order, scalar_contractions, enable_profiling=False
):
    """EXACT copy from first optimization"""
    
    with ProfileTimer("      opt_distance_computation", enable_profiling):
        r_squared = jnp.sum(r_ijs * r_ijs, axis=1)
        r_abs = jnp.sqrt(r_squared)
        profile_component_timings("distances", r_abs, enable_profiling)
    
    with ProfileTimer("      opt_smoothing_computation", enable_profiling):
        valid_mask = r_abs < max_dist
        scaled_smoothing = jnp.where(valid_mask, scaling * (max_dist - r_abs) ** 2, 0.0)
        profile_component_timings("smoothing", scaled_smoothing, enable_profiling)
    
    with ProfileTimer("      opt_chebyshev_computation", enable_profiling):
        radial_basis = optimized_chebyshev_polynomials(r_abs, rb_size, min_dist, max_dist)
        profile_component_timings("radial_basis", radial_basis, enable_profiling)
    
    with ProfileTimer("      opt_einsum_optimization", enable_profiling):
        coeffs = radial_coeffs[itype, jtypes]
        rb_values = optimized_einsum_radial_basis(scaled_smoothing, coeffs, radial_basis)
        profile_component_timings("rb_values", rb_values, enable_profiling)
    
    with ProfileTimer("      opt_basis_computation", enable_profiling):
        basis = optimized_basis_computation(r_ijs, r_abs, rb_values, execution_order, scalar_contractions, enable_profiling)
        profile_component_timings("basis", basis, enable_profiling)
    
    with ProfileTimer("      opt_final_energy_computation", enable_profiling):
        energy = species_coeffs[itype] + jnp.dot(moment_coeffs, basis)
        profile_component_timings("final_energy", energy, enable_profiling)
    
    return energy

@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 12, out_axes=0)
def optimized_calc_local_energy_and_forces(
    r_ijs, itype, jtypes, species_coeffs, moment_coeffs, radial_coeffs,
    scaling, min_dist, max_dist, itypes_shape, itypes_len,
    rb_size, execution_order, scalar_contractions, enable_profiling=False
):
    """EXACT copy from first optimization"""
    
    def energy_function(positions):
        with ProfileTimer("    opt_vmap_local_energy_call", enable_profiling):
            return optimized_local_energy(
                positions, itype, jtypes, species_coeffs, moment_coeffs, 
                radial_coeffs, scaling, min_dist, max_dist,
                rb_size, execution_order, scalar_contractions, enable_profiling
            )
    
    with ProfileTimer("  opt_vmap_value_and_grad_computation", enable_profiling):
        energy, forces = jax.value_and_grad(energy_function)(r_ijs)
        profile_component_timings("vmap_grad_result", (energy, forces), enable_profiling)
        
    with ProfileTimer("  opt_vmap_array_conversion", enable_profiling):
        local_energy = jnp.array([energy], dtype=OUTPUT_DTYPE)
        profile_component_timings("vmap_final", local_energy, enable_profiling)
    
    return local_energy, forces

# ============================================================================
# ONLY CHANGE: Minimal force accumulation fix (no profiling overhead)
# ============================================================================

def minimal_force_accumulation(forces_per_neighbor, all_js, natoms):
    """
    MINIMAL FIX: Same as first optimization but no profiling in critical path
    This was the only regression we needed to fix
    """
    forces = jnp.sum(forces_per_neighbor, axis=1)
    
    forces_flat = forces_per_neighbor.reshape(-1, 3)
    js_flat = all_js.reshape(-1)
    
    reaction_forces = jnp.zeros((natoms, 3), dtype=COMPUTE_DTYPE)
    reaction_forces = reaction_forces.at[js_flat].add(-forces_flat)
    
    return (forces + reaction_forces).astype(OUTPUT_DTYPE)

# ============================================================================
# EXACT COPY: Rest of first optimization
# ============================================================================

def calc_energy_forces_stress_ultra_stable_optimized(
    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume,
    species, scaling, min_dist, max_dist,
    species_coeffs, moment_coeffs, radial_coeffs,
    execution_order, scalar_contractions, enable_profiling=False
):
    """EXACT copy from first optimization with minimal force fix"""
    
    def fromtuple(x, dtype=OUTPUT_DTYPE):
        if isinstance(x, tuple):
            return jnp.array([fromtuple(y, dtype) for y in x], dtype=dtype)
        else:
            return x
    
    with ProfileTimer("opt_coefficient_processing", enable_profiling):
        species_coeffs = fromtuple(species_coeffs, STABLE_DTYPE)
        moment_coeffs = fromtuple(moment_coeffs, STABLE_DTYPE)  
        radial_coeffs = fromtuple(radial_coeffs, COMPUTE_DTYPE)
        profile_component_timings("coefficients", (species_coeffs, moment_coeffs, radial_coeffs), enable_profiling)
    
    with ProfileTimer("opt_main_energy_forces_computation", enable_profiling):
        print(f"      🚀 OPTIMIZED VMAP ANALYSIS ({len(itypes)} atoms):")
        local_energies, forces_per_neighbor = optimized_calc_local_energy_and_forces(
            all_rijs, itypes, all_jtypes, 
            species_coeffs, moment_coeffs, radial_coeffs,
            scaling, min_dist, max_dist, itypes.shape, len(itypes),
            radial_coeffs.shape[3], execution_order, scalar_contractions, enable_profiling
        )
        profile_component_timings("energy_forces_main", (local_energies, forces_per_neighbor), enable_profiling)

    # MINIMAL CHANGE: Use minimal force accumulation (no profiling in critical path)
    with ProfileTimer("opt_force_accumulation", enable_profiling):
        forces = minimal_force_accumulation(forces_per_neighbor, all_js, len(itypes))
        profile_component_timings("force_accumulation", forces, enable_profiling)
    
    with ProfileTimer("opt_stress_computation", enable_profiling):
        stress_tensor = jnp.einsum('aij,aik->jk', 
                                all_rijs.astype(COMPUTE_DTYPE), 
                                forces_per_neighbor.astype(COMPUTE_DTYPE), 
                                optimize=True).astype(STABLE_DTYPE)
        
        def compute_stress_true(stress, volume):
            stress_sym = (stress + stress.T) * (0.5 / volume)
            indices = jnp.array([0, 4, 8, 5, 2, 1])
            return stress_sym.reshape(-1)[indices]

        def compute_stress_false(_):
            return jnp.full(6, jnp.nan, dtype=OUTPUT_DTYPE)
        
        stress_voigt = lax.cond(
            jnp.equal(cell_rank, 3),
            lambda _: compute_stress_true(stress_tensor, volume),
            lambda _: compute_stress_false(stress_tensor),
            operand=None
        )
        profile_component_timings("stress_computation", stress_voigt, enable_profiling)

    return local_energies, forces, stress_voigt

def calc_energy_forces_stress_padded_simple_minimal_optimized(
    itypes,           # Argument 1
    all_js,           # Argument 2  
    all_rijs,         # Argument 3
    all_jtypes,       # Argument 4
    cell_rank,        # Argument 5
    volume,           # Argument 6
    natoms_actual,    # Argument 7
    nneigh_actual,    # Argument 8
    species,          # Static argument 1
    scaling,          # Static argument 2
    min_dist,         # Static argument 3
    max_dist,         # Static argument 4
    species_coeffs,   # Static argument 5
    moment_coeffs,    # Static argument 6
    radial_coeffs,    # Static argument 7
    execution_order,  # Static argument 8
    scalar_contractions, # Static argument 9
    enable_profiling=False  # Optional profiling argument
):
    """EXACT copy from first optimization with minimal change"""
    
    if enable_profiling:
        print(f"\n🚀 MINIMAL OPTIMIZED PROFILING: {len(itypes)} atoms × {all_js.shape[1] if len(all_js.shape) > 1 else 0} neighbors")
    
    with ProfileTimer("OPT_TOTAL_COMPUTATION", enable_profiling):
        energies, forces, stress = calc_energy_forces_stress_ultra_stable_optimized(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume,
            species, scaling, min_dist, max_dist,
            species_coeffs, moment_coeffs, radial_coeffs,
            execution_order, scalar_contractions, enable_profiling
        )
        
        with ProfileTimer("opt_final_output_processing", enable_profiling):
            energy = energies.sum().astype(OUTPUT_DTYPE)
            forces_output = forces.astype(OUTPUT_DTYPE)
            stress_output = stress.astype(OUTPUT_DTYPE)
            profile_component_timings("final_outputs", (energy, forces_output, stress_output), enable_profiling)
    
    return energy, forces_output, stress_output

print("✅ MINIMAL OPTIMIZED JAX-MTP Implementation loaded")
print("   EXACT first optimization + ONLY force accumulation fix")
print("   Target: Beat 0.84ms clean time")
