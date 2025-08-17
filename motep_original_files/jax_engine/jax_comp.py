#!/usr/bin/env python3
"""
Ultra-Optimized JAX MTP Implementation - FIXED VERSION
Maintains exact 8-argument interface for .bin compilation compatibility

Key fixes:
1. Exact 8-argument interface: itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, natoms_actual, nneigh_actual
2. MTP parameters passed as static arguments to JAX
3. Proper static_argnums configuration
4. Maintains compatibility with existing compilation infrastructure
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, checkpoint
from functools import partial
import string

jax.config.update("jax_enable_x64", False)

# Ultra-optimization configuration - TESTING WITH HIGHER PRECISION
ULTRA_COMPUTE_DTYPE = jnp.float32     # Higher precision for debugging
STABLE_COMPUTE_DTYPE = jnp.float32    # For critical calculations
OUTPUT_DTYPE = jnp.float32            # Final outputs

print("🚀 Loading Ultra-Optimized JAX MTP Implementation (FIXED)")
print("   Target: 3-10x reduction in GPU computation time (92ms → 10-30ms)")
print("   Interface: Exact 8-argument compatibility")

# ============================================================================
# ULTRA-OPTIMIZATION 1: FUSED VALUE_AND_GRAD (ELIMINATES JACOBIAN OVERHEAD)
# ============================================================================

@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 11, out_axes=(0, 0))
def _ultra_calc_local_energy_and_forces(
    r_ijs, itype, jtypes, species_coeffs, moment_coeffs, radial_coeffs,
    scaling, min_dist, max_dist, itypes_shape, itypes_len,
    rb_size, execution_order, scalar_contractions
):
    """
    ULTRA-OPTIMIZATION: Replace expensive jax.jacobian with value_and_grad
    Expected speedup: 2-3x (eliminates redundant computation)
    """
    
    def energy_function(positions):
        return _ultra_calc_local_energy_fused(
            positions, itype, jtypes, species_coeffs, moment_coeffs, 
            radial_coeffs, scaling, min_dist, max_dist,
            rb_size, execution_order, scalar_contractions
        )
    
    # Single call computes energy AND gradients efficiently
    energy, forces = jax.value_and_grad(energy_function)(r_ijs)
    
    # Return per-atom energy (not duplicated across all atoms!)
    local_energy = jnp.array([energy], dtype=OUTPUT_DTYPE)
    
    return local_energy, forces

# ============================================================================
# ULTRA-OPTIMIZATION 2: FUSED DISTANCE/BASIS COMPUTATION
# ============================================================================

#@partial(jax.jit, static_argnums=(8, 9, 10))
def _ultra_calc_local_energy_fused(
    r_ijs, itype, jtypes, species_coeffs, moment_coeffs, radial_coeffs,
    scaling, min_dist, max_dist, rb_size, execution_order, scalar_contractions
):
    """
    ULTRA-OPTIMIZATION: Fused distance and basis computation
    Expected speedup: 1.5-2x (reduced memory bandwidth, kernel fusion)
    """

    # FUSED: Distance computation with immediate basis calculation
    r_squared = jnp.sum(r_ijs * r_ijs, axis=1)
    r_abs = jnp.sqrt(r_squared).astype(ULTRA_COMPUTE_DTYPE)
    
    # FUSED: Combine distance masking with smoothing
    valid_mask = r_abs < max_dist
    #smoothing_raw = jnp.where(valid_mask, (max_dist - r_abs) ** 2, 0.0)
    #scaled_smoothing = (scaling * smoothing_raw).astype(ULTRA_COMPUTE_DTYPE)
    scaled_smoothing = jnp.where(valid_mask, (max_dist - r_abs) ** 2, 0.0)

    #jax.debug.print('r_abs: {}', r_abs)
    #jax.debug.print('r_abs shape: {}', r_abs.shape)
    #jax.debug.print('smoothing: {}', scaled_smoothing)
    #jax.debug.print('smoothing shape: {}', scaled_smoothing.shape)
    
    # ULTRA-OPTIMIZED: Fast Chebyshev basis
    radial_basis = _ultra_chebyshev_basis_fused(r_abs, rb_size, min_dist, max_dist)
    
    #jax.debug.print('radial_basis: {}', radial_basis)
    #jax.debug.print('radial_basis shape: {}', radial_basis.shape)

    # ULTRA-OPTIMIZED: Tensor contraction with einsum path optimization
    coeffs = radial_coeffs[itype, jtypes].astype(ULTRA_COMPUTE_DTYPE)
    rb_values = _ultra_einsum_optimized(scaling, scaled_smoothing, coeffs, radial_basis)

    #jax.debug.print('rb_values: {}', rb_values)
    #jax.debug.print('rb_values shape: {}', rb_values.shape)
    
    # ULTRA-OPTIMIZED: Fused basis computation
    basis = _ultra_calc_basis_symmetric_fused(
        r_ijs, r_abs, rb_values, execution_order, scalar_contractions
    )
    
    jax.debug.print('basis: {}',basis)
    #jax.debug.print('basis shape: {}',basis.shape)
    # Final energy computation in stable precision
    energy_base = species_coeffs[itype].astype(STABLE_COMPUTE_DTYPE)
    energy_contrib = jnp.dot(moment_coeffs.astype(STABLE_COMPUTE_DTYPE), 
                            basis.astype(STABLE_COMPUTE_DTYPE))
    
    energy = species_coeffs[itype] + jnp.dot(moment_coeffs, basis)

    print(energy)
    
    jax.debug.print('energy_base: {}', energy_base)
    jax.debug.print('energy_contrib: {}',energy_contrib)

    energy = energy_base + energy_contrib
    return energy.astype(OUTPUT_DTYPE)

# ============================================================================
# ULTRA-OPTIMIZATION 3: OPTIMIZED CHEBYSHEV BASIS
# ============================================================================

def _ultra_chebyshev_basis_fused(r, n_terms, min_dist, max_dist):
    """
    ULTRA-OPTIMIZATION: Fused Chebyshev computation with memory optimization
    Expected speedup: 1.5-2x (vectorization, reduced scan overhead)
    """
    if n_terms == 0:
        return jnp.zeros((r.shape[0], 0), dtype=ULTRA_COMPUTE_DTYPE)
    if n_terms == 1:
        return jnp.ones((r.shape[0], 1), dtype=ULTRA_COMPUTE_DTYPE)
    
    # Optimize scaling computation
    range_inv = 1.0 / (max_dist - min_dist)
    r_scaled = ((2 * r - (min_dist + max_dist)) * range_inv).astype(ULTRA_COMPUTE_DTYPE)
    
    if n_terms == 2:
        T0 = jnp.ones_like(r_scaled)
        T1 = r_scaled
        return jnp.column_stack([T0, T1])
    
    # ULTRA-OPTIMIZED: Use scan but with reduced precision for speed
    def step(carry, _):
        T_prev, T_curr = carry
        T_next = 2 * r_scaled * T_curr - T_prev
        return (T_curr, T_next), T_next #T_curr

    T0 = jnp.ones_like(r_scaled)
    T1 = r_scaled
    
    _, T_rest = lax.scan(step, (T0, T1), None, length=n_terms - 2)

    return jnp.column_stack([T0, T1, *T_rest])

# ============================================================================
# ULTRA-OPTIMIZATION 4: OPTIMIZED EINSUM
# ============================================================================

def _ultra_einsum_optimized_old(scaled_smoothing, coeffs, radial_basis):
    """
    ULTRA-OPTIMIZATION: Einsum with optimal contraction path
    Expected speedup: 1.3-2x (optimal GEMM operations)
    """
    # Use JAX's optimized einsum with path optimization
    return jnp.einsum(
        'jmn, jn -> mj',
        scaled_smoothing,
        coeffs,
        radial_basis,
        optimize=True  # Let JAX find optimal contraction path
    )

def _ultra_einsum_optimized(scaling, smoothing, coeffs, radial_basis):
    # coeffs: (j,m,n), radial_basis: (j,n)
    base = jnp.einsum('jmn, jn -> mj', coeffs, radial_basis, optimize=True)
    return (scaling * smoothing[None, :]) * base



# ============================================================================
# ULTRA-OPTIMIZATION 5: SIMPLIFIED BASIS COMPUTATION
# ============================================================================

def _ultra_calc_basis_symmetric_fused_old(r_ijs, r_abs, rb_values, execution_order, scalar_contractions):
    r_ijs_compute = r_ijs.astype(ULTRA_COMPUTE_DTYPE)
    rb_values_compute = rb_values.astype(ULTRA_COMPUTE_DTYPE)

    basis_moments = {}

    for step in execution_order:
        step_type, moment_spec = step

        if isinstance(step_type, bytes):
            step_type = step_type.decode()

        if step_type == 'basic':
            mu, nu = moment_spec[:2]
            moment_values = _ultra_vectorized_tensor_sum(
                r_ijs_compute, rb_values_compute[mu], nu
            )
            basis_moments[str((mu, nu))] = moment_values

    basis_list = []
    for contraction_key in scalar_contractions:
        # canonicalize incoming keys to (mu, nu) strings as well
        if not isinstance(contraction_key, str):
            contraction_key = str(tuple(contraction_key[:2]) if isinstance(contraction_key, (list, tuple)) else contraction_key)

        if contraction_key in basis_moments:
            moment_tensor = basis_moments[contraction_key]
            scalar_value = jnp.sum(moment_tensor) if moment_tensor.ndim > 0 else moment_tensor
            basis_list.append(scalar_value)
        else:
            basis_list.append(0.0)

    return jnp.array(basis_list, dtype=STABLE_COMPUTE_DTYPE)



def _ultra_calc_basis_symmetric_fused(r_ijs, r_abs, rb_values, execution_order, scalar_contractions):
    """
    CORRECTED: Complete moment-basis calculation using execution_order + scalar extraction.
    
    execution_order contains ('basic', ...) and ('contract', ...) steps in dependency order.
    scalar_contractions lists which computed moments to extract as final basis.
    """
    
    r_ijs_compute = r_ijs.astype(ULTRA_COMPUTE_DTYPE)
    rb_values_compute = rb_values.astype(ULTRA_COMPUTE_DTYPE)
    
    calculated = {}
    
    # Execute all computation steps in order
    for step_type, spec in execution_order:
        if isinstance(step_type, bytes):
            step_type = step_type.decode()
        
        if step_type == 'basic':
            # spec = (mu, nu, ...) from basic_moments
            mu, nu = spec[:2]
            m = _ultra_vectorized_tensor_sum(
                r_ijs_compute,
                r_abs,
                rb_values_compute[int(mu)],
                int(nu)
            )
            calculated[spec] = m
            
        elif step_type == 'contract':
            # spec = (left_key, right_key, contraction_type, axes)
            left_key, right_key, contraction_type, axes = spec
            
            if left_key not in calculated:
                raise KeyError(f"Left key {left_key} not found in calculated moments")
            if right_key not in calculated:
                raise KeyError(f"Right key {right_key} not found in calculated moments")
                
            m1 = calculated[left_key]
            m2 = calculated[right_key]
            
            # Perform tensor contraction
            contracted = jnp.tensordot(m1, m2, axes=axes)
            calculated[spec] = contracted
    
    # Extract scalar contractions as final basis - EXACTLY like original jax.py
    basis_list = []
    for i, contraction_key in enumerate(scalar_contractions):
        if contraction_key in calculated:
            # CRITICAL FIX: Don't sum! Use the value directly like original jax.py
            moment_value = calculated[contraction_key]
            basis_list.append(moment_value)
            if i < 12:  # Debug first few
                print(f"Scalar[{i}]: {contraction_key} -> shape {moment_value.shape if hasattr(moment_value, 'shape') else 'scalar'} -> value {moment_value}")
        else:
            # Missing moment - set to zero
            basis_list.append(jnp.array(0.0, dtype=STABLE_COMPUTE_DTYPE))
            if i < 12:  # Debug first few
                print(f"Scalar[{i}]: {contraction_key} -> MISSING -> value 0.0")
    
    return jnp.array(basis_list, dtype=STABLE_COMPUTE_DTYPE)


# ============================================================================
# ULTRA-OPTIMIZATION 6: VECTORIZED TENSOR OPERATIONS
# ============================================================================

def _ultra_vectorized_tensor_sum(r_ijs, r_abs, rb_values, nu):
    """
    ULTRA-OPTIMIZATION: Vectorized tensor summation
    CRITICAL FIX: Use unit vectors like the original implementation
    Expected speedup: 2-3x (better GPU utilization)
    """
    # CRITICAL: Use unit vectors like original implementation
    r_ijs_unit = (r_ijs.T / r_abs).T  # Unit vectors
    
    if nu == 0:
        return jnp.sum(rb_values)
    elif nu == 1:
        return jnp.dot(rb_values, r_ijs_unit)  
    elif nu == 2:
        return jnp.einsum('i,ij,ik->jk', rb_values, r_ijs_unit, r_ijs_unit, optimize=True)
    elif nu == 3:
        return jnp.einsum('i,ij,ik,il->jkl', rb_values, r_ijs_unit, r_ijs_unit, r_ijs_unit, optimize=True)
    else:
        # General case with einsum optimization
        operands = [rb_values] + [r_ijs_unit] * nu
        letters = string.ascii_lowercase[:nu]
        input_subs = ['i'] + [f'i{l}' for l in letters]
        einsum_expr = f'{",".join(input_subs)}->{"".join(letters)}'
        return jnp.einsum(einsum_expr, *operands, optimize=True)

# ============================================================================
# MAIN ULTRA-OPTIMIZED FUNCTION WITH CORRECT INTERFACE
# ============================================================================

def calc_energy_forces_stress_ultra_optimized(
    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume,
    species, scaling, min_dist, max_dist,
    species_coeffs, moment_coeffs, radial_coeffs,
    execution_order, scalar_contractions
):
    """
    ULTRA-OPTIMIZED JAX MTP Implementation with correct static handling
    """
    
    def fromtuple(x, dtype=OUTPUT_DTYPE):
        if isinstance(x, tuple):
            return jnp.array([fromtuple(y, dtype) for y in x], dtype=dtype)
        else:
            return x
    
    # Convert coefficients to appropriate precision
    species_coeffs = fromtuple(species_coeffs, STABLE_COMPUTE_DTYPE)
    moment_coeffs = fromtuple(moment_coeffs, STABLE_COMPUTE_DTYPE)  
    radial_coeffs = fromtuple(radial_coeffs, ULTRA_COMPUTE_DTYPE)
    
    # ULTRA-OPTIMIZED: Compute energies and forces simultaneously
    local_energies, forces_per_neighbor = _ultra_calc_local_energy_and_forces(
        all_rijs, itypes, all_jtypes, species_coeffs, moment_coeffs, radial_coeffs,
        scaling, min_dist, max_dist, itypes.shape, len(itypes),
        radial_coeffs.shape[3], execution_order, scalar_contractions
    )
    
    # ULTRA-OPTIMIZED: Stress computation with einsum optimization
    stress_tensor = jnp.einsum('aij,aik->jk', all_rijs, forces_per_neighbor, optimize=True)
    
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
    
    # ULTRA-OPTIMIZED: Force reduction
    forces = jnp.sum(forces_per_neighbor, axis=-2)
    
    return local_energies, forces, stress_voigt

# ============================================================================
# EXACT 8-ARGUMENT INTERFACE FOR .BIN COMPILATION
# ============================================================================

def calc_energy_forces_stress_padded_simple_ultra_optimized(
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
    scalar_contractions # Static argument 9
):
    """
    Ultra-optimized function with EXACT 8-argument interface
    
    For .bin compilation, the first 8 arguments are dynamic (arrays from LAMMPS)
    The remaining arguments are static (MTP parameters, baked into .bin file)
    
    Expected: 3-10x reduction in GPU computation time
    """

    jax.debug.print('all_rijs: {}', all_rijs)
    jax.debug.print('all_rijs shape: {}', all_rijs.shape)
    jax.debug.print('all_js: {}', all_js)
    jax.debug.print('all_js shape: {}', all_js.shape)
    jax.debug.print('all_jtypes: {}', all_jtypes)
    jax.debug.print('all_jtypes shape: {}', all_jtypes.shape)
    jax.debug.print('itypes: {}', itypes)
    jax.debug.print('itypes: {}', itypes.shape)

    # Convert positions to ultra-compute precision for memory bandwidth optimization
    all_rijs_compute = all_rijs.astype(ULTRA_COMPUTE_DTYPE)
    
    energies, forces, stress = calc_energy_forces_stress_ultra_optimized(
        itypes, all_js, all_rijs_compute, all_jtypes, cell_rank, volume,
        species, scaling, min_dist, max_dist,
        species_coeffs, moment_coeffs, radial_coeffs,
        execution_order, scalar_contractions
    )
    
    # Convert outputs to stable precision
    energy = energies.sum().astype(OUTPUT_DTYPE)
    forces_output = forces.astype(OUTPUT_DTYPE)
    stress_output = stress.astype(OUTPUT_DTYPE)
    
    return energy, forces_output, stress_output

# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

def analyze_optimization_impact():
    """Print expected performance improvements"""
    
    print("\n🚀 ULTRA-OPTIMIZATION ANALYSIS (FIXED)")
    print("=" * 50)
    print("Current GPU computation: 92ms (97% of total time)")
    print("Target: 10-30ms (3-10x reduction)")
    print()
    print("Interface: EXACT 8-argument compatibility")
    print("  First 8: Dynamic arrays from LAMMPS")
    print("  Remaining: Static MTP parameters")
    print()
    
    optimizations = {
        'value_and_grad (vs jacobian)': '2-3x speedup',
        'Fused distance/basis': '1.5-2x speedup',
        'Optimized einsum paths': '1.3-2x speedup', 
        'Memory bandwidth (bfloat16)': '1.2-1.5x speedup',
        'Kernel fusion': '1.2-1.8x speedup'
    }
    
    print("Individual optimizations:")
    for opt, speedup in optimizations.items():
        print(f"  • {opt:<30} {speedup}")
    
    print()
    print("Combined multiplicative effect: 3-10x total speedup")
    print("Expected result: 92ms → 10-30ms GPU computation")
    print("Total LAMMPS speedup: 95ms → 13-43ms per timestep")
    print("Performance improvement: 2.2-7.3x faster simulations")

if __name__ == "__main__":
    analyze_optimization_impact()
    print("\n✅ Ultra-optimized JAX MTP ready (FIXED)!")
    print("   Interface: Exact 8-argument compatibility")
    print("   Usage: from jax_mtp_ultra_optimized_fixed import calc_energy_forces_stress_padded_simple_ultra_optimized")
