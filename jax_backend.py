#!/usr/bin/env python3
"""
Ultra-Optimized JAX MTP Implementation
Maintains exact 8-argument interface for .bin compilation compatibility
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, checkpoint
from functools import partial
import string

jax.config.update("jax_enable_x64", False)

ULTRA_COMPUTE_DTYPE = jnp.float32
STABLE_COMPUTE_DTYPE = jnp.float32
OUTPUT_DTYPE = jnp.float32

@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 11, out_axes=0)
def _calc_local_energy_and_forces(
    r_ijs, itype, jtypes, species_coeffs, moment_coeffs, radial_coeffs,
    scaling, min_dist, max_dist, itypes_shape, itypes_len,
    rb_size, execution_order, scalar_contractions
):
    """
    Calculate local energy and forces using value_and_grad for efficiency.
    
    Args:
        r_ijs: Relative position vectors
        itype: Central atom type
        jtypes: Neighbor atom types
        species_coeffs: Species-specific coefficients
        moment_coeffs: Moment basis coefficients
        radial_coeffs: Radial basis coefficients
        scaling: Scaling factor
        min_dist: Minimum distance for radial basis
        max_dist: Maximum distance cutoff
        itypes_shape: Shape of atom types array
        itypes_len: Length of atom types array
        rb_size: Radial basis size
        execution_order: Computation order for basis
        scalar_contractions: Scalar contraction indices
        
    Returns:
        energy: Local energy contribution
        forces: Forces on neighbors
    """
    def energy_function(positions):
        return _calc_local_energy(
            positions, itype, jtypes, species_coeffs, moment_coeffs, 
            radial_coeffs, scaling, min_dist, max_dist,
            rb_size, execution_order, scalar_contractions
        )
    
    energy, forces = jax.value_and_grad(energy_function)(r_ijs)
    return energy, forces

def _calc_local_energy(
    r_ijs, itype, jtypes, species_coeffs, moment_coeffs, radial_coeffs,
    scaling, min_dist, max_dist, rb_size, execution_order, scalar_contractions
):
    """
    Calculate local energy contribution for a single atom.
    
    Args:
        r_ijs: Relative position vectors
        itype: Central atom type
        jtypes: Neighbor atom types
        species_coeffs: Species-specific coefficients
        moment_coeffs: Moment basis coefficients
        radial_coeffs: Radial basis coefficients
        scaling: Scaling factor
        min_dist: Minimum distance for radial basis
        max_dist: Maximum distance cutoff
        rb_size: Radial basis size
        execution_order: Computation order for basis
        scalar_contractions: Scalar contraction indices
        
    Returns:
        energy: Local energy contribution
    """
    r_squared = jnp.sum(r_ijs * r_ijs, axis=1)
    r_abs = jnp.sqrt(r_squared).astype(ULTRA_COMPUTE_DTYPE)
    
    valid_mask = r_abs < max_dist
    valid_type = (jtypes >= 0)           
    valid_nbh = valid_type & valid_mask
    
    scaled_smoothing = jnp.where(valid_nbh, (max_dist - r_abs) ** 2, 0.0)
    
    radial_basis = _calc_chebyshev_basis(r_abs, rb_size, min_dist, max_dist)
    
    coeffs = radial_coeffs[itype, jtypes].astype(ULTRA_COMPUTE_DTYPE)
    rb_values = _calc_radial_values(scaling, scaled_smoothing, coeffs, radial_basis)
    
    basis = _calc_basis_symmetric(
        r_ijs, r_abs, rb_values, execution_order, scalar_contractions
    )
    
    energy_base = species_coeffs[itype].astype(STABLE_COMPUTE_DTYPE)
    energy_contrib = jnp.dot(moment_coeffs.astype(STABLE_COMPUTE_DTYPE), 
                            basis.astype(STABLE_COMPUTE_DTYPE))
    
    energy = energy_base + energy_contrib
    return energy.astype(OUTPUT_DTYPE)

def _calc_chebyshev_basis(r, n_terms, min_dist, max_dist):
    """
    Calculate Chebyshev polynomial basis functions.
    
    Args:
        r: Distance values
        n_terms: Number of Chebyshev terms
        min_dist: Minimum distance for scaling
        max_dist: Maximum distance for scaling
        
    Returns:
        Chebyshev basis matrix [n_atoms, n_terms]
    """
    if n_terms == 0:
        return jnp.zeros((r.shape[0], 0), dtype=ULTRA_COMPUTE_DTYPE)
    if n_terms == 1:
        return jnp.ones((r.shape[0], 1), dtype=ULTRA_COMPUTE_DTYPE)
    
    range_inv = 1.0 / (max_dist - min_dist)
    r_scaled = ((2 * r - (min_dist + max_dist)) * range_inv).astype(ULTRA_COMPUTE_DTYPE)
    
    if n_terms == 2:
        T0 = jnp.ones_like(r_scaled)
        T1 = r_scaled
        return jnp.column_stack([T0, T1])
    
    def step(carry, _):
        T_prev, T_curr = carry
        T_next = 2 * r_scaled * T_curr - T_prev
        return (T_curr, T_next), T_next

    T0 = jnp.ones_like(r_scaled)
    T1 = r_scaled
    
    _, T_rest = lax.scan(step, (T0, T1), None, length=n_terms - 2)
    
    return jnp.column_stack([T0, T1, *T_rest])

def _calc_radial_values(scaling, smoothing, coeffs, radial_basis):
    """
    Calculate radial contribution values.
    
    Args:
        scaling: Scaling factor
        smoothing: Smoothing function values
        coeffs: Radial coefficients [n_neighbors, n_moments, n_basis]
        radial_basis: Chebyshev basis [n_neighbors, n_basis]
        
    Returns:
        Radial values [n_moments, n_neighbors]
    """
    base = jnp.einsum('jmn, jn -> mj', coeffs, radial_basis, optimize=True)
    return (scaling * smoothing[None, :]) * base

def _calc_basis_symmetric(r_ijs, r_abs, rb_values, execution_order, scalar_contractions):
    """
    Calculate symmetric moment basis functions with full contraction support.
    
    Args:
        r_ijs: Relative position vectors
        r_abs: Distance values
        rb_values: Radial basis values
        execution_order: Computation order for basis (includes 'basic' and 'contract' steps)
        scalar_contractions: Scalar contraction indices to extract as final basis
        
    Returns:
        Basis vector for energy calculation
    """
    r_ijs_compute = r_ijs.astype(ULTRA_COMPUTE_DTYPE)
    rb_values_compute = rb_values.astype(ULTRA_COMPUTE_DTYPE)
    
    calculated = {}
    
    for step_type, spec in execution_order:
        if isinstance(step_type, bytes):
            step_type = step_type.decode()
        
        if step_type == 'basic':
            mu, nu = spec[:2]
            m = _vectorized_tensor_sum(
                r_ijs_compute,
                r_abs,
                rb_values_compute[int(mu)],
                int(nu)
            )
            calculated[spec] = m
            
        elif step_type == 'contract':
            left_key, right_key, contraction_type, axes = spec
            
            if left_key not in calculated:
                raise KeyError(f"Left key {left_key} not found in calculated moments")
            if right_key not in calculated:
                raise KeyError(f"Right key {right_key} not found in calculated moments")
                
            m1 = calculated[left_key]
            m2 = calculated[right_key]
            
            contracted = jnp.tensordot(m1, m2, axes=axes)
            calculated[spec] = contracted
    
    basis_list = []
    for contraction_key in scalar_contractions:
        if contraction_key in calculated:
            moment_value = calculated[contraction_key]
            basis_list.append(moment_value)
        else:
            basis_list.append(jnp.array(0.0, dtype=STABLE_COMPUTE_DTYPE))
    
    return jnp.array(basis_list, dtype=STABLE_COMPUTE_DTYPE)

def _vectorized_tensor_sum(r_ijs, r_abs, rb_values, nu):
    """
    Calculate vectorized tensor sum for moment calculations using unit vectors.
    
    Args:
        r_ijs: Relative position vectors [n_neighbors, 3]
        r_abs: Distance values [n_neighbors]
        rb_values: Radial basis values [n_neighbors]
        nu: Moment order
        
    Returns:
        Summed moment tensor
    """
    r_ijs_unit = (r_ijs.T / r_abs).T
    
    if nu == 0:
        return jnp.sum(rb_values)
    elif nu == 1:
        return jnp.dot(rb_values, r_ijs_unit)  
    elif nu == 2:
        return jnp.einsum('i,ij,ik->jk', rb_values, r_ijs_unit, r_ijs_unit, optimize=True)
    elif nu == 3:
        return jnp.einsum('i,ij,ik,il->jkl', rb_values, r_ijs_unit, r_ijs_unit, r_ijs_unit, optimize=True)
    else:
        operands = [rb_values] + [r_ijs_unit] * nu
        letters = string.ascii_lowercase[:nu]
        input_subs = ['i'] + [f'i{l}' for l in letters]
        einsum_expr = f'{",".join(input_subs)}->{"".join(letters)}'
        return jnp.einsum(einsum_expr, *operands, optimize=True)

def segment_sum(data, segment_ids, num_segments):
    """
    Sum data according to segment IDs.
    
    Args:
        data: Data to sum [n_items, ...]
        segment_ids: Segment ID for each item [n_items]
        num_segments: Total number of segments
        
    Returns:
        Summed data per segment [num_segments, ...]
    """
    return jax.ops.segment_sum(data, segment_ids, num_segments)

def accumulate_forces(pair_forces, all_js, natoms):
    """
    Accumulate pair forces to atomic forces with Newton's third law.
    
    JAX-compatible force accumulation with correct atom count handling.
    Uses masking to handle only actual local atoms while maintaining 
    static shapes for jax.jit compilation.
    
    Args:
        pair_forces: Forces on each neighbor pair [max_atoms, max_neighbors, 3]
        all_js: Neighbor indices [max_atoms, max_neighbors]
        natoms: Number of real atoms (not padded count)
        
    Returns:
        Total forces on each atom [max_atoms, 3]
    """
    max_atoms, max_neighbors, _ = pair_forces.shape
    
    atom_mask = jnp.arange(max_atoms) < natoms
    atom_mask_3d = atom_mask[:, None, None]
    atom_mask_2d = atom_mask[:, None]
    
    pair_forces_real = jnp.where(atom_mask_3d, pair_forces, 0.0)
    
    Fi = jnp.sum(pair_forces_real, axis=1)
    
    Fflat = pair_forces_real.reshape(-1, 3)
    jflat = all_js.reshape(-1)
    
    valid_i_mask = jnp.repeat(atom_mask, max_neighbors)
    valid_j_mask = jflat < natoms
    valid_pair_mask = valid_i_mask & valid_j_mask
    
    Fflat_masked = jnp.where(valid_pair_mask[:, None], Fflat, 0.0)
    jflat_masked = jnp.where(valid_pair_mask, jflat, 0)
    
    Fj = segment_sum(-Fflat_masked, segment_ids=jflat_masked, num_segments=max_atoms)
    
    total_forces = (Fi + Fj).astype(OUTPUT_DTYPE)
    
    return total_forces

def calc_energy_forces_stress(
    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, natoms_force,
    species, scaling, min_dist, max_dist,
    species_coeffs, moment_coeffs, radial_coeffs,
    execution_order, scalar_contractions
):
    """
    Calculate energy, forces, and stress for the system.
    
    Args:
        itypes: Atom types [max_atoms]
        all_js: Neighbor indices [max_atoms, max_neighbors]
        all_rijs: Relative positions [max_atoms, max_neighbors, 3]
        all_jtypes: Neighbor types [max_atoms, max_neighbors]
        cell_rank: Cell dimensionality (3 for 3D)
        volume: Cell volume
        natoms_force: Number of atoms (local + ghost)
        species: Species list
        scaling: Scaling factor
        min_dist: Minimum distance for radial basis
        max_dist: Maximum distance cutoff
        species_coeffs: Species-specific coefficients
        moment_coeffs: Moment basis coefficients
        radial_coeffs: Radial basis coefficients
        execution_order: Computation order for basis
        scalar_contractions: Scalar contraction indices
        
    Returns:
        local_energies: Energy per atom [max_atoms]
        forces: Forces on each atom [max_atoms, 3]
        stress_voigt: Stress tensor in Voigt notation [6]
    """
    def fromtuple(x, dtype=OUTPUT_DTYPE):
        if isinstance(x, tuple):
            return jnp.array([fromtuple(y, dtype) for y in x], dtype=dtype)
        else:
            return x
    
    species_coeffs = fromtuple(species_coeffs, STABLE_COMPUTE_DTYPE)
    moment_coeffs = fromtuple(moment_coeffs, STABLE_COMPUTE_DTYPE)  
    radial_coeffs = fromtuple(radial_coeffs, ULTRA_COMPUTE_DTYPE)
    
    local_energies, forces_per_neighbor = _calc_local_energy_and_forces(
        all_rijs, itypes, all_jtypes, species_coeffs, moment_coeffs, radial_coeffs,
        scaling, min_dist, max_dist, itypes.shape, len(itypes),
        radial_coeffs.shape[3], execution_order, scalar_contractions
    )
    
    forces = accumulate_forces(forces_per_neighbor, all_js, natoms_force)
    
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
    
    return local_energies, forces, stress_voigt

def calc_energy_forces_stress_padded(
    itypes,
    all_js,
    all_rijs,
    all_jtypes,
    cell_rank,
    volume,
    natoms_energy,
    natoms_force,
    species,
    scaling,
    min_dist,
    max_dist,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
    execution_order,
    scalar_contractions
):
    """
    Main interface function for LAMMPS integration with exact 8-argument interface.
    
    Dynamic Arguments (from LAMMPS):
        itypes: Atom types [max_atoms]
        all_js: Neighbor indices [max_atoms, max_neighbors]
        all_rijs: Relative positions [max_atoms, max_neighbors, 3]
        all_jtypes: Neighbor types [max_atoms, max_neighbors]
        cell_rank: Cell dimensionality
        volume: Cell volume
        natoms_energy: Number of local atoms for energy calculation
        natoms_force: Number of local + ghost atoms for force structure
        
    Static Arguments (MTP parameters):
        species: Species list
        scaling: Scaling factor
        min_dist: Minimum distance for radial basis
        max_dist: Maximum distance cutoff
        species_coeffs: Species-specific coefficients
        moment_coeffs: Moment basis coefficients
        radial_coeffs: Radial basis coefficients
        execution_order: Computation order for basis
        scalar_contractions: Scalar contraction indices
        
    Returns:
        energy: Total system energy (scalar)
        forces: Forces on each atom [max_atoms, 3]
        stress: Stress tensor in Voigt notation [6]
    """
    all_rijs_compute = all_rijs.astype(ULTRA_COMPUTE_DTYPE)
    
    energies, forces, stress = calc_energy_forces_stress(
        itypes, all_js, all_rijs_compute, all_jtypes, cell_rank, volume, natoms_energy,
        species, scaling, min_dist, max_dist,
        species_coeffs, moment_coeffs, radial_coeffs,
        execution_order, scalar_contractions
    )
    
    def compute_energy_masked(energies, natoms_energy):
        """
        Compute total energy from local atoms only.
        
        Args:
            energies: Per-atom energies [max_atoms] or [max_atoms, 1]
            natoms_energy: Number of local atoms
            
        Returns:
            Total energy from local atoms
        """
        if len(energies.shape) > 1:
            energies_flat = energies.flatten()
        else:
            energies_flat = energies
        
        max_atoms_static = energies_flat.shape[0]
        indices = jnp.arange(max_atoms_static)
        energy_mask = indices < natoms_energy
        
        local_energies_only = jnp.where(energy_mask, energies_flat, 0.0)
        total_energy = jnp.sum(local_energies_only)
        
        return total_energy
    
    energy = compute_energy_masked(energies, natoms_energy).astype(OUTPUT_DTYPE)
    forces_output = forces.astype(OUTPUT_DTYPE)  
    stress_output = stress.astype(OUTPUT_DTYPE)
    
    return energy, forces_output, stress_output
