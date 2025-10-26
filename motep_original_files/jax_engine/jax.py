import os
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=4'

from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np

from jax import lax, checkpoint, dtypes
import numpy.typing as npt
from ase import Atoms
import math
from typing import Dict, List, Tuple, Any
import dataclasses

import string
import operator
from collections import defaultdict


def get_types(atoms: Atoms, species: list[int] = None) -> npt.NDArray[np.int64]:
    if species is None:
        return np.array(atoms.numbers, dtype=int)
    else:
        return np.fromiter((species.index(_) for _ in atoms.numbers), dtype=int)


@partial(jax.jit, static_argnames=('execution_order', 'scalar_contractions'))
def calc_energy_forces_stress(
    itypes,
    all_js,
    all_rijs,
    all_jtypes,
    cell_rank,
    volume,
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
    
    def fromtuple(x, dtype=jnp.float32):
        """Convert nested tuple back to JAX array (reverse of totuple)"""
        if isinstance(x, tuple):
            return jnp.array([fromtuple(y, dtype) for y in x], dtype=dtype)
        else:
            return x
    
    species_coeffs =  fromtuple(species_coeffs)
    moment_coeffs = fromtuple(moment_coeffs)
    radial_coeffs = fromtuple(radial_coeffs)
    
    local_energies, forces = _jax_calc_local_energy_and_derivs(
        all_rijs,
        itypes,
        all_jtypes,
        species_coeffs,
        moment_coeffs,
        radial_coeffs,
        scaling,
        min_dist,
        max_dist,
        itypes.shape,
        len(itypes),
        # Static parameters:
        radial_coeffs.shape[3],
        execution_order,
        scalar_contractions
    )
    
        
    stress = jnp.array((all_rijs.transpose((0, 2, 1)) @ forces).sum(axis=0))
    
    def compute_stress_true(stress, volume):
        stress_sym = (stress + stress.T) * 0.5 / volume
        indices = jnp.array([0, 4, 8, 5, 2, 1])
        return stress_sym.reshape(-1)[indices]


    def compute_stress_false(_):
        return jnp.full(6, jnp.nan)
    
    stress_voigt = lax.cond(
        jnp.equal(cell_rank, 3),
        lambda _: compute_stress_true(stress, volume),
        lambda _: compute_stress_false(stress),
        operand=None
    )

    forces = jnp.sum(forces, axis=-2)
    
    return local_energies, forces, stress_voigt



@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 11, out_axes=0)
def _jax_calc_local_energy_and_derivs(
    r_ijs,
    itype,
    jtypes,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
    scaling,
    min_dist,
    max_dist,
    itypes_shape,
    itypes_len,
    # Static parameters:
    rb_size,
    execution_order,
    scalar_contractions,
):
        
    def energy_fn(r_ijs):
        return _jax_calc_local_energy(
            r_ijs,
            itype,
            jtypes,
            species_coeffs,
            moment_coeffs,
            radial_coeffs,
            scaling,
            min_dist,
            max_dist,
            # Static parameters:
            rb_size,
            execution_order,
            scalar_contractions,
        ).sum()  

    total_energy, forces = jax.value_and_grad(energy_fn)(r_ijs)
     
    local_energies = jnp.full(itypes_shape, total_energy / itypes_len)
    
    return local_energies, forces


@partial(jax.jit, static_argnums=(9, 10, 11))
def _jax_calc_local_energy(
    r_ijs,
    itype,
    jtypes,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
    scaling,
    min_dist,
    max_dist,
    # Static parameters:
    rb_size,
    execution_order,
    scalar_contractions
):
    r_abs = jnp.linalg.norm(r_ijs, axis=1)

    radial_basis = _jax_chebyshev_basis(r_abs, rb_size, min_dist, max_dist) 
    smoothing = jnp.where(r_abs < max_dist, (max_dist - r_abs) ** 2, 0)      
    scaled_smoothing = scaling * smoothing                                   

    coeffs = radial_coeffs[itype, jtypes] 
    rb_values = jnp.einsum(
        'j, jmn, jn -> mj',
        scaled_smoothing,
        coeffs,
        radial_basis
    )

    # slower but more memory efficient
    fused_for_checkpoint = lambda r_ijs_in, r_abs_in, rb_values_in: \
        _jax_calc_basis_symmetric_fused(
            r_ijs_in, r_abs_in, rb_values_in,
            execution_order,     
            scalar_contractions   
        )
    basis = checkpoint(fused_for_checkpoint)(r_ijs, r_abs, rb_values)

    #basis = _jax_calc_basis_symmetric_fused(
    #    r_ijs, r_abs, rb_values, execution_order, scalar_contractions
    #)
    energy = species_coeffs[itype] + jnp.dot(moment_coeffs, basis)
    return energy


def _jax_chebyshev_basis(r, n_terms, min_dist, max_dist):
    if n_terms == 0:
        return jnp.zeros((r.shape[0], 0))
    if n_terms == 1:
        return jnp.ones((r.shape[0], 1))

    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)

    def step(carry, _):
        T_prev, T_curr = carry
        T_next = 2 * r_scaled * T_curr - T_prev
        return (T_curr, T_next), T_curr

    T0 = jnp.ones_like(r_scaled)
    T1 = r_scaled
    
    _, T_rest = lax.scan(step, (T0, T1), None, length=n_terms - 2)

    return jnp.column_stack([T0, T1, *T_rest])
    
    
def _safe_tensor_sum(r, rb_values_mu, nu):
    r_bf16 = r.astype(dtypes.bfloat16)
    rb_values_mu_bf16 = rb_values_mu.astype(dtypes.bfloat16)
    
    if nu == 0:
        result = jnp.sum(rb_values_mu_bf16)
    elif nu == 1:
        result = jnp.dot(rb_values_mu_bf16, r_bf16)
    elif nu == 2:
        result = jnp.einsum('i,ij,ik->jk', rb_values_mu_bf16, r_bf16, r_bf16)
    elif nu == 3:
        result = jnp.einsum('i,ij,ik,il->jkl', rb_values_mu_bf16, r_bf16, r_bf16, r_bf16)
    else:
        operands = [rb_values_mu_bf16] + [r_bf16] * nu
        letters = string.ascii_lowercase[:nu]
        input_subs = ['i'] + [f'i{l}' for l in letters]
        result = jnp.einsum(
            f'{",".join(input_subs)}->{"".join(letters)}', 
            *operands
        )
        
    return result.astype(jnp.float32)
    
def _vectorized_safe_tensor_sum(r, rb_values, nu):
    if nu == 0:
        return jnp.sum(rb_values, axis=1)
    elif nu == 1:
        return jnp.dot(rb_values, r)
    else:
        operands = [rb_values, *([r] * nu)]
        letters = string.ascii_lowercase[:nu]
        input_subs = ['mj'] + [f'j{l}' for l in letters]
        return jnp.einsum(f'{",".join(input_subs)}->m{"".join(letters)}', *operands)

def _jax_contract_over_axes(m1, m2, axes):
    m1 = m1.astype(jnp.float32)
    m2 = m2.astype(jnp.float32)
    
    m1_bf16 = m1.astype(dtypes.bfloat16)
    m2_bf16 = m2.astype(dtypes.bfloat16)
    
    result_bf16 = jnp.tensordot(m1_bf16, m2_bf16, axes=axes)
    
    return result_bf16.astype(jnp.float32)

def _jax_calc_basis_symmetric_fused(
    r_ijs, r_abs, rb_values,
    execution_order, scalar_contractions
):
    r = (r_ijs.T / r_abs).T
    results = {}


    basic_moment_keys_by_nu = defaultdict(list)
    for op_type, key in execution_order:
        if op_type == 'basic':
            mu, nu, l = key
            basic_moment_keys_by_nu[nu].append(key)

    for nu, keys in basic_moment_keys_by_nu.items():
        all_nu_tensors = _vectorized_safe_tensor_sum(r, rb_values, nu)

        for key in keys:
            mu, _, _ = key
            results[str(key)] = all_nu_tensors[mu]

    for op_type, key in execution_order:
        if op_type == 'contract':
            key_left, key_right, _, (axes_left, axes_right) = key
            left_val = results[str(key_left)]
            right_val = results[str(key_right)]
            results[str(key)] = _jax_contract_over_axes(left_val, right_val, (axes_left, axes_right))


    basis_vals = [results[str(k)] for k in scalar_contractions]
    return jnp.stack(basis_vals)