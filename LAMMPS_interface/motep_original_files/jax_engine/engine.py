from ase import Atoms

from ..base import EngineBase
from ..data import MTPData

from .conversion import BasisConverter, moments_count_to_level_map
from .jax import calc_energy_forces_stress as jax_calc
from .moment import MomentBasis

import jax


class JaxMTPEngine(EngineBase):
    """MTP Engine in 'full tensor' version based on jax."""

    def __init__(self, *args, **kwargs):
        """Intialize the engine."""
        self.moment_basis = None
        self.basis_converter = None
        super().__init__(*args, **kwargs)


    def _flatten_computation_graph(self, basic_moments, pair_contractions, scalar_contractions):
        execution_order = []
        dependencies = {}
        
        for moment_key in basic_moments:
            execution_order.append(('basic', moment_key))
            dependencies[moment_key] = []
        
        remaining_contractions = list(pair_contractions)
        while remaining_contractions:
            for i, contraction_key in enumerate(remaining_contractions):
                key_left, key_right, _, axes = contraction_key
                if key_left in dependencies and key_right in dependencies:
                    execution_order.append(('contract', contraction_key))
                    dependencies[contraction_key] = [key_left, key_right]
                    remaining_contractions.pop(i)
                    break
            else:
                raise ValueError("Circular dependency in contraction graph")
        
        return execution_order, dependencies

    def update(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        super().update(mtp_data)
        if self.mtp_data.alpha_moments_count is not None:
            level = moments_count_to_level_map[mtp_data.alpha_moments_count]
            if self.moment_basis is None:
                self.moment_basis = MomentBasis(level)
                self.moment_basis.init_moment_mappings()
                self.basis_converter = BasisConverter(self.moment_basis)
            elif self.moment_basis.max_level != level:
                raise RuntimeError(
                    "Changing moments/level is not allowed. "
                    "Use a new instance instead."
                )
            self.basis_converter.remap_mlip_moment_coeffs(self.mtp_data)

            basic_moments = self.moment_basis.basic_moments
            scalar_contractions = self.moment_basis.scalar_contractions
            pair_contractions = self.moment_basis.pair_contractions
            

            execution_order, _ = self._flatten_computation_graph(
                basic_moments, pair_contractions, scalar_contractions
            )            
            self.moment_basis.execution_order = tuple(execution_order)


    def calculate(self, itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params):
        #self.update_neighbor_list(atoms)
        mtp_data = self.mtp_data
        
        # set params for gradient computation
        mtp_data.species_coeffs = params['species']
        mtp_data.radial_coeffs = params['radial']
        self.basis_converter.remapped_coeffs = params['basis']

        energies, forces, stress = jax_calc(
            itypes,
            all_js,
            all_rijs,
            all_jtypes,
            cell_rank,
            volume,
            mtp_data.species,
            mtp_data.scaling,
            mtp_data.min_dist,
            mtp_data.max_dist,
            mtp_data.species_coeffs,
            self.basis_converter.remapped_coeffs,
            mtp_data.radial_coeffs,
            # Static parameters:
            self.moment_basis.execution_order,
            tuple(self.moment_basis.scalar_contractions)
        )
        results = {}
        results["energies"] = energies
        results["energy"] = results["energies"].sum()
        results["forces"] = forces
        results["stress"] = stress
        return results
    
