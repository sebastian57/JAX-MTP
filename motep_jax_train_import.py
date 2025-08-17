import numpy as np
from ase import Atoms
import io
import itertools


import jax
import jax.numpy as jnp
from functools import partial
import pickle
from ase.neighborlist import PrimitiveNeighborList
import os
import textwrap
import numpy.typing as npt
import matplotlib.pyplot as plt
import pathlib
from dataclasses import asdict

# import commands from io.__init__.py
import ase.io
from ase.io.formats import parse_filename
from motep_original_files.old_io.mlip.cfg import read_cfg, write_cfg

jax.config.update('jax_enable_x64', False)

### Data preprocessing function

def create_training_testing_data():
    
    
    
    return 

def _compute_all_offsets(nl: PrimitiveNeighborList, atoms: Atoms):
    '''
    Internal helper function to process neighbor lists from ASE.

    Takes the raw neighbor list and computes the neighbor indices and the
    corresponding offset vectors due to periodic boundary conditions.
    Pads the results so that each atom has the same number of neighbors
    (filling with -1 for indices and zero vectors for offsets), suitable
    for batch processing in JAX.

    :param nl: ASE PrimitiveNeighborList containing neighbor information.
    :param atoms: ASE Atoms object for which the neighbors were computed.
    :return: Tuple containing:
             - np.ndarray: Padded array of neighbor indices (N_atoms, max_neighbors).
             - np.ndarray: Padded array of offset vectors (N_atoms, max_neighbors, 3).
    '''
    cell = atoms.cell
    js = [nl.get_neighbors(i)[0] for i in range(len(atoms))]
    offsets = [nl.get_neighbors(i)[1] @ cell for i in range(len(atoms))]
    num_js = [_.shape[0] for _ in js]
    max_num_js = np.max([_.shape[0] for _ in offsets]) if offsets else 0 
    pads = [(0, max_num_js - n) for n in num_js]
    padded_js = [
        np.pad(js_, pad_width=pad, constant_values=-1) for js_, pad in zip(js, pads)
    ]
    padded_offsets = [
        np.pad(offset, pad_width=(pad, (0, 0)), constant_values=0.0) for offset, pad in zip(offsets, pads) 
    ]

    if not padded_js:
        return np.empty((len(atoms), 0), dtype=int), np.empty((len(atoms), 0, 3), dtype=float)
    return np.array(padded_js, dtype=int), np.array(padded_offsets)

def compute_neighbor_data(atoms, mtp_data):
    '''
    Computes the neighbor list indices and offset vectors for a given Atoms object.

    Uses ASE's PrimitiveNeighborList based on the MTP potential's cutoff
    radius (`mtp_data.max_dist`). It then formats this information using
    `_compute_all_offsets` into padded numpy arrays. This data is essential
    for calculating interatomic distances and MTP features.

    :param atoms: ASE Atoms object representing the atomic configuration.
    :param mtp_data: Object containing MTP potential parameters, including `max_dist`.
    :return: Tuple containing:
             - np.ndarray: Padded array of neighbor indices (N_atoms, max_neighbors).
             - np.ndarray: Padded array of offset vectors (N_atoms, max_neighbors, 3).
    '''
    nl = PrimitiveNeighborList(
        cutoffs=[0.5 * mtp_data.max_dist] * len(atoms),
        skin=0.3,
        self_interaction=False,
        bothways=True,
    )
    if len(atoms) > 0:
        nl.update(atoms.pbc, atoms.cell, atoms.positions)
    else:
        return np.empty((0, 0), dtype=int), np.empty((0, 0, 3), dtype=float)

    all_js, all_offsets = _compute_all_offsets(nl, atoms)
    return all_js, all_offsets

def _get_all_distances(atoms: Atoms, mtp_data, all_js, all_offsets) -> tuple[np.ndarray, np.ndarray]:
    '''
    Internal helper function to calculate interatomic displacement vectors.

    Given an Atoms object and the pre-computed padded neighbor indices (`all_js`)
    and offset vectors (`all_offsets`), this function calculates the displacement
    vector from each atom `i` to its neighbors `j`. Handles padding by assigning
    a large distance vector for padded neighbor slots.

    :param atoms: ASE Atoms object.
    :param mtp_data: Object containing MTP parameters, including `max_dist`.
    :param all_js: Padded array of neighbor indices (N_atoms, max_neighbors).
    :param all_offsets: Padded array of offset vectors (N_atoms, max_neighbors, 3).
    :return: Tuple containing:
             - np.ndarray: The input `all_js` array (passed through).
             - np.ndarray: Array of interatomic displacement vectors `r_ij`
                           (N_atoms, max_neighbors, 3). Padded entries are
                           assigned large values.
    '''
    max_dist = mtp_data.max_dist
    positions = atoms.positions
    offsets = all_offsets
    if all_js.shape[1] == 0: 
        all_r_ijs = np.zeros((len(atoms), 0, 3))
    else:
        all_r_ijs = positions[all_js] + offsets - positions[:, None, :]
        mask = all_js < 0
        all_r_ijs[mask, :] = max_dist 
    return all_js, all_r_ijs

def get_data_for_indices(jax_images, index):
    '''
    Extracts pre-computed data for a specific configuration index from the JAX dataset.

    Retrieves the JAX arrays corresponding to a single atomic configuration
    (specified by `index`) from the larger `jax_images` dictionary, which
    contains the pre-processed data for the entire dataset.

    :param jax_images: Dictionary containing JAX arrays for the entire dataset
                       (e.g., 'itypes', 'all_js', 'all_rijs', 'E', 'F', 'sigma', etc.).
    :param index: The integer index of the desired configuration/image.
    :return: Tuple containing the data arrays for the specified configuration:
             (itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma).
    '''
    itypes = jax_images['itypes'][index]
    all_js = jax_images['all_js'][index]
    all_rijs = jax_images['all_rijs'][index]
    all_jtypes = jax_images['all_jtypes'][index]
    cell_rank = jax_images['cell_ranks'][index]
    volume = jax_images['volumes'][index]
    E = jax_images['E'][index]
    F = jax_images['F'][index]
    sigma = jax_images['sigma'][index]

    return itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma


def get_types(atoms: Atoms, species: list[int] = None) -> npt.NDArray[np.int64]:
    if species is None:
        return np.array(atoms.numbers, dtype=int)
    else:
        return np.fromiter((species.index(_) for _ in atoms.numbers), dtype=int)

def extract_data(atoms, species, mtp_data, all_js, all_offsets):
    '''
    Extracts and computes all necessary data for a single atomic configuration.

    This function takes a single `Atoms` object and pre-computed neighbor data,
    calculates the MTP-specific inputs (atom types, neighbor types, displacement
    vectors using `_get_all_distances`), and retrieves the target properties
    (energy, forces, stress) from the `Atoms` object. Converts results to JAX arrays.

    :param atoms: ASE Atoms object for the configuration.
    :param species: List of atomic numbers defining the species types for the MTP.
    :param mtp_data: Object containing MTP potential parameters.
    :param all_js: Padded array of neighbor indices from `compute_neighbor_data`.
    :param all_offsets: Padded array of offset vectors from `compute_neighbor_data`.
    :return: Tuple containing JAX arrays for the configuration:
             (itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma).
    '''

    itypes = jnp.array(get_types(atoms, species))
    all_js, all_rijs = _get_all_distances(atoms, mtp_data, all_js, all_offsets)
    all_js, all_rijs = jnp.array(all_js), jnp.array(all_rijs)

    if all_js.shape[1] > 0:
        all_jtypes = itypes[jnp.asarray(all_js)]
        all_jtypes = jnp.where(all_js >= 0, all_jtypes, -1)
    else:
        all_jtypes = jnp.empty((len(atoms), 0), dtype=itypes.dtype)


    cell_rank = atoms.cell.rank
    volume = atoms.get_volume()

    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    sigma = atoms.get_stress(voigt=True) 

    return itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma


def save_data_pickle(data, filename):
    '''
    Saves Python data to a file using the pickle module.

    :param data: The Python object (e.g., dictionary, list) to be saved.
    :param filename: The path to the file where the data will be saved.
    :return: None
    '''
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(filename):
    '''
    Loads Python data from a pickle file.

    :param filename: The path to the pickle file to load.
    :return: The Python object loaded from the file.
    '''
    with open(filename, 'rb') as f:
        return pickle.load(f)

def extract_and_save_img_data(images, species, mtp_data, name='jax_images_data'):
    '''
    Processes a list of ASE Atoms objects (images) to extract training data and saves it.

    Iterates through each `Atoms` object in the `images` list. For each image,
    it computes neighbor data (`compute_neighbor_data`), extracts features and
    target properties (`extract_data`), collects all data into lists, converts
    these lists into a dictionary of stacked JAX arrays, and finally saves
    this dictionary to a pickle file ('jax_images_data.pkl') using `save_data_pickle`.
    This pre-processes the entire dataset for efficient use in JAX-based training.

    :param images: A list of ASE `Atoms` objects representing the training configurations.
    :param species: List of atomic numbers defining the species types for the MTP.
    :param mtp_data: Object containing MTP potential parameters (like `max_dist`).
    :return: None
    '''

    all_itypes = []
    all_all_js = []
    all_all_rijs = []
    all_all_jtypes = []
    cell_ranks = []
    volumes = []
    all_E = []
    all_F = []
    all_sigma = []

    max_atoms = max(len(atoms) for atoms in images) if images else 0
    max_neighbors_global = 0 

    all_neighbor_data = []
    for atoms in images:
        all_js, all_offsets = compute_neighbor_data(atoms, mtp_data)
        all_neighbor_data.append({'js': all_js, 'offsets': all_offsets})
        if all_js.shape[1] > max_neighbors_global:
            max_neighbors_global = all_js.shape[1]

    for i, atoms in enumerate(images):
        neighbor_data = all_neighbor_data[i]
        current_max_neighbors = neighbor_data['js'].shape[1]
        pad_width_neighbors = max_neighbors_global - current_max_neighbors
        
        padded_js = np.pad(neighbor_data['js'], ((0, 0), (0, pad_width_neighbors)), constant_values=-1)
        padded_offsets = np.pad(neighbor_data['offsets'], ((0, 0), (0, pad_width_neighbors), (0,0)), constant_values=0.0)

        itypes, all_js_extracted, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = extract_data(atoms, species, mtp_data, padded_js, padded_offsets)

        num_atoms = len(atoms)
        pad_width_atoms = max_atoms - num_atoms

        all_itypes.append(jnp.pad(itypes, (0, pad_width_atoms), constant_values=-1))
        all_all_js.append(jnp.pad(all_js_extracted, ((0, pad_width_atoms), (0, 0)), constant_values=-1))
        all_all_rijs.append(jnp.pad(all_rijs, ((0, pad_width_atoms), (0, 0), (0,0)), constant_values=jnp.nan))
        all_all_jtypes.append(jnp.pad(all_jtypes, ((0, pad_width_atoms), (0, 0)), constant_values=-1))
        all_F.append(jnp.pad(F, ((0, pad_width_atoms), (0, 0)), constant_values=jnp.nan))

        cell_ranks.append(cell_rank)
        volumes.append(volume)
        all_E.append(E)
        all_sigma.append(sigma) 

  
    data_dict = {
        'itypes': jnp.stack(all_itypes),
        'all_js': jnp.stack(all_all_js),
        'all_rijs': jnp.stack(all_all_rijs),
        'all_jtypes': jnp.stack(all_all_jtypes),
        'cell_ranks': jnp.array(cell_ranks),
        'volumes': jnp.array(volumes),
        'E': jnp.array(all_E),
        'F': jnp.stack(all_F),
        'sigma': jnp.stack(all_sigma),
        'n_atoms': jnp.array([len(atoms) for atoms in images]) 
    }


    save_data_pickle(data_dict, f'training_data/{name}.pkl')


def read(filename: str, species: list[int] | None = None) -> list[Atoms]:
    '''Read images.

    Parameters
    ----------
    filename : str
        File name to be read.
        Both the MLIP `.cfg` format and the ASE-recognized formats can be parsed.

        To select a part of images, the ASE `@` syntax can be used as follows.

        https://wiki.fysik.dtu.dk/ase/ase/gui/basics.html#selecting-part-of-a-trajectory

        - `x.traj@0:10:1`: first 10 images
        - `x.traj@0:10`: first 10 images
        - `x.traj@:10`: first 10 images
        - `x.traj@-10:`: last 10 images
        - `x.traj@0`: first image
        - `x.traj@-1`: last image
        - `x.traj@::2`: every second image

        Further, for the ASE database format, i.e., `.json` and `.db`,
        the extended ASE syntax can also be used as follows.

        https://wiki.fysik.dtu.dk/ase/ase/db/db.html#integration-with-other-parts-of-ase

        https://wiki.fysik.dtu.dk/ase/ase/db/db.html#querying

        - `x.db@H>0`: images with hydrogen atoms

    species : list[int]
        List of atomic numbers for the atomic types in the MLIP `.cfg` format.

    Returns
    -------
    list[Atoms]
        List of ASE `Atoms` objects.

    '''
    filename_parsed, index = parse_filename(filename)
    index = ':' if index is None else index
    if isinstance(filename_parsed, str) and filename_parsed.endswith('.cfg'):
        images = read_cfg(filename_parsed, index=index, species=species)
    else:
        # was io.read
        images = ase.read(filename_parsed, index=index)
    return [images] if isinstance(images, Atoms) else images


def write(filename: str, images: list[Atoms], species: list[int] | None = None) -> None:
    '''Write images.

    Parameters
    ----------
    filename : str
        File name to be written.
        Both the MLIP `.cfg` format and the ASE-recognized formats can be parsed.
    images : list[Atoms]
        List of ASE `Atoms` objects.
    species : list[int]
        List of atomic numbers for the atomic types in the MLIP `.cfg` format.

    '''
    if filename.endswith('.cfg'):
        return write_cfg(filename, images, species=species)
    # was io.write
    return ase.write(filename, images)


def read_images(filenames: list[str], species: list[int] | None = None) -> list:
    '''
    Reads atomic configurations (images) from a list of file paths.

    Uses the `io.read` function to load structures from files,
    potentially filtering by atom types if `species` is provided.
    Aggregates structures from all specified files into a single list.

    :param filenames: A list of strings, where each string is a path to a file
                      containing atomic structures (e.g., 'training.cfg').
    :param species: Optional; A list of atomic numbers. If provided, only atoms
                    of these types will be included in the loaded structures.
    :return: A list of ASE `Atoms` objects read from the files.
    '''
    images = []
    for filename in filenames:
        # was io.read
        images.extend(read(filename, species))
    return images

def write_motep_toml(level):
    '''
    Generates and writes the 'motep.toml' configuration file for training.

    Creates a TOML formatted string containing settings for the MOTEP training
    process, such as input/output potential file paths (using the provided `level`),
    loss function weights, and optimizer steps. Overwrites any existing 'motep.toml' file.

    :param level: String identifier for the MTP level (e.g., '04'), used to
                  construct the initial potential filename.
    :return: None
    '''
    content = textwrap.dedent(f'''\
        configurations = 'training.cfg'
        potential_initial = 'untrained_mtps/{level}.mtp'
        potential_final = 'final.mtp'

        seed = 10  # random seed for initializing MTP paramters

        engine = 'jax'  # {{'numpy', 'numba', 'mlippy'}}

        [loss]  # settings for the loss function
        energy_weight = 1.0
        forces_weight = 0.01
        stress_weight = 0.001

        [[steps]]
        method = 'L-BFGS-B'
        optimized = ['scaling', 'species_coeffs', 'radial_coeffs', 'moment_coeffs']

        [[steps]]
        method = 'Nelder-Mead'
        optimized = ['scaling', 'species_coeffs', 'radial_coeffs', 'moment_coeffs']
        [steps.kwargs]
        tol = 1e-7
        [steps.kwargs.options]
        maxiter = 1000
    ''')

    if os.path.exists('motep.toml'):
        os.remove('motep.toml')
    with open('motep.toml', 'w') as file:
        file.write(content)
        
        
def write_mtp(file, data):
    
    def _format_value(value: float | int | list | str) -> str:
        if hasattr(value, "item") and not isinstance(value, (list, str)):
            try:
                scalar = value.item()
            except Exception:
                pass
            else:
                return _format_value(scalar)
        print(f'value: {type(value)}')
        if isinstance(value, float):
            return f"{value:21.15e}"
        if isinstance(value, int):
            return f"{value:d}"
        if isinstance(value, list):
            return _format_list(value)
        if isinstance(value, np.ndarray):
            return _format_list(value.tolist())
        return value.strip()


    def _format_list(value: list) -> str:
        if len(value) == 0:
            return "{}"
        if isinstance(value[0], list):
            return "{" + ", ".join(_format_list(_) for _ in value) + "}"
        return "{" + ", ".join(f"{_format_value(_)}" for _ in value) + "}"
    
    """Write an MLIP .mtp file."""
    keys0 = [
        "version",
        "potential_name",
        "scaling",
        "species_count",
        "potential_tag",
        "radial_basis_type",
    ]
    keys1 = [
        "min_dist",
        "max_dist",
        "radial_basis_size",
        "radial_funcs_count",
    ]
    keys2 = [
        "alpha_moments_count",
        "alpha_index_basic_count",
        "alpha_index_basic",
        "alpha_index_times_count",
        "alpha_index_times",
        "alpha_scalar_moments",
        "alpha_moment_mapping",
        "species_coeffs",
        "moment_coeffs",
    ]
    data = asdict(data)
    species_count = data["species_count"]
    with pathlib.Path(file).open("w", encoding="utf-8") as fd:
        fd.write("MTP\n")
        for key in keys0:
            if data[key] is not None:
                fd.write(f"{key} = {_format_value(data[key])}\n")
        for key in keys1:
            if data[key] is not None:
                fd.write(f"\t{key} = {_format_value(data[key])}\n")
        if data["radial_coeffs"] is not None:
            fd.write("\tradial_coeffs\n")
            for k0, k1 in itertools.product(range(species_count), repeat=2):
                value = data["radial_coeffs"][k0, k1]
                fd.write(f"\t\t{k0}-{k1}\n")
                for _ in range(data["radial_funcs_count"]):
                    fd.write(f"\t\t\t{_format_list(value[_])}\n")
        for key in keys2:
            if data[key] is not None:
                fd.write(f"{key} = {_format_value(data[key])}\n")

        
### Plotting/Analysis functions ###

def plot_timing(levels, elapsed_times, counter, name, folder_name, save='true'):
    
    base_output_dir = 'training_results'
    target_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"Ensured output directory exists: {target_dir}")
    
    fig, ax = plt.subplots()
    ax.plot(levels[0:counter], elapsed_times, c='b')
    ax.scatter(levels[0:counter], elapsed_times, c='r', label='times')
    ax.set_xlabel('level')
    ax.set_ylabel('time in s')
    ax.set_title(f'Timing {name}')
    ax.grid(True)
    plt.legend()
    plt.savefig(f'training_results/{folder_name}/timing_{name}.pdf')
    plt.show()
    plt.close()
    
    if save == 'true':
        data = np.column_stack((levels[0:counter], elapsed_times))
        np.savetxt(f'training_results/{folder_name}/timing_{name}.txt', data, header='level time_in_s', comments='', fmt='%.8f')
    else:
        print('Data not saved')
    
    
def plot_loss(loss, val_loss, steps_performed, name, folder_name, level, min_ind=0, max_ind=-1, save='true'):
    
    base_output_dir = 'training_results'
    target_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"Ensured output directory exists: {target_dir}")
    
    epochs = np.arange(np.sum(steps_performed))
    
    fig, ax = plt.subplots()
    ax.plot(epochs[min_ind:max_ind], loss[min_ind:max_ind], c='b')
    ax.scatter(epochs[min_ind:max_ind], loss[min_ind:max_ind], c='r', label='loss')
    ax.plot(epochs[min_ind:max_ind], val_loss[min_ind:max_ind], c='g')
    ax.scatter(epochs[min_ind:max_ind], val_loss[min_ind:max_ind], c='yellow', label='validation loss')
    ax.axvline(steps_performed[0],color='g',linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RMSE')
    ax.set_title(f'Loss {name}')
    ax.grid(True)
    plt.legend()
    plt.savefig(f'training_results/{folder_name}/loss_{name}_{level}.pdf')
    plt.close()
    plt.show()
    
    if save == 'true':
        data = np.column_stack((loss, val_loss, epochs))
        np.savetxt(f'training_results/{folder_name}/loss_{name}_{level}.txt', data, header='rmse val_rmse epochs', comments='', fmt='%.8f')
    else:
        print('Data not saved')
        
        
def plot_test(E,F,sigma,real_values,name,folder_name,level):
    
    loss_E = jnp.sum((E - real_values[0])**2)
    loss_F = jnp.sum((F - real_values[1])**2)
    loss_sigma = jnp.sum((sigma - real_values[2])**2)
    
    x = np.arange(0,len(E))
        
    fig, ax = plt.subplots()
    
    text_str = (
        f"RMSE E: {float(loss_E):.4e}\n"
        f"RMSE F: {float(loss_F):.4e}\n"
        f"RMSE S: {float(loss_sigma):.4e}"
    )

    ax.text(0.97, 0.97, text_str,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='aliceblue', alpha=0.8)) 
    
    ax.scatter(x,E,c='b',label='prediction')
    ax.scatter(x,real_values[0],c='r',label='real values')
    ax.set_xlabel('CFGs')
    ax.set_ylabel('E')
    ax.set_title(f'Test {name}')
    ax.grid(True)
    plt.legend()
    plt.savefig(f'training_results/{folder_name}/test_{name}_{level}.pdf')
    plt.close()
    plt.show()
    


### LLS test ###
def solve_lls_for_basis(
    prediction_fn,
    params,
    jax_images,
    training_ids,
    weight_e, 
    weight_f,
    weight_s,
    num_basis_params,
    num_targets_per_config,
    num_f_components_per_config, 
    num_s_components_per_config,
    num_configs
):
    
    def flatten_targets(E, F, sigma):
      E_arr = jnp.atleast_1d(E)
      F_flat = F.reshape(-1)
      sigma_flat = sigma.reshape(-1) 
      return jnp.concatenate((E_arr, F_flat, sigma_flat))
   
    def predict_with_separated_params(basis_p, fixed_p, structure_inputs):
        current_params = {
            'species': fixed_p['species'], 'radial': fixed_p['radial'], 'basis': basis_p
        }
        itypes, all_js, all_rijs, all_jtypes, cell_rank, volume = structure_inputs
        targets = prediction_fn(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, current_params
        )
        E_pred, F_pred, sigma_pred = targets['energy'], targets['forces'], targets['stress']
        return flatten_targets(E_pred, F_pred, sigma_pred)

    def calculate_offset_contribution(fixed_p, structure_inputs):
        zero_basis = jnp.zeros((num_basis_params,), dtype=params['basis'].dtype)
        return predict_with_separated_params(zero_basis, fixed_p, structure_inputs)

    get_basis_matrix_single = jax.jacfwd(predict_with_separated_params, argnums=0)

    def get_single_config_data_for_lls(atoms_id):
        itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E_true, F_true, sigma_true = get_data_for_indices(jax_images, atoms_id)
        structure_inputs = (itypes, all_js, all_rijs, all_jtypes, cell_rank, volume)
        true_targets_flat = flatten_targets(E_true, F_true, sigma_true)
        return structure_inputs, true_targets_flat

    all_structure_inputs, all_true_targets_flat = jax.vmap(get_single_config_data_for_lls)(training_ids)

    fixed_params = {'species': params['species'], 'radial': params['radial']}
    all_B_matrices = jax.vmap(get_basis_matrix_single, in_axes=(None, None, 0))(
        params['basis'], fixed_params, all_structure_inputs
    )
    all_offsets = jax.vmap(calculate_offset_contribution, in_axes=(None, 0))(
        fixed_params, all_structure_inputs
    )

    num_configs = len(training_ids)
    total_targets = num_configs * num_targets_per_config
    X = all_B_matrices.reshape(total_targets, num_basis_params)
    y_true_flat = all_true_targets_flat.reshape(total_targets)
    offsets_flat = all_offsets.reshape(total_targets)
    y_prime = y_true_flat - offsets_flat

    sqrt_we = jnp.sqrt(weight_e)
    sqrt_wf = jnp.sqrt(weight_f)
    sqrt_ws = jnp.sqrt(weight_s)

    weights_e_part = jnp.full((1,), sqrt_we)
    weights_f_part = jnp.full((num_f_components_per_config,), sqrt_wf)
    weights_s_part = jnp.full((num_s_components_per_config,), sqrt_ws)
    sqrt_weights_per_config = jnp.concatenate((weights_e_part, weights_f_part, weights_s_part))

    sqrt_weights = jnp.tile(sqrt_weights_per_config, num_configs)

  
    X_weighted = X * sqrt_weights[:, None]
    y_prime_weighted = y_prime * sqrt_weights

    w_basis_optimal, residuals, rank, s = jnp.linalg.lstsq(X_weighted, y_prime_weighted, rcond=None)

    return w_basis_optimal


