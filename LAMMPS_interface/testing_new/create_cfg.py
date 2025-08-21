import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList


def build_l12_ni3al(a=3.57):
    frac = np.array([
        [0.0, 0.0, 0.0],       # Al
        [0.0, 0.5, 0.5],       # Ni
        [0.5, 0.0, 0.5],       # Ni
        [0.5, 0.5, 0.0],       # Ni
    ])
    species = ['Al', 'Ni', 'Ni', 'Ni']
    type_map = {'Al': 0, 'Ni': 1}
    cell = np.diag([a, a, a])
    positions = frac @ cell
    types = np.array([type_map[s] for s in species], int)
    return positions, types, cell


def write_mlip_cfg(filename, positions, types, cell,
                   energy=0.0, forces=None, stress=None):
    N = len(types)
    if forces is None:
        forces = np.zeros_like(positions)
    if stress is None:
        stress = np.zeros(6)

    with open(filename, "w") as f:
        f.write("BEGIN_CFG\n")
        f.write(" Size\n")
        f.write(f"{N:5d}\n")
        f.write(" Supercell\n")
        for i in range(3):
            f.write(f"{cell[i,0]:13.6f}{cell[i,1]:12.6f}{cell[i,2]:12.6f}\n")
        f.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz\n")
        for i in range(N):
            f.write(f"{i+1:13d}{types[i]:5d}"
                    f"{positions[i,0]:14.6f}{positions[i,1]:14.6f}{positions[i,2]:14.6f}"
                    f"{forces[i,0]:12.6f}{forces[i,1]:12.6f}{forces[i,2]:12.6f}\n")
        f.write(" Energy\n")
        f.write(f"{energy:17.12f}\n")
        f.write(" PlusStress:  xx          yy          zz          yz          xz          xy\n")
        f.write(f"{stress[0]:12.5f}{stress[1]:12.5f}{stress[2]:12.5f}{stress[3]:12.5f}{stress[4]:12.5f}{stress[5]:12.5f}\n")
        f.write("END_CFG\n")


def read_mlip_cfg(filename):
    """Minimal reader for MLIP2 .cfg files written by write_mlip_cfg."""
    with open(filename) as f:
        lines = f.readlines()

    # number of atoms: line after " Size"
    size_idx = [i for i, l in enumerate(lines) if l.strip() == "Size"][0]
    n_atoms = int(lines[size_idx + 1].strip())

    # supercell: 3 lines after " Supercell"
    sc_idx = [i for i, l in enumerate(lines) if l.strip() == "Supercell"][0]
    cell = []
    for i in range(3):
        cell.append([float(x) for x in lines[sc_idx + 1 + i].split()])
    cell = np.array(cell)

    # AtomData block
    atomdata_idx = [i for i, l in enumerate(lines) if l.strip().startswith("AtomData")][0] + 1
    positions, types = [], []
    for i in range(n_atoms):
        parts = lines[atomdata_idx + i].split()
        types.append(int(parts[1]))
        positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
    positions = np.array(positions)
    types = np.array(types)

    # reconstruct ASE Atoms
    type_map = {0: "Al", 1: "Ni"}  # adjust if you add more elements
    symbols = [type_map[t] for t in types]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms, types, cell, positions


def analyze_cfg(filename, cutoff=5.0):
    atoms, itypes, cell, positions = read_mlip_cfg(filename)

    # neighbor list
    nl = NeighborList([cutoff]*len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    all_js, all_rijs, all_jtypes = [], [], []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        all_js.append(indices)
        rij = atoms.positions[indices] + np.dot(offsets, atoms.get_cell()) - atoms.positions[i]
        all_rijs.append(rij)
        all_jtypes.append([itypes[j] for j in indices])

    return {
        "itypes": itypes,
        "all_js": all_js,
        "all_rijs": all_rijs,
        "all_jtypes": all_jtypes,
        "cell_rank": np.linalg.matrix_rank(atoms.cell),
        "volume": atoms.get_volume(),
        "n_atoms": len(atoms),
        "n_neighbors": sum(len(js) for js in all_js),
        "cell": cell,
        "positions": positions,
    }



import numpy as np
from ase.io.lammpsdata import read_lammps_data
from ase.neighborlist import NeighborList

def analyze_lammps_data(filename, cutoff=5.0, max_atoms=2889, max_neighbors=150):
    """
    Parse a LAMMPS data file and return tensors for JAX model input.
    
    Parameters
    ----------
    filename : str
        Path to the LAMMPS data file.
    cutoff : float
        Cutoff distance for neighbor list.
    max_atoms : int
        Maximum atoms the model expects.
    max_neighbors : int
        Maximum neighbors per atom the model expects.
    
    Returns
    -------
    dict with keys:
        itypes      (max_atoms,)
        all_js      (max_atoms, max_neighbors)
        all_rijs    (max_atoms, max_neighbors, 3)
        all_jtypes  (max_atoms, max_neighbors)
        cell_rank   int
        volume      float
        natoms_actual int
        nneigh_actual int
        cell, positions for reference
    """
    # Load atoms from LAMMPS data file
    atoms = read_lammps_data(filename, style="atomic")
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    natoms = len(atoms)
    itypes = np.array(atoms.get_atomic_numbers(), dtype=np.int32)

    # Neighbor list setup
    cutoffs = [cutoff / 2.0] * natoms  # ASE expects per-atom cutoff radius
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    # Allocate padded arrays
    all_js = np.full((max_atoms, max_neighbors), -1, dtype=np.int32)
    all_rijs = np.zeros((max_atoms, max_neighbors, 3), dtype=np.float32)
    all_jtypes = np.full((max_atoms, max_neighbors), -1, dtype=np.int32)

    nneigh_actual_total = 0

    for i in range(natoms):
        indices, offsets = nl.get_neighbors(i)
        nneigh = len(indices)
        nneigh_actual_total += nneigh

        if nneigh > max_neighbors:
            raise RuntimeError(
                f"Atom {i} has {nneigh} neighbors, exceeds max_neighbors={max_neighbors}"
            )

        all_js[i, :nneigh] = indices
        rij = atoms.positions[indices] + np.dot(offsets, atoms.get_cell()) - atoms.positions[i]
        all_rijs[i, :nneigh, :] = rij
        all_jtypes[i, :nneigh] = itypes[indices]

    return {
        "itypes": itypes.astype(np.int32),
        "all_js": all_js,
        "all_rijs": all_rijs,
        "all_jtypes": all_jtypes,
        "cell_rank": np.linalg.matrix_rank(cell),
        "volume": atoms.get_volume(),
        "natoms_actual": natoms,
        "nneigh_actual": nneigh_actual_total,
        "cell": cell,
        "positions": positions,
    }


if __name__ == "__main__":
    # build and write
    pos, types, cell = build_l12_ni3al()
    write_mlip_cfg("ni3al.cfg", pos, types, cell)

    # analyze
    info = analyze_cfg("ni3al.cfg")
    print("Analysis:", info)

