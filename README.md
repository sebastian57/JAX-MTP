# JAX-MTP 
A Python implementation of the Moment Tensor Potential model with a custom pair-style LAMMPS interface and training code. JAX represents the main backbone of the code and allows for efficient computation on GPU hardware.

## General Setup for LAMMPS Simulations
### Step 1: Python venv
Create a Python virtual environment with the necessary files:
- python3 -m venv JAX-MTP_env
- source JAX-MTP_env/bin/activate
- python3 -m pip install -U "jax[cuda13]" optax flatbuffers absl_py ase
### Step 2: Set-up and build LAMMPS
Clone the LAMMPS repository and build with the correct modifications
- git clone https://github.com/lammps/lammps.git --> creates lammps directory
- Copy the four files from pair_style_files/: pair_jax_mtp.* and zero_overhead.* to lammps/src/EXTRA-PAIR
- Build LAMMPS with the following steps:
  - Activate Python venv (JAX-MTP_env)
  - Create lammps/build directory
  - Run commands from LAMMPS_build_instructions file
### Step 3: Compile Potential
Determine system size and find trained .mtp file
- Add the correct path for the trained .mtp file and the MTP level at the bottom of the compile_cuda.py file
- Set the required system size specifications in the "CONFIGS" list
- Run the compile_cuda.py script on the target hardware (takes a few seconds per potential file) (If not cuda based, check the optimizer flags set at the beginning of the file)
- This creates a compiled_potentials directory with the corresponding .bin file inisde
### Step 4: Run Simulation
- Create a run script with the following inclusions:
  - Choose correct hardware (same as during compilation)
  - Activate Python venv (JAX-MTP_env)
  - Modify LAMMPS input script to call correct pair-style, e.g.: pair_style jax/mtp compiled_potentials/jaxmtp_potential_cuda_2889_60.bin 2889 60 5.0 0
  -   - Parameters: MAX_ATOMS, MAX_ATOMS_PER_NEIGHBOR, MTP_CUTOFF, DEBUG_LEVEL (0,1,2,3)
  - Potentially set some hardware flags
  - Run on 1 MPI process 

## Notes on Existing Files
- pair-style files
  - Well tested.
  - Currently NPT is still being fixed, but NVT and NVE work
  - Import some functionality from motep_original_files directory. Could be cleaned up
- Training files
  - Work, but use an older version of the JAX backend
  - Either big batch or mini batch version
  - Training works in principle and error converges but needs to be tested
## Future 
- Fix pressure for NPT
- Add Kokkos to pair-style
- Maybe PJRT Api for full C++ support
