# motep_lmp_jax
Python implementation of the Moment Tensor Potential (machine learning potential for atomistic simulations), training code and C++ LAMMPS interface.


## General Description
The current state of the repository includes training code for the python MTP and a complete LAMMPS integration using a custom pair-style. <br>
The training can be tested by running the "run_timing.sh" script. One can test all the .cfg files that are given inside the "training_data" directory and can choose between the normal function and a mini batch version. After training for the chose levels and recording the loss and timing data, one can create a comprehensive report by simply adjusting some variables and the running "run_analysis.sh". The complete output is the found in the training_results directory. Inside the "motep_original_files" directory lies the MTP formulation that is being run for the training, the main function being imported from "jax_engine/jax.py". <br>
Inside the "LAMMPS_interface" directory, all the files that are needed for setting up a LAMMPS simulation with the newest version of the python MTP are found. All the .cpp, .h or .hpp files must be placed inside the lammps/src/EXTRA-PAIR directory of the local LAMMPS build. Then one can build LAMMPS using the system-dependant commands from the "lammps_build_commands.txt" file. After building LAMMPS one still needs to compile the core function into a serialized .bin file. This is automated inside the "compile_ultra_optimized_cuda.py" (cuda version!) file. It calls the "motep_original_files/jax_engine/jax_comp.py" file and creates several .bin files, according to the specifications inside the compilation script. These can then simply be called inside of a LAMMPS script, as shown in "lammps_jaxmtp.in". <br>


## Necessary Pacakges
Jax (built for the specific system), Optax, ASE, abls_py, flatbuffers, Pillow. (Rest should be automatically installed with these main packages). <br>

## ToDo
- The functions used in the training and for LAMMPS currently differ. This is because, the training code developement was put on hold after initial successes. The new "jax_comp.py" file contains a much more highly optimized (both for computational speed and memory layout) version. Soon these will however be integrated, so that the training code also calls the newer version. <br>
- Make sure that the neighbor list computation runs on GPU. Changes need to be made to the .cpp pair-style files. <br>
- Add testing scripts that allow for direct comparison between jax-mtp and mlip2, both for single calculations aswell as longer LAMMPS runs. <br>
- Remove python calls entirely from the pair-style files. Currently it is still briefly being initialized to call the exported .bin file. Want to use the XLA C++ API to run completely in C++. <br>
- Work on a solution for the unique elements problem that the current implementation faces. The number of unique tensor elements rises sub-linearly with increasing level. This leads to a lot of unnecessary memory use for large MTPs. However Jax is very good at handling large dense tensors, so the solution must be very efficient. <br>
