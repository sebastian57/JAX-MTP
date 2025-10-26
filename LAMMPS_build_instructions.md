### For GPU on cuda system
First activate Python venv

#### General system variables
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")

#### Actual LAMMPS build
All commands in the lammps/build directory

cmake ../cmake     
	-DCMAKE_BUILD_TYPE=Release     
	-DBUILD_MPI=ON
	-DBUILD_SHARED_LIBS=ON     
	-DPKG_EXTRA-PAIR=ON
	-DPKG_PYTHON=ON
	-DPKG_MANYBODY=ON
	-DPython3_EXECUTABLE="$PYTHON_EXECUTABLE"
	-DPython3_INCLUDE_DIR="$PYTHON_INCLUDE"
	-DPython3_LIBRARY="$PYTHON_LIBRARY/libpython${PYTHON_VERSION}.so"
	-DPython3_NumPy_INCLUDE_DIR="$NUMPY_INCLUDE"
	-DCMAKE_CXX_FLAGS="-I$PYTHON_INCLUDE -I$NUMPY_INCLUDE"

make -j$(nproc)
