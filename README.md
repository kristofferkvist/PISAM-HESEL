# PISAM-HESEL
PISAM is a full 3D discrete particle model tailored for modeling deuterium atoms and molecules in fusion plasma.
PISAM has been coupled with the 2D turbulent edge plasma model HESEL. PISAM is implemented in Python while HESEL is implement in the C++ framework BOUT++. The two codes are coupled using MPI. 

## Installation

### Installing BOUT++
For information on BOUT++ see, http://boutproject.github.io/, where you can download the source code and get instructions for installation. For HESEL to be supported in its current implementation build BOUT++ using "./configure --options" followed by "make", not using cmake as otherwise recommendet. The current coupling of PISAM is only tested using OpenMPI with gnu compilers, which is the recommended setup.

### Compiling HESEL
To compile HESEL add the line "export BOUT_TOP=/your/BOUT/installation/path" to your bash.rc file, and run "source .bashrc".
HESEL can now be compiled using make.

### Python dependencies
The python dependencies for using PISAM-HESEL are:

numpy
scipy
wheel
pickle5
importlib-metadata
h5py
netCDF4
Pillow
mpi4py
matplotlib
setuptools_scm

If you have python3 set up with pip, you should be able to run "bash install_py_dependencies", and all the nessecary dependencies will be installed.
You might have to point directly to OpenMPI mpicc binary when installing mpi4py, see https://mpi4py.readthedocs.io/en/stable/install.html.

### Setting up PISAM
To setup PISAM for the first time run "make PISAM_setup"

### Running PISAM-HESEL
To run PISAM-HESEL use mpirun -n number_of_cores_HESEL ./hesel : -n number_of_cores_PISAM python3 python_manager.py
This assumes you have a BOUT.inp file in the folder /data. If your BOUT.inp file is located elsewhere use the option -d 'path/to/BOUT.inp/dir'. This option should be used after the hesel AND! python executables.

### Setting of PISAM-HESEL in BOUT.inp
Except for a flag called 

