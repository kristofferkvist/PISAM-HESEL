import numpy as np
from h_atom import H_atom
from h_molecule import H_molecule
from domain import Domain
import pickle
from mpi4py import MPI
from boututils.options import BOUTOptions
import netCDF4 as nc

class Simulator():
    def __init__(self, data_folder, oci, rhos, rank, procs_python, n_init, Te_init, Ti_init, optDIR):
        self.procs_python = procs_python
        self.rank = rank
        self.data_folder = data_folder
        self.optDIR = optDIR
        self.rhos = rhos
        self.oci = oci
        self.n_init = n_init
        self.Te_init = Te_init
        self.Ti_init = Ti_init
        self.set_phys_consts()
        self.read_input_params()
        self.set_options_for_histrograms()
        self.set_dictionaries()
        self.init_objects()

    def init_objects(self):
        self.init_domain()
        self.init_atoms()
        self.init_molecules()

    def initiate_from_fields(self):
        self.run_transient()
        self.obtain_init_sources()
        #self.save_objects([self.data_folder + 'h_atoms_init_' + str(self.rank) + '.nc', self.data_folder + 'h_molecules_init_' + str(self.rank) + '.nc', self.data_folder + 'domain_init_' + str(self.rank) + '.nc'])
        if (self.rank == 0):
            print("########################################Max inds after convergence on rank 0:#######################################")
            print(self.h_atoms.max_ind)
            print(self.h_molecules.max_ind)
        self.finish_init()

    def initiate_from_files(self):
        self.load_objects([self.data_folder + 'h_atoms_init_' + str(self.rank) + '.nc', self.data_folder + 'h_molecules_init_' + str(self.rank) + '.nc', self.data_folder + 'domain_init_' + str(self.rank) + '.nc'])
        self.finish_init()

    def restart(self):
        self.load_objects([self.data_folder + 'h_atoms_restart_' + str(self.rank) + '.nc', self.data_folder + 'h_molecules_restart_' + str(self.rank) + '.nc', self.data_folder + 'domain_restart_' + str(self.rank) + '.nc'])
        self.finish_init()

    def finish_init(self):
        self.domain.dt = self.domain.step_length
        self.h_atoms.domain = self.domain
        self.h_molecules.domain = self.domain
        self.h_molecules.diss_product = self.h_atoms
        self.species_list = [self.h_atoms, self.h_molecules]

    def set_fields(self, n, Te, Ti):
        self.domain.n_mesh = n
        self.domain.Te_mesh = Te
        self.domain.Ti_mesh = Ti

    def get_sources_electrons(self):
        return self.domain.electron_source_particle, self.domain.electron_source_momentum_x, self.domain.electron_source_momentum_y, self.domain.electron_source_energy

    def get_sources_ions(self):
        return self.domain.ion_source_particle, self.domain.ion_source_momentum_x, self.domain.ion_source_momentum_y, self.domain.ion_source_energy

    def sim_step(self, t_plasma):
        self.domain.set_sources_zero()
        full_steps = np.floor((t_plasma - self.domain.t)/self.domain.step_length).astype(np.int32)
        t_remaining = t_plasma - self.domain.t - self.domain.step_length*full_steps
        self.domain.dt = t_remaining
        self.domain.t += self.domain.dt
        for s in self.species_list:
            s.step_save()
        self.domain.dt = self.domain.step_length
        for _ in np.arange(full_steps):
            self.domain.t += self.domain.dt
            for s in self.species_list:
                s.step()

#########################INITIALIZATION###########################
    def set_phys_consts(self):
        #Physical constants
        self.MASS_D_ATOM = 2.01410177811*1.660539e-27
        self.MASS_D_MOLECULE = 2*2.014102*1.660539e-27
        self.MASS_ELECTRON = 9.1093837e-31
        self.EV = 1.60217663e-19

    def read_input_params(self):
        #Read parameters
        myOpts = BOUTOptions(self.optDIR)
        #Read options from root dictionary
        root_dict = myOpts.root
        mxg = int(root_dict['mxg'])
        self.mxg = mxg

        #Read options from mesh dictionary
        mesh_dict = myOpts.mesh
        nx = int(eval(mesh_dict['nx']))
        self.nx = nx-2*mxg
        nz = int(mesh_dict['nz'])
        self.ny = nz #The axes are switched its confusing
        Lx = float(mesh_dict['Lx'])
        self.Lx = Lx*self.rhos
        Lz = float(mesh_dict['Lx'])
        self.Lz = Lz*self.rhos

        self.x_min = 0
        self.x_max = self.Lx
        self.y_min = 0  #The axes are switched its confusing
        self.y_max = self.Lz  #The axes are switched its confusing

        #Read options from mesh dictionary
        hesel_dict = myOpts.hesel
        self.r_minor = float(eval(hesel_dict['Rminor']))
        self.r_major = float(eval(hesel_dict['Rmajor']))
        self.n0 = float(eval(hesel_dict['n0']))

        #Read options from kinetic neutrals dictionary
        kinetic_neutral_dict = myOpts.kinetic_neutrals
        self.step_length = float(eval(kinetic_neutral_dict['step_length']))/self.oci
        self.init_source_time = float(eval(kinetic_neutral_dict['init_source_time']))/self.oci
        self.H_atom_injection_rate = float(eval(kinetic_neutral_dict['H_atom_injection_rate']))
        self.H_atom_N_max = int(eval(kinetic_neutral_dict['H_atom_N_max']))
        self.H_molecule_temperature = float(eval(kinetic_neutral_dict['H_molecule_temperature']))
        self.H_atom_temperature = float(eval(kinetic_neutral_dict['H_atom_temperature']))
        self.H_molecule_injection_rate = int(eval(kinetic_neutral_dict['H_molecule_injection_rate']))
        self.H_molecule_N_max = int(eval(kinetic_neutral_dict['H_molecule_N_max']))
        self.absorbtion_coefficient_atom = float(eval(kinetic_neutral_dict['wallAbsorbtionFractionAtom']))
        self.absorbtion_coefficient_molecule = float(eval(kinetic_neutral_dict['wallAbsorbtionFractionMolecule']))

    def set_options_for_histrograms(self):
        n_bins = 50
        self.bin_edges_x = np.linspace(self.x_min, self.x_max, self.nx+1)
        self.bin_edges_vs_atom = np.linspace(0, 2.5*np.sqrt(8*100*1.602e-19/(np.pi*self.MASS_D_ATOM)), n_bins)
        self.bin_edges_vs_molecule = np.linspace(0, 2.5*np.sqrt(8*self.H_molecule_temperature*1.602e-19/(np.pi*self.MASS_D_MOLECULE)), n_bins)

    def set_dictionaries(self):
        with open(self.data_folder + 'h_atom_dict' + '.pkl', 'rb') as f:
            self.dict_atom = pickle.load(f)
        with open(self.data_folder + 'h_molecule_dict' + '.pkl', 'rb') as f:
            self.dict_molecule = pickle.load(f)

    def init_domain(self):
        self.domain = Domain(self.x_min, self.x_max, self.y_min, self.y_max, self.r_minor, self.r_major, self.nx, self.ny, self.step_length, self.n_init, self.Te_init, self.Ti_init, self.n0)
        self.bin_edges_T = np.arange(np.max(self.Te_init)+1)

    def init_atoms(self):
        self.h_atoms = H_atom(int(self.H_atom_injection_rate/self.procs_python), self.H_atom_N_max, self.H_atom_temperature, self.domain, self.dict_atom, self.bin_edges_x, self.bin_edges_vs_atom, 5, self.bin_edges_T, self.absorbtion_coefficient_atom)

    def init_molecules(self):
        self.h_molecules = H_molecule(int(self.H_molecule_injection_rate/self.procs_python), self.H_molecule_N_max, self.H_molecule_temperature, self.domain, self.h_atoms, self.dict_molecule, self.bin_edges_x, self.bin_edges_vs_molecule, 5, self.bin_edges_T, self.absorbtion_coefficient_molecule)

    """
    def run_transient(self):
        self.domain.dt = self.domain.step_length*100
        self.species_list = [self.h_atoms, self.h_molecules]
        self.h_molecules.step()
        for _ in np.arange(100):
            for s in self.species_list:
                s.step()
    """

    def run_transient(self):
        self.domain.dt = self.domain.step_length*100
        self.species_list = [self.h_atoms, self.h_molecules]
        self.h_molecules.step()
        convergence_count = 0
        max_ind_molecules = 0
        max_ind_atoms = 0
        while (convergence_count < 10):
            max_ind_molecules = self.h_molecules.max_ind
            max_ind_atoms = self.h_atoms.max_ind
            for s in self.species_list:
                s.step()
            if (max_ind_molecules == self.h_molecules.max_ind and max_ind_atoms == self.h_atoms.max_ind):
                convergence_count = convergence_count + 1
            else:
                convergence_count = 0
            #print("Max_ind_atoms = " + str(self.h_atoms.max_ind))
            #print("Max_ind_molecules = " + str(self.h_molecules.max_ind))
        self.domain.dt = self.domain.step_length
        for i in np.arange(10):
            for s in self.species_list:
                s.step()
            #print("Step " + str(i) + " of 1000")

    def obtain_init_sources(self):
        self.domain.set_sources_zero()
        full_steps = np.floor(self.init_source_time/self.domain.step_length).astype(np.int32)
        t_remaining = self.init_source_time - self.domain.step_length*full_steps
        self.domain.dt = t_remaining
        for s in self.species_list:
            s.step_save()
        self.domain.dt = self.domain.step_length
        for _ in np.arange(full_steps):
            for s in self.species_list:
                s.step()

#############################################IO OPERATIONS######################
    def save_objects(self, filenames):
        self.h_atoms.save_object_nc(filenames[0])
        self.h_molecules.save_object_nc(filenames[1])
        self.domain.save_object_nc(filenames[2])

    def load_objects(self, filenames):
        self.h_atoms.load_object_nc(filenames[0])
        self.h_molecules.load_object_nc(filenames[1])
        self.domain.load_object_nc(filenames[2])

    def finalize(self, output_datafolder, last_plasma_timestep_length):
        self.domain.last_plasma_timestep_length = last_plasma_timestep_length
        #self.h_atoms.save_hist_results(output_datafolder, self.rank)
        #self.h_molecules.save_hist_results(output_datafolder, self.rank)
        self.save_objects([self.data_folder + 'h_atoms_restart_' + str(self.rank) + '.nc', self.data_folder + 'h_molecules_restart_' + str(self.rank) + '.nc', self.data_folder + 'domain_restart_' + str(self.rank) + '.nc'])
