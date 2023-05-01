import numpy as np
from h_atom import H_atom
from h_molecule import H_molecule
from domain import Domain
import pickle
from mpi4py import MPI
from boututils.options import BOUTOptions
import netCDF4 as nc
import matplotlib.pyplot as plt

class Simulator():
    def __init__(self, data_folder, oci, rhos, rank, sub_comm, procs_python, n_init, Te_init, Ti_init, u0_x_ion_init, u0_y_ion_init, optDIR, weight):
        self.procs_python = procs_python
        self.rank = rank
        self.sub_comm = sub_comm
        self.data_folder = data_folder
        self.optDIR = optDIR
        self.rhos = rhos
        self.oci = oci
        self.n_init = n_init
        self.Te_init = Te_init
        self.Ti_init = Ti_init
        self.u0_x_ion_init = u0_x_ion_init
        self.u0_y_ion_init = u0_y_ion_init
        self.weight = weight
        self.set_phys_consts()
        self.read_input_params()
        self.set_options_for_histrograms()
        self.set_dictionaries()
        self.init_objects()
        self.initiate_recording()

    def init_objects(self):
        self.init_domain()
        self.init_atoms()
        self.init_molecules()

    def initiate_sim(self, restart, flag_initialized):
        if restart:
            self.restart()
        elif flag_initialized:
            self.initiate_from_files()
            self.domain.t = 0
        else:
            self.initiate_from_fields()
            self.domain.t = 0

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

    def initiate_recording(self):
        self.record_iter = 0
        if (self.rank == 0):
            self.hist_buff_atom_density = np.empty(self.h_atoms.bin_edges_x.size-1).astype(np.float32)
            self.hist_buff_atom_density_cx = np.empty(self.h_atoms.bin_edges_x.size-1).astype(np.float32)
            self.hist_buff_molecule_density = np.empty(self.h_molecules.bin_edges_x.size-1).astype(np.float32)
            self.dense_dat_atoms = np.zeros((self.nout, self.hist_buff_atom_density.size)).astype(np.float32)
            self.dense_dat_atoms_cx = np.zeros((self.nout, self.hist_buff_atom_density_cx.size)).astype(np.float32)
            self.dense_dat_molecules = np.zeros((self.nout, self.hist_buff_molecule_density.size)).astype(np.float32)

    def set_fields(self, n, Te, Ti, u0_x_ion, u0_y_ion):
        self.domain.n_mesh = n
        self.domain.Te_mesh = Te
        self.domain.Ti_mesh = Ti
        self.domain.u0_x_ion_mesh = u0_x_ion
        self.domain.u0_y_ion_mesh = u0_y_ion

    def get_sources_electrons(self):
        return self.domain.electron_source_particle, self.domain.electron_source_energy

    def get_sources_ions(self):
        return self.domain.ion_source_momentum_x, self.domain.ion_source_momentum_y, self.domain.ion_source_energy

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
        self.nout = int(root_dict['nout'])
        self.timestep = eval(root_dict['timestep'])
        mxg = int(root_dict['mxg'])
        self.mxg = mxg

        #Read options from mesh dictionary
        mesh_dict = myOpts.mesh
        nx = int(eval(mesh_dict['nx']))
        self.nx = nx-2*mxg
        ny = int(mesh_dict['nz'])
        self.ny = ny #The axes are switched its confusing
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
        self.domain = Domain(self.x_min, self.x_max, self.y_min, self.y_max, self.r_minor, self.r_major, self.nx, self.ny, self.step_length, self.n0)
        self.set_fields(self.n_init, self.Te_init, self.Ti_init, self.u0_x_ion_init, self.u0_y_ion_init)
        self.bin_edges_T = np.arange(np.max(self.Te_init)+1)

    def init_atoms(self):
        self.h_atoms = H_atom(int(self.H_atom_injection_rate/self.procs_python), self.H_atom_N_max, self.H_atom_temperature, self.domain, self.dict_atom, self.bin_edges_x, self.bin_edges_vs_atom, 5, self.bin_edges_T, self.absorbtion_coefficient_atom)

    def init_molecules(self):
        self.h_molecules = H_molecule(int(self.H_molecule_injection_rate/self.procs_python), self.H_molecule_N_max, self.H_molecule_temperature, self.domain, self.h_atoms, self.dict_molecule, self.bin_edges_x, self.bin_edges_vs_molecule, 5, self.bin_edges_T, self.absorbtion_coefficient_molecule)

    def run_transient(self):
        self.domain.dt = self.domain.step_length*1
        self.species_list = [self.h_atoms, self.h_molecules]
        self.h_molecules.step()
        convergence_count = 0
        max_ind_molecules = 0
        max_ind_atoms = 0
        while (convergence_count < 50):
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
        for i in np.arange(1):
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
    def record(self, t):
        if (t > (self.record_iter+1)*self.timestep):
            x_pos_atoms = self.h_atoms.x[0:self.h_atoms.max_ind][self.h_atoms.active[0:self.h_atoms.max_ind]]
            x_pos_atoms_cxed = self.h_atoms.x[0:self.h_atoms.max_ind][self.h_atoms.cxed[0:self.h_atoms.max_ind]]
            x_pos_molecules = self.h_molecules.x[0:self.h_molecules.max_ind][self.h_molecules.active[0:self.h_molecules.max_ind]]
            hist_atoms, _ = np.histogram(x_pos_atoms, bins=self.h_atoms.bin_edges_x)
            hist_atoms_cxed, _ = np.histogram(x_pos_atoms_cxed, bins=self.h_atoms.bin_edges_x)
            hist_molecules, _ =  np.histogram(x_pos_molecules, bins=self.h_molecules.bin_edges_x)
            #Diagnnostics
            if (self.rank == 0):
                self.sub_comm.Reduce([hist_atoms.astype(np.float32), MPI.FLOAT], [self.hist_buff_atom_density, MPI.FLOAT], op = MPI.SUM, root = 0)
                self.sub_comm.Reduce([hist_atoms_cxed.astype(np.float32), MPI.FLOAT], [self.hist_buff_atom_density_cx, MPI.FLOAT], op = MPI.SUM, root = 0)
                self.sub_comm.Reduce([hist_molecules.astype(np.float32), MPI.FLOAT], [self.hist_buff_molecule_density, MPI.FLOAT], op = MPI.SUM, root = 0)
                self.dense_dat_atoms[self.record_iter, :] = self.hist_buff_atom_density
                self.dense_dat_atoms_cx[self.record_iter, :] = self.hist_buff_atom_density_cx
                self.dense_dat_molecules[self.record_iter, :] = self.hist_buff_molecule_density
            else:
                self.sub_comm.Reduce([hist_atoms.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
                self.sub_comm.Reduce([hist_atoms_cxed.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
                self.sub_comm.Reduce([hist_molecules.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
            self.record_iter = self.record_iter + 1


    def save_objects(self, filenames):
        self.h_atoms.save_object_nc(filenames[0])
        self.h_molecules.save_object_nc(filenames[1])
        self.domain.save_object_nc(filenames[2])

    def load_objects(self, filenames):
        self.h_atoms.load_object_nc(filenames[0])
        self.h_molecules.load_object_nc(filenames[1])
        self.domain.load_object_nc(filenames[2])

    def finalize(self):
        dense_dat_atoms = self.weight*self.dense_dat_atoms/(self.Lz*self.domain.dx)
        dense_dat_atoms_cx = self.weight*self.dense_dat_atoms_cx/(self.Lz*self.domain.dx)
        dense_dat_molecules = self.weight*self.dense_dat_molecules/(self.Lz*self.domain.dx)
        nc_dat = nc.Dataset(self.optDIR + '/kinetic_hists.nc', 'w', 'NETCDF4')
        nc_dat.createDimension('x', self.bin_edges_x.size-1)
        nc_dat.createDimension('t', self.nout)
        n_atom_var = nc_dat.createVariable('n_atom_x_t', 'float32', ('t', 'x'))
        n_atom_var[:, :] = dense_dat_atoms
        n_atom_var_cx = nc_dat.createVariable('n_atom_cx_x_t', 'float32', ('t', 'x'))
        n_atom_var_cx[:, :] = dense_dat_atoms_cx
        n_molecule_var = nc_dat.createVariable('n_molecule_x_t', 'float32', ('t', 'x'))
        n_molecule_var[:, :] = dense_dat_molecules
        nc_dat.close()
