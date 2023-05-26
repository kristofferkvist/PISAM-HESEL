"""
The simulator class is responsible the PISAM part of the simulation across all
the ranks of the python part. One simulator object is created per rank used for
PISAM to conduct a simulation.

Its main responsibilities are:

*Initiate PISAM by running it with the initial plasma fields until the domain is saturated with neutrals.

*Call the all important step function of each species with the appropriate step length

*Monitor and extract the neutral data i.e. density, mean free path, velocity dist etc.
"""

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
        self.get_dictionaries()
        self.init_objects()
        self.initiate_recording()

    #Wrapper for initialization of the domain and species objects
    def init_objects(self):
        self.init_domain()
        self.init_atoms()
        self.init_molecules()

    #Apply the chosen initialization procedure
    def initiate_sim(self, restart, flag_initialized):
        if restart:
            self.restart()
        elif flag_initialized:
            self.initiate_from_files()
            self.domain.t = 0
        else:
            self.initiate_from_fields()
            self.domain.t = 0

    #Default for initialization. No prior runs demanded
    def initiate_from_fields(self):
        self.run_transient()
        self.obtain_init_sources()
        #self.save_objects([self.data_folder + 'h_atoms_init_' + str(self.rank) + '.nc', self.data_folder + 'h_molecules_init_' + str(self.rank) + '.nc', self.data_folder + 'domain_init_' + str(self.rank) + '.nc'])
        if (self.rank == 0):
            print("########################################Max inds after convergence on rank 0:#######################################")
            print(self.h_atoms.max_ind)
            print(self.h_molecules.max_ind)
        self.finish_init()

    #Load the result of an initialization
    def initiate_from_files(self):
        self.load_objects([self.data_folder + 'h_atoms_init_' + str(self.rank) + '.nc', self.data_folder + 'h_molecules_init_' + str(self.rank) + '.nc', self.data_folder + 'domain_init_' + str(self.rank) + '.nc'])
        self.finish_init()

    #Load the end state of neutrals from a prior PISAM-HESEL sim
    def restart(self):
        self.load_objects([self.data_folder + 'h_atoms_restart_' + str(self.rank) + '.nc', self.data_folder + 'h_molecules_restart_' + str(self.rank) + '.nc', self.data_folder + 'domain_restart_' + str(self.rank) + '.nc'])
        self.finish_init()

    #Set the appropriate step length in domain and the relations between the objects.
    def finish_init(self):
        self.domain.dt = self.domain.step_length
        self.h_atoms.domain = self.domain
        self.h_molecules.domain = self.domain
        self.h_molecules.diss_product = self.h_atoms
        self.species_list = [self.h_atoms, self.h_molecules]

    #Initiate the recording by creating buffers for reduction between the parallel processes
    #and creating a netCDF-file with the parameters you want to save.
    def initiate_recording(self):
        self.init_speed_dists()
        self.record_iter = -1
        self.n_atoms = np.zeros((self.domain.plasma_dim_x, self.domain.plasma_dim_y)).astype(np.float32)
        self.n_atoms_cxed = np.zeros((self.domain.plasma_dim_x, self.domain.plasma_dim_y)).astype(np.float32)
        self.n_molecules = np.zeros((self.domain.plasma_dim_x, self.domain.plasma_dim_y)).astype(np.float32)
        if (self.rank == 0):
            self.buff_atom_density = np.zeros((self.domain.plasma_dim_x, self.domain.plasma_dim_y)).astype(np.float32)
            self.buff_atom_density_cx = np.zeros((self.domain.plasma_dim_x, self.domain.plasma_dim_y)).astype(np.float32)
            self.buff_molecule_density = np.zeros((self.domain.plasma_dim_x, self.domain.plasma_dim_y)).astype(np.float32)
            self.buff_atom_density_diff = np.zeros(self.domain.plasma_dim_x).astype(np.float32)
            self.buff_atom_cx_density_diff = np.zeros(self.domain.plasma_dim_x).astype(np.float32)
            self.buff_molecule_density_diff = np.zeros(self.domain.plasma_dim_x).astype(np.float32)
            self.buff_atom_source = np.zeros(self.domain.plasma_dim_x).astype(np.float32)
            self.buff_atom_cx_source = np.zeros(self.domain.plasma_dim_x).astype(np.float32)
            self.buff_molecule_source = np.zeros(self.domain.plasma_dim_x).astype(np.float32)
            self.buff_speed_dist_atom = np.zeros((self.velocity_domains, self.n_bins_atom)).astype(np.float32)
            self.buff_speed_dist_atom_cx = np.zeros((self.velocity_domains, self.n_bins_atom)).astype(np.float32)
            self.buff_speed_dist_mol = np.zeros((self.velocity_domains, self.n_bins_mol)).astype(np.float32)

            self.n_atoms_x_before = np.zeros(self.domain.plasma_dim_x).astype(np.float32)
            self.n_atoms_cxed_x_before = np.zeros(self.domain.plasma_dim_x).astype(np.float32)
            self.n_molecules_x_before = np.zeros(self.domain.plasma_dim_x).astype(np.float32)

            self.nc_dat = nc.Dataset(self.optDIR + '/neutral_diagnostics.nc', 'w', 'NETCDF4')
            self.nc_dat.createDimension('x', self.domain.plasma_dim_x)
            self.nc_dat.createDimension('y', self.domain.plasma_dim_y)
            self.nc_dat.createDimension('t', self.nout+1)
            self.nc_dat.createDimension('scalar', 1)
            self.nc_dat.createDimension('v_doms', self.velocity_domains)
            self.nc_dat.createDimension('n_bins_atom', self.n_bins_atom)
            self.nc_dat.createDimension('n_bins_mol', self.n_bins_mol)
            self.nc_dat.createDimension('bin_edges_atom', self.bin_edges_atom.size)
            self.nc_dat.createDimension('bin_edges_mol', self.bin_edges_mol.size)

            self.molecule_injection_rate_var = self.nc_dat.createVariable('molecule_injection_rate', 'float32', ('scalar', ))
            self.molecule_injection_rate_var[0] = np.array(self.H_molecule_injection_rate).astype(np.float32)
            self.weight_var = self.nc_dat.createVariable('weight', 'float32', ('scalar', ))
            self.weight_var[0] = np.array(self.weight).astype(np.float32)
            self.Ly_var = self.nc_dat.createVariable('Ly', 'float32', ('scalar', ))
            self.Ly_var[0] = np.array(self.Ly).astype(np.float32)
            #E dist data
            self.bin_edges_atom_var = self.nc_dat.createVariable('bin_edges_atom', 'float32', ('bin_edges_atom', ))
            self.bin_edges_atom_var[:] = self.bin_edges_atom.astype(np.float32)
            self.bin_edges_mol_var = self.nc_dat.createVariable('bin_edges_mol', 'float32', ('bin_edges_mol', ))
            self.bin_edges_mol_var[:] = self.bin_edges_mol.astype(np.float32)
            self.speed_dist_atom_var = self.nc_dat.createVariable('speed_dist_atom', 'float32', ('t', 'v_doms', 'n_bins_atom'))
            self.speed_dist_atom_cx_var = self.nc_dat.createVariable('speed_dist_atom_cx', 'float32', ('t', 'v_doms', 'n_bins_atom'))
            self.speed_dist_mol_var = self.nc_dat.createVariable('speed_dist_mol', 'float32', ('t', 'v_doms', 'n_bins_mol'))

            self.step_length_var = self.nc_dat.createVariable('dt', 'float32', ('t', ))
            self.n_atom_var = self.nc_dat.createVariable('n_atom', 'float32', ('t', 'x', 'y'))
            self.n_atom_cx_var = self.nc_dat.createVariable('n_atom_cx', 'float32', ('t', 'x', 'y'))
            self.n_molecule_var = self.nc_dat.createVariable('n_molecule', 'float32', ('t', 'x', 'y'))

            """self.n_atom_diff_var = self.nc_dat.createVariable('n_atom_x_diff', 'float32', ('t', 'x'))
            self.n_atom_cx_diff_var = self.nc_dat.createVariable('n_atom_cx_x_diff', 'float32', ('t', 'x'))
            self.n_molecule_diff_var = self.nc_dat.createVariable('n_molecule_x_diff', 'float32', ('t', 'x'))
            self.n_atom_source_var = self.nc_dat.createVariable('n_atom_x_source', 'float32', ('t', 'x'))
            self.n_atom_cx_source_var = self.nc_dat.createVariable('n_atom_cx_x_source', 'float32', ('t', 'x'))
            self.n_molecule_source_var = self.nc_dat.createVariable('n_molecule_x_source', 'float32', ('t', 'x'))"""
        else:
            self.buff_atom_density =           None
            self.buff_atom_density_cx =        None
            self.buff_molecule_density =       None
            self.buff_atom_density_diff =      None
            self.buff_atom_cx_density_diff =   None
            self.buff_molecule_density_diff =  None
            self.buff_atom_source =            None
            self.buff_atom_cx_source =         None
            self.buff_molecule_source =        None
            self.buff_speed_dist_atom =        None
            self.buff_speed_dist_atom_cx =     None
            self.buff_speed_dist_mol =         None

    #Set the plasma fields seen by the neutrals of PISAM
    def set_fields(self, n, Te, Ti, u0_x_ion, u0_y_ion):
        self.domain.n_mesh = n
        self.domain.Te_mesh = Te
        self.domain.Ti_mesh = Ti
        self.domain.u0_x_ion_mesh = u0_x_ion
        self.domain.u0_y_ion_mesh = u0_y_ion

    #Returns electron sources
    def get_sources_electrons(self):
        return self.domain.electron_source_particle, self.domain.electron_source_energy

    #Returns ion sources
    def get_sources_ions(self):
        return self.domain.ion_source_momentum_x, self.domain.ion_source_momentum_y, self.domain.ion_source_energy

    #Steps PISAM through one whole neutral step as seen from HESEL.
    def sim_step(self, t_plasma):
        self.domain.set_sources_zero()
        #######Diagnostics#########
        #self.h_molecules.Sn_this_step.fill(0)
        #self.h_atoms.Sn_this_step.fill(0)
        #self.h_atoms.Sn_this_step_cx.fill(0)
        #######Diagnostics#########
        #Calculate the amount of steps that can be conducted with the current step_length
        #without overshooting the current time in HESEL.
        full_steps = np.floor((t_plasma - self.domain.t)/self.domain.step_length).astype(np.int32)
        #Calculate what remains after this number of full steps.
        t_remaining = t_plasma - self.domain.t - self.domain.step_length*full_steps
        #If no full steps can be taken, which only happens in the very last simulation step
        #set the step length to be t_remaining. Else take one "long step" which is
        #step_length + t_remaining. This is justified since t_remaining is usually very short.
        if full_steps == 0:
            self.domain.dt = t_remaining
        else:
            self.domain.dt = t_remaining + self.domain.step_length
        self.domain.t += self.domain.dt
        for s in self.species_list:
            s.step()
        #Do the remaining full step if any - this will usually not be the case.
        if full_steps > 1:
            self.domain.dt = self.domain.step_length
            for _ in np.arange(full_steps-1):
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

    #Read the parameters of interest from the input file BOUT.inp
    def read_input_params(self):
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
        Ly = float(mesh_dict['Lz'])
        self.Ly = Ly*self.rhos

        self.x_min = 0
        self.x_max = self.Lx
        self.y_min = -self.Ly/2  #Using standard right handed coord. system in PISAM
        self.y_max = self.Ly/2   #Using standard right handed coord. system in PISAM

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
        self.H_molecule_injection_rate = float(eval(kinetic_neutral_dict['H_molecule_injection_rate']))
        self.H_molecule_N_max = int(eval(kinetic_neutral_dict['H_molecule_N_max']))
        self.absorbtion_coefficient_atom = float(eval(kinetic_neutral_dict['wallAbsorbtionFractionAtom']))
        self.absorbtion_coefficient_molecule = float(eval(kinetic_neutral_dict['wallAbsorbtionFractionMolecule']))
        self.T_wall = float(eval(kinetic_neutral_dict['T_wall']))
        self.wall_boundary_atom = bool(eval(kinetic_neutral_dict['wall_boundary_atom']))
        self.wall_boundary_molecule = bool(eval(kinetic_neutral_dict['wall_boundary_molecule']))
        self.min_weight_atom = float(eval(kinetic_neutral_dict['min_weight_atom']))

    #Load ther dictionaries with all the tables holding reaction rates and various distributions
    def get_dictionaries(self):
        with open(self.data_folder + 'h_atom_dict' + '.pkl', 'rb') as f:
            self.dict_atom = pickle.load(f)
        with open(self.data_folder + 'h_molecule_dict' + '.pkl', 'rb') as f:
            self.dict_molecule = pickle.load(f)

    #Initialize the domain and set the initial plasma fields.
    def init_domain(self):
        self.domain = Domain(self.x_min, self.x_max, self.y_min, self.y_max, self.r_minor, self.r_major, self.nx, self.ny, self.step_length, self.n0, self.T_wall)
        self.set_fields(self.n_init, self.Te_init, self.Ti_init, self.u0_x_ion_init, self.u0_y_ion_init)
        self.bin_edges_T = np.arange(np.max(self.Te_init)+1)

    def init_atoms(self):
        self.h_atoms = H_atom(self.H_atom_injection_rate/self.procs_python, self.H_atom_N_max, self.H_atom_temperature, self.domain, self.dict_atom, self.absorbtion_coefficient_atom, self.min_weight_atom, self.wall_boundary_atom)

    def init_molecules(self):
        self.h_molecules = H_molecule(self.H_molecule_injection_rate/self.procs_python, self.H_molecule_N_max, self.H_molecule_temperature, self.domain, self.h_atoms, self.dict_molecule, self.absorbtion_coefficient_molecule, self.wall_boundary_molecule)

    #Run the transient until the chosen convergence criterion.
    #Various approaches can be chosen by adjusting the long and short
    #time step and the iteration parameters (convergence count).
    def run_transient(self):
        #Convergence parameters - just play with them and ajust after your needs.
        self.transient_step_lengths = np.array([50, 10, 1])
        self.convergence_thresholds = np.array([0.05, 0.01, 0.01])
        self.convergence_max_counts = np.array([3, 5, 10])
        loop_size = 100
        self.species_list = [self.h_molecules, self.h_atoms]
        for i in np.arange(self.transient_step_lengths.size):
            """if self.rank == 0:
                print("Convergence loop = " + str(i))"""
            self.domain.dt = self.domain.step_length*self.transient_step_lengths[i]
            convergence_count = 0
            while (convergence_count < self.convergence_max_counts[i]):
                N_molecules_old = np.sum(self.h_molecules.norm_weight[0:self.h_molecules.max_ind][self.h_molecules.active[0:self.h_molecules.max_ind]])
                N_atoms_old = np.sum(self.h_atoms.norm_weight[0:self.h_atoms.max_ind][self.h_atoms.active[0:self.h_atoms.max_ind]])
                for _ in np.arange(loop_size):
                    self.domain.t = self.domain.t + self.domain.dt
                    for s in self.species_list:
                        s.step()
                N_molecules_new = np.sum(self.h_molecules.norm_weight[0:self.h_molecules.max_ind][self.h_molecules.active[0:self.h_molecules.max_ind]])
                N_atoms_new = np.sum(self.h_atoms.norm_weight[0:self.h_atoms.max_ind][self.h_atoms.active[0:self.h_atoms.max_ind]])
                """if self.rank == 0:
                    print("N molecules = " + str(N_molecules_new))
                    print("N atoms = " + str(N_atoms_new))"""
                if (N_molecules_old > 0 and N_atoms_old > 0) and ((np.abs(N_molecules_old-N_molecules_new)/N_molecules_old < self.convergence_thresholds[i]) and (np.abs(N_atoms_old-N_atoms_new)/N_atoms_old < 4*self.convergence_thresholds[i])):
                    convergence_count = convergence_count + 1
                else:
                    convergence_count = 0


    #After saturation a timestep is run to obtain the initial sources to send to HESEL
    #Which can then begin its work.
    def obtain_init_sources(self):
        self.domain.set_sources_zero()
        full_steps = np.floor(self.init_source_time/self.domain.step_length).astype(np.int32)
        t_remaining = self.init_source_time - self.domain.step_length*full_steps
        self.domain.dt = t_remaining
        for s in self.species_list:
            s.step()
        self.domain.dt = self.domain.step_length
        for _ in np.arange(full_steps):
            for s in self.species_list:
                s.step()

    #Calculate the number density from the positions of the neutrals.
    #np.add.at allows for dublicates in the coordinates.
    def get_density(self, species, active, output):
        mi = species.max_ind
        output.fill(0)
        x_inds = species.plasma_inds_x[0:mi][active[0:mi]]
        y_inds = species.plasma_inds_y[0:mi][active[0:mi]]
        np.add.at(output, (x_inds, y_inds), species.norm_weight[0:mi][active[0:mi]])

#############################################IO OPERATIONS######################
    #Records the neutral densities at the beginning af a PISAM iteration i.e. before stepping.
    def record_before_step(self):
        self.get_density(self.h_atoms, self.h_atoms.active, self.n_atoms)
        self.get_density(self.h_atoms, self.h_atoms.cxed, self.n_atoms_cxed)
        self.get_density(self.h_molecules, self.h_molecules.active, self.n_molecules)

        self.get_speed_dist_atom()
        self.get_speed_dist_mol()

        self.sub_comm.Reduce([self.n_atoms.astype(np.float32), MPI.FLOAT], [self.buff_atom_density, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([self.n_atoms_cxed.astype(np.float32), MPI.FLOAT], [self.buff_atom_density_cx, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([self.n_molecules.astype(np.float32), MPI.FLOAT], [self.buff_molecule_density, MPI.FLOAT], op = MPI.SUM, root = 0)

        self.sub_comm.Reduce([self.speed_dist_atom.astype(np.float32), MPI.FLOAT], [self.buff_speed_dist_atom, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([self.speed_dist_atom_cx.astype(np.float32), MPI.FLOAT], [self.buff_speed_dist_atom_cx, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([self.speed_dist_mol.astype(np.float32), MPI.FLOAT], [self.buff_speed_dist_mol, MPI.FLOAT], op = MPI.SUM, root = 0)
        if (self.rank == 0):
            """self.n_atoms_x_before = np.sum(self.buff_atom_density, axis = 1)
            self.n_atoms_cxed_x_before = np.sum(self.buff_atom_density_cx, axis = 1)
            self.n_molecules_x_before = np.sum(self.buff_molecule_density, axis = 1)"""

            self.n_atom_var[self.record_iter+1, :, :] = self.weight*self.buff_atom_density/(self.domain.dx*self.domain.dy)
            self.n_atom_cx_var[self.record_iter+1, :, :] = self.weight*self.buff_atom_density_cx/(self.domain.dx*self.domain.dy)
            self.n_molecule_var[self.record_iter+1, :, :] = self.weight*self.buff_molecule_density/(self.domain.dx*self.domain.dy)

            self.speed_dist_atom_var[self.record_iter+1, :, :] = self.buff_speed_dist_atom
            self.speed_dist_atom_cx_var[self.record_iter+1, :, :] = self.buff_speed_dist_atom_cx
            self.speed_dist_mol_var[self.record_iter+1, :, :] = self.buff_speed_dist_mol

        self.n_atoms.fill(0)
        self.n_atoms_cxed.fill(0)
        self.n_molecules.fill(0)

    #Record densities after along with the losses and gains due to particle creation and removal (includes edge losses).
    #This allows us to calculate the particle flux.
    def record_after_step(self, step_length):
        """self.get_density(self.h_atoms, self.h_atoms.active, self.n_atoms)
        self.get_density(self.h_atoms, self.h_atoms.cxed, self.n_atoms_cxed)
        self.get_density(self.h_molecules, self.h_molecules.active, self.n_molecules)"""
        if self.rank == 0:
            """self.sub_comm.Reduce([self.n_atoms.astype(np.float32), MPI.FLOAT], [self.buff_atom_density, MPI.FLOAT], op = MPI.SUM, root = 0)
            self.sub_comm.Reduce([self.n_atoms_cxed.astype(np.float32), MPI.FLOAT], [self.buff_atom_density_cx, MPI.FLOAT], op = MPI.SUM, root = 0)
            self.sub_comm.Reduce([self.n_molecules.astype(np.float32), MPI.FLOAT], [self.buff_molecule_density, MPI.FLOAT], op = MPI.SUM, root = 0)

            self.sub_comm.Reduce([self.h_atoms.Sn_this_step.astype(np.float32), MPI.FLOAT], [self.buff_atom_source, MPI.FLOAT], op = MPI.SUM, root = 0)
            self.sub_comm.Reduce([self.h_atoms.Sn_this_step_cx.astype(np.float32), MPI.FLOAT], [self.buff_atom_cx_source, MPI.FLOAT], op = MPI.SUM, root = 0)
            self.sub_comm.Reduce([self.h_molecules.Sn_this_step.astype(np.float32), MPI.FLOAT], [self.buff_molecule_source, MPI.FLOAT], op = MPI.SUM, root = 0)

            self.n_atom_diff_var[self.record_iter+1, :] = np.sum(self.buff_atom_density, axis = 1) - self.n_atoms_x_before
            self.n_atom_cx_diff_var[self.record_iter+1, :] = np.sum(self.buff_atom_density_cx, axis = 1) - self.n_atoms_cxed_x_before
            self.n_molecule_diff_var[self.record_iter+1, :] = np.sum(self.buff_molecule_density, axis = 1) - self.n_molecules_x_before
            self.n_atom_source_var[self.record_iter+1, :] = self.buff_atom_source
            self.n_atom_cx_source_var[self.record_iter+1, :] = self.buff_atom_cx_source
            self.n_molecule_source_var[self.record_iter+1, :] = self.buff_molecule_source"""
            self.step_length_var[self.record_iter+1] = step_length
        """else:
            self.sub_comm.Reduce([self.n_atoms.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
            self.sub_comm.Reduce([self.n_atoms_cxed.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
            self.sub_comm.Reduce([self.n_molecules.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)

            self.sub_comm.Reduce([self.h_atoms.Sn_this_step.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
            self.sub_comm.Reduce([self.h_atoms.Sn_this_step_cx.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
            self.sub_comm.Reduce([self.h_molecules.Sn_this_step.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.n_atoms.fill(0)
        self.n_atoms_cxed.fill(0)
        self.n_molecules.fill(0)"""

    def init_speed_dists(self):
        self.velocity_domains = 10
        self.n_bins_atom = 2000
        self.n_bins_mol = 200
        self.bin_edges_atom = np.linspace(0, np.sqrt(2*350*self.domain.EV/(self.domain.d_ion_mass)), self.n_bins_atom+1)
        self.bin_edges_mol = np.linspace(0, np.sqrt(2*15*self.domain.EV/(self.domain.d_molecule_mass)), self.n_bins_mol+1)
        self.speed_dist_atom = np.zeros((self.velocity_domains, self.n_bins_atom))
        self.speed_dist_atom_cx = np.zeros((self.velocity_domains, self.n_bins_atom))
        self.speed_dist_mol = np.zeros((self.velocity_domains, self.n_bins_mol))

    def get_speed_dist_atom(self):
        s = self.h_atoms
        for i in np.arange(self.velocity_domains):
            larger_mask = s.x[0:s.max_ind] > (self.domain.x_min + i*(self.domain.x_max-self.domain.x_min)/self.velocity_domains)
            smaller_mask = s.x[0:s.max_ind] < (self.domain.x_min + (i+1)*(self.domain.x_max-self.domain.x_min)/self.velocity_domains)
            mask = (larger_mask) & (smaller_mask) & (s.active[0:s.max_ind])
            hist, _ = np.histogram(np.sqrt(2*s.E[0:s.max_ind][mask]*self.domain.EV/(s.mass)), bins=self.bin_edges_atom, weights=s.norm_weight[0:s.max_ind][mask])
            self.speed_dist_atom[i, :] = hist
            mask = (mask) & (s.cxed[0:s.max_ind] == 1)
            hist_cx, _ = np.histogram(np.sqrt(2*s.E[0:s.max_ind][mask]*self.domain.EV/(s.mass)), bins=self.bin_edges_atom, weights=s.norm_weight[0:s.max_ind][mask])
            self.speed_dist_atom_cx[i, :] = hist_cx

    def get_speed_dist_mol(self):
        s = self.h_molecules
        for i in np.arange(self.velocity_domains):
            larger_mask = s.x[0:s.max_ind] > (self.domain.x_min + i*(self.domain.x_max-self.domain.x_min)/self.velocity_domains)
            smaller_mask = s.x[0:s.max_ind] < (self.domain.x_min + (i+1)*(self.domain.x_max-self.domain.x_min)/self.velocity_domains)
            mask = (larger_mask) & (smaller_mask) & (s.active[0:s.max_ind])
            hist, _ = np.histogram(np.sqrt(2*s.E[0:s.max_ind][mask]*self.domain.EV/(s.mass)), bins=self.bin_edges_mol, weights=s.norm_weight[0:s.max_ind][mask])
            self.speed_dist_mol[i, :] = hist



    #Loading and saving object for IO based initialization procedures
    def save_objects(self, filenames):
        self.h_atoms.save_object_nc(filenames[0])
        self.h_molecules.save_object_nc(filenames[1])
        self.domain.save_object_nc(filenames[2])

    def load_objects(self, filenames):
        self.h_atoms.load_object_nc(filenames[0])
        self.h_molecules.load_object_nc(filenames[1])
        self.domain.load_object_nc(filenames[2])

    #Print the accounting of particles and energy to check the conservation. Close the netCDF file
    def finalize(self):
        if self.rank == 0:
            self.buf_reflection_loss = np.array([0]).astype(np.float32)
            self.buf_inner_edge_loss = np.array([0]).astype(np.float32)
            self.buf_reflection_loss_mol = np.array([0]).astype(np.float32)
            self.buf_inner_edge_loss_mol = np.array([0]).astype(np.float32)
            self.buf_total_source = np.array([0]).astype(np.float32)
            self.buf_inflow = np.array([0]).astype(np.float32)
            self.buf_weight_atoms = np.array(0).astype(np.float32)
            self.buf_weight_molecules = np.array(0).astype(np.float32)
            self.buf_E_init = np.array(0).astype(np.float32)
            self.buf_E_absorb = np.array(0).astype(np.float32)
            self.buf_E_IE = np.array(0).astype(np.float32)
            self.buf_E_loss_ion = np.array(0).astype(np.float32)
            self.buf_E_gain_cx = np.array(0).astype(np.float32)
            self.buf_E_in_system = np.array(0).astype(np.float32)
        else:
            self.buf_reflection_loss = None
            self.buf_inner_edge_loss = None
            self.buf_reflection_loss_mol = None
            self.buf_inner_edge_loss_mol = None
            self.buf_total_source = None
            self.buf_inflow = None
            self.buf_weight_atoms = None
            self.buf_weight_molecules = None
            self.buf_E_init = None
            self.buf_E_absorb = None
            self.buf_E_IE = None
            self.buf_E_loss_ion = None
            self.buf_E_gain_cx = None
            self.buf_E_in_system = None

        self.sub_comm.Reduce([np.array([self.h_atoms.reflection_loss]).astype(np.float32), MPI.FLOAT], [self.buf_reflection_loss, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array([self.h_atoms.inner_edge_loss]).astype(np.float32), MPI.FLOAT], [self.buf_inner_edge_loss, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array([self.h_molecules.reflection_loss]).astype(np.float32), MPI.FLOAT], [self.buf_reflection_loss_mol, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array([self.h_molecules.inner_edge_loss]).astype(np.float32), MPI.FLOAT], [self.buf_inner_edge_loss_mol, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array([self.domain.total_plasma_source]).astype(np.float32), MPI.FLOAT], [self.buf_total_source, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array([self.h_molecules.total_inflow]).astype(np.float32), MPI.FLOAT], [self.buf_inflow, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.sum(self.h_atoms.norm_weight[0:self.h_atoms.max_ind][self.h_atoms.active[0:self.h_atoms.max_ind]]).astype(np.float32), MPI.FLOAT], [self.buf_weight_atoms, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.sum(self.h_molecules.norm_weight[0:self.h_molecules.max_ind][self.h_molecules.active[0:self.h_molecules.max_ind]]).astype(np.float32), MPI.FLOAT], [self.buf_weight_molecules, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array(self.h_atoms.E_init).astype(np.float32), MPI.FLOAT], [self.buf_E_init, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array(self.h_atoms.E_absorb).astype(np.float32), MPI.FLOAT], [self.buf_E_absorb, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array(self.h_atoms.E_loss_IE).astype(np.float32), MPI.FLOAT], [self.buf_E_IE, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array(self.h_atoms.E_cx_gain).astype(np.float32), MPI.FLOAT], [self.buf_E_gain_cx, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.array(self.h_atoms.E_loss_ion).astype(np.float32), MPI.FLOAT], [self.buf_E_loss_ion, MPI.FLOAT], op = MPI.SUM, root = 0)
        self.sub_comm.Reduce([np.sum(self.h_atoms.E[0:self.h_atoms.max_ind][self.h_atoms.active[0:self.h_atoms.max_ind]]*self.h_atoms.norm_weight[0:self.h_atoms.max_ind][self.h_atoms.active[0:self.h_atoms.max_ind]]).astype(np.float32), MPI.FLOAT], [self.buf_E_in_system, MPI.FLOAT], op = MPI.SUM, root = 0)

        if self.rank == 0:
            print("Atom loss due to wall absorbtion:")
            print(self.buf_reflection_loss)
            print("Atom loss at inner edge:")
            print(self.buf_inner_edge_loss)
            print("Molecule loss due to wall absorbtion:")
            print(self.buf_reflection_loss_mol)
            print("Molecule loss at inner edge:")
            print(self.buf_inner_edge_loss_mol)
            print("Total source:")
            print(self.buf_total_source)
            print("Total inflow of molecules:")
            print(self.buf_inflow)
            print("Atoms in system:")
            print(self.buf_weight_atoms)
            print("Molecules in system:")
            print(self.buf_weight_molecules)
            print("Total ledger for particles:")
            print(2*self.buf_inflow - self.buf_reflection_loss - self.buf_inner_edge_loss - self.buf_total_source - self.buf_weight_atoms - 2*self.buf_weight_molecules)
            print("/******************************************************************************************/")
            print("Initial Energy of atoms:")
            print(self.buf_E_init)
            print("Energy gained in cx:")
            print(self.buf_E_gain_cx)
            print("Energy lost from absorbtion:")
            print(self.buf_E_absorb)
            print("Energy lost at inner edge:")
            print(self.buf_E_IE)
            print("Energy transferred to plasma from ionization:")
            print(self.buf_E_loss_ion)
            print("Atom energy in system:")
            print(self.buf_E_in_system)
            print("Total Energy Ledger:")
            print(self.buf_E_init + self.buf_E_gain_cx - self.buf_E_absorb - self.buf_E_IE - self.buf_E_loss_ion - self.buf_E_in_system)
            self.nc_dat.close()
