"""
@Kristoffer Kvist (Orcid ID: 0009-0006-4494-7281)
The code of this document is developed and written by Kristoffer Kvist affiliated
to the physics department of the Technical University of Denmark. The content of the
code can be applied by any third party, given that article "A direct Monte Carlo
approach for the modeling of neutrals at the plasma edge and its self-consistent
coupling with the 2D fluid plasma edge turbulence model HESEL" published in
"Physics of Plasmas" in 2024 is cited accordingly.
"""

"""
This class is responsible for the binding between PISAM and HESEL.
Its main purpose is to communicate the plasma fields and source terms.
"""

import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
#MPI
from mpi4py import MPI
#BOUT utils
from boututils.options import BOUTOptions

class Manager():
    def __init__(self, optDIR, restart):
        #Physical constants
        self.electron_mass = 9.1093837e-31
        self.mi = 2.01410177811*1.660539e-27 - self.electron_mass
        #Number of source arrays passed to HESEL
        self.n_sources = 5
        #Directory of BOUT.inp
        self.optDIR = optDIR
        self.restart = restart
        self.initiate_mpi()
        self.read_dict_options()
        self.initialize_buffers()
        self.initial_communication()
        self.make_kernel()

    #Communication
    def send_sources(self, dt):
        #Obtain and normalize sources
        Sn_electron, SP_electron = self.simulator.get_sources_electrons()
        Su_x_ion, Su_z_ion, SP_ion = self.simulator.get_sources_ions()
        Sn_electron = self.normalize_sources_n(Sn_electron, dt)
        SP_electron = self.normalize_sources_p(SP_electron, dt)
        Su_x_ion = self.normalize_sources_u(Su_x_ion, dt)
        Su_z_ion = self.normalize_sources_u(Su_z_ion, dt)
        SP_ion = self.normalize_sources_p(SP_ion, dt)

        #Make source list for iteration
        source_list = [Sn_electron, SP_electron, Su_x_ion, Su_z_ion, SP_ion]

        #Blur and send to hesel
        i = 0
        while i < self.n_sources:
            loop_size = np.min(np.array([self.n_sources-i, self.procs_python]))
            loop_end = i + loop_size
            #Reduce the repsective source terms to the corresponding rank of the current chunk (loop size).
            #Usually the number of processors running PISAM is larger than the number of sources (5)
            #and thus only one chunk is nessecary (loop_size = self.n_sources).
            while i < loop_end:
                s = source_list[i]
                if self.rank == i%self.procs_python:
                    self.sub_comm.Reduce([s.astype(np.float64), MPI.DOUBLE], [self.send_buf, MPI.DOUBLE], op = MPI.SUM, root = i%self.procs_python)
                else:
                    self.sub_comm.Reduce([s.astype(np.float64), MPI.DOUBLE], [None, MPI.DOUBLE], op = MPI.SUM, root = i%self.procs_python)
                i = i+1
            #With the source terms reduced they can be blurred in parallel
            #and then communicated to HESEL.
            if self.rank < loop_size:
                self.send_buf = self.convolve_blur(self.send_buf)
                self.intercomm.Scatter([np.reshape(self.send_buf, (self.procs_cpp, self.data_per_proc)).astype(np.float64), MPI.DOUBLE], None, root = MPI.ROOT)

    #Initialization
    def initiate_mpi(self):
        world_comm = MPI.COMM_WORLD
        rank = world_comm.Get_rank()
        world_size = world_comm.Get_size()
        #Get the application number for the split operation
        app_id = MPI.APPNUM
        #Make the sub communicators belonging to python and HESEL.
        self.sub_comm = world_comm.Split(app_id, rank)
        #Create the intercommunicator from these subcommunicators
        self.intercomm = self.sub_comm.Create_intercomm(0, world_comm, 0, tag=1)
        #Override rank with the rank of the python part of the program, starting from 0.
        self.rank = self.intercomm.Get_rank()
        self.procs_python = self.intercomm.Get_size()
        #Calculate the number of C++ cores used in the current simulation.
        self.procs_cpp = world_size - self.procs_python

    #This method reads the options for "kinetic neutrals" in BOUT.inp
    def read_dict_options(self):
        #Get BOUTOptions instance
        myOpts = BOUTOptions(self.optDIR)

        #Read options from root section
        root_dict = myOpts.root
        self.nout = int(root_dict['nout'])
        mxg = int(root_dict['mxg'])
        self.mxg = mxg
        self.timestep = eval(root_dict['timestep'])
        self.total_sim_time = self.nout*self.timestep

        #Read options from mesh section
        #Here, unlike BOUT++, x is the radial axis and y is the poloidal axis
        mesh_dict = myOpts.mesh
        self.nx = int(eval(mesh_dict['nx']))
        self.ny = int(mesh_dict['nz'])
        self.Lx_norm = float(mesh_dict['Lx'])
        #I use a right handed coordinate system in all of PISAM.
        #It is left handed in HESEL, hence the Ly/Lz confusion in the following line.
        self.Ly_norm = float(mesh_dict['Lz'])

        self.send_total = (self.nx-2*self.mxg)*self.ny
        #Check if number of processors is adequate
        if self.rank == 0:
            if (self.send_total%self.procs_cpp != 0):
                raise Exception("###X-dimension not dividable between the given number of processors###")
        self.data_per_proc = int(self.send_total/self.procs_cpp)

        #Read options kinetic neutral section
        kinetic_neutral_dict = myOpts.kinetic_neutrals
        self.flag_initialized = int(eval(kinetic_neutral_dict['flag_initialized']))
        self.phys_injection_rate_molecules = int(eval(kinetic_neutral_dict['phys_injection_rate_molecules']))
        self.H_molecule_injection_rate = int(eval(kinetic_neutral_dict['H_molecule_injection_rate']))
        self.init_source_time = int(eval(kinetic_neutral_dict['init_source_time']))
        r_std_blur = float(eval(kinetic_neutral_dict['r_std_blur']))
        self.n_std_blur_radial = r_std_blur*self.nx/self.Lx_norm
        self.n_std_blur_poloidal = r_std_blur*self.ny/self.Ly_norm

        #Read options from hesel section
        hesel_dict = myOpts.hesel
        self.n0 = float(eval(hesel_dict['n0']))
        self.Te0 = float(eval(hesel_dict['Te0']))
        self.u0 = np.sqrt(self.Te0*1.602e-19/self.mi)

    #Inititalize the buffers used for internal as well as external mpi calls.
    def initialize_buffers(self):
        self.oci = np.array(0).astype(np.float64)
        self.rhos = np.array(0).astype(np.float64)
        self.t = np.array(0).astype(np.float64)
        #Only ranks lower than n_sources collects the data to be send in the send buf.
        #Due to the lack of pointers in python all ranks recv field through
        #broadcasting on the sub communicator, and thus they all need recv_buf
        if self.rank < self.n_sources:
            self.send_buf = np.empty((self.nx-2*self.mxg, self.ny)).astype(np.float64)
        else:
            self.send_buf = None
        self.recv_buf_n = np.empty(self.send_total).astype(np.float64)
        self.recv_buf_Te = np.empty(self.send_total).astype(np.float64)
        self.recv_buf_Ti = np.empty(self.send_total).astype(np.float64)
        self.recv_buf_u0_x_ion = np.empty(self.send_total).astype(np.float64)
        self.recv_buf_u0_y_ion = np.empty(self.send_total).astype(np.float64)

    #Recieve the Bohm normalization parameters and calculate the particle weight.
    def initial_communication(self):
        self.intercomm.Bcast([self.oci, MPI.DOUBLE], root = 0)
        self.intercomm.Bcast([self.rhos, MPI.DOUBLE], root = 0)
        self.Lx_metric = self.Lx_norm*self.rhos
        self.Ly_metric = self.Ly_norm*self.rhos
        self.dx = self.Lx_metric/self.nx
        self.dy = self.Ly_metric/self.ny
        self.weight = self.phys_injection_rate_molecules*self.Ly_metric/(self.H_molecule_injection_rate)
        #Receive fields and normalize.
        self.receive_fields()
        self.format_fields()

    #Send the source terms obtained as the last step in initializing PISAM to HESEL
    def send_initial_source_terms(self):
        #Init the timekeeper array if the PISAM simulation.
        self.t_old_new = np.zeros(2)
        if (self.restart):
            if not (self.simulator.domain.last_plasma_timestep_length > 0):
                raise Exception("#######################The timestep under which sources for restart objects was obtained, is not set properly####################")
            start_time = self.simulator.domain.t*self.oci
            self.total_sim_time = total_sim_time + start_time
            self.t_old_new[0] = start_time
            self.t_old_new[1] = start_time
            if (self.rank == 0):
                self.intercomm.Bcast([np.array(start_time).astype(np.float64), MPI.DOUBLE], root=MPI.ROOT)
            self.send_sources(self.simulator.domain.last_plasma_timestep_length)
        else:
            self.send_sources(self.init_source_time)

    #Build the gaussian kernel that is used for the blurring
    def build_gauss(self, rows, cols, std_rad, std_pol):
        dist_1d_cols = np.arange(-int(cols/2), -int(cols/2)+cols)
        hori_mat = np.power(np.tile(dist_1d_cols, (rows, 1)), 2)
        dist_1d_rows = np.arange(-int(rows/2), -int(rows/2)+rows)
        vert_mat = np.power(np.transpose(np.tile(dist_1d_rows, (cols, 1))), 2)
        base_mat = hori_mat/(2*std_pol*std_pol) + vert_mat/(2*std_rad*std_rad)
        gauss = np.exp(-base_mat)
        return gauss/np.sum(gauss)

    def make_kernel(self):
        #Make number of grid spacing per standand deviation an equal integer
        self.std_radial_equal = (2*np.round(self.n_std_blur_radial/2)).astype(np.int32)
        self.std_poloidal_equal = (2*np.round(self.n_std_blur_poloidal/2)).astype(np.int32)
        self.kernel = self.build_gauss(5*self.std_radial_equal+1, 5*self.std_poloidal_equal+1, self.std_radial_equal, self.std_poloidal_equal)

    #Perform a gaussian convolution by multipliying with the gaussian kernel in the fourier domain.
    def convolve_blur(self, img):
        #Apply mirror padding to avoid edge losses
        padded = np.pad(img, (((2.5*self.std_radial_equal).astype(np.int32), (2.5*self.std_radial_equal).astype(np.int32)), ((2.5*self.std_poloidal_equal).astype(np.int32), (2.5*self.std_poloidal_equal).astype(np.int32))), mode='symmetric')
        #The "valid" option means that only values within the padding region is kept
        return convolve(padded, self.kernel, mode='valid')

    #The following functions implement bohm normalization of the sources.
    def convert_source_dims(self, s, dt):
        dv = self.dx*self.dy #z-dim is chosen to be one meter
        return s*self.weight/(dv*dt)

    def normalize_sources_n(self, s, dt):
        s = self.convert_source_dims(s, dt)
        return s/self.n0

    def normalize_sources_u(self, s, dt):
        s = self.convert_source_dims(s, dt)
        s = s/self.n0
        return s/np.sqrt(self.Te0*1.602e-19*self.mi)

    def normalize_sources_p(self, s, dt):
        s = self.convert_source_dims(s, dt)
        return s/(self.n0*self.Te0)

    #Receive the field using MPI gather on rank zero and broadcast it to the rest of the python ranks
    def receive_field(self, recv_buf):
        if (self.intercomm.Get_rank() == 0):
            self.intercomm.Gather(None, recv_buf, root = MPI.ROOT)
        self.sub_comm.Bcast([recv_buf, MPI.DOUBLE], root = 0)
        return recv_buf

    #Wrapper for receiving plasma fields
    def receive_fields(self):
        self.n = self.receive_field(self.recv_buf_n)
        self.Te = self.receive_field(self.recv_buf_Te)
        self.Ti = self.receive_field(self.recv_buf_Ti)
        self.u0_x_ion = self.receive_field(self.recv_buf_u0_x_ion)
        self.u0_y_ion = self.receive_field(self.recv_buf_u0_y_ion)

    #Reshapes and scales according to normalization.
    def format_field(self, field, normalization):
        return np.reshape(field, [self.nx-2*self.mxg, self.ny])*normalization

    #Wrapper for field formatting
    def format_fields(self):
        #As n0 is very large, densities are only unnormalized when needed
        #The other fields are unnormalized before passing them to the simulator
        self.n = self.format_field(self.n, 1)
        self.Te = self.format_field(self.Te, self.Te0)
        self.Ti = self.format_field(self.Ti, self.Te0)
        self.u0_x_ion = self.format_field(self.u0_x_ion, self.u0)
        self.u0_y_ion = self.format_field(self.u0_y_ion, self.u0)
