from mpi4py import MPI
import os
import sys, getopt
import numpy as np
from boututils.options import BOUTOptions
from simulator import Simulator
from matplotlib import pyplot as plt
import time
from scipy.ndimage import fourier_gaussian
from datetime import datetime
import netCDF4 as nc


def convert_source_dims(s, dt, dx, dy, oci, rhos, n_particles):
    dv = dx*dy #z-dim is chosen to be one meter
    return s*n_particles/(dv*dt*oci)

def normalize_sources_n(sn, dt, dx, dy, oci, rhos, n0, n_particles):
    sn = convert_source_dims(sn, dt, dx, dy, oci, rhos, n_particles)
    return sn/n0

def normalize_sources_u(sn, dt, dx, dy, oci, rhos, n0, Te0, n_particles, mi):
    su = convert_source_dims(sn, dt, dx, dy, oci, rhos, n_particles)
    su = su/n0
    return su/np.sqrt(Te0*1.602e-19*mi)

def normalize_sources_p(sp, dt, dx, dy, oci, rhos, n0, Te0, n_particles):
    sp = convert_source_dims(sp, dt, dx, dy, oci, rhos, n_particles)
    p0 = n0*Te0
    return sp/p0

def send_source(intercomm, send_buf, n_procs, size):
    #Send data using scatter
    intercomm.Scatter([send_buf, MPI.DOUBLE], None, root = MPI.ROOT)

def receive_field(sub_comm, intercomm, recv_buf):
    if (intercomm.Get_rank() == 0):
        intercomm.Gather(None, recv_buf, root = MPI.ROOT)
    sub_comm.Bcast([recv_buf, MPI.DOUBLE], root = 0)
    return recv_buf

def format_field(field, normalization, size_x, size_z):
    return np.reshape(field, [size_x,  size_z])*normalization


def fourier_blur(img, sigma):
    input_ = np.fft.fft2(img)
    result = fourier_gaussian(input_, sigma=sigma)
    return np.fft.ifft2(result).real

#Read input file
def main(argv):
    optDIR = 'data'
    restart = False

    opts, args = getopt.getopt(argv,"rd:")
    for opt, arg in opts:
        if opt in ("-r", "-restart"):
            restart = True
        if opt in ("-d"):
            optDIR = arg


    world_comm = MPI.COMM_WORLD
    rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()
    app_id = MPI.APPNUM
    sub_comm = world_comm.Split(app_id, rank)
    intercomm = sub_comm.Create_intercomm(0, world_comm, 0, tag=1)
    #Override rank with the rank of the python part of the program, starting from 0.
    rank = intercomm.Get_rank()

    data_folder = 'input_data/'
    output_datafolder = optDIR

    procs_python = intercomm.Get_size()
    procs_cpp = world_size - procs_python

    #Get BOUTOptions instance
    myOpts = BOUTOptions(optDIR)

    #Read options from root section
    root_dict = myOpts.root
    nout = int(root_dict['nout'])
    mxg = int(root_dict['mxg'])
    timestep = eval(root_dict['timestep'])
    total_sim_time = nout*timestep

    #Read options from mesh section
    mesh_dict = myOpts.mesh
    nx = int(eval(mesh_dict['nx']))
    ny = int(mesh_dict['ny'])
    nz = int(mesh_dict['nz'])
    Lx = float(mesh_dict['Lx'])
    Ly = float(mesh_dict['Lz'])

    #Read options kinetic neutral section
    kinetic_neutral_dict = myOpts.kinetic_neutrals
    flag_initialized = int(eval(kinetic_neutral_dict['flag_initialized']))
    phys_injection_rate_molecules = int(eval(kinetic_neutral_dict['phys_injection_rate_molecules']))
    H_molecule_injection_rate = int(eval(kinetic_neutral_dict['H_molecule_injection_rate']))
    init_source_time = int(eval(kinetic_neutral_dict['init_source_time']))

    r_std_blur = float(eval(kinetic_neutral_dict['r_std_blur']))

    #Read options from hesel section
    hesel_dict = myOpts.hesel
    n0 = float(eval(hesel_dict['n0']))
    Te0 = float(eval(hesel_dict['Te0']))
    Ti0 = float(eval(hesel_dict['Ti0']))

    #Check if nummber of processors is adequate.
    send_total = (nx-2*mxg)*ny*nz
    if (rank == 0):
        rows_x_per_proc = int(nx-2*mxg/procs_cpp)
        if (send_total%procs_cpp != 0):
            raise Exception("###############################X-dimension not dividable between the given number of processors##################")
    data_per_proc = int(send_total/procs_cpp)

    #Initialize buffers
    oci = np.array(0).astype(np.double)
    rhos = np.array(0).astype(np.double)
    t = np.array(0).astype(np.double)
    #Only rank zero collects the data to be send in the send buf.
    #Due to the lack of pointers in python all ranks recv field through
    #broadcasting on the sub communicator, and thus they all need recv_buf
    if (rank == 0):
        send_buf = np.empty([procs_cpp, data_per_proc]).astype(np.double)

    recv_buf_n = np.empty(send_total).astype(np.double)
    recv_buf_Te = np.empty(send_total).astype(np.double)
    recv_buf_Ti = np.empty(send_total).astype(np.double)
    #ts = np.array([0])
    #***************************************************************************
    #Wrapper for format field
    def format_field_wrapper(field, normalization):
        return format_field(field, normalization, nx-2*mxg, nz)
    #***************************************************************************

    intercomm.Bcast([oci, MPI.DOUBLE], root = 0)
    #print("######################################OCI###############################" + str(oci))
    intercomm.Bcast([rhos, MPI.DOUBLE],root = 0)
    n_init = receive_field(sub_comm, intercomm, recv_buf_n)
    n_init = format_field_wrapper(n_init, 1)
    Te_init = receive_field(sub_comm, intercomm, recv_buf_Te)
    Te_init = format_field_wrapper(Te_init, Te0)
    Ti_init = receive_field(sub_comm, intercomm, recv_buf_Ti)
    Ti_init = format_field_wrapper(Ti_init, Ti0)

    super_particle_size = phys_injection_rate_molecules*Ly*rhos/(H_molecule_injection_rate) #Lz is set to 1 m, hence this expression

    #Initiate neutral simulator
    simulator = Simulator(data_folder, oci, rhos, rank, procs_python, n_init, Te_init, Ti_init, optDIR)
    n_std_blur_x = rhos*r_std_blur/simulator.domain.dx
    n_std_blur_y = rhos*r_std_blur/simulator.domain.dy
    if (restart):
        simulator.restart()
    else:
        if (flag_initialized):
            simulator.initiate_from_files()
            simulator.domain.t = 0
        else:
            simulator.initiate_from_fields()
            simulator.domain.t = 0

    #***************************************************************************
    #Wrapper for send_source:
    def send_source_wrapper(dt):
        #For both species the sources are ordered as: Particle, momentum_x, momentum_z, temperature.
        Sn_electron, Su_x_electron, Su_z_electron, SP_electron = simulator.get_sources_electrons()
        Sn_ion, Su_x_ion, Su_z_ion, SP_ion = simulator.get_sources_ions()
        Sn_electron = normalize_sources_n(Sn_electron, dt, simulator.domain.dx, simulator.domain.dy, oci, rhos, n0, super_particle_size)
        SP_electron = normalize_sources_p(SP_electron, dt, simulator.domain.dx, simulator.domain.dy, oci, rhos, n0, Te0, super_particle_size)
        Su_x_ion = normalize_sources_u(Su_x_ion, dt, simulator.domain.dx, simulator.domain.dy, oci, rhos, n0, Te0, super_particle_size, simulator.domain.d_ion_mass)
        Su_z_ion = normalize_sources_u(Su_z_ion, dt, simulator.domain.dx, simulator.domain.dy, oci, rhos, n0, Te0, super_particle_size, simulator.domain.d_ion_mass)
        SP_ion = normalize_sources_p(SP_ion, dt, simulator.domain.dx, simulator.domain.dy, oci, rhos, n0, Te0, super_particle_size)
        Sn_electron = fourier_blur(Sn_electron, n_std_blur_x)
        SP_electron = fourier_blur(SP_electron, n_std_blur_x)
        Su_x_ion = fourier_blur(Su_x_ion, n_std_blur_x)
        Su_z_ion = fourier_blur(Su_z_ion, n_std_blur_x)
        SP_ion = fourier_blur(SP_ion, n_std_blur_x)
        source_list = [Sn_electron, SP_electron, Su_x_ion, Su_z_ion, SP_ion]
        #Reducing all sources to send_buf of rank 0, and using rank 0 only
        #to communicate to hesel.
        for i in np.arange(len(source_list)):
            s = source_list[i]
            if (rank == 0):
                source_shape = s.shape
                sub_comm.Reduce([np.reshape(s, (procs_cpp, data_per_proc)).astype(np.double), MPI.DOUBLE], [send_buf, MPI.DOUBLE], op = MPI.SUM, root = 0)
                send_source(intercomm, send_buf, procs_cpp, data_per_proc)
            else:
                sub_comm.Reduce([np.reshape(s, (procs_cpp, data_per_proc)).astype(np.double), MPI.DOUBLE], [None, MPI.DOUBLE], op = MPI.SUM, root = 0)
        return
    #***************************************************************************

    t_old_new = np.zeros(2)
    if (restart):
        if not (simulator.domain.last_plasma_timestep_length > 0):
            raise Exception("#######################The timestep under which sources for restart objects was obtained, is not set properly####################")
        start_time = simulator.domain.t*oci
        total_sim_time = total_sim_time + start_time
        t_old_new[0] = start_time
        t_old_new[1] = start_time
        if (rank == 0):
            intercomm.Bcast([np.array(start_time).astype(np.double), MPI.DOUBLE], root=MPI.ROOT)
        send_source_wrapper(simulator.domain.last_plasma_timestep_length)
    else:
        send_source_wrapper(init_source_time/oci)
        t_old_new[0] = 0
        t_old_new[1] = 0

    #Histogram recording stuff
    record_iter = 0
    if (rank == 0):
        hist_buff_atom_density = np.empty(simulator.h_atoms.bin_edges_x.size-1).astype(np.float32)
        hist_buff_atom_density_cx = np.empty(simulator.h_atoms.bin_edges_x.size-1).astype(np.float32)
        hist_buff_molecule_density = np.empty(simulator.h_molecules.bin_edges_x.size-1).astype(np.float32)
        dense_dat_atoms = np.zeros((nout, hist_buff_atom_density.size)).astype(np.float32)
        dense_dat_atoms_cx = np.zeros((nout, hist_buff_atom_density_cx.size)).astype(np.float32)
        dense_dat_molecules = np.zeros((nout, hist_buff_molecule_density.size)).astype(np.float32)

    if (rank == 0):
        runtimes = np.zeros(5)
        times = np.zeros(6)
#**************************RUNNING THE SIMULATION******************************#
    while (t_old_new[1] < total_sim_time):
        #Get updated time from solver
        if (rank == 0):
            times[0] = time.time()
        intercomm.Bcast([t, MPI.DOUBLE], root = 0)
        if (t > (record_iter+1)*timestep):
            x_pos_atoms = simulator.h_atoms.x[0:simulator.h_atoms.max_ind][simulator.h_atoms.active[0:simulator.h_atoms.max_ind]]
            x_pos_atoms_cxed = simulator.h_atoms.x[0:simulator.h_atoms.max_ind][simulator.h_atoms.cxed[0:simulator.h_atoms.max_ind]]
            x_pos_molecules = simulator.h_molecules.x[0:simulator.h_molecules.max_ind][simulator.h_molecules.active[0:simulator.h_molecules.max_ind]]
            hist_atoms, _ = np.histogram(x_pos_atoms, bins=simulator.h_atoms.bin_edges_x)
            hist_atoms_cxed, _ = np.histogram(x_pos_atoms_cxed, bins=simulator.h_atoms.bin_edges_x)
            hist_molecules, _ =  np.histogram(x_pos_molecules, bins=simulator.h_molecules.bin_edges_x)
            #Diagnnostics
            if (rank == 0):
                sub_comm.Reduce([hist_atoms.astype(np.float32), MPI.FLOAT], [hist_buff_atom_density, MPI.FLOAT], op = MPI.SUM, root = 0)
                sub_comm.Reduce([hist_atoms_cxed.astype(np.float32), MPI.FLOAT], [hist_buff_atom_density_cx, MPI.FLOAT], op = MPI.SUM, root = 0)
                sub_comm.Reduce([hist_molecules.astype(np.float32), MPI.FLOAT], [hist_buff_molecule_density, MPI.FLOAT], op = MPI.SUM, root = 0)
                dense_dat_atoms[record_iter, :] = hist_buff_atom_density
                dense_dat_atoms_cx[record_iter, :] = hist_buff_atom_density_cx
                dense_dat_molecules[record_iter, :] = hist_buff_molecule_density
            else:
                sub_comm.Reduce([hist_atoms.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
                sub_comm.Reduce([hist_atoms_cxed.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
                sub_comm.Reduce([hist_molecules.astype(np.float32), MPI.FLOAT], [None, MPI.FLOAT], op = MPI.SUM, root = 0)
            record_iter = record_iter + 1
        t_old_new[0] = t_old_new[1]
        if t > total_sim_time:
            t_old_new[1] = total_sim_time
        else:
            t_old_new[1] = t
        if (rank == 0):
            times[1] = time.time()
        n = receive_field(sub_comm, intercomm, recv_buf_n)
        Te = receive_field(sub_comm, intercomm, recv_buf_Te)
        Ti = receive_field(sub_comm, intercomm, recv_buf_Ti)
        if (rank == 0):
            times[2] = time.time()
        simulator.set_fields(format_field_wrapper(n, 1), format_field_wrapper(Te, Te0), format_field_wrapper(Ti, Ti0))
        if (rank == 0):
            times[3] = time.time()
        simulator.sim_step(t_old_new[1]/oci)
        if (rank == 0):
            times[4] = time.time()
        #Calculate and return sources.
        send_source_wrapper((t_old_new[1] - t_old_new[0])/oci)
        if (rank == 0):
            times[5] = time.time()
            runtimes = runtimes + np.diff(times)

#*****************************FINALIZING***************************************#
    if (rank == 0):
        dense_dat_atoms = super_particle_size*dense_dat_atoms/(simulator.Lz*simulator.domain.dx)
        dense_dat_atoms_cx = super_particle_size*dense_dat_atoms_cx/(simulator.Lz*simulator.domain.dx)
        dense_dat_molecules = super_particle_size*dense_dat_molecules/(simulator.Lz*simulator.domain.dx)
        nc_dat = nc.Dataset(optDIR + '/kinetic_hists.nc', 'w', 'NETCDF4')
        nc_dat.createDimension('x', simulator.bin_edges_x.size-1)
        nc_dat.createDimension('t', nout)
        n_atom_var = nc_dat.createVariable('n_atom_x_t', 'float32', ('t', 'x'))
        n_atom_var[:, :] = dense_dat_atoms
        n_atom_var_cx = nc_dat.createVariable('n_atom_cx_x_t', 'float32', ('t', 'x'))
        n_atom_var_cx[:, :] = dense_dat_atoms_cx
        n_molecule_var = nc_dat.createVariable('n_molecule_x_t', 'float32', ('t', 'x'))
        n_molecule_var[:, :] = dense_dat_molecules
        nc_dat.close()
        print("############DOMAIN TIME ENDS AT:" + str(simulator.domain.t))
        print("###########################WALL TIMES - ON RANK 0 ##########################")
        print(runtimes)

    #simulator.finalize(output_datafolder, (t_old_new[1] - t_old_new[0])/oci)


if __name__ == "__main__":
   main(sys.argv[1:])
