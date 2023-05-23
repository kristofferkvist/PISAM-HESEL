#Standard Libraries
import os
import sys, getopt
import numpy as np
from matplotlib import pyplot as plt
#Read and write netCDF files
import netCDF4 as nc
#MPI
from mpi4py import MPI
#Timing
import time
from datetime import datetime
#My Classes
#Add the PISAM directory temporarily to your path
sys.path.insert(0, os.getcwd() + '/PISAM')
from simulator import Simulator
from manager import Manager

#The main method of a full simulation.
def main(argv):
    optDIR = 'data'
    #Default is not restarting
    restart = False

    #Read options from command line
    opts, args = getopt.getopt(argv,"rd:")
    for opt, arg in opts:
        if opt in ("-r", "-restart"):
            restart = True
        if opt in ("-d"):
            optDIR = arg

    #Set the directory in which all tables and dicts are stored.
    data_folder = 'PISAM/input_data/'

    #Initiate Manager instance
    manager = Manager(optDIR, restart)
    #Get MPI rank
    rank = manager.rank
    #Initiate Simulator instance
    simulator = Simulator(data_folder, manager.oci, manager.rhos, rank, manager.sub_comm, manager.procs_python, manager.n, manager.Te, manager.Ti, manager.u0_x_ion, manager.u0_y_ion, optDIR, manager.weight)
    manager.simulator = simulator
    simulator.initiate_sim(restart, manager.flag_initialized)
    #Send the source terms obtained as part of simulator initiation
    manager.send_initial_source_terms()

#**************************TIMING THE SIMULATION*******************************#
    if (rank == 0):
        runtimes = np.zeros(7)
        times = np.zeros(8)
#**************************RUNNING THE SIMULATION******************************#
    while (manager.t_old_new[1] < manager.total_sim_time):
        #Get updated time from solver
        if (rank == 0):
            times[0] = time.time()
        #Receive the current time in the HESEL simulation
        manager.intercomm.Bcast([manager.t, MPI.DOUBLE], root = 0)
        if (rank == 0):
            times[1] = time.time()
        #Recording data for density and flux monitoration
        simulator.record_before_step(manager.t)
        #Save the end-time of the previous timestep
        manager.t_old_new[0] = manager.t_old_new[1]
        #Set the new sim time of the HESEL simulation
        #and avoid overshooting the last timestep (Important for restart)
        if manager.t > manager.total_sim_time:
            manager.t_old_new[1] = manager.total_sim_time
        else:
            manager.t_old_new[1] = manager.t
        if (rank == 0):
            times[2] = time.time()
        #Receive the current plasma fields and unnormalize them
        manager.receive_fields()
        manager.format_fields()
        if (rank == 0):
            times[3] = time.time()
        #Set the fields of the simulator
        simulator.set_fields(manager.n, manager.Te, manager.Ti, manager.u0_x_ion, manager.u0_y_ion)
        if (rank == 0):
            times[4] = time.time()
        #Perform the simulation step
        simulator.sim_step(manager.t_old_new[1]/manager.oci)
        if (rank == 0):
            times[5] = time.time()
        #Recording data for flux monitoration
        simulator.record_after_step(manager.t, (manager.t_old_new[1]-manager.t_old_new[0])/manager.oci)
        if (rank == 0):
            times[6] = time.time()
        #Send the new sources to HESEL sources. This step involves reducing, normalizing and smoothing.
        manager.send_sources(manager.t_old_new[1] - manager.t_old_new[0])
        if (rank == 0):
            times[7] = time.time()
            runtimes = runtimes + np.diff(times)

#*****************************FINALIZING***************************************#
    if (rank == 0):
        #Close the netCDF file
        #Print the runtimes
        print(" ")
        print("############DOMAIN TIME ENDS AT:" + str(simulator.domain.t))
        print("###########################WALL TIMES - ON RANK 0 ##########################")
        print("Time spent waiting for HESEL:                       " + str(runtimes[0]))
        print("First Recording:                                    " + str(runtimes[1]))
        print("Recieve and format fields:                          " + str(runtimes[2]))
        print("Set fields:                                         " + str(runtimes[3]))
        print("Perform the PISAM simulation step:                  " + str(runtimes[4]))
        print("Recording after step:                               " + str(runtimes[5]))
        print("Blurring and sending sources:                       " + str(runtimes[6]))
        print("###################Wall times of PISAM on rank 0 ###########################")
        print("Inflow routine:                                     " + str(simulator.domain.wall_times[0]))
        print("Stripping:                                          " + str(simulator.domain.wall_times[1]))
        print("Set plasma fields for the individual neutrals:      " + str(simulator.domain.wall_times[2]))
        print("Obtain reaction rates:                              " + str(simulator.domain.wall_times[3]))
        print("Calculate corresponding propabilities:              " + str(simulator.domain.wall_times[4]))
        print("Sample interactions from propabilities:             " + str(simulator.domain.wall_times[5]))
        print("Avoid doing operations on inactive particles:       " + str(simulator.domain.wall_times[6]))
        print("Perform the interactions:                           " + str(simulator.domain.wall_times[7]))
        print("Translate routine including boundary conditions:    " + str(simulator.domain.wall_times[8]))
        print("Set new plasma indices for the individual neutrals: " + str(simulator.domain.wall_times[9]))


    simulator.finalize()


if __name__ == "__main__":
    #Call main with command line arguments, avoid parsing first argument
    #which is allways program name
    main(sys.argv[1:])
