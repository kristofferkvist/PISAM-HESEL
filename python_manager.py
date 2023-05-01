#Standard Libraries
import os
import sys, getopt

sys.path.insert(0, '/home/kristoffer/Desktop/KU/speciale/PISAM-HESEL/PISAM')

import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
#MPI
from mpi4py import MPI
#Timing
import time
from datetime import datetime
#My Classes
from simulator import Simulator
from manager import Manager

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

    data_folder = 'input_data/'

    manager = Manager(optDIR, restart)
    rank = manager.rank
    simulator = Simulator(data_folder, manager.oci, manager.rhos, rank, manager.sub_comm, manager.procs_python, manager.n, manager.Te, manager.Ti, manager.u0_x_ion, manager.u0_y_ion, optDIR, manager.weight)
    manager.simulator = simulator
    simulator.initiate_sim(restart, manager.flag_initialized)
    manager.send_initial_source_terms()

#**************************TIMING THE SIMULATION*******************************#
    if (rank == 0):
        runtimes = np.zeros(5)
        times = np.zeros(6)
#**************************RUNNING THE SIMULATION******************************#
    while (manager.t_old_new[1] < manager.total_sim_time):
        #Get updated time from solver
        if (rank == 0):
            times[0] = time.time()
        manager.intercomm.Bcast([manager.t, MPI.DOUBLE], root = 0)
        simulator.record(manager.t)
        manager.t_old_new[0] = manager.t_old_new[1]
        if manager.t > manager.total_sim_time:
            manager.t_old_new[1] = manager.total_sim_time
        else:
            manager.t_old_new[1] = manager.t
        if (rank == 0):
            times[1] = time.time()
        manager.receive_fields()
        manager.format_fields()
        if (rank == 0):
            times[2] = time.time()
        simulator.set_fields(manager.n, manager.Te, manager.Ti, manager.u0_x_ion, manager.u0_y_ion)
        if (rank == 0):
            times[3] = time.time()
        simulator.sim_step(manager.t_old_new[1]/manager.oci)
        if (rank == 0):
            times[4] = time.time()
        #Calculate and return sources.
        manager.send_sources(manager.t_old_new[1] - manager.t_old_new[0])
        if (rank == 0):
            times[5] = time.time()
            runtimes = runtimes + np.diff(times)
            print("Max" + str(np.max(simulator.h_atoms.percentage[0:simulator.h_atoms.max_ind])))
            print("Min" + str(np.min(simulator.h_atoms.percentage[0:simulator.h_atoms.max_ind])))
            print("Mean" + str(np.sum(simulator.h_atoms.percentage[0:simulator.h_atoms.max_ind])))
            print("Active" + str(np.sum(simulator.h_atoms.active)))

#*****************************FINALIZING***************************************#
    if (rank == 0):
        simulator.finalize()
        print("############DOMAIN TIME ENDS AT:" + str(simulator.domain.t))
        print("###########################WALL TIMES - ON RANK 0 ##########################")
        print(runtimes)


if __name__ == "__main__":
   main(sys.argv[1:])
