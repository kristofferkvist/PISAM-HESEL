#Standard Libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#My Classes
from parallel_lowpass import Parallel_lowpass
from mpi4py import MPI

world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
rank = world_comm.Get_rank()

if rank == 0:
    img_file = '/home/kristoffer/Downloads/moon.png'
    image = Image.open(img_file)
    image.show()
    img_arr = np.array(image.convert('L'), dtype=np.float64)
    print(np.sum(img_arr))
    n1 = np.array(img_arr.shape[0], dtype=np.int32)
    n2 = np.array(img_arr.shape[1], dtype=np.int32)
else:
    n1 = np.array(0, dtype=np.int32)
    n2 = np.array(0, dtype=np.int32)
    img_arr = None

world_comm.Bcast([n1, MPI.INT], root = 0)
world_comm.Bcast([n2, MPI.INT], root = 0)

lowpasser = Parallel_lowpass(world_comm, n1, n2, 10, 1)
img_new = lowpasser.blur(img_arr)

if rank == 0:
    print(np.sum(img_new))
    image_new = Image.fromarray(img_new)
    image_new.show()
