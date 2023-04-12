#Standard Libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
#My Classes
from parallel_lowpass import Parallel_lowpass
from mpi4py import MPI

world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
rank = world_comm.Get_rank()

if rank == 0:
    img_file = '/home/kristoffer/Downloads/moon.png'
    image = Image.open(img_file)
    image = image.resize((1024, 1024))
    image.show()
    img_arr = np.array(image.convert('L'), dtype=np.float64)
    print(img_arr.shape)
    n1 = np.array(img_arr.shape[0], dtype=np.int32)
    n2 = np.array(img_arr.shape[1], dtype=np.int32)
else:
    n1 = np.array(0, dtype=np.int32)
    n2 = np.array(0, dtype=np.int32)
    img_arr = None

world_comm.Bcast([n1, MPI.INT], root = 0)
world_comm.Bcast([n2, MPI.INT], root = 0)

lowpasser = Parallel_lowpass(world_comm, n1, n2, 10, 1)

if rank == 0:
    t1 = time.time()
img_new = lowpasser.blur(img_arr)
if rank == 0:
    t2 = time.time()
    print(t2-t1)



if rank == 0:
    image_new = Image.fromarray(img_new)
    image_new.show()
