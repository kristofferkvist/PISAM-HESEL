#Standard Libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
#My Classes
from parallel_lowpass import Parallel_lowpass
from mpi4py import MPI
from scipy.ndimage import fourier_gaussian
from scipy.signal import convolve

def build_gauss(M, N, std_x, std_y):
    dist_1d_N = np.arange(-int(N/2), -int(N/2)+N)
    hori_mat = np.power(np.tile(dist_1d_N, (M, 1)), 2)
    dist_1d_M = np.arange(-int(M/2), -int(M/2)+M)
    vert_mat = np.power(np.transpose(np.tile(dist_1d_M, (N, 1))), 2)
    base_mat = hori_mat/(2*std_x*std_x) + vert_mat/(2*std_y*std_y)
    gauss = np.exp(-base_mat)
    return gauss/np.sum(gauss)

def fourier_blur(img, sigma):
    input_ = np.fft.fft2(img)
    print(input_.shape)
    result = fourier_gaussian(input_, sigma=sigma)
    return np.fft.ifft2(result).real

def fourier_blur_real(img, sigma):
    input_ = np.fft.rfft2(img)
    print(input_.shape)
    result = fourier_gaussian(input_, sigma=sigma)
    return np.fft.irfft2(result)

def convolve_blur(img, kernel):
    return convolve(img, kernel, mode='valid')

world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
rank = world_comm.Get_rank()

if rank == 0:
    img_file = '/home/kristoffer/Downloads/moon.png'
    image = Image.open(img_file)
    image = image.resize((int(1024/2), int(1024/2)))
    #image.show()
    img_arr = np.array(image.convert('L'), dtype=np.float64)
    plt.figure()
    plt.imshow(img_arr)
    n1 = np.array(img_arr.shape[0], dtype=np.int32)
    n2 = np.array(img_arr.shape[1], dtype=np.int32)
else:
    n1 = np.array(0, dtype=np.int32)
    n2 = np.array(0, dtype=np.int32)
    img_arr = None

world_comm.Bcast([n1, MPI.INT], root = 0)
world_comm.Bcast([n2, MPI.INT], root = 0)

lowpasser = Parallel_lowpass(world_comm, n1, n2, 8, 9)

if rank == 0:
    t1 = time.time()
img_new1 = lowpasser.blur(img_arr)
if rank == 0:
    t2 = time.time()
    print(t2-t1)

if rank == 0:
    t1 = time.time()
    img_new2 = fourier_blur(img_arr, 8)
    t2 = time.time()
    print(t2-t1)

if rank == 0:
    std_x = 15.3/2
    std_y = 17.3/2
    std_x = (2*np.round(std_x/2)).astype(np.int32)
    std_y = (2*np.round(std_y/2)).astype(np.int32)
    kernel = build_gauss(5*std_y+1, 5*std_x+1, std_x, std_y)
    t1 = time.time()
    padded = np.pad(img_arr, (((2.5*std_y).astype(np.int32), (2.5*std_y).astype(np.int32)), ((2.5*std_x).astype(np.int32), (2.5*std_x).astype(np.int32))), mode='symmetric')
    img_new4 = convolve_blur(padded, kernel)
    t2 = time.time()
    print(t2-t1)

if rank == 0:
    print(np.sum(img_arr))
    plt.figure()
    plt.imshow(img_new1)
    print(np.sum(img_new1))
    plt.figure()
    plt.imshow(img_new2)
    plt.figure()
    print(np.sum(img_new4))
    print(img_new4.shape)
    plt.imshow(img_new4)
    plt.show()
