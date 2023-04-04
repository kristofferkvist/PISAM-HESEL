import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
from PIL import Image
import matplotlib.pyplot as plt

class Parallel_lowpass():
    def __init__(self, comm, n1, n2, std_x, std_y):
        self.comm = comm
        self.comm_size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.n1 = n1
        self.n2 = n2
        self.P = 2*n1
        self.Q = 2*n2
        self.std_x = std_x
        self.std_y = std_y
        self.initiate_shifting_arrray()
        self.initiate_lowpass()

    def dist_mat(self, M, N):
        dist_1d_N = np.arange(-int(N/2), -int(N/2)+N)
        hori_mat = np.tile(dist_1d_N, (M, 1))
        dist_1d_M = np.arange(-int(M/2), -int(M/2)+M)
        vert_mat = np.transpose(np.tile(dist_1d_M, (N, 1)))
        return np.sqrt(np.power(hori_mat, 2) + np.power(vert_mat, 2))

    def build_gauss(self, M, N):
        dist_1d_N = np.arange(-int(N/2), -int(N/2)+N)
        hori_mat = np.power(np.tile(dist_1d_N, (M, 1)), 2)
        dist_1d_M = np.arange(-int(M/2), -int(M/2)+M)
        vert_mat = np.power(np.transpose(np.tile(dist_1d_M, (N, 1))), 2)
        base_mat = hori_mat/(2*self.std_x*self.std_x) + vert_mat/(2*self.std_y*self.std_y)
        gauss = np.exp(-base_mat)
        return gauss/np.sum(gauss)

    def build_shifting_array(self, M, N):
        arr_hor = -1*np.ones((M, N))
        arr_vert = -1*np.ones((M, N))
        equal_first_axis = np.arange(1, M, 2)
        equal_second_axis = np.arange(1, N, 2)
        arr_hor[equal_first_axis, :] = 1
        arr_vert[:, equal_second_axis] = 1
        shifting_array = arr_hor*arr_vert
        return shifting_array

    def distribute_arr(self, arr):
        N = np.array([self.P, self.Q], dtype=int)
        fft = PFFT(self.comm, N, axes=(0, 1), dtype=np.float64, grid=(-1,))
        u = newDistArray(fft, False)

        #Make buffer for partition sizes
        if self.rank == 0:
            row_count_buf = np.empty(self.comm_size, dtype=np.int32)
        else:
            row_count_buf = None

        #Communicate number of rows in partition to root and calculate total partition size
        self.comm.Gather(np.array(u.shape[0], dtype=np.int32), row_count_buf, root = 0)
        if self.rank == 0:
            row_count_buf = self.Q*row_count_buf
            displs = np.cumsum(row_count_buf)-row_count_buf

        #Distribute the image to the separate partitions
        recv_buf = np.empty(u.size, dtype=np.float64)
        if self.rank == 0:
            self.comm.Scatterv(sendbuf=[arr.flatten(), row_count_buf, displs, MPI.DOUBLE], recvbuf=[recv_buf, MPI.DOUBLE], root=0)
        else:
            self.comm.Scatterv(sendbuf=None, recvbuf=[recv_buf, MPI.DOUBLE], root=0)
        u[:] = np.reshape(recv_buf, u.shape)
        return u, fft, row_count_buf

    def initiate_shifting_arrray(self):
        if self.rank == 0:
            shifting_array = self.build_shifting_array(self.P, self.Q)
        else:
            shifting_array = None
        self.shift, _, _ = self.distribute_arr(shifting_array)

    def initiate_lowpass(self):
        if self.rank == 0:
            gauss = self.build_gauss(self.P, self.Q)
        else:
            gauss = None
        gauss_dist, fft, _ = self.distribute_arr(gauss)
        gauss_dist[:] = gauss_dist*self.shift
        #Make the forward transform
        gauss_fft = fft.forward(gauss_dist, normalize=True) # Note that normalize=True is default and can be omitted
        #Take absolute value
        gauss_fft[:] = np.abs(gauss_fft)
        self.lowpass = gauss_fft

    def blur(self, img):
        #Pad Image
        if self.rank == 0:
            padded_img = np.zeros((self.P, self.Q)).astype(np.float64)
            padded_img[0:self.n1, 0:self.n2] = img
            padded_img[0:self.n1, self.n2:] = np.flip(img, axis = 1)
            padded_img[self.n1:, 0:self.n2] = np.flip(img, axis = 0)
            padded_img[self.n1:, self.n2:] = np.flip(img, axis = None)
        else:
            padded_img = None
        u, fft, part_size_count = self.distribute_arr(padded_img)
        #Shift the image
        u[:] = u*self.shift
        #Make the forward transform
        u_fft = fft.forward(u, normalize=False) # Note that normalize=True is default and can be omitted
        #Apply lowpass filter
        u_fft_filtered = u_fft*self.lowpass
        #Make inverse transform
        u_new = np.zeros_like(u)
        u_new = fft.backward(u_fft_filtered, u_new)
        #Shift to obtain blurred image
        u_new[:] = np.real(u_new*self.shift)

        #Gather blurred image in rank 0
        if self.rank == 0:
            img_buf = np.empty(self.Q*self.P, dtype=np.float64)
        else:
            img_buf = None

        self.comm.Gatherv(sendbuf=[u_new, MPI.DOUBLE], recvbuf=(img_buf, part_size_count), root = 0)
        if self.rank == 0:
            img_new = np.reshape(img_buf, (self.P, self.Q))
            return img_new[0:self.n1, 0:self.n2]
        else:
            return None
