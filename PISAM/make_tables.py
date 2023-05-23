import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d

#A minimal table where table data is inputted directly. It is meant for 1D
#distributions that needs to be sampled from during simulation.
class Table_simple:
    def __init__(self, min, dx, data, table_filename):
        self.min = min
        self.dx = dx
        self.data = data
        self.table_filename = table_filename

    def save_object(self):
        with open(self.table_filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)

#A table for reactions rates that are only dependent on plasma temperature.
class Tables_1d:
    def __init__(self, T_min, T_max, T_res, base, sum_coefs, table_filename):
        self.T_min = T_min
        self.T_max = T_max
        self.T_res = T_res
        self.base = base
        self.T_log_min = np.log(T_min)/np.log(base)
        self.T_log_max = np.log(T_max)/np.log(base)
        self.dT_log = (self.T_log_max - self.T_log_min)/(T_res-1)
        self.sum_coefs = sum_coefs
        self.table_filename = table_filename
        self.Ts = np.logspace(np.log(T_min)/np.log(base), np.log(T_max)/np.log(base), T_res, base=base)

    def get_inds_T(self, Ts):
        inds = ((np.log(Ts)/np.log(self.base)-self.T_log_min)/self.dT_log).astype(np.int32)
        #Dealing with Ts outside range
        inds[inds < 0] = 0
        inds[inds > self.T_res - 1] = self.T_res - 1
        return inds

    def calc_electron_col_rate(self):
        rate = np.zeros_like(self.Ts)
        ln_Te_power = np.ones_like(self.Ts)
        ln_Te = np.log(self.Ts)
        for i in np.arange(self.sum_coefs.shape[0]):
            rate = rate + self.sum_coefs[i]*ln_Te_power
            ln_Te_power = np.multiply(ln_Te_power, ln_Te)
        rate = np.exp(rate)*1e-6
        self.rates = rate

    def save_object(self):
        with open(self.table_filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)

#A 2D table for reactions rates that depend on plasma temperature and neutral energy.
class Tables_2d:
    def __init__(self, T_min, T_max, dT, E_min, E_max, dE, sum_coefs, table_filename):
        self.T_min = T_min
        self.T_max = T_max
        self.dT = dT
        self.E_min = E_min
        self.E_max = E_max
        self.dE = dE
        self.sum_coefs = sum_coefs
        self.table_filename = table_filename
        self.Ts = np.arange(T_min, T_max, dT)
        self.Es = np.arange(E_min, E_max, dE)

    def get_inds_T(self, Ts):
        inds = ((Ts - self.T_min)/self.dT).astype(np.int32)
        #Dealing with Ts outside range
        inds[inds < 0] = 0
        inds[inds > np.size(self.Ts) - 1] = np.size(self.Ts) - 1
        return inds

    def get_inds_E(self, Es):
        inds = ((Es - self.E_min)/self.dE).astype(np.int32)
        #Dealing with Es outside range
        inds[inds < 0] = 0
        inds[inds > np.size(self.Es) - 1] = np.size(self.Es) - 1
        return inds

    def calc_heavy_col_rate(self):
        E_mesh, T_mesh = np.meshgrid(self.Es, self.Ts, indexing='ij')
        rate = np.zeros_like(E_mesh)
        for E in np.arange(self.sum_coefs.shape[0]):
            for T in np.arange(self.sum_coefs.shape[1]):
                rate = rate + self.sum_coefs[E, T]*np.multiply(np.power(np.log(E_mesh), E), np.power(np.log(T_mesh), T))
        rate = np.exp(rate)*1e-6
        self.rates = rate

    def save_object(self):
        with open(self.table_filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)

#A 2D table for reactions rates that depend on plasma temperature and plasma density.
class Tables_2d_T_n:
    def __init__(self, T_min, T_max, T_res, n_min, n_max, n_res, sum_coefs, table_filename):
        self.T_min = T_min
        self.T_max = T_max
        self.T_res = T_res
        self.n_min = n_min
        self.n_max = n_max
        self.n_res = n_res
        self.sum_coefs = sum_coefs
        self.table_filename = table_filename
        self.Ts = np.linspace(T_min, T_max, T_res)
        self.ns = np.linspace(n_min, n_max, n_res)
        self.dT = self.Ts[1]- self.Ts[0]
        self.dn = self.ns[1]- self.ns[0]

    def get_inds_n(self, ns):
        inds = ((ns-self.n_min)/self.dn).astype(np.int32)
        inds[inds < 0] = 0
        inds[inds > (self.n_res-1)] = self.n_res-1
        return inds

    def get_inds_T(self, Ts):
        inds = ((Ts-self.T_min)/self.dT).astype(np.int32)
        inds[inds < 0] = 0
        inds[inds > (self.T_res-1)] = self.T_res-1
        return inds

    def calc_rate(self):
        n_mesh, T_mesh = np.meshgrid(self.ns, self.Ts, indexing='ij')
        n_mesh = n_mesh*1e-14
        rate = np.zeros_like(n_mesh)
        for n in np.arange(self.sum_coefs.shape[0]):
            for T in np.arange(self.sum_coefs.shape[1]):
                rate = rate + self.sum_coefs[n, T]*np.multiply(np.power(np.log(n_mesh), n), np.power(np.log(T_mesh), T))
        rate = np.exp(rate)*1e-6
        self.rates = rate

    def save_object(self):
        with open(self.table_filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)

#This table calculates the velocity distribution of ions going into charge exchange
#reactions in accordance with the subsection "Charge Exchange of Deuterium Atoms and Deuterium Ions"
#in chapter 5 of my thesis and the subsection "Ion-Neutral Collisions" in chapter 3 of my thesis.
#The distribution depends on the ion temperature and the velocity of the neutral in the rest frame of the ions.
class Table_cx_sampling_3D:
    def __init__(self, T_i_min, T_i_max, T_i_res, E_n_min, E_n_max, E_n_res, base, d_alpha, n_standard_deviations, v_res, sum_coefs, table_filename, mass):
        self.T_i_min = T_i_min
        self.T_i_max = T_i_max
        self.E_n_min = E_n_min
        self.E_n_max = E_n_max
        self.d_alpha = d_alpha
        self.T_i_res = T_i_res
        self.E_n_res = E_n_res
        #Sum coefficients for the fir used for the cross section.
        self.sum_coefs = sum_coefs
        self.table_filename = table_filename
        self.base = base
        self.Ts = np.logspace(np.log(T_i_min)/np.log(base), np.log(T_i_max)/np.log(base), T_i_res, base=base)
        self.Es = np.logspace(np.log(E_n_min)/np.log(base), np.log(E_n_max)/np.log(base), E_n_res, base=base)
        self.d_alpha = d_alpha
        self.alphas = np.arange(0, np.pi, d_alpha)
        self.mass = mass
        self.v_res = v_res
        self.n_standard_deviations = n_standard_deviations
        #Temperatures and energies are sampled logarithmically to have the best resolution at the places with the strongest dependence of T and E
        self.T_log_min = np.log(T_i_min)/np.log(base)
        self.T_log_max = np.log(T_i_max)/np.log(base)
        self.dT_log = (self.T_log_max - self.T_log_min)/(T_i_res-1)
        self.E_log_min = np.log(E_n_min)/np.log(base)
        self.E_log_max = np.log(E_n_max)/np.log(base)
        self.dE_log = (self.E_log_max - self.E_log_min)/(E_n_res-1)
        self.dvs = n_standard_deviations*np.sqrt(self.Ts*1.602e-19/mass)/(v_res-1)

    def get_inds_T(self, Ts):
        inds = ((np.log(Ts)/np.log(self.base)-self.T_log_min)/self.dT_log).astype(np.int32)
        #Dealing with Ts outside range
        inds[inds < 0] = 0
        inds[inds > self.T_i_res - 1] = self.T_i_res - 1
        return inds

    def get_inds_E(self, Es):
        inds = ((np.log(Es)/np.log(self.base)-self.E_log_min)/self.dE_log).astype(np.int32)
        #Dealing with Es outside range
        inds[inds < 0] = 0
        inds[inds > self.E_n_res - 1] = self.E_n_res - 1
        return inds

    #Calculate the relative velocity from the magnitude of v_n and v_i and the polar angle og v_i in the d-frame.
    def v_rel(self, v_n_grid, v_i_grid, alpha_grid):
        v_rel = np.sqrt(np.power(v_n_grid, 2) + np.power(v_i_grid, 2) - 2*v_n_grid*v_i_grid*np.cos(alpha_grid))
        return v_rel

    #Calculate cross section based on the relative velocity
    def cx_cross_section(self, v_rel_grid):
        E_grid = 0.5*np.power(v_rel_grid, 2)*self.mass/1.602e-19
        E_grid_valid = E_grid > 0.1
        ln_E_grid = np.log(E_grid)
        ln_cross_section_grid = np.zeros_like(ln_E_grid)
        for i in np.arange(self.sum_coefs.size):
            ln_cross_section_grid[E_grid_valid] = ln_cross_section_grid[E_grid_valid] + self.sum_coefs[i]*np.power(ln_E_grid[E_grid_valid], i)
        #ln_cross_section_grid[E_grid_valid] = -3.294589355000e+01 - 1.713112000000e-01*ln_E_grid[E_grid_valid]
        cross_section_grid = np.exp(ln_cross_section_grid)*1e-4 #Conversion from cm^2 to m^2
        cross_section_grid[np.invert(E_grid_valid)] = 0
        return cross_section_grid

    #Calculate the value of the integrand of the reaction rate in spherical
    #coordinates of the d-frame. See the subsection "Ion-Neutral Collisions" in chapter 3 of my thesis.
    def rate_contrib_T(self, T, alpha_grid, v_i_grid, v_rel_grid):
        cross_section_grid = self.cx_cross_section(v_rel_grid)
        cross_vrel = np.multiply(cross_section_grid, v_rel_grid)
        cross_vrel_v_i = np.multiply(cross_vrel, np.power(v_i_grid, 2))
        exp_grid = np.exp(-(self.mass*np.power(v_i_grid, 2))/(2*T*1.602e-19))
        exp_sin_alpha = np.multiply(exp_grid, np.sin(alpha_grid))
        return 2*np.pi*np.power(self.mass/(2*np.pi*T*1.602e-19), 1.5)*np.multiply(exp_sin_alpha, cross_vrel_v_i)

    #For a certain T make 3D mesh grids from the three 1D arrays containing the
    #ion velocities relevant to the temperature, the relative neutral velocities
    #corresponding to the chosen range of energy and the chosen range of polar angles, alpha.
    def make_grids(self, T):
        v_is = np.linspace(0, self.n_standard_deviations*np.sqrt(T*1.602e-19/self.mass), self.v_res)
        v_ns = np.sqrt(2*self.Es*1.602e-19/self.mass)
        v_n_grid, v_i_grid, alpha_grid = np.meshgrid(v_ns, v_is, self.alphas, indexing='ij')
        v_rel_grid = self.v_rel(v_n_grid, v_i_grid, alpha_grid)
        return alpha_grid, v_i_grid, v_rel_grid

    #Fill the table by calculating the 3D contributions corresponding to each value of T
    #Lastly the cumsum (inclusive) is taken such that the table is ready for sampling. Moreover
    #zeros are inserted in the beginning of the axis over which the cumsum runs to ready for the
    #linear interpolation process used in the sampling.
    def tabulate(self):
        self.table = np.zeros((self.Ts.size, self.Es.size, self.v_res+1, self.alphas.size))
        for i in np.arange(self.Ts.size):
            alpha_grid, v_i_grid, v_rel_grid = self.make_grids(self.Ts[i])
            self.table[i, :, :, :] = np.insert(np.cumsum(self.rate_contrib_T(self.Ts[i], alpha_grid, v_i_grid, v_rel_grid), axis = 1), 0, 0, axis = 1)

    def save_object(self):
        with open(self.table_filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)

#This table calculates the contribution to reaction rates electrons of various
#velocities from a spline to cross section data. As for the previous table the
#temperatures are sampled logarithmically, and the ion evaluation velocities depends
#on temperature. From these contributions the rate is trivially calculated and stored.
#The "distribution part" of the table is not currently used but can be applied to
#sample the electrons going into the relevant reaction, which can be usefull if electron
#energy loss or fragment energy in strongly dependent on impact energy.
class Table_light_heavy_sampling_from_spline_clean:
    def __init__(self, T_e_min, T_e_max, T_e_res, base, n_standard_deviations, v_res, input_file, table_filename):
        self.T_e_min = T_e_min
        self.T_e_max = T_e_max
        self.T_e_res = T_e_res
        self.file = input_file
        self.table_filename = table_filename
        self.base = base
        self.Ts = np.logspace(np.log(T_e_min)/np.log(base), np.log(T_e_max)/np.log(base), T_e_res, base=base)
        self.mass = 9.1093837e-31
        self.v_res = v_res
        self.n_standard_deviations = n_standard_deviations
        self.T_log_min = np.log(T_e_min)/np.log(base)
        self.T_log_max = np.log(T_e_max)/np.log(base)
        self.dT_log = (self.T_log_max - self.T_log_min)/(T_e_res-1)
        self.dvs = n_standard_deviations*np.sqrt(self.Ts*1.602e-19/self.mass)/(v_res-1)
        with open(input_file + '.pkl', 'rb') as f:
            self.cross_spline = pickle.load(f)

    def get_inds_T(self, Ts):
        inds = ((np.log(Ts)/np.log(self.base)-self.T_log_min)/self.dT_log).astype(np.int32)
        #Dealing with Ts outside range
        inds[inds < 0] = 0
        inds[inds > self.T_e_res - 1] = self.T_e_res - 1
        return inds

    def cross_section(self, v_s):
        E_s = 0.5*np.power(v_s, 2)*self.mass/1.602e-19
        return self.cross_spline(E_s)

    def rate_contrib_T(self, T, v_s):
        cross_section_array = self.cross_section(v_s)
        cross_v = np.multiply(cross_section_array, np.power(v_s, 3))
        exp_array = np.exp(-(self.mass*np.power(v_s, 2))/(2*T*1.602e-19))
        return np.sqrt(2/np.pi)*np.power(self.mass/(T*1.602e-19), 1.5)*np.multiply(exp_array, cross_v)

    def make_v_s(self, T):
        v_s = np.linspace(0, self.n_standard_deviations*np.sqrt(T*1.602e-19/self.mass), self.v_res)
        return v_s

    def tabulate(self):
        self.table = np.zeros((self.T_e_res, self.v_res+1))
        for i in np.arange(self.Ts.size):
            v_s = self.make_v_s(self.Ts[i])
            self.table[i, :] = np.insert(np.cumsum(self.rate_contrib_T(self.Ts[i], v_s)), 0, 0)
        self.rates = np.multiply(self.table[:, -1], self.dvs)

    def save_object(self):
        with open(self.table_filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)
