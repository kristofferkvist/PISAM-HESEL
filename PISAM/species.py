import numpy as np
import time
from domain import Domain
import netCDF4 as nc
import pickle

class Species():
    def __init__(self, mass, inflow_rate, N_max, temperature_init, domain, absorbtion_coefficient, bin_edges_x, bin_edges_vs, velocity_domains):
        self.N_max = N_max
        self.vacant_indices = np.array([]).astype(np.int32)
        self.max_ind = 0
        self.active = np.zeros(N_max).astype(np.bool_)
        #self.active[0:self.max_ind] = np.array([]).astype(np.int32)
        self.domain = domain
        self.x = (domain.x_max+domain.dx)*np.ones(N_max)
        self.y = (domain.y_max+domain.dy)*np.ones(N_max)
        self.z = np.zeros(N_max)
        self.vx = np.zeros(N_max)
        self.vy = np.zeros(N_max)
        self.vz = np.zeros(N_max)
        self.E = np.zeros(N_max).astype(np.float32)
        self.plasma_inds_x = np.ones(N_max).astype(np.int32)
        self.plasma_inds_y = np.ones(N_max).astype(np.int32)
        self.percentage = np.zeros(N_max)
        self.mass = mass
        self.inflow_rate = inflow_rate
        self.T_init = temperature_init
        self.born_x = np.zeros(N_max)
        self.born_y = np.zeros(N_max)
        self.domain = domain
        self.Te = np.zeros(N_max).astype(np.float32)
        self.Ti = np.zeros(N_max).astype(np.float32)
        self.n = np.zeros(N_max).astype(np.float32)
        self.absorbtion_coefficient = absorbtion_coefficient
        self.bin_edges_x = bin_edges_x
        self.bin_edges_free_path = self.bin_edges_x*3
        self.bin_edges_vs = bin_edges_vs
        self.velocity_domains = velocity_domains
        self.hist_x = np.zeros(bin_edges_x.size-1)
        self.hist_vs = np.zeros((velocity_domains, bin_edges_vs.size-1))
        self.hist_free_path = self.hist_x = np.zeros(self.bin_edges_free_path.size-1)
        self.x_max_wall_dist_poloidal = self.domain.r_minor - np.sqrt(self.domain.r_minor*self.domain.r_minor - (self.domain.y_max - self.domain.y_min)/8)

    def save_object_nc(self, filename):
        nc_dat = nc.Dataset(filename, 'w', 'NETCDF4') # using netCDF4 for output format
        nc_dat.createDimension('particles', self.N_max)
        nc_dat.createDimension('hist_space', self.bin_edges_x.size-1)
        nc_dat.createDimension('hist_velocity', self.bin_edges_vs.size-1)
        nc_dat.createDimension('hist_velocity_domains', self.velocity_domains)
        nc_dat.createDimension('hist_space_bins', self.bin_edges_x.size)
        nc_dat.createDimension('hist_velocity_bins', self.bin_edges_vs.size)
        nc_dat.createDimension('scalars', 1)
        nc_dat.createDimension('vacant_ind_dim', self.vacant_indices.size)
        x = nc_dat.createVariable('x', 'float64', ('particles'))
        x[:] = self.x
        y = nc_dat.createVariable('y', 'float64', ('particles'))
        y[:] = self.y
        z = nc_dat.createVariable('z', 'float64', ('particles'))
        z[:] = self.z
        vx = nc_dat.createVariable('vx', 'float64', ('particles'))
        vx[:] = self.vx
        vy = nc_dat.createVariable('vy', 'float64', ('particles'))
        vy[:] = self.vy
        vz = nc_dat.createVariable('vz', 'float64', ('particles'))
        vz[:] = self.vz
        E = nc_dat.createVariable('E', 'float64', ('particles'))
        E[:] = self.E
        n = nc_dat.createVariable('n', 'float64', ('particles'))
        n[:] = self.n
        Te = nc_dat.createVariable('Te', 'float64', ('particles'))
        Te[:] = self.Te
        Ti = nc_dat.createVariable('Ti', 'float64', ('particles'))
        Ti[:] = self.Ti
        born_x = nc_dat.createVariable('born_x', 'float64', ('particles'))
        born_x[:] = self.born_x
        born_y = nc_dat.createVariable('born_y', 'float64', ('particles'))
        born_y[:] = self.born_y
        vacant_indices = nc_dat.createVariable('vacant_indices', 'int32', ('vacant_ind_dim'))
        vacant_indices[:] = self.vacant_indices
        active = nc_dat.createVariable('active', 'int32', ('particles'))
        active[:] = self.active
        max_ind = nc_dat.createVariable('max_ind', 'int32', ('scalars'))
        max_ind[:] = self.max_ind
        bin_edges_x = nc_dat.createVariable('bin_edges_x', 'float64', ('hist_space_bins'))
        bin_edges_x[:] = self.bin_edges_x
        bin_edges_vs = nc_dat.createVariable('bin_edges_vs', 'float64', ('hist_velocity_bins'))
        bin_edges_vs[:] = self.bin_edges_vs
        hist_x = nc_dat.createVariable('hist_x', 'float64', ('hist_space'))
        bin_edges_free_path = nc_dat.createVariable('bin_edges_free_path', 'float64', ('hist_space_bins'))
        bin_edges_free_path[:] = self.bin_edges_free_path
        hist_x[:] = self.hist_x
        hist_vs = nc_dat.createVariable('hist_vs', 'float64', ('hist_velocity_domains', 'hist_velocity'))
        hist_vs[:] = self.hist_vs
        hist_free_path = nc_dat.createVariable('hist_free_path', 'float64', ('hist_space'))
        hist_free_path[:] = self.hist_free_path
        self.save_hists(nc_dat)

    def load_object_nc(self, filename):
        nc_dat = nc.Dataset(filename)
        self.x = np.array(nc_dat['x']).astype(np.float64)
        self.y = np.array(nc_dat['y']).astype(np.float64)
        self.z = np.array(nc_dat['y']).astype(np.float64)
        self.vx = np.array(nc_dat['vx']).astype(np.float64)
        self.vy = np.array(nc_dat['vy']).astype(np.float64)
        self.vz = np.array(nc_dat['vz']).astype(np.float64)
        self.E = np.array(nc_dat['E']).astype(np.float64)
        self.n = np.array(nc_dat['n']).astype(np.float64)
        self.Te = np.array(nc_dat['Te']).astype(np.float64)
        self.Ti = np.array(nc_dat['Ti']).astype(np.float64)
        self.vacant_indices = np.array(nc_dat['vacant_indices']).astype(np.int32)
        self.active = np.array(nc_dat['active']).astype(np.bool_)
        self.max_ind = np.array(nc_dat['max_ind']).astype(np.int32)[0]
        self.born_x = np.array(nc_dat['born_x']).astype(np.float64)
        self.born_y = np.array(nc_dat['born_y']).astype(np.float64)


    def maxwellian(self, T, m, n):
        return np.random.normal(0, np.sqrt(T*1.602e-19/m), n)

    def initialize_x(self, n):
        return (self.domain.x_max-self.domain.dx/2)*np.ones(n)

    def initialize_y(self, n):
        return np.random.uniform(self.domain.y_min, self.domain.y_max, n)

    def initialize_vx(self, n):
        return -1*np.abs(self.maxwellian(self.T_init, self.mass, n))

    def initialize_vyz(self, n):
        return self.maxwellian(self.T_init, self.mass, n)

    def set_plasma_inds(self):
        active_x = self.x[0:self.max_ind][self.active[0:self.max_ind]]
        active_y = self.y[0:self.max_ind][self.active[0:self.max_ind]]
        self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]] = ((active_x-self.domain.x_min)/self.domain.dx).astype(np.int32)
        self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]] = ((active_y-self.domain.y_min)/self.domain.dy).astype(np.int32)

    def set_Te(self):
        self.Te[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.Te_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    def set_Ti(self):
        self.Ti[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.Ti_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    def set_n(self):
        self.n[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.n_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    def init_pos_vs(self):
        n_inflow = int(self.inflow_rate*self.domain.dt)
        inflow_percentage = np.ones(n_inflow)*self.inflow_rate*self.domain.dt/n_inflow
        return self.initialize_x(n_inflow), self.initialize_y(n_inflow), self.initialize_vx(n_inflow), self.initialize_vyz(n_inflow), self.initialize_vyz(n_inflow), inflow_percentage

    def inflow(self, xs, ys, vxs, vys, vzs, inflow_percentage):
        lv = self.vacant_indices.size
        n_inflow = xs.size
        if lv < n_inflow:
            if lv > 0:
                self.active[self.vacant_indices] = True
                self.x[self.vacant_indices] = xs[0:lv]
                self.y[self.vacant_indices] = ys[0:lv]
                self.born_x[self.vacant_indices] = xs[0:lv]
                self.born_y[self.vacant_indices] = ys[0:lv]
                self.vx[self.vacant_indices] = vxs[0:lv]
                self.vy[self.vacant_indices] = vys[0:lv]
                self.vz[self.vacant_indices] = vzs[0:lv]
                self.E[self.vacant_indices] = self.domain.kinetic_energy(vxs[0:lv], vys[0:lv], vzs[0:lv], self.mass)
                self.percentage[self.vacant_indices] = inflow_percentage[0:lv]
                self.vacant_indices = np.array([]).astype(np.int32)
            max_ind_new = self.max_ind+n_inflow-lv
            self.active[self.max_ind:(max_ind_new)] = True
            self.x[self.max_ind:(max_ind_new)] = xs[lv:]
            self.y[self.max_ind:(max_ind_new)] = ys[lv:]
            self.born_x[self.max_ind:(max_ind_new)] = xs[lv:]
            self.born_y[self.max_ind:(max_ind_new)] = ys[lv:]
            self.vx[self.max_ind:(max_ind_new)] = vxs[lv:]
            self.vy[self.max_ind:(max_ind_new)] = vys[lv:]
            self.vz[self.max_ind:(max_ind_new)] = vzs[lv:]
            self.E[self.max_ind:(max_ind_new)] = self.domain.kinetic_energy(vxs[lv:], vys[lv:], vzs[lv:], self.mass)
            self.percentage[self.max_ind:(max_ind_new)] = inflow_percentage[lv:]
            self.max_ind = max_ind_new
        else:
            self.active[self.vacant_indices[0:n_inflow]] = True
            self.x[self.vacant_indices[0:n_inflow]] = xs
            self.y[self.vacant_indices[0:n_inflow]] = ys
            self.born_x[self.vacant_indices[0:n_inflow]] = xs
            self.born_y[self.vacant_indices[0:n_inflow]] = ys
            self.vx[self.vacant_indices[0:n_inflow]] = vxs
            self.vy[self.vacant_indices[0:n_inflow]] = vys
            self.vz[self.vacant_indices[0:n_inflow]] = vzs
            self.E[self.vacant_indices[0:n_inflow]] = self.domain.kinetic_energy(vxs, vys, vzs, self.mass)
            self.percentage[self.vacant_indices[0:n_inflow]] = inflow_percentage
            self.vacant_indices = self.vacant_indices[n_inflow:]

    def rand_search_func(self, a):
        r = np.random.uniform(0, 1, 1)
        return np.searchsorted(a, r)

    def calc_interaction(self, probabilities):
        cum_probs = np.cumsum(probabilities[:, 0:self.max_ind], axis = 0)
        rands = np.random.uniform(0, 1, self.max_ind)
        inds = np.sum(cum_probs < rands, axis=0) + 1
        no_react = inds == (probabilities.shape[0]+1)
        inds[no_react] = 0
        return inds

#    def step(self, ne, T_e, T_i):
#        raise NotImplementedError("Step is not implemented for the species you are calling it on.")

    def translate(self):
        #Move particles
        self.x[0:self.max_ind][self.active[0:self.max_ind]] = self.x[0:self.max_ind][self.active[0:self.max_ind]]+self.vx[0:self.max_ind][self.active[0:self.max_ind]]*self.domain.dt
        self.y[0:self.max_ind][self.active[0:self.max_ind]] = self.y[0:self.max_ind][self.active[0:self.max_ind]]+self.vy[0:self.max_ind][self.active[0:self.max_ind]]*self.domain.dt
        self.z[0:self.max_ind][self.active[0:self.max_ind]] = self.z[0:self.max_ind][self.active[0:self.max_ind]]+self.vz[0:self.max_ind][self.active[0:self.max_ind]]*self.domain.dt
        #Periodic boundary in y
        self.y[0:self.max_ind][self.active[0:self.max_ind]] = (self.y[0:self.max_ind][self.active[0:self.max_ind]]-self.domain.y_min)%(self.domain.y_max - self.domain.y_min) + self.domain.y_min
        upper_boundary = self.y[0:self.max_ind][self.active[0:self.max_ind]] > self.domain.y_max
        #upper_boundary = np.multiply(upper_boundary, self.active[0:self.max_ind])
        self.born_y[0:self.max_ind][self.active[0:self.max_ind]][upper_boundary] = self.born_y[0:self.max_ind][self.active[0:self.max_ind]][upper_boundary] - (self.domain.y_max - self.domain.y_min)
        lower_boundary = self.y[0:self.max_ind][self.active[0:self.max_ind]] < self.domain.y_min
        #lower_boundary = np.multiply(lower_boundary, self.active[0:self.max_ind])
        self.born_y[0:self.max_ind][self.active[0:self.max_ind]][lower_boundary] = self.born_y[0:self.max_ind][self.active[0:self.max_ind]][lower_boundary] + (self.domain.y_max - self.domain.y_min)
        #Reflect at outer wall
        reflected_rands = np.random.uniform(0, 1, self.max_ind)
        reflected_mask = reflected_rands > self.absorbtion_coefficient
        outer_wall_mask = self.x[0:self.max_ind] > self.domain.x_max #Particles that move beyond the outer boundary
        #Remove those which are not reflected
        mask = (self.active[0:self.max_ind]) & (outer_wall_mask) & (reflected_mask != True)
        self.remove(mask)
        #Reflect those that are reflected
        mask = (self.active[0:self.max_ind]) & (outer_wall_mask) & (reflected_mask)
        self.born_x[0:self.max_ind][mask] = - self.born_x[0:self.max_ind][mask]
        self.x[0:self.max_ind][mask] = 2*self.domain.x_max - self.x[0:self.max_ind][mask]
        self.vx[0:self.max_ind][mask] = -self.vx[0:self.max_ind][mask]
        #If they move beyond lcfs assume ionized
        lcfs = self.x[0:self.max_ind] < self.domain.x_min
        self.remove(lcfs)

        #Deal with particles hitting the wall due to curvature of walls.
        #First deal with poloidal direction
        y_dist = np.abs(self.y[0:self.max_ind] - self.born_y[0:self.max_ind])
        x_from_wall = self.domain.x_max - self.x[0:self.max_ind][self.active[0:self.max_ind]] + self.x_max_wall_dist_poloidal
        placeholder = np.zeros(self.max_ind)
        placeholder[self.active[0:self.max_ind]] = np.sqrt(2*self.domain.r_minor*x_from_wall - np.power(x_from_wall, 2))
        outer_wall_mask = (self.active[0:self.max_ind]) & (y_dist > placeholder) & (self.vx[0:self.max_ind] < 0)
        mask = outer_wall_mask & (reflected_mask)
        self.bounce(mask)
        mask = outer_wall_mask & (reflected_mask != True)
        self.remove(mask)

        #Then do
        z_dist = np.abs(self.z[0:self.max_ind])
        x_from_wall = self.domain.x_max - self.x[0:self.max_ind][self.active[0:self.max_ind]]
        placeholder[self.active[0:self.max_ind]] = np.sqrt(2*self.domain.r_major*x_from_wall - np.power(x_from_wall, 2))
        outer_wall_mask = (self.active[0:self.max_ind]) & (z_dist > placeholder) & (self.vx[0:self.max_ind] < 0)
        mask = outer_wall_mask & (reflected_mask)
        self.bounce(mask)
        mask = outer_wall_mask & (reflected_mask != True)
        self.remove(mask)

    def bounce(self, bounced):
        self.x[0:self.max_ind][bounced] = self.born_x[0:self.max_ind][bounced]
        self.y[0:self.max_ind][bounced] = self.born_y[0:self.max_ind][bounced]
        self.z[0:self.max_ind][bounced] = 0

    def remove(self, removed):
        self.save_to_hist_free_path(removed)
        mi = self.max_ind
        self.active[0:mi][removed] = False
        self.x[0:mi][removed] = self.domain.x_max+self.domain.dx
        self.y[0:mi][removed] = self.domain.y_max+self.domain.dy
        self.vx[0:mi][removed] = 0
        self.vy[0:mi][removed] = 0
        self.vz[0:mi][removed] = 0
        self.percentage[0:mi][removed] = 0
        self.vacant_indices = np.append(self.vacant_indices, np.nonzero(removed)[0])

    def probs(self, rates):
        col_freq = np.multiply(self.n[0:self.max_ind], rates)*5e+19
        return 1 - np.exp(-col_freq*self.domain.dt)

    def get_probs_from_rates(self, rate_arr):
        #Get the total rate for each particle
        rates_tot = np.sum(rate_arr, axis = 0)
        #Get the total decay probability for each particle
        probs_tot = self.probs(rates_tot)
        #For each particle calculate the fractional rate compared to the total rate.
        #Multiply these fractional rate with the total probability.
        return (rate_arr/rates_tot)*probs_tot


    def probs_electron_collisions_lookup(self, table):
        inds = table.get_inds_T(self.Te[0:self.max_ind])
        rates = table.rates[inds]
        return rates

    def probs_heavy_collisions_lookup(self, table):
        inds_E = table.get_inds_E(self.E[0:self.max_ind])
        inds_T = table.get_inds_T(self.Ti[0:self.max_ind])
        rates = table.rates[inds_E, inds_T]
        return rates

    def probs_T_n(self, table):
        n_inds = table.get_inds_n(self.n[0:self.max_ind]*5e+19)
        T_inds = table.get_inds_T(self.Te[0:self.max_ind])
        rates = table.rates[n_inds, T_inds]
        return rates

    def step_body(self):
        new_xs, new_ys, new_vxs, new_vys, new_vzs, inflow_percentage = self.init_pos_vs()
        self.inflow(new_xs, new_ys, new_vxs, new_vys, new_vzs, inflow_percentage)
        self.domain.time_array[0] = time.time()
        self.set_plasma_inds()
        self.domain.time_array[1] = time.time()
        self.set_Te()
        self.set_Ti()
        self.set_n()
        self.domain.time_array[2] = time.time()
        self.get_probs()
        self.domain.time_array[3] = time.time()
        self.probs_arr[:, 0:self.max_ind] = self.get_probs_from_rates(self.probs_arr[:, 0:self.max_ind])
        self.domain.time_array[4] = time.time()
        specify_interaction = self.calc_interaction(self.probs_arr)
        self.domain.time_array[5] = time.time()
        specify_interaction[~self.active[0:self.max_ind]] = 0
        self.domain.time_array[6] = time.time()
        return specify_interaction

    def step(self):
        specify_interaction = self.step_body()
        self.do_interaction(specify_interaction)
        self.domain.time_array[7] = time.time()
        self.translate()
        self.domain.time_array[8] = time.time()
        self.domain.wall_times = self.domain.wall_times + np.diff(self.domain.time_array)

    def step_save(self):
        self.save_to_hist_x()
        self.save_to_hist_vs()
        specify_interaction = self.step_body()
        self.do_interaction_save(specify_interaction)
        self.domain.time_array[8] = time.time()
        self.domain.wall_times = self.domain.wall_times + np.diff(self.domain.time_array)

    def save_to_hist_x(self):
        hist_data, _ = np.histogram(self.x[self.active==1], bins=self.bin_edges_x)
        self.hist_x = self.hist_x+hist_data

    def save_to_hist_reaction(self, hist, reacted):
        hist_data, _ = np.histogram(self.x[0:self.max_ind][reacted], bins=self.bin_edges_x)
        return hist + hist_data

    def save_to_hist_reaction_temperature(self, hist, reacted):
        hist_data, _ = np.histogram(self.Te[0:self.max_ind][reacted], bins=self.bin_edges_T)
        return hist + hist_data

    def save_to_hist_free_path(self, removed):
        diff_x = self.x[0:self.max_ind][removed] - self.born_x[0:self.max_ind][removed]
        diff_y = self.y[0:self.max_ind][removed] - self.born_y[0:self.max_ind][removed]
        free_path = np.sqrt(np.power(diff_x, 2) + np.power(diff_y, 2))
        hist_data, _ = np.histogram(free_path, bins=self.bin_edges_free_path)
        return hist_data

    def sample_inds(self, cumsum_array):
        samples = cumsum_array.shape[0]
        rands = np.random.uniform(np.zeros((samples, 1)), np.reshape(cumsum_array[:, -1], (samples, 1)))
        inds = np.sum(cumsum_array < rands, axis=1) - 1
        lefts = np.reshape(rands, (samples)) - cumsum_array[np.arange(samples), inds]
        return inds, lefts

    def linear_spline(self, cumsum_array, inds, lefts, ds):
        diffs = cumsum_array[np.arange(cumsum_array.shape[0]), inds+1] - cumsum_array[np.arange(cumsum_array.shape[0]), inds]
        return ds*inds + ds*lefts/diffs

    def save_object(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
