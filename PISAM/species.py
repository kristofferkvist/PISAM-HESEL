"""
The species class holds the minimal attributes required for each type of
particle added to the system e.g. mass, inflow rate, inflow temperature, etc.
Furthermore, it stores the positions, velocities, and surrounding temperatures
for each particle of the species it represents. It implements the step()
function called for each species in each time step. This function calls
all the processes making up a simulation step.
"""

import numpy as np
import time
from domain import Domain
import netCDF4 as nc
import pickle

class Species():
    def __init__(self, mass, inflow_rate, N_max, temperature_init, domain, absorbtion_coefficient, wall_boundary):
        #Array size for memory allocation
        self.N_max = N_max
        #An array of all vacant indices i.e. indices of non-active particles less than max_ind
        self.vacant_indices = np.array([]).astype(np.int32)
        #The largest indice of an active particle
        self.max_ind = 0
        #Boolean array. True for active particles False for inactive particles.
        self.active = np.zeros(N_max).astype(np.bool_)
        #Set relation to domain instance
        self.domain = domain
        #Neutral positions
        self.x = (domain.x_max+domain.dx)*np.ones(N_max)
        self.true_y = (domain.y_max+domain.dy)*np.ones(N_max)
        self.y = (domain.y_max+domain.dy)*np.ones(N_max)
        self.z = np.zeros(N_max)
        #Neutral velocities
        self.vx = np.zeros(N_max)
        self.vy = np.zeros(N_max)
        self.vz = np.zeros(N_max)
        #Neutral energies
        self.E = np.zeros(N_max).astype(np.float32)
        #Indices of neutrals on the HESEL grid.
        self.plasma_inds_x = np.ones(N_max).astype(np.int32)
        self.plasma_inds_y = np.ones(N_max).astype(np.int32)
        #The normalized weight of a particle. The normalization is with respect to the weight
        #calculated in the Manager class.
        self.norm_weight = np.zeros(N_max)
        self.mass = mass
        #Inflow rate of super particles of the relevant species
        self.inflow_rate = inflow_rate
        self.T_init = temperature_init
        #Birth places used to monitor mean free paths.
        self.born_x = np.zeros(N_max)
        self.born_y = np.zeros(N_max)
        #limit for wall boundary
        self.wall_lim = np.zeros(N_max)
        #Plasma fields seen by the individual particles
        self.Te = np.zeros(N_max).astype(np.float32)
        self.Ti = np.zeros(N_max).astype(np.float32)
        self.n = np.zeros(N_max).astype(np.float32)
        #Wall absorption coefficient
        self.absorbtion_coefficient = absorbtion_coefficient
        self.wall_boundary = wall_boundary
        #Test stuff
        self.total_inflow = 0
        self.inner_edge_loss = 0
        self.reflection_loss = 0
        self.E_loss_IE = 0
        self.E_init = 0
        self.E_absorb = 0
        #Some diagnostics not currently in use
        """
        self.bin_edges_x = bin_edges_x
        self.bin_edges_free_path = self.bin_edges_x*3
        self.bin_edges_vs = bin_edges_vs
        self.velocity_domains = velocity_domains
        self.hist_x = np.zeros(bin_edges_x.size-1)
        self.hist_vs = np.zeros((velocity_domains, bin_edges_vs.size-1))
        self.hist_free_path = self.hist_x = np.zeros(self.bin_edges_free_path.size-1)
        self.weights_add_wall = np.array([]).astype(np.float32)
        """

    #This functions compacts all the species arrays by removing the vacant indices.
    #Example:
    #Before strip:
    #self.x = np.array([2.1, 15, 15, 1.2, 5.1, 1.9, 15, 15, 15]) (15 is an arbitrary fill value for inactive particles)
    #self.active = np.array([True, False, False, True, True, True, False, False, False]).astype(np.bool_)
    #After strip:
    #self.x = [2.1  1.2  5.1  1.9 15.  15.  15.  15.  15. ]
    #self.active = [True  True  True  True False False False False False]
    def strip(self):
        mi_old = self.max_ind
        active_old = np.zeros(mi_old).astype(np.bool_) #This is used for species-specific arrays like cxed in h_atoms
        active_old[:] = self.active[0:mi_old]
        inactive = np.sum(~self.active[0:mi_old])
        self.max_ind = self.max_ind - inactive
        mi_new = self.max_ind
        self.x[0:mi_new] = self.x[0:mi_old][self.active[0:mi_old]]
        self.x[mi_new:mi_old] = (self.domain.x_max+self.domain.dx)

        self.true_y[0:mi_new] = self.true_y[0:mi_old][self.active[0:mi_old]]
        self.true_y[mi_new:mi_old] = (self.domain.y_max+self.domain.dy)

        self.y[0:mi_new] = self.y[0:mi_old][self.active[0:mi_old]]
        self.y[mi_new:mi_old] = (self.domain.y_max+self.domain.dy)

        self.z[0:mi_new] = self.z[0:mi_old][self.active[0:mi_old]]
        self.z[mi_new:mi_old] = 0

        self.vx[0:mi_new] = self.vx[0:mi_old][self.active[0:mi_old]]
        self.vx[mi_new:mi_old] = 0

        self.vy[0:mi_new] = self.vy[0:mi_old][self.active[0:mi_old]]
        self.vy[mi_new:mi_old] = 0

        self.vz[0:mi_new] = self.vz[0:mi_old][self.active[0:mi_old]]
        self.vz[mi_new:mi_old] = 0

        self.E[0:mi_new] = self.E[0:mi_old][self.active[0:mi_old]]
        self.E[mi_new:mi_old] = 0

        self.norm_weight[0:mi_new] = self.norm_weight[0:mi_old][self.active[0:mi_old]]
        self.norm_weight[mi_new:mi_old] = 0

        self.born_x[0:mi_new] = self.born_x[0:mi_old][self.active[0:mi_old]]
        self.born_x[mi_new:mi_old] = 0

        self.born_y[0:mi_new] = self.born_y[0:mi_old][self.active[0:mi_old]]
        self.born_y[mi_new:mi_old] = 0

        self.active[0:mi_new] = True
        self.active[mi_new:mi_old] = False
        self.vacant_indices = np.array([]).astype(np.int32)
        return active_old, mi_new, mi_old


    #Create maxwellian distributed velocities along one dimension according to temperature T
    def maxwellian(self, T, m, n):
        return np.random.normal(0, np.sqrt(T*self.domain.EV/m), n)

    #Place new particles just inside the domain in the x-direction
    def initialize_x(self, n):
        return (self.domain.x_max-self.domain.dx/100)*np.ones(n)

    #Initiate new particles uniformly in the y direction
    def initialize_y(self, n):
        return np.random.uniform(self.domain.y_min, self.domain.y_max, n)

    #Initial x-velocities must be positive
    def initialize_vx(self, n, T):
        return -1*np.abs(self.maxwellian(T, self.mass, n))

    def initialize_vyz(self, n, T):
        return self.maxwellian(T, self.mass, n)

    #Calculate and set the plasma indices of the HESEL grid corresponding to the
    #positions of the individual neutral particles.
    def set_plasma_inds(self):
        active_x = self.x[0:self.max_ind][self.active[0:self.max_ind]]
        active_y = self.y[0:self.max_ind][self.active[0:self.max_ind]]
        self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]] = ((active_x-self.domain.x_min)/self.domain.dx).astype(np.int32)
        self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]] = ((active_y-self.domain.y_min)/self.domain.dy).astype(np.int32)

    #Set the plasma fields of the HESEL corresponding to the
    #positions of the individual neutral particles.
    def set_Te(self):
        self.Te[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.Te_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    def set_Ti(self):
        self.Ti[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.Ti_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    def set_n(self):
        self.n[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.n_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    #Sample positions and velocities of injected particles.
    def init_pos_vs(self):
        n_inflow = int(self.inflow_rate*self.domain.dt)
        init_x = self.initialize_x(n_inflow)
        init_y = self.initialize_y(n_inflow)
        init_vx = self.initialize_vx(n_inflow, self.T_init)
        init_vy = self.initialize_vyz(n_inflow, self.T_init)
        init_vz = self.initialize_vyz(n_inflow, self.T_init)
        #Avoid rounding error by asserting an initial weight different from one (allways larger than one, but usually only a little)
        inflow_norm_weight = np.ones(n_inflow)*self.inflow_rate*self.domain.dt/n_inflow
        #print("dt = " + str(self.domain.dt))
        #print("n_inflow = " + str(n_inflow))
        #print("sum of norm weight = " + str(np.sum(inflow_norm_weight)))
        inflow_inds_x = (np.ones(n_inflow)*self.domain.plasma_dim_x-1).astype(np.int32)
        inflow_inds_y = ((init_y-self.domain.y_min)/self.domain.dy).astype(np.int32)
        return init_x, init_y, init_vx, init_vy, init_vz, inflow_norm_weight, inflow_inds_x, inflow_inds_y

    #The inflow method sets the relevant values of new particles. It does so at
    #vacant indices i.e. indices of inactive particles. If there are not enough vacant
    #indices available it adds particles by increasing self.max_ind
    def inflow(self, xs, ys, vxs, vys, vzs, inflow_norm_weight, inflow_inds_x, inflow_inds_y):
        #######Diagnostics#########
        #np.add.at(self.Sn_this_step, inflow_inds_x, inflow_norm_weight)
        self.total_inflow = self.total_inflow + np.sum(inflow_norm_weight)
        self.E_init = self.E_init + np.sum(self.domain.kinetic_energy(vxs, vys, vzs, self.mass)*inflow_norm_weight)
        #######Diagnostics#########
        lv = self.vacant_indices.size
        n_inflow = xs.size
        if lv < n_inflow:
            if lv > 0:
                self.active[self.vacant_indices] = True
                self.x[self.vacant_indices] = xs[0:lv]
                self.true_y[self.vacant_indices] = ys[0:lv]
                self.born_x[self.vacant_indices] = xs[0:lv]
                self.born_y[self.vacant_indices] = ys[0:lv]
                self.vx[self.vacant_indices] = vxs[0:lv]
                self.vy[self.vacant_indices] = vys[0:lv]
                self.vz[self.vacant_indices] = vzs[0:lv]
                self.E[self.vacant_indices] = self.domain.kinetic_energy(vxs[0:lv], vys[0:lv], vzs[0:lv], self.mass)
                self.norm_weight[self.vacant_indices] = inflow_norm_weight[0:lv]
                self.plasma_inds_x[self.vacant_indices] = inflow_inds_x[0:lv]
                self.plasma_inds_y[self.vacant_indices] = inflow_inds_y[0:lv]
                self.vacant_indices = np.array([]).astype(np.int32)
            max_ind_new = self.max_ind+n_inflow-lv
            self.active[self.max_ind:(max_ind_new)] = True
            self.x[self.max_ind:(max_ind_new)] = xs[lv:]
            self.true_y[self.max_ind:(max_ind_new)] = ys[lv:]
            self.born_x[self.max_ind:(max_ind_new)] = xs[lv:]
            self.born_y[self.max_ind:(max_ind_new)] = ys[lv:]
            self.vx[self.max_ind:(max_ind_new)] = vxs[lv:]
            self.vy[self.max_ind:(max_ind_new)] = vys[lv:]
            self.vz[self.max_ind:(max_ind_new)] = vzs[lv:]
            self.E[self.max_ind:(max_ind_new)] = self.domain.kinetic_energy(vxs[lv:], vys[lv:], vzs[lv:], self.mass)
            self.norm_weight[self.max_ind:(max_ind_new)] = inflow_norm_weight[lv:]
            self.plasma_inds_x[self.max_ind:(max_ind_new)] = inflow_inds_x[lv:]
            self.plasma_inds_y[self.max_ind:(max_ind_new)] = inflow_inds_y[lv:]
            self.max_ind = max_ind_new
        else:
            self.active[self.vacant_indices[0:n_inflow]] = True
            self.x[self.vacant_indices[0:n_inflow]] = xs
            self.true_y[self.vacant_indices[0:n_inflow]] = ys
            self.born_x[self.vacant_indices[0:n_inflow]] = xs
            self.born_y[self.vacant_indices[0:n_inflow]] = ys
            self.vx[self.vacant_indices[0:n_inflow]] = vxs
            self.vy[self.vacant_indices[0:n_inflow]] = vys
            self.vz[self.vacant_indices[0:n_inflow]] = vzs
            self.E[self.vacant_indices[0:n_inflow]] = self.domain.kinetic_energy(vxs, vys, vzs, self.mass)
            self.norm_weight[self.vacant_indices[0:n_inflow]] = inflow_norm_weight
            self.plasma_inds_x[self.vacant_indices[0:n_inflow]] = inflow_inds_x
            self.plasma_inds_y[self.vacant_indices[0:n_inflow]] = inflow_inds_y
            self.vacant_indices = self.vacant_indices[n_inflow:]

    #Move the particles accordin to their velocities and apply boundary conditions
    def translate(self):
        #Move particles
        self.x[0:self.max_ind][self.active[0:self.max_ind]] = self.x[0:self.max_ind][self.active[0:self.max_ind]]+self.vx[0:self.max_ind][self.active[0:self.max_ind]]*self.domain.dt
        self.true_y[0:self.max_ind][self.active[0:self.max_ind]] = self.true_y[0:self.max_ind][self.active[0:self.max_ind]]+self.vy[0:self.max_ind][self.active[0:self.max_ind]]*self.domain.dt
        #self.y[0:self.max_ind][self.active[0:self.max_ind]] = self.y[0:self.max_ind][self.active[0:self.max_ind]]+self.vy[0:self.max_ind][self.active[0:self.max_ind]]*self.domain.dt
        self.z[0:self.max_ind][self.active[0:self.max_ind]] = self.z[0:self.max_ind][self.active[0:self.max_ind]]+self.vz[0:self.max_ind][self.active[0:self.max_ind]]*self.domain.dt
        #Periodic boundary in y
        self.y[0:self.max_ind][self.active[0:self.max_ind]] = (self.true_y[0:self.max_ind][self.active[0:self.max_ind]]-self.domain.y_min)%(self.domain.y_max - self.domain.y_min) + self.domain.y_min
        #Reflect at outer wall
        reflected_rands = np.random.uniform(0, 1, self.max_ind)
        reflected_mask = reflected_rands > self.absorbtion_coefficient
        outer_wall_mask = self.x[0:self.max_ind] > self.domain.x_max #Particles that move beyond the outer boundary
        #Remove those which are not reflected
        mask = (self.active[0:self.max_ind]) & (outer_wall_mask) & (reflected_mask != True)
        self.reflection_loss = self.reflection_loss + np.sum(self.norm_weight[0:self.max_ind][mask])
        self.E_absorb = self.E_absorb + np.sum(self.E[0:self.max_ind][mask]*self.norm_weight[0:self.max_ind][mask])
        self.remove(mask)
        #Reflect those that are reflected
        mask = (self.active[0:self.max_ind]) & (outer_wall_mask) & (reflected_mask)
        self.born_x[0:self.max_ind][mask] = - self.born_x[0:self.max_ind][mask]
        self.x[0:self.max_ind][mask] = 2*self.domain.x_max - self.x[0:self.max_ind][mask]
        self.vx[0:self.max_ind][mask] = -self.vx[0:self.max_ind][mask]
        #If they move beyond lcfs assume ionized
        lcfs = self.x[0:self.max_ind] < self.domain.x_min
        self.inner_edge_loss = self.inner_edge_loss + np.sum(self.norm_weight[0:self.max_ind][lcfs])
        self.E_loss_IE = self.E_loss_IE + np.sum(self.E[0:self.max_ind][lcfs]*self.norm_weight[0:self.max_ind][lcfs])
        self.remove(lcfs)

        if self.wall_boundary:
            #Deal with particles hitting the wall due to curvature of walls.
            #First deal with poloidal direction
            x_from_wall = self.domain.x_max - self.x[0:self.max_ind] + self.domain.x_max_wall_dist_poloidal
            #y_lim = np.zeros(self.max_ind)
            self.wall_lim[0:self.max_ind] = np.sqrt(2*self.domain.r_minor*x_from_wall - np.power(x_from_wall, 2))
            outer_wall_mask = (self.active[0:self.max_ind]) & (np.logical_or(self.true_y[0:self.max_ind] > self.wall_lim[0:self.max_ind], self.true_y[0:self.max_ind] < -self.wall_lim[0:self.max_ind])) & (self.vx[0:self.max_ind] < 0)
            #mask = outer_wall_mask & reflected_mask
            #self.bounce_y(mask, x_from_wall, self.wall_lim[0:self.max_ind])
            mask = outer_wall_mask & (reflected_mask != True)
            self.absorb(mask)

            #Thn the toroidal direction
            self.wall_lim[0:self.max_ind] = np.sqrt(2*self.domain.r_minor*x_from_wall - np.power(x_from_wall, 2))
            outer_wall_mask = (self.active[0:self.max_ind]) & (np.logical_or(self.z[0:self.max_ind] > self.wall_lim[0:self.max_ind], self.z[0:self.max_ind] < -self.wall_lim[0:self.max_ind])) & (self.vx[0:self.max_ind] < 0)
            #mask = outer_wall_mask & reflected_mask
            #self.bounce_z(mask, x_from_wall, self.wall_lim[0:self.max_ind])
            mask = outer_wall_mask & (reflected_mask != True)
            self.absorb(mask)

    """#Perform the reflection and reset the relevant values
    def bounce_y(self, bounced, x_from_wall, y_lim):
        self.bounce(bounced, x_from_wall, y_lim)
        new_x = self.initialize_x(np.sum(bounced))
        new_y = self.initialize_y(np.sum(bounced))
        self.x[0:self.max_ind][bounced] = new_x
        self.true_y[0:self.max_ind][bounced] = new_y
        self.y[0:self.max_ind][bounced] = new_y
        self.born_x[0:self.max_ind][bounced] = new_x
        self.born_y[0:self.max_ind][bounced] = new_y

    #Perform the reflection and reset the relevant values
    def bounce_z(self, bounced, x_from_wall, y_lim):
        self.bounce(bounced, x_from_wall, y_lim)
        new_x = self.initialize_x(np.sum(bounced))
        self.x[0:self.max_ind][bounced] = new_x
        self.z[0:self.max_ind][bounced] = 0
        self.born_x[0:self.max_ind][bounced] = new_x
        self.born_y[0:self.max_ind][bounced] = self.y[0:self.max_ind][bounced]

    #This method perform the actual 3D relfection
    def bounce(self, bounced, x_from_wall, y_lim):
        n_bounced = np.sum(bounced)
        x_from_magnetic_axis = self.domain.r_minor-x_from_wall[bounced]
        y_bounce = y_lim[bounced]*np.sign(self.y[0:self.max_ind][bounced])
        #Let the polar angle of the bounce be defined from the ratio of the
        #x and y coordinates, in a manner such the n_hat i always normalized.
        #The latter is a requirement for energy conservation!
        tan_phi = y_bounce/x_from_magnetic_axis
        cos_phi = 1/np.sqrt(np.power(tan_phi, 2) + 1)
        sin_phi = tan_phi/np.sqrt(np.power(tan_phi, 2) + 1)
        theta = self.z[0:self.max_ind][bounced]/(self.domain.r_major + self.domain.r_minor)
        n_hat_x = -cos_phi*np.cos(theta)
        n_hat_y = -sin_phi
        n_hat_z = -cos_phi*np.sin(theta)
        #Calculate the projection of the current velocity onto the normal vector
        v_dot_n_hat = self.vx[0:self.max_ind][bounced]*n_hat_x + self.vy[0:self.max_ind][bounced]*n_hat_y + self.vz[0:self.max_ind][bounced]*n_hat_z
        proj_x = v_dot_n_hat*n_hat_x
        proj_y = v_dot_n_hat*n_hat_y
        proj_z = v_dot_n_hat*n_hat_z
        #Calculate the new velocities by v_reflected = v_old - 2*v_proj
        v_mag_before = np.sqrt(np.power(self.vx[0:self.max_ind][bounced], 2) + np.power(self.vy[0:self.max_ind][bounced], 2) + np.power(self.vz[0:self.max_ind][bounced], 2))
        self.vx[0:self.max_ind][bounced] = self.vx[0:self.max_ind][bounced] - 2*proj_x
        self.vy[0:self.max_ind][bounced] = self.vy[0:self.max_ind][bounced] - 2*proj_y
        self.vz[0:self.max_ind][bounced] = self.vz[0:self.max_ind][bounced] - 2*proj_z
        v_mag_after = np.sqrt(np.power(self.vx[0:self.max_ind][bounced], 2) + np.power(self.vy[0:self.max_ind][bounced], 2) + np.power(self.vz[0:self.max_ind][bounced], 2))
        np.testing.assert_allclose(v_mag_before, v_mag_after)"""

    #Particles that are absorbed due to the "wall" condition are reemitted at wall temperature
    def absorb(self, absorbed):
        n_absorbed = np.sum(absorbed)
        new_x = self.initialize_x(n_absorbed)
        new_y = self.initialize_y(n_absorbed)
        self.x[0:self.max_ind][absorbed] = new_x
        self.true_y[0:self.max_ind][absorbed] = new_y
        self.y[0:self.max_ind][absorbed] = new_y
        self.z[0:self.max_ind][absorbed] = 0
        self.born_x[0:self.max_ind][absorbed] = new_x
        self.born_y[0:self.max_ind][absorbed] = new_y
        self.vx[0:self.max_ind][absorbed] = self.initialize_vx(n_absorbed, self.domain.T_wall)
        self.vy[0:self.max_ind][absorbed] = self.initialize_vyz(n_absorbed, self.domain.T_wall)
        self.vz[0:self.max_ind][absorbed] = self.initialize_vyz(n_absorbed, self.domain.T_wall)

    #Removes particles when they escape the domain or react to create new particles
    def remove(self, removed):
        #self.save_to_hist_free_path(removed)
        mi = self.max_ind
        inds_x = self.plasma_inds_x[0:mi][removed]
        #np.add.at(self.Sn_this_step, inds_x, -self.norm_weight[0:mi][removed])
        self.active[0:mi][removed] = False
        self.x[0:mi][removed] = self.domain.x_max+self.domain.dx
        self.y[0:mi][removed] = self.domain.y_max+self.domain.dy
        self.true_y[0:mi][removed] = 0
        self.vx[0:mi][removed] = 0
        self.vy[0:mi][removed] = 0
        self.vz[0:mi][removed] = 0
        self.norm_weight[0:mi][removed] = 0
        self.vacant_indices = np.append(self.vacant_indices, np.nonzero(removed)[0])

    #Calculate probability from reaction rate.
    #See the subsection Calculating Decay Probabilities from Decay Frequencies of
    #chapter 6 in my thesis for details
    def probs(self, rates):
        col_freq = np.multiply(self.n[0:self.max_ind], rates)*5e+19
        return 1 - np.exp(-col_freq*self.domain.dt)

    #Calculate the probabilities of each individual reaction.
    #See the subsection Calculating Decay Probabilities from Decay Frequencies of
    #chapter 6 in my thesis for details
    def get_probs_from_rates(self, rate_arr):
        #Get the total rate for each particle
        rates_tot = np.sum(rate_arr, axis = 0)
        #Get the total decay probability for each particle
        probs_tot = self.probs(rates_tot)
        #For each particle calculate the fractional rate compared to the total rate.
        #Multiply these fractional rate with the total probability.
        return (rate_arr/rates_tot)*probs_tot

    #Sample what reactions happen given the array of probabilities for each reaction.
    def calc_interaction(self, probabilities):
        cum_probs = np.cumsum(probabilities[:, 0:self.max_ind], axis = 0)
        rands = np.random.uniform(0, 1, self.max_ind)
        inds = np.sum(cum_probs < rands, axis=0) + 1
        no_react = inds == (probabilities.shape[0]+1)
        inds[no_react] = 0
        return inds

    #Lookup reation rates for collisions depending only on electron temperature
    def probs_electron_collisions_lookup(self, table):
        inds = table.get_inds_T(self.Te[0:self.max_ind])
        rates = table.rates[inds]
        return rates

    #Lookup reation rates for collisions depending on neutral energy and ion temperature
    def probs_heavy_collisions_lookup(self, table):
        inds_E = table.get_inds_E(self.E[0:self.max_ind])
        inds_T = table.get_inds_T(self.Ti[0:self.max_ind])
        rates = table.rates[inds_E, inds_T]
        return rates

    #Lookup reation rates for collisions depending on plasma density and electron temperature
    def probs_T_n(self, table):
        n_inds = table.get_inds_n(self.n[0:self.max_ind]*5e+19)
        T_inds = table.get_inds_T(self.Te[0:self.max_ind])
        rates = table.rates[n_inds, T_inds]
        return rates

    #This method samples from the cumulative sum of a distribution.
    #It return the sampled index as well as the "overshoot" which is then used
    #in the linear_spline() method below.
    #See the subsection "Charge Exchange of Deuterium Atoms and Deuterium
    #Ions" in chapter 5 of my thesis for details on the sampling procedure.
    def sample_inds(self, cumsum_array):
        samples = cumsum_array.shape[0]
        rands = np.random.uniform(np.zeros((samples, 1)), np.reshape(cumsum_array[:, -1], (samples, 1)))
        inds = np.sum(cumsum_array < rands, axis=1) - 1
        lefts = np.reshape(rands, (samples)) - cumsum_array[np.arange(samples), inds]
        return inds, lefts

    def linear_spline(self, cumsum_array, inds, lefts, ds):
        diffs = cumsum_array[np.arange(cumsum_array.shape[0]), inds+1] - cumsum_array[np.arange(cumsum_array.shape[0]), inds]
        return ds*inds + ds*lefts/diffs

    #The all important step function wrapping all the processes of a full timestep
    #of the species in question. It records the wall time of the different processes.
    def step(self):
        self.domain.time_array[0] = time.time()
        init_x, init_y, init_vx, init_vy, init_vz, inflow_norm_weight, inflow_inds_x, inflow_inds_y = self.init_pos_vs()
        self.inflow(init_x, init_y, init_vx, init_vy, init_vz, inflow_norm_weight, inflow_inds_x, inflow_inds_y)
        self.domain.time_array[1] = time.time()
        if (np.sum(self.active[0:self.max_ind]) > 1 and self.vacant_indices.size/np.sum(self.active[0:self.max_ind]) > 0.05):
            self.strip()
        self.domain.time_array[2] = time.time()
        self.set_Te()
        self.set_Ti()
        self.set_n()
        self.domain.time_array[3] = time.time()
        self.get_probs()
        self.domain.time_array[4] = time.time()
        self.probs_arr[:, 0:self.max_ind] = self.get_probs_from_rates(self.probs_arr[:, 0:self.max_ind])
        self.domain.time_array[5] = time.time()
        specify_interaction = self.calc_interaction(self.probs_arr)
        self.domain.time_array[6] = time.time()
        specify_interaction[~self.active[0:self.max_ind]] = 0
        self.domain.time_array[7] = time.time()
        self.do_interaction(specify_interaction)
        self.domain.time_array[8] = time.time()
        self.translate()
        self.domain.time_array[9] = time.time()
        self.set_plasma_inds()
        self.domain.time_array[10] = time.time()
        self.domain.wall_times = self.domain.wall_times + np.diff(self.domain.time_array)

    def save_to_hist_free_path(self, removed):
        diff_x = self.x[0:self.max_ind][removed] - self.born_x[0:self.max_ind][removed]
        diff_y = self.y[0:self.max_ind][removed] - self.born_y[0:self.max_ind][removed]
        free_path = np.sqrt(np.power(diff_x, 2) + np.power(diff_y, 2))
        hist_data, _ = np.histogram(free_path, bins=self.bin_edges_free_path)
        return hist_data

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
