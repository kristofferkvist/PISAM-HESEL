"""
A subclass of species implementing all the reactions specific for hydrogen/deuterium atoms.
This includes calculation of reactions rates, calculation of sources for reactions
that include atoms, and the changes to the atom superparticle velocity and/or weight
that a reaction might lead to.
"""

import numpy as np
from species import Species
import pickle
from make_tables import Tables_1d
from make_tables import Tables_2d
from make_tables import Tables_2d_T_n

class H_atom(Species):
    def __init__(self, inflow_rate, N_max, temperature, domain, table_dict, absorbtion_coefficient, min_weight, wall_boundary):
        #Call the init function of the parent class Species
        super().__init__(domain.d_ion_mass, inflow_rate, N_max, temperature, domain, absorbtion_coefficient, wall_boundary)
        #The number of reactions for atoms, not including ionization which is treated by weight reduction.
        self.num_react = 2
        #Associate the names of the included reactions with a unique integer in the range [1-n]
        self.CX, self.EXCITATION = range(1,self.num_react+1, 1)
        """
        If you want to add a reaction increase self.num_react by 1, and add the name of your reaction in the line above.
        Then implement the nessecary methods for "get_probs" and "do_interaction".
        """
        #Initiate the array holding the probabilities of each interaction
        #for each atom present in the domain.
        self.probs_arr = np.zeros((self.num_react, N_max))
        #Read the tables from the dictionary
        self.ion_table = table_dict['ion_rate']
        self.cx_table = table_dict['cx_rate']
        self.cx_dist_table = table_dict['cx_cross'] #Dimensions of the .table array represent T, E, v, theta_prime respectively
        self.excitation_rate = table_dict['1s_to_2p_rate']
        """
        cx_dist_table.table hold the cumsum along the axis of variyng velocity.
        self.cx_dist_table.table[T_ind, E_ind, -1, :] is thus the theta_prime dist
        at the relevant neutral energy and plasma temperature.
        We cumsum these (inclusive) to be able to sample theta_primes directly.
        Zeros are inserted before the cumsum rows for the linear interpolation
        to work.
        See the subsection "Charge Exchange of Deuterium Atoms and Deuterium Ions"
        in chapter 5 of my thesis, for an explanation of the 4D CX_dist_table.
        """
        self.vs_cumsum_table = np.insert(np.cumsum(self.cx_dist_table.table[:, :, -1, :], axis = 2), 0, 0, axis=2)
        #Velocity dist histogram
        #self.hist_vs_cx = np.zeros_like(self.hist_vs)
        #The species classe monitors the fields variables n, Te, Ti at the position of each neutral.
        #In addition the atoms uses the ion plasma fluid velocity.
        self.u0_x_ion = np.zeros(N_max).astype(np.float32)
        self.u0_y_ion = np.zeros(N_max).astype(np.float32)
        #######Diagnostics#########
        #Keep track of atoms that have undergone charge exchange for diagnostic purposes
        self.cxed = np.zeros(N_max).astype(np.bool_)
        #self.Sn_this_step = np.zeros(self.domain.plasma_dim_x)
        #self.Sn_this_step_cx = np.zeros(self.domain.plasma_dim_x)
        self.E_cx_gain = 0
        self.E_loss_ion = 0
        #######Diagnostics#########
        self.min_weight = min_weight

    def strip(self):
        active_old, mi_new, mi_old = super().strip()
        self.cxed[0:mi_new] = self.cxed[0:mi_old][active_old]
        self.cxed[mi_new:mi_old] = False

    #Override remove for atoms, to monitor cx'ed and not cx'ed atoms individually for diagnostic purposes
    def remove(self, removed):
        mi = self.max_ind
        self.cxed[0:mi][removed] = False
        #cx_removed_mask = removed & self.cxed[0:mi]
        #inds_x_cx = self.plasma_inds_x[0:mi][cx_removed_mask]
        #np.add.at(self.Sn_this_step_cx, inds_x_cx, -self.norm_weight[0:mi][cx_removed_mask])
        super().remove(removed)

    def absorb(self, absorbed):
        super().absorb(absorbed)
        self.cxed[0:self.max_ind][absorbed] = False

    #Set the fields values for ion plasma fluid velocity for each active atom.
    def set_u0_x_ion(self):
        self.u0_x_ion[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.u0_x_ion_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    def set_u0_y_ion(self):
        self.u0_y_ion[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.u0_y_ion_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    #Calculate the probability of CX for each atom
    def probs_chargeexchange(self):
        self.probs_arr[self.CX-1, 0:self.max_ind] = self.probs_heavy_collisions_lookup(self.cx_table)

    #Calculate the probability of excitation to 2p for each atom
    def probs_excitation_2p(self):
        self.probs_arr[self.EXCITATION-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.excitation_rate)

    #Wrapper for calculating reaction probs
    def get_probs(self):
        self.probs_chargeexchange()
        self.probs_excitation_2p()

    #Ionization procedure implemented through weight reduction
    def ionize(self):
        #Calculate the ionization probability
        rates_ionization = self.probs_T_n(self.ion_table)
        probs_ionization = self.probs(rates_ionization)
        #Calculate the number of norm_weight points lost
        norm_weight_loss = self.norm_weight[0:self.max_ind]*probs_ionization
        #Reduce norm_weight
        self.norm_weight[0:self.max_ind] = self.norm_weight[0:self.max_ind] - norm_weight_loss
        #Check for particles with a weight below min_weight
        mask = (self.norm_weight[0:self.max_ind] < self.min_weight) & self.active[0:self.max_ind]
        #Add the weights of these to the source calculations
        norm_weight_loss[mask] = norm_weight_loss[mask] + self.norm_weight[0:self.max_ind][mask]
        #Calc sources
        self.ion_sources(norm_weight_loss)
        #Remove the particles that are below the threshold
        self.remove(mask)

    #Calculate sources from ionizations and add them to the source arrays in domain
    def ion_sources(self, norm_weight_loss):
        inds_x = self.plasma_inds_x[0:self.max_ind]
        inds_y = self.plasma_inds_y[0:self.max_ind]
        #Add the sources at the relevant indices to the already existing sources of this timestep.
        #Using np.add.at rather than self.source[x_inds, y_inds] = self.source[x_inds, y_inds] + norm_weight
        #is important to allow for identical pairs of indices i.e. neutrals reacting in the same grid cell
        #of the plasma sim.
        self.domain.total_plasma_source = self.domain.total_plasma_source + np.sum(norm_weight_loss)
        np.add.at(self.domain.electron_source_particle, (inds_x, inds_y), norm_weight_loss)
        np.add.at(self.domain.ion_source_momentum_x, (inds_x, inds_y), self.domain.d_ion_mass*self.vx[0:self.max_ind]*norm_weight_loss)
        np.add.at(self.domain.ion_source_momentum_y, (inds_x, inds_y), self.domain.d_ion_mass*self.vy[0:self.max_ind]*norm_weight_loss)
        np.add.at(self.domain.electron_source_energy, (inds_x, inds_y), -13.6*norm_weight_loss)
        np.add.at(self.domain.ion_source_energy, (inds_x, inds_y), self.E[0:self.max_ind]*norm_weight_loss)
        self.E_loss_ion = self.E_loss_ion + np.sum(self.E[0:self.max_ind]*norm_weight_loss)
        #######Diagnostics#########
        #np.add.at(self.Sn_this_step, inds_x, -norm_weight_loss)
        #inds_x_cx = self.plasma_inds_x[0:self.max_ind][self.cxed[0:self.max_ind]]
        #norm_weight_loss_cx = norm_weight_loss[self.cxed[0:self.max_ind]]
        #np.add.at(self.Sn_this_step_cx, inds_x_cx, -norm_weight_loss_cx)
        #######Diagnostics#########

    #Helper methos for cx. It rotates the axis of a coordinate system theta_prime degrees around
    # the y-axis and phi degrees around the z-axis, in that order.
    def rotate_axes(self, vxp, vyp, vzp, phis, theta_primes):
        cos_phi = np.cos(phis)
        sin_phi = np.sin(phis)
        cos_theta_prime = np.cos(theta_primes)
        sin_theta_prime = np.sin(theta_primes)
        #I calculated the matrix product of R(phi)*R(theta), and then I "hard coded" the result for efficiency
        vxs = cos_phi*cos_theta_prime*vxp+vyp*sin_phi-vzp*cos_phi*sin_theta_prime
        vys = -sin_phi*cos_theta_prime*vxp+vyp*cos_theta_prime+vzp*sin_phi*cos_theta_prime
        vzs = sin_theta_prime*vxp+vzp*cos_theta_prime
        return vxs, vys, vzs

    #The purpose of this function is to sample the ion velocity of the ion going into
    #each of the individual cx reactions happening i the given time step. This sampling
    #depends on neutral velocity, ion temperature, and ion fluid velocity.
    #A mathematical account of the implementation of CX in this work is given
    #in the subsection "Ion-Neutral Collisions" in chapter 3 of my thesis.
    def cx(self, cxed):
        #######Diagnostics#########
        #Check which atoms that are cxed for the first time.
        #new_cx_mask = cxed & (self.cxed[0:self.max_ind] != True)
        #np.add.at(self.Sn_this_step_cx, self.plasma_inds_x[0:self.max_ind][new_cx_mask], self.norm_weight[0:self.max_ind][new_cx_mask])
        self.cxed[0:self.max_ind][cxed] = True
        #Change the birth place of the particle. This is used for mean free path diagnostics
        self.born_x[0:self.max_ind][cxed] = self.x[0:self.max_ind][cxed]
        self.born_y[0:self.max_ind][cxed] = self.y[0:self.max_ind][cxed]
        #######Diagnostics#########
        n_cxed = np.sum(cxed)
        #Get field values at cxed neutrals
        T_s_cxed = self.Ti[0:self.max_ind][cxed]
        self.set_u0_x_ion()
        self.set_u0_y_ion()
        u0_x_ion_cxed = self.u0_x_ion[0:self.max_ind][cxed]
        u0_y_ion_cxed = self.u0_y_ion[0:self.max_ind][cxed]
        #Get table indices for the relevant temperatures
        T_inds = self.cx_dist_table.get_inds_T(T_s_cxed)
        #Calculate the energy of the neutral in the rest frame of the ion fluid
        v_rel_x_s = self.vx[0:self.max_ind][cxed]-u0_x_ion_cxed
        v_rel_y_s = self.vy[0:self.max_ind][cxed]-u0_y_ion_cxed
        v_rel_square = np.power(v_rel_x_s, 2) + np.power(v_rel_y_s, 2) + np.power(self.vz[0:self.max_ind][cxed], 2)
        E_rel_cxed = 0.5*self.mass/(1.6022e-19)*v_rel_square
        #Get table indices for the relevant temperatures
        E_inds = self.cx_dist_table.get_inds_E(E_rel_cxed)
        #Sample theta_primes accroding to the distributions using a linear interpolations technique.
        #See the subsection "Charge Exchange of Deuterium Atoms and Deuterium Ions"
        #in chapter 5 of my thesis, for an explanation of the interpolation procedure.
        theta_prime_dist_cumsums = self.vs_cumsum_table[T_inds, E_inds, :]
        theta_prime_inds, theta_prime_lefts = self.sample_inds(theta_prime_dist_cumsums)
        theta_primes = self.linear_spline(theta_prime_dist_cumsums, theta_prime_inds, theta_prime_lefts, self.cx_dist_table.d_alpha)
        #Sample theta_primes accroding to the distributions using a linear interpolations technique.
        #Note that the evaluation velocoties of the cx dist table depends on the temperature.
        v_dist_cumsums = self.cx_dist_table.table[T_inds, E_inds, :, theta_prime_inds]
        v_inds, v_lefts = self.sample_inds(v_dist_cumsums)
        v_s = self.linear_spline(v_dist_cumsums, v_inds, v_lefts, self.cx_dist_table.dvs[T_inds])
        #Sample Phi uniformly.
        phis = np.random.uniform(0, 2*np.pi, n_cxed)
        #Velocities in primed coordinates
        vxp = v_s*np.sin(theta_primes)*np.cos(phis)
        vyp = v_s*np.sin(theta_primes)*np.sin(phis)
        vzp = v_s*np.cos(theta_primes)
        #/* Calculate the angles for the rotation */#
        #Avoid error in arccos due to floating point uncertaincies for relatvie velocities almost exactly parallel to the z-axis.
        vz_ratio = self.vz[0:self.max_ind][cxed]/np.sqrt(v_rel_square)
        vz_mask = vz_ratio <= -1
        vz_ratio[vz_mask] = -0.995
        vz_mask = vz_ratio >= 1
        vz_ratio[vz_mask] = 0.995
        theta0s = np.arccos(vz_ratio)
        phi0s = np.arctan2(v_rel_y_s, v_rel_x_s)
        #Rotate the primed coordinate axes back onto the lab frame - hence the minuses in the line below
        vxs, vys, vzs = self.rotate_axes(vxp, vyp, vzp, -phi0s, -theta0s)
        #Shift the velocity back to the rest frame of the lab.
        vxs = vxs + u0_x_ion_cxed
        vys = vys + u0_y_ion_cxed
        #Calc and add sources
        self.cx_sources(cxed, vxs, vys, vzs)
        #Update velocity and energy
        self.vx[0:self.max_ind][cxed] = vxs
        self.vy[0:self.max_ind][cxed] = vys
        self.vz[0:self.max_ind][cxed] = vzs
        self.E[0:self.max_ind][cxed] = self.domain.kinetic_energy(vxs, vys, vzs, self.mass)
        return theta_primes


    #Calculate sources from cx and add them to the source arrays in domain
    #Same ideas as in ion_sources().
    def cx_sources(self, cxed, vx_new, vy_new, vz_new):
        inds_x = self.plasma_inds_x[0:self.max_ind][cxed]
        inds_y = self.plasma_inds_y[0:self.max_ind][cxed]
        vx_old = self.vx[0:self.max_ind][cxed]
        vy_old = self.vy[0:self.max_ind][cxed]
        vz_old = self.vz[0:self.max_ind][cxed]
        np.add.at(self.domain.ion_source_momentum_x, (inds_x, inds_y), self.domain.d_ion_mass*(-vx_new + vx_old)*self.norm_weight[0:self.max_ind][cxed])
        np.add.at(self.domain.ion_source_momentum_y, (inds_x, inds_y), self.domain.d_ion_mass*(-vy_new + vy_old)*self.norm_weight[0:self.max_ind][cxed])
        np.add.at(self.domain.ion_source_energy, (inds_x, inds_y), (self.E[0:self.max_ind][cxed] - self.domain.kinetic_energy(vx_new, vy_new, vz_new, self.domain.d_ion_mass))*self.norm_weight[0:self.max_ind][cxed])
        self.E_cx_gain = self.E_cx_gain - np.sum((self.E[0:self.max_ind][cxed] - self.domain.kinetic_energy(vx_new, vy_new, vz_new, self.domain.d_ion_mass))*self.norm_weight[0:self.max_ind][cxed])

    #Calculate sources from excitations and add them to the source arrays in domain
    #This reaction is simple as it only acts as an electron heat sink. Nothing else.
    def excite(self, excited):
        inds_x = self.plasma_inds_x[0:self.max_ind][excited]
        inds_y = self.plasma_inds_y[0:self.max_ind][excited]
        np.add.at(self.domain.electron_source_energy, (inds_x, inds_y), -10.2*self.norm_weight[0:self.max_ind][excited])

    #Wrapper for performing the interactions
    def do_interaction(self, specify_interaction):
        cxed = specify_interaction == self.CX
        self.cx(cxed)
        excited = specify_interaction == self.EXCITATION
        self.excite(excited)
        self.ionize()

    """
    def save_to_hist_vs(self):
        for i in np.arange(self.velocity_domains):
            larger_mask = self.x[0:self.max_ind] > (self.domain.x_min + i*(self.domain.x_max-self.domain.x_min)/self.velocity_domains)
            smaller_mask = self.x[0:self.max_ind] < (self.domain.x_min + (i+1)*(self.domain.x_max-self.domain.x_min)/self.velocity_domains)
            mask = (larger_mask) & (smaller_mask) & (self.active[0:self.max_ind])# & (self.cxed[0:self.max_ind] == 1)
            velocities = np.sqrt(np.power(self.vx[0:self.max_ind][mask], 2) + np.power(self.vy[0:self.max_ind][mask], 2) + np.power(self.vz[0:self.max_ind][mask], 2))
            hist_data, _ = np.histogram(velocities, bins=self.bin_edges_vs)
            self.hist_vs[i, :] = self.hist_vs[i, :] + hist_data
            mask = (mask) & (self.cxed[0:self.max_ind] == 1)
            velocities = np.sqrt(np.power(self.vx[0:self.max_ind][mask], 2) + np.power(self.vy[0:self.max_ind][mask], 2) + np.power(self.vz[0:self.max_ind][mask], 2))
            hist_data, _ = np.histogram(velocities, bins=self.bin_edges_vs)
            self.hist_vs_cx[i, :] = self.hist_vs_cx[i, :] + hist_data

    def save_hists(self, nc_dat):
        #nc_dat.createDimension('hist_n', self.bin_edges_n.size-1)
        #nc_dat.createDimension('hist_E', self.bin_edges_E.size-1)
        nc_dat.createDimension('hist_T', self.bin_edges_T.size-1)
        nc_dat.createDimension('hist_T_bins', self.bin_edges_T.size)
        nc_dat.createDimension('hist_theta_prime', self.bin_edges_theta_prime.size-1)
        nc_dat.createDimension('hist_theta_prime_bins', self.bin_edges_theta_prime.size)
        #nc_dat.createDimension('hist_n_bins', self.bin_edges_n.size)
        #nc_dat.createDimension('hist_E_bins', self.bin_edges_E.size)
        #bin_edges_n = nc_dat.createVariable('bin_edges_n', 'float64', ('hist_n_bins'))
        #bin_edges_n[:] = self.hist_bin_edges_n
        #bin_edges_E = nc_dat.createVariable('bin_edges_E', 'float64', ('hist_E_bins'))
        #bin_edges_E[:] = self.hist_bin_edges_E
        bin_edges_T = nc_dat.createVariable('bin_edges_T', 'float32', ('hist_T_bins'))
        bin_edges_T[:] = self.bin_edges_T
        bin_edges_theta_prime = nc_dat.createVariable('bin_edges_theta_prime', 'float64', ('hist_theta_prime_bins'))
        bin_edges_theta_prime[:] = self.bin_edges_theta_prime
        hist_vs_cx = nc_dat.createVariable('hist_vs_cx', 'float32', ('hist_velocity_domains', 'hist_velocity'))
        hist_vs_cx[:] = self.hist_vs_cx
        hist_ion = nc_dat.createVariable('hist_ion', 'float32', ('hist_space'))
        hist_ion[:] = self.hist_ion
        hist_cx = nc_dat.createVariable('hist_cx', 'float32', ('hist_space'))
        hist_cx[:] = self.hist_cx
        hist_theta_prime = nc_dat.createVariable('hist_theta_prime', 'float32', ('hist_theta_prime'))
        hist_theta_prime[:] = self.hist_theta_prime
        hist_ion_temp = nc_dat.createVariable('hist_ion_temp', 'float32', ('hist_T'))
        hist_ion_temp[:] = self.hist_ion_temp
    """
