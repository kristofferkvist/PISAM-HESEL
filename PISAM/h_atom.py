import numpy as np
from species import Species
import pickle
from make_tables import Tables_1d
from make_tables import Tables_2d
from make_tables import Tables_2d_T_n

class H_atom(Species):
    def __init__(self, inflow_rate, N_max, temperature, domain, table_dict, bin_edges_x, bin_edges_vs, velocity_domains, bin_edges_T, absorbtion_coefficient):
        super().__init__(domain.d_ion_mass, inflow_rate, N_max, temperature, domain, absorbtion_coefficient, bin_edges_x, bin_edges_vs, velocity_domains)
        self.num_react = 2
        self.CX = 1
        self.EXCITATION = 2
        self.probs_arr = np.zeros((self.num_react, N_max))
        self.ion_table = table_dict['ion_rate']
        self.cx_table = table_dict['cx_rate']
        self.cx_dist_table = table_dict['cx_cross'] #Dimensions of the .table array represent T, E, v, alpha respectively
        self.excitation_rate = table_dict['1s_to_2p_rate']
        self.vs_cumsum_table = np.insert(np.cumsum(self.cx_dist_table.table[:, :, -1, :], axis = 2), 0, 0, axis=2)
        self.bin_edges_T = bin_edges_T
        self.hist_vs_cx = np.zeros_like(self.hist_vs)
        self.hist_ion = np.zeros(bin_edges_x.size-1)
        self.hist_cx = np.zeros(bin_edges_x.size-1)
        self.bin_edges_alpha = np.linspace(0, np.pi, 100)
        self.hist_alpha = np.zeros(self.bin_edges_alpha.size-1)
        self.hist_ion_temp = np.zeros(bin_edges_T.size-1)
        self.cxed = np.zeros(N_max).astype(np.bool_)
        self.u0_x_ion = np.zeros(N_max).astype(np.float32)
        self.u0_y_ion = np.zeros(N_max).astype(np.float32)

    def set_u0_x_ion(self):
        self.u0_x_ion[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.u0_x_ion_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    def set_u0_y_ion(self):
        self.u0_y_ion[0:self.max_ind][self.active[0:self.max_ind]] = self.domain.u0_y_ion_mesh[self.plasma_inds_x[0:self.max_ind][self.active[0:self.max_ind]], self.plasma_inds_y[0:self.max_ind][self.active[0:self.max_ind]]]

    def probs_ionization(self):
        rates_ionization = self.probs_T_n(self.ion_table)
        probs_ionization = self.probs(rates_ionization)
        percentage_loss = self.percentage[0:self.max_ind]*probs_ionization
        self.percentage[0:self.max_ind] = self.percentage[0:self.max_ind] - percentage_loss
        self.ion_sources(percentage_loss)

    def probs_chargeexchange(self):
        self.probs_arr[self.CX-1, 0:self.max_ind] = self.probs_heavy_collisions_lookup(self.cx_table)

    def probs_excitation_2p(self):
        self.probs_arr[self.EXCITATION-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.excitation_rate)

    def get_probs(self):
        self.probs_ionization()
        self.probs_chargeexchange()
        self.probs_excitation_2p()

    def ion_sources(self, percentage_loss):
        inds_x = self.plasma_inds_x[0:self.max_ind]
        inds_y = self.plasma_inds_y[0:self.max_ind]
        self.domain.electron_source_particle[inds_x, inds_y] = self.domain.electron_source_particle[inds_x, inds_y] + percentage_loss
        self.domain.ion_source_momentum_x[inds_x, inds_y] = self.domain.ion_source_momentum_x[inds_x, inds_y] + self.domain.d_ion_mass*self.vx[0:self.max_ind]*percentage_loss
        self.domain.ion_source_momentum_y[inds_x, inds_y] = self.domain.ion_source_momentum_y[inds_x, inds_y] + self.domain.d_ion_mass*self.vy[0:self.max_ind]*percentage_loss
        self.domain.electron_source_energy[inds_x, inds_y] = self.domain.electron_source_energy[inds_x, inds_y] + (self.E[0:self.max_ind]*self.domain.electron_mass/self.domain.d_ion_mass - 13.6)*percentage_loss
        self.domain.ion_source_energy[inds_x, inds_y] = self.domain.ion_source_energy[inds_x, inds_y] + self.E[0:self.max_ind]*percentage_loss

    def cx_sources(self, cxed, vx_new, vy_new, vz_new):
        inds_x = self.plasma_inds_x[0:self.max_ind][cxed]
        inds_y = self.plasma_inds_y[0:self.max_ind][cxed]
        vx_old = self.vx[0:self.max_ind][cxed]
        vy_old = self.vy[0:self.max_ind][cxed]
        vz_old = self.vz[0:self.max_ind][cxed]
        self.domain.ion_source_momentum_x[inds_x, inds_y] = self.domain.ion_source_momentum_x[inds_x, inds_y] + self.domain.d_ion_mass*(-vx_new + vx_old)*self.percentage[0:self.max_ind][cxed]
        self.domain.ion_source_momentum_y[inds_x, inds_y] = self.domain.ion_source_momentum_y[inds_x, inds_y] + self.domain.d_ion_mass*(-vy_new + vy_old)*self.percentage[0:self.max_ind][cxed]
        self.domain.ion_source_energy[inds_x, inds_y] = self.domain.ion_source_energy[inds_x, inds_y] + (self.E[0:self.max_ind][cxed] - self.domain.kinetic_energy(vx_new, vy_new, vz_new, self.domain.d_ion_mass))*self.percentage[0:self.max_ind][cxed]

    #This function rotates the axis of a coordinate system alpha degrees around
    # the y-axis and phi degrees around the z-axis, in that order.
    def rotate_axes(self, vxp, vyp, vzp, phis, alphas):
        cos_phi = np.cos(phis)
        sin_phi = np.sin(phis)
        cos_alpha = np.cos(alphas)
        sin_alpha = np.sin(alphas)
        #I calculated the matrix product of R(phi)*R(theta), and then I "hard coded" the result for efficiency
        vxs = cos_phi*cos_alpha*vxp+vyp*sin_phi-vzp*cos_phi*sin_alpha
        vys = -sin_phi*cos_alpha*vxp+vyp*cos_alpha+vzp*sin_phi*cos_alpha
        vzs = sin_alpha*vxp+vzp*cos_alpha
        return vxs, vys, vzs

    def cx(self, cxed):
        self.cxed[0:self.max_ind][cxed] = True
        self.cxed[0:self.max_ind] = np.multiply(self.cxed[0:self.max_ind], self.active[0:self.max_ind])
        self.born_x[0:self.max_ind][cxed] = self.x[0:self.max_ind][cxed]
        self.born_y[0:self.max_ind][cxed] = self.y[0:self.max_ind][cxed]
        n_cxed = np.sum(cxed)
        T_s_cxed = self.Ti[0:self.max_ind][cxed]
        u0_x_s_cxed = self.u0_x_ion[0:self.max_ind][cxed]
        u0_y_s_cxed = self.u0_y_ion[0:self.max_ind][cxed]
        T_inds = self.cx_dist_table.get_inds_T(T_s_cxed) #Incidices are the last value in table Ts that is smaller than the T you are searching the indicie for. This could be refined
        #Calculate the energy of the neutral in the rest frame of the ion fluid
        self.set_u0_x_ion()
        self.set_u0_y_ion()
        u0_x_ion_cxed = self.u0_x_ion[0:self.max_ind][cxed]
        u0_y_ion_cxed = self.u0_y_ion[0:self.max_ind][cxed]
        v_rel_x_s = self.vx[0:self.max_ind][cxed]-u0_x_ion_cxed
        v_rel_y_s = self.vy[0:self.max_ind][cxed]-u0_y_ion_cxed
        v_rel_square = np.power(v_rel_x_s, 2) + np.power(v_rel_y_s, 2) + np.power(self.vz[0:self.max_ind][cxed], 2)
        E_rel_cxed = 0.5*self.mass/(1.6022e-19)*v_rel_square
        E_inds = self.cx_dist_table.get_inds_E(E_rel_cxed) #Same as for T
        cumsum_vs_at_inds = self.vs_cumsum_table[T_inds, E_inds, :]
        alpha_inds, alpha_lefts = self.sample_inds(cumsum_vs_at_inds)
        alphas = self.linear_spline(cumsum_vs_at_inds, alpha_inds, alpha_lefts, self.cx_dist_table.d_alpha)
        cumsum_alphas = self.cx_dist_table.table[T_inds, E_inds, :, alpha_inds]
        v_inds, v_lefts = self.sample_inds(cumsum_alphas)
        v_s = self.linear_spline(cumsum_alphas, v_inds, v_lefts, self.cx_dist_table.dvs[T_inds])
        phis = np.random.uniform(0, 2*np.pi, n_cxed)
        #Velocities in primed coordinates
        vxp = v_s*np.sin(alphas)*np.cos(phis)
        vyp = v_s*np.sin(alphas)*np.sin(phis)
        vzp = v_s*np.cos(alphas)
        #/* Calculate the angles for the rotation */#
        #Avoid error in arccos due to floating point uncertaincies for relatvie velocities almost exactly parallel to the z-axis.
        vz_ratio = self.vz[0:self.max_ind][cxed]/np.sqrt(v_rel_square)
        vz_mask = vz_ratio <= -1
        vz_ratio[vz_mask] = -0.995
        vz_mask = vz_ratio >= 1
        vz_ratio[vz_mask] = 0.995
        theta0s = np.arccos(vz_ratio)
        phi0s = np.arctan2(v_rel_y_s, v_rel_x_s)
        #I rotate the primed coordinate axes back onto the lab frame - hence the minuses in the line below
        vxs, vys, vzs = self.rotate_axes(vxp, vyp, vzp, -phi0s, -theta0s)
        #I also shift the velocity back to the rest frame of the lab.
        vxs = vxs + u0_x_ion_cxed
        vys = vys + u0_y_ion_cxed
        self.cx_sources(cxed, vxs, vys, vzs)
        self.vx[0:self.max_ind][cxed] = vxs
        self.vy[0:self.max_ind][cxed] = vys
        self.vz[0:self.max_ind][cxed] = vzs
        self.E[0:self.max_ind][cxed] = self.domain.kinetic_energy(vxs, vys, vzs, self.mass)
        return alphas

    #def cx_save(self, cxed):
    #    self.cxed_save_x, self.cxed_save_y, self.n_cxed = self.save(self.cxed_save_x, self.cxed_save_y, self.n_cxed, cxed)
    #    self.cx(cxed)

    def cx_save(self, cxed):
        self.hist_cx = self.save_to_hist_reaction(self.hist_cx, cxed)
        alphas = self.cx(cxed)
        alpha_hist, _ = np.histogram(alphas, self.bin_edges_alpha)
        self.hist_alpha = self.hist_alpha + alpha_hist

    def excite(self, excited):
        inds_x = self.plasma_inds_x[0:self.max_ind][excited]
        inds_y = self.plasma_inds_y[0:self.max_ind][excited]
        self.domain.electron_source_energy[inds_x, inds_y] = self.domain.electron_source_energy[inds_x, inds_y] - 10.2*self.percentage[0:self.max_ind][excited]

    def do_interaction(self, specify_interaction):
        cxed = specify_interaction == self.CX
        self.cx(cxed)
        excited = specify_interaction == self.EXCITATION
        self.excite(excited)

    def do_interaction_save(self, specify_interaction):
        cxed = specify_interaction == self.CX
        self.cx_save(cxed)
        excited = specify_interaction == self.EXCITATION
        self.excite(excited)

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
        nc_dat.createDimension('hist_alpha', self.bin_edges_alpha.size-1)
        nc_dat.createDimension('hist_alpha_bins', self.bin_edges_alpha.size)
        #nc_dat.createDimension('hist_n_bins', self.bin_edges_n.size)
        #nc_dat.createDimension('hist_E_bins', self.bin_edges_E.size)
        #bin_edges_n = nc_dat.createVariable('bin_edges_n', 'float64', ('hist_n_bins'))
        #bin_edges_n[:] = self.hist_bin_edges_n
        #bin_edges_E = nc_dat.createVariable('bin_edges_E', 'float64', ('hist_E_bins'))
        #bin_edges_E[:] = self.hist_bin_edges_E
        bin_edges_T = nc_dat.createVariable('bin_edges_T', 'float32', ('hist_T_bins'))
        bin_edges_T[:] = self.bin_edges_T
        bin_edges_alpha = nc_dat.createVariable('bin_edges_alpha', 'float64', ('hist_alpha_bins'))
        bin_edges_alpha[:] = self.bin_edges_alpha
        hist_vs_cx = nc_dat.createVariable('hist_vs_cx', 'float32', ('hist_velocity_domains', 'hist_velocity'))
        hist_vs_cx[:] = self.hist_vs_cx
        hist_ion = nc_dat.createVariable('hist_ion', 'float32', ('hist_space'))
        hist_ion[:] = self.hist_ion
        hist_cx = nc_dat.createVariable('hist_cx', 'float32', ('hist_space'))
        hist_cx[:] = self.hist_cx
        hist_alpha = nc_dat.createVariable('hist_alpha', 'float32', ('hist_alpha'))
        hist_alpha[:] = self.hist_alpha
        hist_ion_temp = nc_dat.createVariable('hist_ion_temp', 'float32', ('hist_T'))
        hist_ion_temp[:] = self.hist_ion_temp


    def save_hist_results(self, output_datafolder, rank):
        np.savetxt(output_datafolder + 'hist_x_atom_' + str(rank) + '.txt', self.hist_x)
        np.savetxt(output_datafolder + 'hist_vs_atom_' + str(rank) + '.txt', self.hist_vs)
        np.savetxt(output_datafolder + 'hist_vs_cx_atom_' + str(rank) + '.txt', self.hist_vs_cx)
        np.savetxt(output_datafolder + 'hist_ion_' + str(rank) + '.txt', self.hist_ion)
        np.savetxt(output_datafolder + 'hist_cx_' + str(rank) + '.txt', self.hist_cx)
        np.savetxt(output_datafolder + 'hist_free_path_atom_' + str(rank) + '.txt', self.hist_free_path)
        np.savetxt(output_datafolder + 'hist_alpha_atom_' + str(rank) + '.txt', self.hist_alpha)
        np.savetxt(output_datafolder + 'hist_ion_temp_' + str(rank) + '.txt', self.hist_ion_temp)
        if (rank == 0):
            np.savetxt(output_datafolder + 'hist_alpha_atom_bin_edges.txt', self.bin_edges_alpha)
            np.savetxt(output_datafolder + 'hist_x_atom_bin_edges.txt', self.bin_edges_x)
            np.savetxt(output_datafolder + 'hist_vs_atom_bin_edges.txt', self.bin_edges_vs)

    def save_ionization_results(self):
        np.savetxt('ionized_x.txt', self.ionized_save_x[0:self.n_ionized])
        np.savetxt('ionized_y.txt', self.ionized_save_y[0:self.n_ionized])

    def save_cxed_results(self):
        np.savetxt('cxed_x.txt', self.cxed_save_x[0:self.n_cxed])
        np.savetxt('cxed_y.txt', self.cxed_save_y[0:self.n_cxed])
