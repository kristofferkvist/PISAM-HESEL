import numpy as np
from species import Species
import pickle
from make_tables import Tables_1d

class H_molecule(Species):
    def __init__(self, inflow_rate, N_max, temperature, domain, diss_product, table_dict, bin_edges_x, bin_edges_vs, velocity_domains, bin_edges_T, absorbtion_coefficient):
        super().__init__(domain.d_molecule_mass, inflow_rate, N_max, temperature, domain, absorbtion_coefficient, bin_edges_x, bin_edges_vs, velocity_domains)
        self.num_react = 5
        self.ASSISTET_IONIZATION = 1
        self.DISSOCIATION_B1_C1 = 2
        self.DISSOCIATION_Bp1_D1 = 3
        self.DISSOCIATION_a3_c3 = 4
        self.DISSOCIATION_b3 = 5
        self.probs_arr = np.zeros((self.num_react, N_max))
        self.diss_product = diss_product
        self.bin_edges_T = bin_edges_T
        self.ass_ion_table = table_dict['effective_ion_rate']
        self.ass_ion_fragment_KE_table = table_dict['ass_ion_fragment_KE']
        self.diss_table_B1_C1 = table_dict['B1_C1_table']
        self.diss_table_Bp1_D1 = table_dict['Bp1_D1_table']
        self.diss_table_a3_c3 = table_dict['a3_c3_table']
        self.diss_table_b3 = table_dict['b3_table']
        self.bin_edges_n = np.linspace(0.8e+18, 1e+20, 100)
        self.hist_ns_ass_ion = np.zeros(self.bin_edges_n.size-1)
        self.hist_ns_diss = np.zeros(self.bin_edges_n.size-1)
        self.bin_edges_E = np.linspace(0, 20, 50)
        self.hist_Es_ass_ion = np.zeros(self.bin_edges_E.size-1)
        self.hist_ass_ion = np.zeros(bin_edges_x.size-1)
        self.hist_diss = np.zeros(bin_edges_x.size-1)
        self.hist_ass_ion_temp = np.zeros(bin_edges_T.size-1)
        self.hist_diss_temp = np.zeros(bin_edges_T.size-1)
        self.energy_loss_B1_C1 = 12.75
        self.fragment_energy_B1_C1= 0.17
        self.energy_loss_Bp1_D1 = 17.25
        self.fragment_energy_Bp1_D1= 0.3
        self.energy_loss_a3_c3 = 12.64
        self.fragment_energy_a3_c3= 0.75
        self.energy_loss_b3 = 10.62
        self.fragment_energy_b3= 2.75


    def probs_ass_ionization(self):
        self.probs_arr[self.ASSISTET_IONIZATION-1, 0:self.max_ind] = self.probs_T_n(self.ass_ion_table)

    def probs_dissociations(self):
        self.probs_arr[self.DISSOCIATION_B1_C1-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.diss_table_B1_C1)
        self.probs_arr[self.DISSOCIATION_Bp1_D1-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.diss_table_B1_C1)
        self.probs_arr[self.DISSOCIATION_a3_c3-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.diss_table_a3_c3)
        self.probs_arr[self.DISSOCIATION_b3-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.diss_table_b3)

    def get_probs(self):
        self.probs_ass_ionization()
        self.probs_dissociations()

    def sample_angles(self, n):
        phis = np.random.uniform(0, 2*np.pi, n)
        thetas = np.arccos(1-2*np.random.uniform(0, 1, n))
        return phis, thetas

    def ass_ionize(self, ass_ionized):
        n_ass_ionized = np.sum(ass_ionized)
        #Get positions for breeding the new D-atoms
        xs = self.x[0:self.max_ind][ass_ionized]
        ys = self.y[0:self.max_ind][ass_ionized]
        percentages = self.percentage[0:self.max_ind][ass_ionized]
        #Get the neutral velocities going into the reaction
        vxs_incoming = self.vx[0:self.max_ind][ass_ionized]
        vys_incoming = self.vy[0:self.max_ind][ass_ionized]
        vzs_incoming = self.vz[0:self.max_ind][ass_ionized]
        #Get the velocities of the fragments in the neutral/"CM" frame
        rands = np.random.uniform(0, self.ass_ion_fragment_KE_table.data[-1], n_ass_ionized)
        inds = np.sum(self.ass_ion_fragment_KE_table.data < np.reshape(rands, (n_ass_ionized, 1)), axis = 1)
        #Sample fragment energy in eV
        Es_fragment_cm = self.ass_ion_fragment_KE_table.min + self.ass_ion_fragment_KE_table.dx*inds
        vs_fragment_cm = np.sqrt(2*Es_fragment_cm*1.602e-19/self.domain.d_ion_mass)
        #Making a random orientation, by proper uniform sampling of the unit sphere
        phis, thetas = self.sample_angles(n_ass_ionized)
        vxs_fragment_cm = np.sin(thetas)*np.cos(phis)*vs_fragment_cm
        vys_fragment_cm = np.sin(thetas)*np.sin(phis)*vs_fragment_cm
        vzs_fragment_cm = np.cos(thetas)*vs_fragment_cm
        fragment_E = 0.5*self.domain.d_ion_mass*(np.power(vxs_fragment_cm, 2)+np.power(vys_fragment_cm, 2)+np.power(vzs_fragment_cm, 2))/1.602e-19
        self.hist_Es_ass_ion = self.hist_Es_ass_ion + np.histogram(fragment_E, self.bin_edges_E)[0]
        #Add to incoming velocities
        vxs_new = vxs_incoming + vxs_fragment_cm
        vys_new = vys_incoming + vys_fragment_cm
        vzs_new = vzs_incoming + vzs_fragment_cm
        #Breed the new atoms
        self.diss_product.inflow(xs, ys, vxs_new, vys_new, vzs_new, self.percentage[0:self.max_ind][ass_ionized])
        inds_x = self.plasma_inds_x[0:self.max_ind][ass_ionized]
        inds_y = self.plasma_inds_y[0:self.max_ind][ass_ionized]
        #The Energy loss from the ionization of H2 is 15.4 eV.
        #The energy transferred from the electron to the h2+ ion is 2 times the
        #fragment energy plus the FC and cross sec weighted dissociation energy
        #at 15 eV. This last value it found to be 1.614 eV
        self.domain.electron_source_particle[inds_x, inds_y] = self.domain.electron_source_particle[inds_x, inds_y] + percentages
        #Remember that fragments are flying opposite to each other, hence the (-)
        self.domain.ion_source_momentum_x[inds_x, inds_y] = self.domain.ion_source_momentum_x[inds_x, inds_y] + self.domain.d_ion_mass*(-1)*vxs_new*percentages
        self.domain.ion_source_momentum_y[inds_x, inds_y] = self.domain.ion_source_momentum_y[inds_x, inds_y] + self.domain.d_ion_mass*(-1)*vys_new*percentages
        self.domain.ion_source_energy[inds_x, inds_y] = self.domain.ion_source_energy[inds_x, inds_y] + Es_fragment_cm*percentages
        self.domain.electron_source_energy[inds_x, inds_y] = self.domain.electron_source_energy[inds_x, inds_y] - (2*Es_fragment_cm + 1.61 + 15.4)*percentages + self.domain.kinetic_energy(self.vx[0:self.max_ind][ass_ionized], self.vy[0:self.max_ind][ass_ionized], self.vz[0:self.max_ind][ass_ionized], self.domain.electron_mass)*percentages
        self.remove(ass_ionized)

    def ass_ionize_save(self, ass_ionized):
        self.hist_ass_ion = self.save_to_hist_reaction(self.hist_ass_ion, ass_ionized)
        self.hist_ass_ion_temp = self.save_to_hist_reaction_temperature(self.hist_ass_ion_temp, ass_ionized)
        ns = self.n[0:self.max_ind][ass_ionized]*5e+19
        self.hist_ns_ass_ion = self.hist_ns_ass_ion + np.histogram(ns, self.bin_edges_n)[0]
        self.ass_ionize(ass_ionized)

    def dissociate(self, dissociated, energy_loss, fragment_energy):
        n_dissociated = np.sum(dissociated)
        #Get positions for breeding the new D-atoms
        xs = np.tile(self.x[0:self.max_ind][dissociated], 2)
        ys = np.tile(self.y[0:self.max_ind][dissociated], 2)
        percentages = self.percentage[0:self.max_ind][dissociated]
        #Get the neutral velocities going into the reaction
        vxs_incoming = self.vx[0:self.max_ind][dissociated]
        vys_incoming = self.vy[0:self.max_ind][dissociated]
        vzs_incoming = self.vz[0:self.max_ind][dissociated]
        #Get the velocities of the fragments in the neutral/"CM" frame
        vs_fragment_cm = np.ones(n_dissociated)*np.sqrt(2*fragment_energy*1.602e-19/self.domain.d_ion_mass)
        #Making a random orientation, by proper uniform sampling of the unit sphere
        phis, thetas = self.sample_angles(n_dissociated)
        vxs_fragment_cm = np.sin(thetas)*np.cos(phis)*vs_fragment_cm
        vys_fragment_cm = np.sin(thetas)*np.sin(phis)*vs_fragment_cm
        vzs_fragment_cm = np.cos(thetas)*vs_fragment_cm
        #Add to incoming velocities
        vxs_new = np.concatenate((vxs_incoming + vxs_fragment_cm, vxs_incoming - vxs_fragment_cm))
        vys_new = np.concatenate((vys_incoming + vys_fragment_cm, vys_incoming - vys_fragment_cm))
        vzs_new = np.concatenate((vzs_incoming + vzs_fragment_cm, vzs_incoming - vzs_fragment_cm))
        #Breed the new atoms
        self.diss_product.inflow(xs, ys, vxs_new, vys_new, vzs_new, np.tile(percentages, 2))
        #Remove the dissociated molecules
        self.remove(dissociated)
        inds_x = self.plasma_inds_x[0:self.max_ind][dissociated]
        inds_y = self.plasma_inds_y[0:self.max_ind][dissociated]
        self.domain.electron_source_energy[inds_x, inds_y] = self.domain.electron_source_energy[inds_x, inds_y] - energy_loss*percentages

    def dissociate_save(self, dissociated, energy_loss, fragment_energy):
        self.hist_diss = self.save_to_hist_reaction(self.hist_diss, dissociated)
        self.hist_diss_temp = self.save_to_hist_reaction_temperature(self.hist_diss_temp, dissociated)
        self.dissociate(dissociated, energy_loss, fragment_energy)

    def do_interaction(self, specify_interaction):
        ass_ionized = specify_interaction == self.ASSISTET_IONIZATION
        self.ass_ionize_save(ass_ionized)
        dissociated_B1_C1 = specify_interaction == self.DISSOCIATION_B1_C1
        self.dissociate_save(dissociated_B1_C1, self.energy_loss_B1_C1, self.fragment_energy_B1_C1)
        dissociated_Bp1_D1 = specify_interaction == self.DISSOCIATION_Bp1_D1
        self.dissociate_save(dissociated_Bp1_D1, self.energy_loss_Bp1_D1, self.fragment_energy_Bp1_D1)
        dissociated_a3_c3 = specify_interaction == self.DISSOCIATION_a3_c3
        self.dissociate_save(dissociated_a3_c3, self.energy_loss_a3_c3, self.fragment_energy_a3_c3)
        dissociated_b3 = specify_interaction == self.DISSOCIATION_b3
        self.dissociate_save(dissociated_b3, self.energy_loss_b3, self.fragment_energy_b3)

    def do_interaction_save(self, specify_interaction):
        ass_ionized = specify_interaction == self.ASSISTET_IONIZATION
        self.ass_ionize_save(ass_ionized)
        dissociated_B1_C1 = specify_interaction == self.DISSOCIATION_B1_C1
        self.dissociate_save(dissociated_B1_C1, self.energy_loss_B1_C1, self.fragment_energy_B1_C1)
        dissociated_Bp1_D1 = specify_interaction == self.DISSOCIATION_Bp1_D1
        self.dissociate_save(dissociated_Bp1_D1, self.energy_loss_Bp1_D1, self.fragment_energy_Bp1_D1)
        dissociated_a3_c3 = specify_interaction == self.DISSOCIATION_a3_c3
        self.dissociate_save(dissociated_a3_c3, self.energy_loss_a3_c3, self.fragment_energy_a3_c3)
        dissociated_b3 = specify_interaction == self.DISSOCIATION_b3
        self.dissociate_save(dissociated_b3, self.energy_loss_b3, self.fragment_energy_b3)

    def save_to_hist_vs(self):
        for i in np.arange(self.velocity_domains):
            larger_mask = self.x[0:self.max_ind] > (self.domain.x_min + i*(self.domain.x_max-self.domain.x_min)/self.velocity_domains)
            smaller_mask = self.x[0:self.max_ind] < (self.domain.x_min + (i+1)*(self.domain.x_max-self.domain.x_min)/self.velocity_domains)
            mask = (larger_mask) & (smaller_mask) & (self.active[0:self.max_ind])
            velocities = np.sqrt(np.power(self.vx[0:self.max_ind][mask], 2) + np.power(self.vy[0:self.max_ind][mask], 2) + np.power(self.vz[0:self.max_ind][mask], 2))
            hist_data, _ = np.histogram(velocities, bins=self.bin_edges_vs)
            self.hist_vs[i, :] = self.hist_vs[i, :] + hist_data

    def save_hists(self, nc_dat):
        nc_dat.createDimension('hist_n', self.bin_edges_n.size-1)
        nc_dat.createDimension('hist_E', self.bin_edges_E.size-1)
        nc_dat.createDimension('hist_T', self.bin_edges_T.size-1)
        nc_dat.createDimension('hist_n_bins', self.bin_edges_n.size)
        nc_dat.createDimension('hist_E_bins', self.bin_edges_E.size)
        nc_dat.createDimension('hist_T_bins', self.bin_edges_T.size)
        bin_edges_n = nc_dat.createVariable('bin_edges_n', 'float32', ('hist_n_bins'))
        bin_edges_n[:] = self.bin_edges_n
        bin_edges_E = nc_dat.createVariable('bin_edges_E', 'float32', ('hist_E_bins'))
        bin_edges_E[:] = self.bin_edges_E
        bin_edges_T = nc_dat.createVariable('bin_edges_T', 'float32', ('hist_T_bins'))
        bin_edges_T[:] = self.bin_edges_T
        hist_ass_ion = nc_dat.createVariable('hist_ass_ion', 'float32', ('hist_space'))
        hist_ass_ion[:] = self.hist_ass_ion
        hist_ass_ion_temp = nc_dat.createVariable('hist_ass_ion_temp', 'float32', ('hist_T'))
        hist_ass_ion_temp[:] = self.hist_ass_ion_temp
        hist_Es_ass_ion = nc_dat.createVariable('hist_Es_ass_ion', 'float32', ('hist_E'))
        hist_Es_ass_ion[:] = self.hist_Es_ass_ion
        hist_ns_ass_ion = nc_dat.createVariable('hist_ns_ass_ion', 'float32', ('hist_n'))
        hist_ns_ass_ion[:] = self.hist_ns_ass_ion
        hist_diss = nc_dat.createVariable('hist_diss', 'float64', ('hist_space'))
        hist_diss[:] = self.hist_diss
        hist_diss_temp = nc_dat.createVariable('hist_diss_temp', 'float32', ('hist_T'))
        hist_diss_temp[:] = self.hist_diss_temp

    def save_hist_results(self, output_datafolder, rank):
        np.savetxt(output_datafolder + 'hist_x_molecule_' + str(rank) + '.txt', self.hist_x)
        np.savetxt(output_datafolder + 'hist_vs_molecule_' + str(rank) + '.txt', self.hist_vs)
        np.savetxt(output_datafolder + 'hist_ass_ion_' + str(rank) + '.txt', self.hist_ass_ion)
        np.savetxt(output_datafolder + 'hist_diss_' + str(rank) + '.txt', self.hist_diss)
        np.savetxt(output_datafolder + 'hist_ass_ion_temp_' + str(rank) + '.txt', self.hist_ass_ion_temp)
        np.savetxt(output_datafolder + 'hist_diss_temp_' + str(rank) + '.txt', self.hist_diss_temp)
        np.savetxt(output_datafolder + 'hist_Es_ass_ion_' + str(rank) + '.txt', self.hist_Es_ass_ion)
        np.savetxt(output_datafolder + 'hist_ns_ass_ion_' + str(rank) + '.txt', self.hist_ns_ass_ion)
        np.savetxt(output_datafolder + 'hist_ns_diss_' + str(rank) + '.txt', self.hist_ns_diss)
        if (rank == 0):
            np.savetxt(output_datafolder + 'n_bin_edges.txt', self.bin_edges_n)
            np.savetxt(output_datafolder + 'E_bin_edges.txt', self.bin_edges_E)
            np.savetxt(output_datafolder + 'T_bin_edges.txt', self.bin_edges_T)
            np.savetxt(output_datafolder + 'hist_x_molecule_bin_edges.txt', self.bin_edges_x)
            np.savetxt(output_datafolder + 'hist_vs_molecule_bin_edges.txt', self.bin_edges_vs)
