import numpy as np
from species import Species
import pickle
from make_tables import Tables_1d

class H_molecule(Species):
    def __init__(self, inflow_rate, N_max, temperature, domain, diss_product, table_dict, bin_edges_x, bin_edges_vs, velocity_domains, bin_edges_T, absorbtion_coefficient, wall_boundary):
        #Call the init function of the parent class Species
        super().__init__(domain.d_molecule_mass, inflow_rate, N_max, temperature, domain, absorbtion_coefficient, wall_boundary, bin_edges_x, bin_edges_vs, velocity_domains)
        #The number of reactions for atoms, not including ionization which is treated by weight reduction.
        self.num_react = 5
        #Associate the names of the included reactions with a unique integer in the range [1-n]
        self.MID, self.DISSOCIATION_B1_C1, self.DISSOCIATION_Bp1_D1, self.DISSOCIATION_a3_c3, self.DISSOCIATION_b3 = range(1,self.num_react+1, 1)
        #Initiate the array holding the probabilities of each interaction
        #for each atom present in the domain.
        self.probs_arr = np.zeros((self.num_react, N_max))
        #Set the dissociation product equal to the passed instance of H-atoms.
        self.diss_product = diss_product
        #Read the tables from the dictionary
        self.MID_table = table_dict['effective_ion_rate']
        self.MID_fragment_KE_table = table_dict['MID_fragment_KE']
        self.diss_table_B1_C1 = table_dict['B1_C1_table']
        self.diss_table_Bp1_D1 = table_dict['Bp1_D1_table']
        self.diss_table_a3_c3 = table_dict['a3_c3_table']
        self.diss_table_b3 = table_dict['b3_table']
        #Define electron energy loss for dissociation reactions
        #See the section "Molecules" in chapter 5 of my thesis for the calculattions of these.
        self.energy_loss_B1_C1 = 12.75
        self.fragment_energy_B1_C1= 0.17
        self.energy_loss_Bp1_D1 = 17.25
        self.fragment_energy_Bp1_D1= 0.3
        self.energy_loss_a3_c3 = 12.64
        self.fragment_energy_a3_c3= 0.75
        self.energy_loss_b3 = 10.62
        self.fragment_energy_b3= 2.75
        #######Diagnostics#########
        self.Sn_this_step = np.zeros(self.domain.plasma_dim_x)
        #######Diagnostics#########

    #Calculate the probability of molecular ion dissociation (MID) for each molecule
    def probs_MID(self):
        self.probs_arr[self.MID-1, 0:self.max_ind] = self.probs_T_n(self.MID_table)

    #Calculate the probability of molecular dissociation for each molecule
    def probs_dissociations(self):
        self.probs_arr[self.DISSOCIATION_B1_C1-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.diss_table_B1_C1)
        self.probs_arr[self.DISSOCIATION_Bp1_D1-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.diss_table_B1_C1)
        self.probs_arr[self.DISSOCIATION_a3_c3-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.diss_table_a3_c3)
        self.probs_arr[self.DISSOCIATION_b3-1, 0:self.max_ind] = self.probs_electron_collisions_lookup(self.diss_table_b3)

    #Wrapper for calculating probabilities
    def get_probs(self):
        self.probs_MID()
        self.probs_dissociations()

    #Uniform sampling of unit sphere. Return n sets of angles.
    def sample_angles(self, n):
        phis = np.random.uniform(0, 2*np.pi, n)
        thetas = np.arccos(1-2*np.random.uniform(0, 1, n))
        return phis, thetas

    #Perform MID
    def do_MID(self, MIDed):
        n_MIDed = np.sum(MIDed)
        #Get positions for breeding the new D-atoms
        xs = self.x[0:self.max_ind][MIDed]
        ys = self.y[0:self.max_ind][MIDed]
        #Get the current norm_weight of the MID'ed molecules
        norm_weights = self.norm_weight[0:self.max_ind][MIDed]
        #Get the neutral velocities going into the reaction
        vxs_incoming = self.vx[0:self.max_ind][MIDed]
        vys_incoming = self.vy[0:self.max_ind][MIDed]
        vzs_incoming = self.vz[0:self.max_ind][MIDed]
        #Sample the velocities of the fragments in the neutral/"CM" frame
        rands = np.random.uniform(0, self.MID_fragment_KE_table.data[-1], n_MIDed)
        inds = np.sum(self.MID_fragment_KE_table.data < np.reshape(rands, (n_MIDed, 1)), axis = 1)
        #Sample fragment energy in eV
        Es_fragment_cm = self.MID_fragment_KE_table.min + self.MID_fragment_KE_table.dx*inds
        vs_fragment_cm = np.sqrt(2*Es_fragment_cm*1.602e-19/self.domain.d_ion_mass)
        #Making a random orientation, by proper uniform sampling of the unit sphere
        phis, thetas = self.sample_angles(n_MIDed)
        #Calculate the cartesion components of the sampled velocity
        vxs_fragment_cm = np.sin(thetas)*np.cos(phis)*vs_fragment_cm
        vys_fragment_cm = np.sin(thetas)*np.sin(phis)*vs_fragment_cm
        vzs_fragment_cm = np.cos(thetas)*vs_fragment_cm
        fragment_E = self.domain.kinetic_energy(vxs_fragment_cm, vys_fragment_cm, vzs_fragment_cm, self.domain.d_ion_mass)
        #Add to incoming velocities
        vxs_new = vxs_incoming + vxs_fragment_cm
        vys_new = vys_incoming + vys_fragment_cm
        vzs_new = vzs_incoming + vzs_fragment_cm
        #Get the positional indices of the MID's on the HESEL grid
        inds_x = self.plasma_inds_x[0:self.max_ind][MIDed]
        inds_y = self.plasma_inds_y[0:self.max_ind][MIDed]
        #Breed the new atoms
        self.diss_product.inflow(xs, ys, vxs_new, vys_new, vzs_new, self.norm_weight[0:self.max_ind][MIDed], inds_x, inds_y)
        #**********
        #The Energy loss from the ionization of H2 is 15.4 eV.
        #The energy transferred from the electron to the h2+ ion is 2 times the
        #fragment energy plus the FC and cross sec weighted dissociation energy
        #at 15 eV. This last value it found to be 1.614 eV
        #See the subsection Molecular Ion Dissociation (MID) in chapter 5 of my
        #thesis for the full calculation.
        #**********
        #Add the sources at the relevant indices to the already existing sources of this timestep.
        #Using np.add.at rather than self.source[x_inds, y_inds] = self.source[x_inds, y_inds] + norm_weight
        #is important to allow for identical pairs of indices i.e. neutrals reacting in the same grid cell
        #of the plasma sim.
        np.add.at(self.domain.electron_source_particle, (inds_x, inds_y), norm_weights)
        #Remember that fragments are flying opposite to each other, hence the (-)
        np.add.at(self.domain.ion_source_momentum_x, (inds_x, inds_y), self.domain.d_ion_mass*(-1)*vxs_new*norm_weights)
        np.add.at(self.domain.ion_source_momentum_y, (inds_x, inds_y), self.domain.d_ion_mass*(-1)*vys_new*norm_weights)
        np.add.at(self.domain.ion_source_energy, (inds_x, inds_y), Es_fragment_cm*norm_weights)
        np.add.at(self.domain.electron_source_energy, (inds_x, inds_y), -(2*Es_fragment_cm + 1.61 + 15.4)*norm_weights)
        self.remove(MIDed)

    def dissociate(self, dissociated, energy_loss, fragment_energy):
        n_dissociated = np.sum(dissociated)
        #Get positions for breeding the new D-atoms - Two per molecule
        xs = np.tile(self.x[0:self.max_ind][dissociated], 2)
        ys = np.tile(self.y[0:self.max_ind][dissociated], 2)
        #Get the current norm_weight of the MID'ed molecules
        norm_weights = self.norm_weight[0:self.max_ind][dissociated]
        #Get the neutral velocities going into the reaction
        vxs_incoming = self.vx[0:self.max_ind][dissociated]
        vys_incoming = self.vy[0:self.max_ind][dissociated]
        vzs_incoming = self.vz[0:self.max_ind][dissociated]
        #Get the velocities of the fragments in the neutral/"CM" frame
        vs_fragment_cm = np.ones(n_dissociated)*np.sqrt(2*fragment_energy*1.602e-19/self.domain.d_ion_mass)
        #Making a random orientation, by proper uniform sampling of the unit sphere
        phis, thetas = self.sample_angles(n_dissociated)
        #Calculate the cartesion components of the sampled velocity
        vxs_fragment_cm = np.sin(thetas)*np.cos(phis)*vs_fragment_cm
        vys_fragment_cm = np.sin(thetas)*np.sin(phis)*vs_fragment_cm
        vzs_fragment_cm = np.cos(thetas)*vs_fragment_cm
        #Add to incoming velocities
        vxs_new = np.concatenate((vxs_incoming + vxs_fragment_cm, vxs_incoming - vxs_fragment_cm))
        vys_new = np.concatenate((vys_incoming + vys_fragment_cm, vys_incoming - vys_fragment_cm))
        vzs_new = np.concatenate((vzs_incoming + vzs_fragment_cm, vzs_incoming - vzs_fragment_cm))
        #Get the positional indices of the dissociations on the HESEL grid
        inds_x = self.plasma_inds_x[0:self.max_ind][dissociated]
        inds_y = self.plasma_inds_y[0:self.max_ind][dissociated]
        #Breed the new atoms
        self.diss_product.inflow(xs, ys, vxs_new, vys_new, vzs_new, np.tile(norm_weights, 2), np.tile(inds_x, 2), np.tile(inds_y, 2))
        #Add the source, which is purely electron energy sink
        np.add.at(self.domain.electron_source_energy, (inds_x, inds_y), -energy_loss*norm_weights)
        #Remove the dissociated molecules
        self.remove(dissociated)

    #Wrapper for performin the different interactions
    def do_interaction(self, specify_interaction):
        MIDed = specify_interaction == self.MID
        self.do_MID(MIDed)
        dissociated_B1_C1 = specify_interaction == self.DISSOCIATION_B1_C1
        self.dissociate(dissociated_B1_C1, self.energy_loss_B1_C1, self.fragment_energy_B1_C1)
        dissociated_Bp1_D1 = specify_interaction == self.DISSOCIATION_Bp1_D1
        self.dissociate(dissociated_Bp1_D1, self.energy_loss_Bp1_D1, self.fragment_energy_Bp1_D1)
        dissociated_a3_c3 = specify_interaction == self.DISSOCIATION_a3_c3
        self.dissociate(dissociated_a3_c3, self.energy_loss_a3_c3, self.fragment_energy_a3_c3)
        dissociated_b3 = specify_interaction == self.DISSOCIATION_b3
        self.dissociate(dissociated_b3, self.energy_loss_b3, self.fragment_energy_b3)

    """
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
        hist_MID_temp = nc_dat.createVariable('hist_MID_temp', 'float32', ('hist_T'))
        hist_MID_temp[:] = self.hist_MID_temp
        hist_Es_ass_ion = nc_dat.createVariable('hist_Es_ass_ion', 'float32', ('hist_E'))
        hist_Es_ass_ion[:] = self.hist_Es_ass_ion
        hist_ns_ass_ion = nc_dat.createVariable('hist_ns_ass_ion', 'float32', ('hist_n'))
        hist_ns_ass_ion[:] = self.hist_ns_ass_ion
        hist_diss = nc_dat.createVariable('hist_diss', 'float64', ('hist_space'))
        hist_diss[:] = self.hist_diss
        hist_diss_temp = nc_dat.createVariable('hist_diss_temp', 'float32', ('hist_T'))
        hist_diss_temp[:] = self.hist_diss_temp
    """
