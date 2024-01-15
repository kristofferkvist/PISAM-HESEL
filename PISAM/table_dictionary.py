"""
@Kristoffer Kvist (Orcid ID: 0009-0006-4494-7281)
The code of this document is developed and written by Kristoffer Kvist affiliated
to the physics department of the Technical University of Denmark. The content of the
code can be applied by any third party, given that article "A direct Monte Carlo
approach for the modeling of neutrals at the plasma edge and its self-consistent
coupling with the 2D fluid plasma edge turbulence model HESEL" published in
"Physics of Plasmas" in 2024 is cited accordingly.
"""


"""This short scrips simply collects all the calculated tables in species-specific
dictionarys that are passes to each species at initialization."""


import pickle
data_folder = 'PISAM/input_data/'

def set_entry_from_pkl(dict, entry_name, filename):
    with open(data_folder + filename + '.pkl', 'rb') as f:
        dict[entry_name] = pickle.load(f)

def make_dict_atom():
    print('COMPILING ATOM TABLE DICTIONARY')
    dict = {}
    filenames = ['effective_ion_rate', 'cx_2d_table', 'cx_rate_integrand_3D', '1s_to_2p']
    entry_names = ['ion_rate', 'cx_rate', 'cx_cross', '1s_to_2p_rate']
    for i in range(len(filenames)):
        filename = filenames[i]
        entry_name = entry_names[i]
        set_entry_from_pkl(dict, entry_name, filename)
    with open(data_folder + 'h_atom_dict.pkl', 'wb') as f:
        pickle.dump(dict, f)

def make_dict_molecule():
    print('COMPILING MOLECULE TABLE DICTIONARY')
    dict = {}
    filenames = ['effective_ion_rate_molecule', 'ass_ion_fragment_KE', 'B1_C1_table', 'Bp1_D1_table', 'a3_c3_table', 'b3_table']
    entry_names = ['effective_ion_rate', 'MID_fragment_KE', 'B1_C1_table', 'Bp1_D1_table', 'a3_c3_table', 'b3_table']
    for i in range(len(filenames)):
        filename = filenames[i]
        entry_name = entry_names[i]
        set_entry_from_pkl(dict, entry_name, filename)
    with open(data_folder + 'h_molecule_dict.pkl', 'wb') as f:
        pickle.dump(dict, f)

make_dict_atom()
make_dict_molecule()
