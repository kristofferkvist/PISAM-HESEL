import pickle
#from make_tables import Table_cx_sampling_3D

data_folder = '/home/kristoffer/Desktop/KU/speciale/python_stuff/input_data/'

def set_entry_from_pkl(dict, entry_name, filename):
    with open(data_folder + filename + '.pkl', 'rb') as f:
        dict[entry_name] = pickle.load(f)

def make_dict_atom():
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
    dict = {}
    filenames = ['effective_ion_rate_molecule', 'ass_ion_fragment_KE', 'B1_C1_table', 'Bp1_D1_table', 'a3_c3_table', 'b3_table']
    entry_names = ['effective_ion_rate', 'ass_ion_fragment_KE', 'B1_C1_table', 'Bp1_D1_table', 'a3_c3_table', 'b3_table']
    for i in range(len(filenames)):
        filename = filenames[i]
        entry_name = entry_names[i]
        set_entry_from_pkl(dict, entry_name, filename)
    with open(data_folder + 'h_molecule_dict.pkl', 'wb') as f:
        pickle.dump(dict, f)

def test_dict():
    with open(data_folder + 'h_atom_dict' + '.pkl', 'rb') as f:
        dict_atom = pickle.load(f)
    cx_cross = dict_atom['cx_cross']
    print(cx_cross.table.shape)

make_dict_atom()
make_dict_molecule()
