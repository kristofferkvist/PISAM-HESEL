"""
This script uses the classes implemented in make_tables.py to create the tables
used in PISAM simulations. The data loaded to make the tables are in the form
of fit parameters or data points for reactions rates and cross sections. The data is
stored in the directory "Collision data for tables". I you add a reaction is it
recommended to follow the procedure of calculating a table of reaction rates
prefferably using one of the classes provided in make_tables.py, and then saving
this table in the dictionary holding the tables of the relevant species, by adjusting
"table_dictionary.py".

Certain table ranges and resolutions are chosen in the script below. These are by
no means universal and can be adjusted depending on your purpose.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d
import os
from make_tables import Tables_1d
from make_tables import Tables_2d
from make_tables import Table_cx_sampling_3D
from make_tables import Tables_2d_T_n
from make_tables import Table_simple
from make_tables import Table_light_heavy_sampling_from_spline_clean

path = 'PISAM/input_data'
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
data_folder = path + '/'
coef_folder = 'PISAM/collision_data_for_tables/'
mccc_fit_folder = coef_folder + 'MCCC-fits/'

MASS_D_ATOM = 2.01410177811*1.660539e-27
MASS_ELECTRON = 9.1093837e-31
bohr_radius = 5.29177210903e-11

def load_dats_make_spline(filenames, object_name, ratios):
    E_plot = np.linspace(0, 300, 1000)
    final_cross = np.zeros(1000)
    for i in range(len(filenames)):
        filename = filenames[i]
        dat = np.loadtxt(mccc_fit_folder + filename)
        Es = dat[:, 0]
        cross = dat[:, 1]*bohr_radius*bohr_radius*ratios[i]
        spline = interp1d(Es, cross, kind='cubic', bounds_error = False, fill_value = 0)
        final_cross = final_cross + spline(E_plot)
    final_spline = interp1d(E_plot, final_cross, kind='cubic', bounds_error = False, fill_value = 0)
    with open(data_folder + object_name + '.pkl', 'wb') as f:
        pickle.dump(final_spline, f)

def make_1D_Tables(T_min, T_max, T_res, base, sum_coefs, table_filenames):
    for i in np.arange(sum_coefs.shape[0]):
        table = Tables_1d(T_min, T_max, T_res, base, sum_coefs[i], data_folder + table_filenames[i])
        table.calc_electron_col_rate()
        table.save_object()

def make_fragment_energy_table_ass_ion():
    el_ghaz_E_0 = 10
    el_ghaz_E_10 = 208
    dE = 10/(el_ghaz_E_10-el_ghaz_E_0)
    el_ghaz_cross_0 = 184
    el_ghaz_cross_10 = 41
    d_cross = 10/(el_ghaz_cross_0-el_ghaz_cross_10)
    el_ghaz_E = np.array([12.1, 13.57, 15.89, 18.9, 24.26, 29.17, 36.06, 43.5, 52.8, 62.64, 73, 85, 98, 111.4, 126.2, 143.3, 160, 177.1, 208])/2
    el_ghaz_cross = np.array([143, 74, 63.6, 70, 95.14, 95.91, 105.69, 104.5, 103.11, 107.89, 117.7, 132.22, 145.5, 158.13, 158, 161.48, 167, 172, 180])
    el_ghaz_E_real = (el_ghaz_E-el_ghaz_E_0)*dE
    el_ghaz_cross_real = -1*(el_ghaz_cross-el_ghaz_cross_0)*d_cross
    spline = interp1d(el_ghaz_E_real, el_ghaz_cross_real, kind='cubic', bounds_error = False, fill_value = 0)
    E_plot = np.linspace(0, 7, 500)
    dE = E_plot[1]-E_plot[0]
    e_min = 0
    table = Table_simple(e_min, dE, np.cumsum(spline(E_plot)), data_folder + 'ass_ion_fragment_KE')
    table.save_object()

#Make the table of CX reaction rates, from det double polynomial fit with coefficients saved in the file 'cx_rate.txt'
def make_cx_rate_table():
    cx_sum_coefs = np.loadtxt(coef_folder + 'cx_rate.txt')
    cx = Tables_2d(0.1, 300, 0.1, 0.1, 300, 0.1, cx_sum_coefs, data_folder + 'cx_2d_table')
    cx.calc_heavy_col_rate()
    #Adjust for the factor 2 mass difference
    cx.rates = cx.rates/np.sqrt(2)
    cx.save_object()

#Make the table of the distribution of ions going into cx reactions
def make_cx_integrand_table():
    sum_coefs_cx_cross = np.array([-3.274123792568e+01, -8.916456579806e-02, -3.016990732025e-02, 9.205482406462e-03, 2.400266568315e-03, -1.927122311323e-03, 3.654750340106e-04, -2.788866460622e-05, 7.422296363524e-07])
    cx_rate_integrand = Table_cx_sampling_3D(0.1, 300, 100, 0.1, 300, 100, 3, np.pi/100, 5, 100, sum_coefs_cx_cross, data_folder + 'cx_rate_integrand_3D', MASS_D_ATOM)
    cx_rate_integrand.tabulate()
    cx_rate_integrand.save_object()

def make_effective_atom_ionization_rate_table():
    effective_ion_rate_coefs = np.loadtxt(coef_folder + 'effective_ion_rate.txt')
    effective_ion_rate_table = Tables_2d_T_n(0.1, 300, 1000, 5e+17, 1e+20, 1000, effective_ion_rate_coefs, data_folder + 'effective_ion_rate')
    effective_ion_rate_table.calc_rate()
    effective_ion_rate_table.save_object()

def make_1s_2p_table():
    H1s_to_H2p_coefs = np.array([-2.814949375869e+01, 1.009828023274e+01, -4.771961915818e+00, 1.467805963618e+00, -2.979799374553e-01, 3.861631407174e-02, -3.051685780771e-03, 1.335472720988e-04, -2.476088392502e-06])
    make_1D_Tables(1, 300, 10000, 3, np.array([H1s_to_H2p_coefs]), ['1s_to_2p'])

def make_heavy_light_tables_from_splines(T_min, T_max, T_res, base, n_standard_deviations, v_res, spline_names, table_filenames):
    for i in np.arange(spline_names.shape[0]):
        table = Table_light_heavy_sampling_from_spline_clean(T_min, T_max, T_res, base, n_standard_deviations, v_res, data_folder + spline_names[i], data_folder + table_filenames[i])
        table.tabulate()
        table.save_object()

def make_diss_splines():
    load_dats_make_spline(['MCCC-el-D2-B1Su_total.X1Sg_vi=0.txt', 'MCCC-el-D2-C1Pu_total.X1Sg_vi=0.txt'], 'B1_C1_spline', np.array([0.26, 0.02]))
    load_dats_make_spline(['MCCC-el-D2-Bp1Su_total.X1Sg_vi=0.txt', 'MCCC-el-D2-D1Pu_total.X1Sg_vi=0.txt'], 'Bp1_D1_spline', np.array([0.4, 0.285]))
    load_dats_make_spline(['MCCC-el-D2-a3Sg_total.X1Sg_vi=0.txt', 'MCCC-el-D2-c3Pu_total.X1Sg_vi=0.txt'], 'a3_c3_spline', np.array([1, 1]))
    load_dats_make_spline(['MCCC-el-D2-b3Su_DE.X1Sg_vi=0.txt'], 'b3_spline', np.array([1]))

def make_tables_for_diss():
    make_diss_splines()
    make_heavy_light_tables_from_splines(0.1, 300, 1000, 4, 3, 1000, np.array(['B1_C1_spline', 'Bp1_D1_spline', 'a3_c3_spline', 'b3_spline']), ['B1_C1_table', 'Bp1_D1_table', 'a3_c3_table', 'b3_table'])

def make_eff_molecule_ion_rate_table():
    eff_ion_mol_sum_coefs = np.loadtxt(coef_folder + 'effective_ion_molecule.txt')
    effective_ion_rate_molecule_table = Tables_2d_T_n(0.1, 300, 1000, 5e+17, 1e+20, 1000, eff_ion_mol_sum_coefs, data_folder + 'effective_ion_rate_molecule')
    effective_ion_rate_molecule_table.calc_rate()
    effective_ion_rate_molecule_table.save_object()

print("Making Fragment Energy Table for MID")
make_fragment_energy_table_ass_ion()
print("Making CX RATE TABLE")
make_cx_rate_table()
print("Making CX INTEGRAND TABLE")
make_cx_integrand_table()
print("Making EFF. ATOM IONIZATION TABLE")
make_effective_atom_ionization_rate_table()
print("Making 1S->2P TABLE")
make_1s_2p_table()
print("Making TABLES FOR MD")
make_tables_for_diss()
print("Making EFF. MOLECULE IONIZATION TABLE")
make_eff_molecule_ion_rate_table()
