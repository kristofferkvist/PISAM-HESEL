from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d
from make_tables import Tables_1d
from make_tables import Tables_2d
from make_tables import Table_cx_sampling_2D
from make_tables import Table_cx_sampling_3D
from make_tables import Table_light_heavy_sampling
from make_tables import Tables_2d_T_n
from make_tables import Table_light_heavy_sampling_mccc
from make_tables import Table_light_heavy_sampling_from_spline
from make_tables import Table_simple
from matplotlib import rc
from make_tables import Table_light_heavy_sampling_from_spline_clean

SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 14

rc('axes', axisbelow=True)
rc('font', family='serif', size=SMALL_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

path_to_figures = '/home/kristoffer/Desktop/KU/speciale/figures/'
data_folder = '/home/kristoffer/BOUT-HESEL_kinetic_neutral/input_data/'
coef_folder = '/home/kristoffer/Desktop/KU/speciale/python_stuff/rate_coefficient/'
mccc_fit_folder = '/home/kristoffer/Desktop/KU/speciale/python_stuff/MCCC-fits/'

MASS_D_ATOM = 2.01410177811*1.660539e-27
MASS_ELECTRON = 9.1093837e-31
#HYDHEL and Janev
ion_sum_coefs = np.array([-3.271396786375e+1 , 1.353655609057e+1, -5.739328757388, 1.563154982022, -2.877056004391e-1, 3.482559773737e-2,-2.631976175590e-3, 1.119543953861e-4,-2.039149852002e-6])
ass_ion_sum_coefs = np.array([-3.568640293666e+1, 1.733468989961e+1, -7.767469363538, 2.211579405415, -4.169840174384e-1, 5.088289820867e-02, -3.832737518325e-03, 1.612863120371e-04, -2.893391904431e-06])
#NEW from HYDHEL
diss_sum_coefs = np.array([-2.787217511174e+01, 1.052252660075e+01, -4.973212347860e+00, 1.451198183114e+00, -3.062790554644e-01, 4.433379509258e-02, -4.096344172875e-03, 2.159670289222e-04, -4.928545325189e-06])

ion_2s_sum_coefs = np.array([-1.973476726029e+01, 3.992702671457e+00, -1.773436308973e+00, 5.331949621358e-01, -1.181042453190e-01, 1.763136575032e-02, -1.616005335321e-03, 8.093908992682e-05, -1.686664454913e-06])

s_to_p_sum_coefs = np.array([-1.219616012805e+01, -3.859057071006e-01, -6.509976401685e-03, 4.981099209058e-04, -4.184102479407e-05, 3.054358926267e-06, -1.328567638366e-07, 8.974535105058e-10, 1.010269574757e-10])

#a = Tables_1d(0.1, 300, 10000, 3, s_to_p_sum_coefs, data_folder + 's_to_p_table')
#ass_ion = Tables_1d(0.1, 300, 10000, 3, ass_ion_sum_coefs, data_folder + 'ass_ion_table')
#diss = Tables_1d(0.1, 300, 10000, 3, diss_sum_coefs, data_folder + 'diss_table')
#a.calc_electron_col_rate()
#ass_ion.calc_electron_col_rate()
#diss.calc_electron_col_rate()
#a.save_object()
#ass_ion.save_object()
#iss.save_object()
"""
cx_sum_coefs = np.loadtxt(coef_folder + 'cx_rate.txt')
cx = Tables_2d(0.1, 300, 0.1, 0.1, 1000, 0.1, cx_sum_coefs, data_folder + 'cx_2d_table')
cx.calc_heavy_col_rate()
#Adjust for the factor 2 mass difference
cx.rates = cx.rates/np.sqrt(2)
cx.save_object()
"""
#sum_coefs_cx_cross = np.array([-3.274123792568e+01, -8.916456579806e-02, -3.016990732025e-02, 9.205482406462e-03, 2.400266568315e-03, -1.927122311323e-03, 3.654750340106e-04, -2.788866460622e-05, 7.422296363524e-07])
#cx_rate_integrand = Table_cx_sampling_3D(0.1, 300, 100, 0.1, 1000, 100, 3, np.pi/100, 5, 100, sum_coefs_cx_cross, data_folder + 'cx_rate_integrand_3D', MASS_D_ATOM)
#cx_rate_integrand.tabulate()
#print(cx_rate_integrand.table.shape)
#cx_rate_integrand.save_object()

#sum_coefs_diss_cross = np.array([-1.019870329452e+05, 2.252601430192e+05, -2.158143676206e+05, 1.171042848075e+05, -3.936494849617e+04, 8.395340835067e+03, -1.109486871647e+03, 8.308421522823e+01, -2.699781210407e+00])
#table_diss = Table_light_heavy_sampling(0.1, 300, 1000, 3, 3, 1000, sum_coefs_diss_cross, data_folder + 'diss_rate_integrand', MASS_ELECTRON, 8.5)
#table_diss.tabulate()
#table_diss.save_object()

#effective_ion_rate_coefs = np.loadtxt(coef_folder + 'effective_ion_rate.txt')

#effective_ion_rate_table = Tables_2d_T_n(0.1, 300, 1000, 5e+17, 1e+20, 1000, effective_ion_rate_coefs, data_folder + 'effective_ion_rate')
#effective_ion_rate_table.calc_rate()
#effective_ion_rate_table.save_object()



#table_b3 = Table_light_heavy_sampling_mccc(0.1, 300, 1000, 3, 3, 1000, mccc_fit_folder + 'MCCC-el-D2-b3Su.X1Sg_vi=0_fit.txt', data_folder + 'b3_v0_cross_section')
#table_b3.tabulate()
#table_b3.save_object()

#table_triplets = Table_light_heavy_sampling_from_spline(0.1, 300, 1000, 3, 3, 1000, data_folder + 'triplet_spline', data_folder + 'triplet_cross', 0, 1400)
#table_triplets.tabulate()
#table_triplets.save_object()

#table_singlets = Table_light_heavy_sampling_from_spline(0.1, 300, 1000, 3, 3, 1000, data_folder + 'singlet_spline', data_folder + 'singlet_cross', 0, 1400)
#table_singlets.tabulate()
#table_singlets.save_object()
"""
table_De = Table_light_heavy_sampling_from_spline(0.1, 300, 1000, 3, 3, 1000, data_folder + 'DE_FC_spline', data_folder + 'DE_FC_cross', 0.3, 1000)
table_De.tabulate()
table_De.save_object()
"""
#table_Dr = Table_light_heavy_sampling_from_spline(0.1, 300, 1000, 3, 3, 1000, data_folder + 'DR_FC_spline', data_folder + 'DR_FC_cross', 0.2, 10)
#table_Dr.tabulate()
#table_Dr.save_object()

def fragment_energy_table_ass_ion():
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
    plt.figure()
    plt.plot(E_plot, spline(E_plot))
    table = Table_simple(e_min, dE, np.cumsum(spline(E_plot)), data_folder + 'ass_ion_fragment_KE')
    table.save_object()
    plt.show()

#fragment_energy_table_ass_ion()

def c3_cross_table():
    def cross_section(E):
        dE = 0.017
        x = E/0.017
        return 2.08/np.power(x, 1.20)*np.power(1-1/x, 3.80)*1e-15

    E_s = np.linspace(0.02, 200, 1000)
    cross = cross_section(E_s)
    spline = interp1d(E_s, cross, kind = 'cubic')
    with open(data_folder + 'c3_spline.pkl', 'wb') as f:
        pickle.dump(spline, f)
    plt.figure()
    plt.yscale('log')
    plt.plot(E_s, spline(E_s))
    table = Table_light_heavy_sampling_from_spline(0.1, 300, 1000, 4, 3, 1000, data_folder + 'c3_spline', data_folder +'c3_cross', 0.02, 200)
    table.tabulate()
    table.save_object()

def make_1D_Tables(T_min, T_max, T_res, base, sum_coefs, table_filenames, labels, figure_name):
    plt.figure()
    plt.title('Hydrogen 3n State Reaction Rates')
    plt.yscale('log')
    #plt.xscale('log')
    plt.grid(True, 'both')
    plt.xlim(2, 50)
    plt.ylim(1e-14, 5e-12)
    #plt.ylim(1e-14, 7e-12)
    plt.xlabel(r'$T_e$, [eV]')
    plt.ylabel(r'$\langle\sigma v\rangle$, [$\frac{m^3}{s}$]')
    plt.tight_layout()
    for i in np.arange(sum_coefs.shape[0]):
        table = Tables_1d(T_min, T_max, T_res, base, sum_coefs[i], data_folder + table_filenames[i])
        table.calc_electron_col_rate()
        table.save_object()
        plt.plot(table.Ts, table.rates, label = labels[i])
    plt.legend(loc = 'lower right')
    plt.savefig(path_to_figures + figure_name + '.eps', format = 'eps')


def hydrogen_stuff():
    #Hydrogen atom stuff
    H1s_to_H2p_coefs = np.array([-2.814949375869e+01, 1.009828023274e+01, -4.771961915818e+00, 1.467805963618e+00, -2.979799374553e-01, 3.861631407174e-02, -3.051685780771e-03, 1.335472720988e-04, -2.476088392502e-06])
    H1s_to_H2s_coefs = np.array([-2.833259375256e+01, 9.587356325603e+00, -4.833579851041e+00, 1.415863373520e+00, -2.537887918825e-01, 2.800713977946e-02, -1.871408172571e-03, 6.986668318407e-05, -1.123758504195e-06])
    H1s_ion_coefs = np.array([-3.271396786375e+01, 1.353655609057e+01, -5.739328757388e+00, 1.563154982022e+00, -2.877056004391e-01, 3.482559773737e-02, -2.631976175590e-03, 1.119543953861e-04, -2.039149852002e-06])
    H1s_to_H3_coefs = np.array([-3.113714569232e+01, 1.170494035550e+01, -5.598117886823e+00, 1.668467661343e+00, -3.186788446245e-01, 3.851704802605e-02, -2.845199866183e-03, 1.171512424827e-04, -2.059295818495e-06])
    sum_coef_arr = np.array([H1s_to_H2p_coefs, H1s_to_H2s_coefs, H1s_to_H3_coefs, H1s_ion_coefs])
    #make_1D_Tables(1, 300, 10000, 3, sum_coef_arr, ['1s_to_2p', '1s_to_2s', '1s_to_3n', '1s_direct_ion'], [r'$H(1s)\rightarrow H(2p)$', r'$H(1s)\rightarrow H(2s)$', r'$H(1s)\rightarrow H(3n)$', r'$H(1s)\rightarrow H^{+} + e$'], 'hydrogen_rate')

    H2s_to_H2p_coefs = np.array([-1.219616012805e+01, -3.859057071006e-01, -6.509976401685e-03, 4.981099209058e-04, -4.184102479407e-05, 3.054358926267e-06, -1.328567638366e-07, 8.974535105058e-10, 1.010269574757e-10])
    H2s_to_H3n_coefs = np.array([-1.515830911091e+01, 1.923956400537e+00, -9.275338417712e-01, 3.370367299915e-01, -8.758162223598e-02, 1.409066167839e-02, -1.325225954526e-03, 6.672025878086e-05, -1.387615199713e-06])
    H2s_ion_coefs = np.array([-1.973476726029e+01, 3.992702671457e+00, -1.773436308973e+00, 5.331949621358e-01, -1.181042453190e-01, 1.763136575032e-02, -1.616005335321e-03, 8.093908992682e-05, -1.686664454913e-06])
    sum_coef_arr_2s = np.array([H2s_to_H2p_coefs, H2s_to_H3n_coefs, H2s_ion_coefs])
    #make_1D_Tables(1, 300, 10000, 3, sum_coef_arr_2s, ['2s_to_2p', '2s_to_3n', '2s_direct_ion'], [r'$H(2s)\rightarrow H(2p)$', r'$H(2s)\rightarrow H(3n)$', r'$H(2s)\rightarrow H^{+} + e$'], 'hydrogen_2s_rate')

    H3n_ion_coefs = np.array([-1.566968719411e+01, 1.719661170920e+00, -8.365041963678e-01, 2.642794957304e-01, -6.527754894629e-02, 1.066883130107e-02, -1.041488149422e-03, 5.457216484634e-05, -1.177539827071e-06])
    sum_coefs_arr_3n = np.array([H3n_ion_coefs])
    make_1D_Tables(1, 300, 10000, 3, sum_coefs_arr_3n, ['H3n_ion'], [r'$H(3n)\rightarrow H^{+} + e$'], 'hydrogen_3n_rate')

def make_heavy_light_tables_from_splines(T_min, T_max, T_res, base, n_standard_deviations, v_res, spline_names, table_filenames, labels, figure_name):
    plt.figure()
    plt.title('Dissociation Rates for Molecules')
    plt.yscale('log')
    plt.grid(True, 'both')
    plt.xlim(0, 40)
    plt.ylim(1e-16, 3e-13)
    plt.xlabel(r'$T_e$, [eV]')
    plt.ylabel(r'$\langle \sigma v\rangle$, [$\frac{m^3}{s}$]')
    plt.tight_layout()
    for i in np.arange(spline_names.shape[0]):
        table = Table_light_heavy_sampling_from_spline_clean(T_min, T_max, T_res, base, n_standard_deviations, v_res, data_folder + spline_names[i], data_folder + table_filenames[i])
        table.tabulate()
        table.save_object()
        plt.plot(table.Ts, table.rates, label = labels[i])
    plt.legend(loc = 'lower right')
    plt.savefig(path_to_figures + figure_name + '.eps', format = 'eps')

def make_heavy_light_tables_from_splines_ratio(T_min, T_max, T_res, base, n_standard_deviations, v_res, spline_names, table_filenames, labels, figure_name):
    fig, ax = plt.subplots()
    ax.set_title('DE1 Rates for the Molecular Deuterium Ion')
    ax2 = ax.twinx()
    ax2.set_ylabel('Ratio')
    ax2.set_ylim(0, 30)
    ax.set_yscale('log')
    ax.grid(True, 'both')
    ax.set_xlim(0, 40)
    ax.set_ylim(1e-16, 3e-13)
    ax.set_xlabel(r'$T_e$, [eV]')
    ax.set_ylabel(r'$\langle \sigma v\rangle$, [$\frac{m^3}{s}$]')
    fig.tight_layout()
    for i in np.arange(spline_names.shape[0]):
        table = Table_light_heavy_sampling_from_spline_clean(T_min, T_max, T_res, base, n_standard_deviations, v_res, data_folder + spline_names[i], table_filenames[i])
        table.tabulate()
        table.save_object()
        ax.plot(table.Ts, table.rates, label = labels[i])
        if i == 0:
            table0 = table
        if i == 1:
            table1 = table
    ind_T = table0.get_inds_T(np.array([4.2]))[0]
    ax2.plot(table0.Ts[ind_T:], table0.rates[ind_T:]/table1.rates[ind_T:],  'k', label = r'$\frac{\langle \sigma v \rangle_{2p\sigma_u}}{\langle \sigma v \rangle_{2p\pi_u}}$')
    spline_rat = interp1d(table0.Ts[ind_T:], table0.rates[ind_T:]/table1.rates[ind_T:], kind='cubic', bounds_error = False, fill_value = 0)
    with open(data_folder + 'spline_rat_DR_DE' + '.pkl', 'wb') as f:
        pickle.dump(spline_rat, f)
    ax.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    #fig.savefig(path_to_figures + figure_name + '.eps', format = 'eps')

make_heavy_light_tables_from_splines(0.1, 300, 1000, 4, 3, 1000, np.array(['B1_C1_spline', 'Bp1_D1_spline', 'a3_c3_spline', 'b3_spline']), ['B1_C1_table', 'Bp1_D1_table', 'a3_c3_table', 'b3_table'], ['B1_C1', 'B\'1_D1', 'a3_c3', 'b3'], 'composite rates')
plt.show()
#make_heavy_light_tables_from_splines(0.1, 300, 1000, 4, 3, 1000, np.array(['a3_spline', 'b3_spline', 'c3_spline', 'd3_spline', 'e3_spline', 'h3_spline']), ['a3_table', 'b3_table', 'c3_table', 'd3_table', 'e3_table', 'h3_table'], ['a3', 'b3', 'c3', 'd3', 'e3', 'h3'], 'triplet_rates')

#make_heavy_light_tables_from_splines_ratio(0.1, 300, 1000, 4, 3, 1000, np.array(['sigma_spline', 'pi_spline']), ['sigma_table', 'pi_table'], [r'$2p\sigma_u$', r'$2p\pi_u$'], 'DE_rates_mole_ratio')
#make_heavy_light_tables_from_splines_ratio(0.1, 300, 1000, 4, 3, 1000, np.array(['DE_FC_spline', 'DR_FC_spline', 'diss_ion_table']), ['DE_mole', 'DR_mole', 'DI_mole'], ['DE', 'DR', 'DI'], 'ass_ion_rates_mole')

#b = np.loadtxt(coef_folder + 'effective_ion_molecule.txt')

#effective_ion_rate_molecule_table = Tables_2d_T_n(0.1, 300, 1000, 5e+17, 1e+20, 1000, b, data_folder + 'effective_ion_rate_molecule')
#effective_ion_rate_molecule_table.calc_rate()
#effective_ion_rate_molecule_table.save_object()

#n_ind = effective_ion_rate_molecule_table.get_inds_n(np.array([1.0e+19]))
#plt.figure()
#plt.plot(effective_ion_rate_molecule_table.Ts, effective_ion_rate_molecule_table.rates[n_ind[0], :])
#hydrogen_stuff()
#plt.show()
