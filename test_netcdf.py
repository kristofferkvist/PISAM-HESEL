import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib import rc

MASS_D_ATOM = 2.01410177811*1.660539e-27
MASS_D_MOLECULE = 2*2.014102*1.660539e-27

path_to_figures = '/home/kristoffer/Desktop/KU/speciale/figures/'

SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 14

my_dpi = 96
n_pixels = 800

rc('axes', axisbelow=True)
rc('font', family='serif', size=SMALL_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_velocity_hists(hists_vs, bin_edges, species, legend, x_min, x_max, fit=False, mass=MASS_D_ATOM, T_guess = 100, fit_gamma = False, guess_gamma = np.array([2, 2])):
    num_domains = hists_vs.shape[0]
    for i in np.arange(num_domains):
        title = 'Velocity distribution of ' + species +'. \n Domain: ' + f'{x_min*100 + (x_max-x_min)/num_domains*i*100:.1f}' + 'cm to ' + f'{x_min*100 + (x_max-x_min)/num_domains*(i+1)*100:.1f}' + 'cm'
        if fit:
            if fit_gamma:
                plot_bar(hists_vs[i, :], bin_edges, title, '[m/s]', legend, fit=fit, mass=mass, T_guess=T_guess,  fit_gamma = fit_gamma, guess_gamma = guess_gamma)
            else:
                plot_bar(hists_vs[i, :], bin_edges, title, '[m/s]', legend, fit=fit, mass=mass, T_guess=T_guess)
        else:
            if fit_gamma:
                plot_bar(hists_vs[i, :], bin_edges, title, '[m/s]', legend, fit_gamma = fit_gamma, guess_gamma = guess_gamma)
            else:
                plot_bar(hists_vs[i, :], bin_edges, title, '[m/s]', legend)

def plot_bar(hist_data, bin_edges, title, xlabel, legend='', fit=False, mass=1, T_guess = 100, fit_gamma = False, guess_gamma = np.array([100, 0.01])):
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths/2
    #hist_data = hist_data*bin_centers
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(r'$P(T)$')
    #plt.xlim(0, 120)
    plt.grid(True, 'both')
    plt.ticklabel_format(axis = 'both', style = 'sci', scilimits = (0, 0), useMathText = True)
    plt.tight_layout()
    if fit:
        popt, _ = curve_fit(fit_func(mass), bin_centers, hist_data/(np.sum(hist_data)*bin_widths[0]), p0 = T_guess)
        v_plot = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        plt.plot(v_plot, fit_func(mass)(v_plot, popt[0]), color='tab:orange', linestyle='-', linewidth=2.0, label='Fit to MB, T_fit = ' + f'{popt[0]:.2f}' + ' eV')
    if fit_gamma:
        bin_edges_fit = bin_edges/(bin_edges[-1])*20
        bin_widths_fit = np.diff(bin_edges_fit)
        bin_centers_fit = bin_edges_fit[:-1] + bin_widths_fit/2
        popt, _ = curve_fit(gamma_dist, bin_centers_fit, hist_data/(np.sum(hist_data)*bin_widths_fit[0]), p0 = guess_gamma, bounds = ([0, 0], [100000, 100000]))
        gamma_vs = np.linspace(bin_edges_fit[0], bin_edges_fit[-1], 1000)
        gamma_plot = gamma_dist(gamma_vs, popt[0], popt[1])
        v_plot = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        plt.plot(v_plot, gamma_plot/bin_edges[-1]*20, color='tab:orange', linestyle='--', linewidth=2.0, label='Fit to Gamma dist')
    plt.bar(bin_centers, hist_data/(np.sum(hist_data)*bin_widths[0]), color='tab:blue', width=bin_widths, edgecolor='k', linewidth=1, label=legend)
    plt.legend()
    #plt.savefig(path_to_figures + 'mol_ion_temp' + '.eps', format = 'eps')

def plot_velocity_hists_combined(hist_vs, hist_vs_cx, bin_edges, x_min, x_max, mass = MASS_D_ATOM, T_guess = 5):
    hist_not_cx = hist_vs-hist_vs_cx
    num_domains = hist_not_cx.shape[0]
    hists = [hist_vs_cx, hist_not_cx]
    colors = ['tab:blue', 'tab:orange']
    labels = [r'$P(v)_{cx}$', r'$P(v)_{Franck Condon}$']
    styles = ['-', '--']
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths/2
    for i in np.arange(num_domains):
        plt.figure()
        plt.title('Velocity distribution of atoms.\n Domain: ' + f'{x_min*100 + (x_max-x_min)/num_domains*i*100:.1f}' + 'cm to ' + f'{x_min*100 + (x_max-x_min)/num_domains*(i+1)*100:.1f}' + 'cm')
        plt.xlabel('[m/s]')
        plt.ylabel(r'$P(v)$')
        plt.ticklabel_format(axis = 'both', style = 'sci', scilimits = (0, 0), useMathText = True)
        plt.tight_layout()
        for j in range(len(hists)):
            hist = hists[j]
            #popt, _ = curve_fit(fit_func(mass), bin_centers, hist[i, :]/(np.sum(hist[i, :])*bin_widths[0]), p0 = T_guess)
            #v_plot = np.linspace(bin_edges[0], bin_edges[-1], 1000)
            #plt.plot(v_plot, fit_func(mass)(v_plot, popt[0]), color='k', linestyle=styles[j], linewidth=2.0, label='Fit to MB, T_fit = ' + f'{popt[0]:.2f}' + ' eV')
            plt.bar(bin_centers, hist[i, :]/(np.sum(hist[i, :])*bin_widths[0]), color=colors[j], width=bin_widths, edgecolor='k', linewidth=1, label=labels[j])
        #plt.axis([0, 200000, 0, 1.1*np.max(hist[i, :]/(np.sum(hist[i, :])*bin_widths[0]))])
        plt.legend()

def calc_fluxes(dts, n_diff, n_sources, Ly, Lz):
    fluxes = np.zeros_like(n_diff)
    for i in np.arange(fluxes.shape[0]):
        for j in np.flip(np.arange(fluxes.shape[1])):
            if j < (fluxes.shape[1] - 1):
                fluxes[i, j] = (n_diff[i, j] - n_sources[i, j] + fluxes[i, j+1])
            else:
                fluxes[i, j] = (n_diff[i, j] - n_sources[i, j])
        fluxes[i, :] = fluxes[i, :]/dts[i]
    return fluxes/(Ly*Lz)

def plot_fluxes(fluxes, label):
    plt.figure()
    plt.title('Flux ' + label)
    plt.grid(True, 'both')
    plt.plot(np.arange(fluxes.shape[1]), np.mean(fluxes, axis = 0))

def plot_densities(densities, label):
    plt.figure()
    plt.title('Desnity ' + label)
    plt.grid(True, 'both')
    plt.plot(np.arange(densities.shape[1]), np.mean(densities, axis = 0))

def lin_func(x, a, b):
    return a*x + b

def transport_coefficients(fluxes, densities, t_min, ind_min, ind_max, dx, xs, label, fit):
    flux = np.mean(fluxes[t_min:, ind_min:ind_max], axis = 0)
    density = np.mean(densities[t_min:, ind_min:ind_max], axis = 0)
    grad = np.gradient(density, dx)
    lambda_recip = -grad/density
    plt.figure()
    plt.title('Density')
    plt.grid(True, 'both')
    plt.tight_layout()
    plt.xlabel(r'Radial coordinate, [m]')
    plt.ylabel(r'$n$, [$m^{-3}$]')
    plt.plot(xs, np.mean(densities[t_min:, :], axis = 0), 'r.', markersize = 5, label=label)
    y_min = np.min(np.mean(densities[t_min:, :], axis = 0))
    y_max = np.max(np.mean(densities[t_min:, :], axis = 0))
    diff = y_max-y_min
    y_min = y_min-0.1*diff
    y_max = y_max+0.1*diff
    plt.plot([xs[ind_min], xs[ind_min]], [y_min, y_max], 'k')
    plt.plot([xs[ind_max], xs[ind_max]], [y_min, y_max], 'k')
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.title('Flux')
    plt.grid(True, 'both')
    plt.xlabel(r'Radial coordinate, [m]')
    plt.ylabel(r'$\Gamma$, [$\frac{1}{m^2s}$]')
    plt.plot(xs, np.mean(fluxes[t_min:, :], axis = 0), 'r.', markersize = 5, label=label)
    y_min = np.min(np.mean(fluxes[t_min:, :], axis = 0))
    y_max = np.max(np.mean(fluxes[t_min:, :], axis = 0))
    diff = y_max-y_min
    y_min = y_min-0.1*diff
    y_max = y_max+0.1*diff
    plt.plot([xs[ind_min], xs[ind_min]], [y_min, y_max], 'k')
    plt.plot([xs[ind_max], xs[ind_max]], [y_min, y_max], 'k')
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.tight_layout()

    if fit:
        a, b, _, _, _ = linregress(lambda_recip, flux/density)

    plt.figure()
    plt.title('Transport Coefficients')
    plt.xlabel(r'$\frac{1}{\lambda_n}$')
    plt.ylabel(r'$\frac{\Gamma}{n}$, [$\frac{m}{s}$]')
    plt.grid(True, 'both')
    plt.plot(lambda_recip, flux/density, 'r.', markersize = 5, label=label)
    if fit:
        plt.plot(lambda_recip, lin_func(lambda_recip, a, b), 'k', label = 'Fit')
        print("slope = " + str(a))
        print("Intersect = " + str(b))
    plt.legend()
    plt.tight_layout()

def downsample(hist, bins, n):
    hist_reshape = np.reshape(hist, (hist.shape[0], int(hist.shape[1]/n), n))
    new_hist = np.sum(hist_reshape, axis = -1)
    bins_reshape = np.reshape(bins[0:-1], (int(bins[0:-1].size/n), n))
    new_bins = np.zeros(int((bins.size-1)/n) + 1)
    new_bins[0:-1] = np.min(bins_reshape, axis = -1)
    new_bins[-1] = bins[-1]
    return new_hist, new_bins

#ds = nc.Dataset('/home/kristoffer/Desktop/KU/speciale/python_stuff/netcdf_data/neutral_diagnostics.nc')
ds = nc.Dataset('data/neutral_diagnostics.nc')
print(ds.variables.keys())

dts = np.array(ds['dt'])
molecule_injection_rate = np.array(ds['molecule_injection_rate'])[0]
weight = np.array(ds['weight'])[0]
Ly = np.array(ds['Ly'])[0]
dx = Ly/512
xs = np.arange(512)*dx
Lz = 1

""" Precautions for the one wrong dataset
n_atoms_x = np.mean(ds['n_atom_x_y_t'], axis = 2)*0.12598911
n_atoms_cx_x = np.mean(ds['n_atom_cx_x_y_t'], axis = 2)*0.12598911
n_molecules_x = np.mean(ds['n_molecule_x_y_t'], axis = 2)*0.12598911
"""

n_atoms_x = np.mean(ds['n_atom'], axis = 2)
n_atoms_cx_x = np.mean(ds['n_atom_cx'], axis = 2)
n_molecules_x = np.mean(ds['n_molecule'], axis = 2)
n_atoms_not_cx_x = n_atoms_x - n_atoms_cx_x

speed_dist_atom = np.mean(ds['speed_dist_atom'], axis = 0)
speed_dist_atom_cx = np.mean(ds['speed_dist_atom_cx'], axis = 0)
speed_dist_mol = np.mean(ds['speed_dist_mol'], axis = 0)

bins_atom = np.array(ds['bin_edges_atom'])
bins_mol = np.array(ds['bin_edges_mol'])

"""n_atom_sources = np.array(ds['n_atom_x_source'])*weight
n_atom_cx_sources = np.array(ds['n_atom_cx_x_source'])*weight
n_molecule_sources = np.array(ds['n_molecule_x_source'])*weight

n_atom_diff = np.array(ds['n_atom_x_diff'])*weight
n_atom_cx_diff = np.array(ds['n_atom_cx_x_diff'])*weight
n_molecule_diff = np.array(ds['n_molecule_x_diff'])*weight

n_atoms_not_cx_x = n_atoms_x - n_atoms_cx_x
n_atom_not_cx_sources = n_atom_sources - n_atom_cx_sources
n_atom_not_cx_diff = n_atom_diff - n_atom_cx_diff"""

#atom_not_cx_fluxes = calc_fluxes(dts, n_atom_not_cx_diff, n_atom_not_cx_sources, Ly, Lz)
#atom_cx_fluxes = calc_fluxes(dts, n_atom_cx_diff, n_atom_cx_sources, Ly, Lz)
#molecule_fluxes = calc_fluxes(dts, n_molecule_diff, n_molecule_sources, Ly, Lz)

#plot_fluxes(molecule_fluxes, 'Molecules')
#plot_fluxes(atom_not_cx_fluxes, 'Atoms')
#plot_fluxes(atom_cx_fluxes, 'CX')

plot_densities(n_molecules_x, 'Molecules')
plot_densities(n_atoms_not_cx_x, 'Atoms')
plot_densities(n_atoms_cx_x, 'CX')

#plot_velocity_hists(speed_dist_atom, bins_atom, 'Atoms', 'Atoms', 0, Ly)
hist_atoms, bins = downsample(speed_dist_atom, bins_atom, 20)
hist_atoms_cx, _ = downsample(speed_dist_atom_cx, bins_atom, 20)
plot_velocity_hists_combined(hist_atoms, hist_atoms_cx, bins, 0, Ly)

#plot_velocity_hists(hist_atoms, bins, 'Atoms', 'Atoms', 0, Ly)
#plot_velocity_hists(hist_atoms_cx, bins, 'Atoms_cx', 'Atoms_cx', 0, Ly)

#hist_mols, bins_mol = downsample(speed_dist_mol, bins_mol, 5)
#plot_velocity_hists(hist_mols, bins_mol, 'Molecules', 'Molecules', 0, Ly)

#transport_coefficients(atom_not_cx_fluxes, n_atoms_not_cx_x, 100, 100, 500, dx, xs, 'Warm Atoms', False)
#transport_coefficients(atom_cx_fluxes, n_atoms_cx_x, 100, 20, 500, dx, xs, 'CX Atoms', False)
#transport_coefficients(molecule_fluxes, n_molecules_x, 100, 200, 500, dx, xs, 'Molecules', False)

#transport_coefficients(atom_not_cx_fluxes, n_atoms_not_cx_x, 100, 360, 405, dx, xs, 'Warm Atoms', True)
#transport_coefficients(atom_cx_fluxes, n_atoms_cx_x, 100, 250, 405, dx, xs, 'CX Atoms', True)
#transport_coefficients(molecule_fluxes, n_molecules_x, 100, 250, 320, dx, xs, 'Molecules', True)

plt.show()
