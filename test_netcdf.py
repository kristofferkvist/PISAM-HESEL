import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib import rc

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

ds = nc.Dataset('/home/kristoffer/Desktop/KU/speciale/python_stuff/netcdf_data/neutral_diagnostics.nc')

dts = np.array(ds['dt'])
molecule_injection_rate = np.array(ds['molecule_injection_rate'])[0]
weight = np.array(ds['weight'])[0]
Ly = np.array(ds['Ly'])[0]
dx = Ly/512
xs = np.arange(512)*dx
Lz = 1

n_atoms_x = np.mean(ds['n_atom_x_y_t'], axis = 2)*0.12598911
n_atoms_cx_x = np.mean(ds['n_atom_cx_x_y_t'], axis = 2)*0.12598911
n_molecules_x = np.mean(ds['n_molecule_x_y_t'], axis = 2)*0.12598911

n_atom_sources = np.array(ds['n_atom_x_source'])*weight
n_atom_cx_sources = np.array(ds['n_atom_cx_x_source'])*weight
n_molecule_sources = np.array(ds['n_molecule_x_source'])*weight

n_atom_diff = np.array(ds['n_atom_x_diff'])*weight
n_atom_cx_diff = np.array(ds['n_atom_cx_x_diff'])*weight
n_molecule_diff = np.array(ds['n_molecule_x_diff'])*weight

n_atoms_not_cx_x = n_atoms_x - n_atoms_cx_x
n_atom_not_cx_sources = n_atom_sources - n_atom_cx_sources
n_atom_not_cx_diff = n_atom_diff - n_atom_cx_diff

atom_not_cx_fluxes = calc_fluxes(dts, n_atom_not_cx_diff, n_atom_not_cx_sources, Ly, Lz)
atom_cx_fluxes = calc_fluxes(dts, n_atom_cx_diff, n_atom_cx_sources, Ly, Lz)
molecule_fluxes = calc_fluxes(dts, n_molecule_diff, n_molecule_sources, Ly, Lz)

#plot_fluxes(molecule_fluxes, 'Molecules')
#plot_fluxes(atom_not_cx_fluxes, 'Atoms')
#plot_fluxes(atom_cx_fluxes, 'CX')

#plot_densities(n_molecules_x, 'Molecules')
#plot_densities(n_atoms_not_cx_x, 'Atoms')
#plot_densities(n_atoms_cx_x, 'CX')

#transport_coefficients(atom_not_cx_fluxes, n_atoms_not_cx_x, 100, 100, 500, dx, xs, 'Warm Atoms', False)
#transport_coefficients(atom_cx_fluxes, n_atoms_cx_x, 100, 20, 500, dx, xs, 'CX Atoms', False)
#transport_coefficients(molecule_fluxes, n_molecules_x, 100, 200, 500, dx, xs, 'Molecules', False)

transport_coefficients(atom_not_cx_fluxes, n_atoms_not_cx_x, 100, 360, 405, dx, xs, 'Warm Atoms', True)
transport_coefficients(atom_cx_fluxes, n_atoms_cx_x, 100, 250, 405, dx, xs, 'CX Atoms', True)
transport_coefficients(molecule_fluxes, n_molecules_x, 100, 250, 320, dx, xs, 'Molecules', True)

plt.show()
