import numpy as np
import netCDF4 as nc
import pickle

class Domain:
    def __init__(self, x_min, x_max, y_min, y_max, r_minor, r_major, plasma_dim_x, plasma_dim_y, step_length, n0):
        self.electron_mass = 9.1093837e-31
        self.d_ion_mass = 2.01410177811*1.660539e-27 - self.electron_mass
        self.d_molecule_mass = 2*2.014102*1.660539e-27
        self.EV = 1.60217663e-19
        self.t = 0
        self.step_length = step_length
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.dx = (x_max-x_min)/plasma_dim_x
        self.dy = (y_max-y_min)/plasma_dim_y
        self.dt = step_length
        self.r_minor = r_minor
        self.r_major = r_major
        self.n0 = n0
        self.plasma_dim_x = plasma_dim_x
        self.plasma_dim_y = plasma_dim_y
        self.electron_source_particle = np.zeros((plasma_dim_x, plasma_dim_y))
        self.electron_source_energy = np.zeros_like(self.electron_source_particle)
        self.ion_source_momentum_x = np.zeros_like(self.electron_source_particle)
        self.ion_source_momentum_y = np.zeros_like(self.electron_source_particle)
        self.ion_source_energy = np.zeros_like(self.electron_source_particle)
        self.time_array = np.zeros(9)
        self.wall_times = np.zeros(8)
        self.last_plasma_timestep_length = 0

    def set_sources_zero(self):
        self.electron_source_particle.fill(0)
        self.electron_source_energy.fill(0)
        self.ion_source_momentum_x.fill(0)
        self.ion_source_momentum_y.fill(0)
        self.ion_source_energy.fill(0)

    def kinetic_energy(self, vx, vy, vz, mass):
        return 0.5*mass*(np.power(vx, 2) + np.power(vy, 2) + np.power(vz, 2))/self.EV

    def add_to_time(self):
        self.wall_times = self.wall_times + np.abs(np.diff(self.time_array))
        self.time_array = 0

    """
    def save_object(self, filename):
        nc_dat = nc.Dataset(filename, 'w', 'NETCDF4') # using netCDF4 for output format
        nx, ny = self.electron_source_particle.shape
        nc_dat.createDimension('nx', nx)
        nc_dat.createDimension('nx', ny)
        electron_source_particle = nc_dat.createVariable('electron_source_particle', 'float32', ('nx', 'ny'))
        electron_source_particle[:] = self.electron_source_particle
        electron_source_energy = nc_dat.createVariable('electron_source_energy', 'float32', ('nx', 'ny'))
        electron_source_energy[:] = self.electron_source_energy
        ion_source_energy = nc_dat.createVariable('ion_source_energy', 'float32', ('nx', 'ny'))
        ion_source_energy[:] = self.ion_source_energy
    """

    def save_object(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def save_object_nc(self, filename):
        nc_dat = nc.Dataset(filename, 'w', 'NETCDF4') # using netCDF4 for output format
        nc_dat.createDimension('nx', self.plasma_dim_x)
        nc_dat.createDimension('ny', self.plasma_dim_y)
        nc_dat.createDimension('time', 1)
        t = nc_dat.createVariable('t', 'float32', ('time'))
        t[:] = self.t
        last_plasma_timestep_length = nc_dat.createVariable('last_plasma_timestep_length', 'float32', ('time'))
        last_plasma_timestep_length[:] = self.last_plasma_timestep_length
        electron_source_particle = nc_dat.createVariable('electron_source_particle', 'float32', ('nx', 'ny'))
        electron_source_particle[:] = self.electron_source_particle
        electron_source_energy = nc_dat.createVariable('electron_source_energy', 'float32', ('nx', 'ny'))
        electron_source_energy[:] = self.electron_source_energy
        ion_source_energy = nc_dat.createVariable('ion_source_energy', 'float32', ('nx', 'ny'))
        ion_source_energy[:] = self.ion_source_energy

    def load_object_nc(self, filename):
        nc_dat = nc.Dataset(filename)
        self.electron_source_particle = np.array(nc_dat['electron_source_particle'])
        self.electron_source_energy = np.array(nc_dat['electron_source_energy'])
        self.ion_source_energy = np.array(nc_dat['ion_source_energy'])
        self.t = np.array(nc_dat['t'])
        self.last_plasma_timestep_length = np.array(nc_dat['last_plasma_timestep_length'])
