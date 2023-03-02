import numpy as np
from boutdata import collect
import netCDF4 as nc
from datetime import datetime
import sys

def main(argv):
    #dt_string = datetime.now().strftime("date_%d_%m_%Y_time_%H_%M_%S")
    field_list  = ['t_array', 'n', 'rhos', 'dx', 'nx', 'dz', 'nz', 'oci', 'te', 'ti', 'n0', 'Te0', 'Ti0', 'B0', 'x_lcfs', 'B', 'phi']
    #n_files = int(argv[1])
    #name_list = map(int, argv[1].strip('[]').split(','))
    name_list = [48, 49]
    for i in name_list:
        print('#############################NAME' + str(i)+ '#######################')
        #if (len(argv) == 0):
        #    path        = 'data'
        #    filename = 'profiles/profiles_test_' + dt_string + '.nc'
        #    t_ind_min = 0
        if (argv[0] == 'no_neutrals'):
            path = '../kristoffer/no_neutrals' + str(i)
            filename = 'profiles/profiles_no_neutrals' + str(i) + '.nc'
            t_ind_min = 100
        elif (argv[0] == 'fluid'):
            path = '../kristoffer/fluid_neutrals' + str(i)
            filename = 'profiles/profiles_fluid' + str(i) + '.nc'
            field_list = field_list + ['Ncold', 'Nwarm', 'Nhot', 'Sn', 'Spe', 'Spi']
            t_ind_min = 0
        elif (argv[0] == 'kinetic'):
            path = '../kristoffer/kinetic_neutrals' + str(i)
            filename = 'profiles/profiles_kinetic' + str(i) + '.nc'
            field_list = field_list + ['Sn', 'Spe', 'Spi']
            t_ind_min = 100

    #ds = nc.Dataset('data/BOUT.dmp.0.nc')
    #print("NO NEUTRALS:")
    #print(ds.variables.keys())
    #ds = nc.Dataset('../kristoffer/fluid_neutrals/BOUT.dmp.0.nc')
    #print("WITH NEUTRALS")
    #print(ds.variables.keys())

        data = {}

        for _field in field_list:
            data[_field] = collect(_field, path = path, xguards = (2, 0, 0))

        n0 = np.array(data['n0'])
        Te0 = np.array(data['Te0'])
        Ti0 = np.array(data['Ti0'])
        B0 = np.array(data['B0'])

        dx = np.array(data['dx'])[0]
        nx = np.array(data['nx']).astype(np.int32)
        dy = np.array(data['dz'])
        ny = np.array(data['nz']).astype(np.int32)
        rhos = np.array(data['rhos'])
        xs = np.arange(nx)*dx*rhos
        oci = np.array(data['oci'])
        ts = np.array(data['t_array'])/oci
        print(ts.size)
        ts = ts[t_ind_min:]
        print(n0)
        print(Te0)
        print(Ti0)
        print(B0)
        print(oci)
        print(rhos)
        x_lcfs = np.array(data['x_lcfs'])
        x_lcfs_ind = np.ceil(x_lcfs*nx).astype(np.int32)
        xs = xs - xs[x_lcfs_ind]
        n = np.squeeze(data['n'])*n0
        n = n[t_ind_min:, :, :]
        n_x_t = np.mean(n, axis = 2)
        te = np.squeeze(data['te'])*Te0
        te = te[t_ind_min:, :, :]
        te_x_t = np.mean(te, axis = 2)
        ti = np.squeeze(data['ti'])*Ti0
        ti = ti[t_ind_min:, :, :]
        ti_x_t = np.mean(ti, axis = 2)
        pe = np.multiply(n, te)
       	pe_x_t = np.mean(pe, axis = 2)
        pi = np.multiply(n, ti)
        pi_x_t = np.mean(pi, axis = 2)
        phi = np.squeeze(data['phi'])*Te0
        phi = phi[t_ind_min:, :, :]
        phi_x_t = np.mean(phi, axis=2)
        B = np.squeeze(np.array(data['B']))*B0
        _, Ex, Ey = np.gradient(phi)
        Ex = -Ex/(dx*rhos)
        Ey = -Ey/(dy*rhos)
        Ex_x_t = np.mean(Ex, axis = 2)
        Ey_x_t = np.mean(Ey, axis = 2)
        rad_flux_x_t = np.mean(np.multiply(n, Ey), axis = 2)/B
        rad_flux_Ee_x_t = np.mean(np.multiply(pe, Ey), axis = 2)/B
        rad_flux_Ei_x_t = np.mean(np.multiply(pi, Ey), axis = 2)/B

        #Densities fluid neutrals
        if ((len(argv) > 0) and (argv[0] == 'fluid')):
            N_cold_x_t = np.mean(data['Ncold'], axis = 3)
            N_cold_x_t = np.squeeze(N_cold_x_t[t_ind_min:, 2:-2])*n0
            N_warm_x_t = np.mean(data['Nwarm'], axis = 3)
            N_warm_x_t = np.squeeze(N_warm_x_t[t_ind_min:, 2:-2])*n0
            N_hot_x_t = np.mean(data['Nhot'], axis = 3)
            N_hot_x_t = np.squeeze(N_hot_x_t[t_ind_min:, 2:-2])*n0

        #Densities Kinetic neutrals
        if ((len(argv) > 0) and (argv[0] == 'kinetic')):
            ds = nc.Dataset(path + '/kinetic_hists.nc')
            print(ds.variables.keys())
            n_atoms_x_t = np.array(ds['n_atom_x_t'])[t_ind_min-1:, :]
            n_atoms_cx_x_t = np.array(ds['n_atom_cx_x_t'])[t_ind_min-1:, :]
            n_molecules_x_t = np.array(ds['n_molecule_x_t'])[t_ind_min-1:, :]
            N_cold_x_t = n_molecules_x_t
            N_warm_x_t = n_atoms_x_t - n_atoms_cx_x_t
            N_hot_x_t = n_atoms_cx_x_t

        #Sources if neutrals are included
        if ((len(argv) > 0) and (argv[0] == 'fluid' or argv[0] == 'kinetic')):
            Sn_x_t = np.mean(data['Sn'], axis = 3)
            Sn_x_t = np.squeeze(Sn_x_t[t_ind_min:, :])*n0*oci
            Spe_x_t = np.mean(data['Spe'], axis = 3)
            Spe_x_t = np.squeeze(Spe_x_t[t_ind_min:, :])*n0*Te0*oci
            Spi_x_t = np.mean(data['Spi'], axis = 3)
            Spi_x_t = np.squeeze(Spi_x_t[t_ind_min:, :])*n0*Ti0*oci

        nc_dat = nc.Dataset(filename, 'w', 'NETCDF4')
        nc_dat.createDimension('x', xs.size)
        nc_dat.createDimension('x_ng', xs.size-4)
        nc_dat.createDimension('t', ts.size)
        t_var = nc_dat.createVariable('ts', 'float32', ('t'))
        t_var[:] = ts
        x_var = nc_dat.createVariable('xs', 'float32', ('x'))
        x_var[:] = xs
        n_var = nc_dat.createVariable('n_x_t', 'float32', ('t', 'x'))
        n_var[:, :] = n_x_t
        te_var = nc_dat.createVariable('te_x_t', 'float32', ('t', 'x'))
        te_var[:, :] = te_x_t
        ti_var = nc_dat.createVariable('ti_x_t', 'float32', ('t', 'x'))
        ti_var[:, :] = ti_x_t
        pe_var = nc_dat.createVariable('pe_x_t', 'float32', ('t', 'x'))
        pe_var[:, :] = pe_x_t
        pi_var = nc_dat.createVariable('pi_x_t', 'float32', ('t', 'x'))
        pi_var[:, :] = pi_x_t
        phi_var = nc_dat.createVariable('phi_x_t', 'float32', ('t', 'x'))
        phi_var[:, :] = phi_x_t
        Ex_var = nc_dat.createVariable('Ex_x_t', 'float32', ('t', 'x'))
        Ex_var[:, :] = Ex_x_t
        Ey_var = nc_dat.createVariable('Ey_x_t', 'float32', ('t', 'x'))
        Ey_var[:, :] = Ey_x_t
        B_var = nc_dat.createVariable('B', 'float32', ('x'))
        B_var[:] = B
        rad_flux_var = nc_dat.createVariable('rad_flux_x_t', 'float32', ('t', 'x'))
        rad_flux_var[: , :] = rad_flux_x_t
        rad_flux_Ee_var = nc_dat.createVariable('rad_flux_Ee_x_t', 'float32', ('t', 'x'))
        rad_flux_Ee_var[: , :] = rad_flux_Ee_x_t
        rad_flux_Ei_var = nc_dat.createVariable('rad_flux_Ei_x_t', 'float32', ('t', 'x'))
        rad_flux_Ei_var[: , :] = rad_flux_Ei_x_t
        if ((len(argv) > 0) and (argv[0] == 'fluid' or argv[0] == 'kinetic')):
            Ncold_var = nc_dat.createVariable('Ncold_x_t', 'float32', ('t', 'x_ng'))
            Ncold_var[:, :] = N_cold_x_t
            Nwarm_var = nc_dat.createVariable('Nwarm_x_t', 'float32', ('t', 'x_ng'))
            Nwarm_var[:, :] = N_warm_x_t
            Nhot_var = nc_dat.createVariable('Nhot_x_t', 'float32', ('t', 'x_ng'))
            Nhot_var[:, :] = N_hot_x_t
            Sn_var = nc_dat.createVariable('Sn_x_t', 'float32', ('t', 'x'))
            Sn_var[:, :] = Sn_x_t
            Spe_var = nc_dat.createVariable('Spe_x_t', 'float32', ('t', 'x'))
            Spe_var[:, :] = Spe_x_t
            Spi_var = nc_dat.createVariable('Spi_x_t', 'float32', ('t', 'x'))
            Spi_var[:, :] = Spi_x_t
        nc_dat.close()

if __name__ == "__main__":
   main(sys.argv[1:])
