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
            #path = '../kristoffer/kinetic_neutrals' + str(i)
            #filename = 'profiles/profiles_kinetic' + str(i) + '.nc'
            field_list = field_list + ['Sn', 'Spe', 'Spi', 'Sux', 'Suz']
            #t_ind_min = 100
            path = 'data'
            filename = 'profile_test.nc'
            t_ind_min = 0

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
        EV = 1.602e-19
        mi = 2*1.67e-27

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
        n_t_x = np.mean(n, axis = 2)
        te = np.squeeze(data['te'])*Te0
        te = te[t_ind_min:, :, :]
        te_t_x = np.mean(te, axis = 2)
        ti = np.squeeze(data['ti'])*Ti0
        ti = ti[t_ind_min:, :, :]
        ti_t_x = np.mean(ti, axis = 2)
        pe = np.multiply(n, te)
       	pe_t_x = np.mean(pe, axis = 2)
        pi = np.multiply(n, ti)
        pi_t_x = np.mean(pi, axis = 2)
        phi = np.squeeze(data['phi'])*Te0
        phi = phi[t_ind_min:, :, :]
        phi_t_x = np.mean(phi, axis=2)
        print("Phi shape = " + str(phi_t_x.shape))
        B = np.squeeze(np.array(data['B']))*B0
        _, Ex, Ey = np.gradient(phi)
        Ex = -Ex/(dx*rhos)
        Ey = -Ey/(dy*rhos)
        Ex_t_x = np.mean(Ex, axis = 2)
        Ey_t_x = np.mean(Ey, axis = 2)
        rad_flux_t_x = np.mean(np.multiply(n, Ey), axis = 2)/B
        rad_flux_Ee_t_x = np.mean(np.multiply(pe, Ey), axis = 2)/B
        rad_flux_Ei_t_x = np.mean(np.multiply(pi, Ey), axis = 2)/B

        #Densities fluid neutrals
        if ((len(argv) > 0) and (argv[0] == 'fluid')):
            N_cold_t_x = np.mean(data['Ncold'], axis = 3)
            N_cold_t_x = np.squeeze(N_cold_t_x[t_ind_min:, 2:-2])*n0
            N_warm_t_x = np.mean(data['Nwarm'], axis = 3)
            N_warm_t_x = np.squeeze(N_warm_t_x[t_ind_min:, 2:-2])*n0
            N_hot_t_x = np.mean(data['Nhot'], axis = 3)
            N_hot_t_x = np.squeeze(N_hot_t_x[t_ind_min:, 2:-2])*n0

        #Densities and speed dists Kinetic neutrals
        if ((len(argv) > 0) and (argv[0] == 'kinetic')):
            ds = nc.Dataset(path + '/neutral_diagnostics.nc')
            print(ds.variables.keys())
            N_atom_t_x = np.mean(ds['n_atom'], axis = 2)
            N_hot_t_x = np.mean(ds['n_atom_cx'], axis = 2)
            N_cold_t_x = np.mean(ds['n_molecule'], axis = 2)
            N_warm_t_x = N_atom_t_x - N_hot_t_x

            speed_dist_atom = np.mean(ds['speed_dist_atom'], axis = 0)
            speed_dist_atom_cx = np.mean(ds['speed_dist_atom_cx'], axis = 0)
            speed_dist_mol = np.mean(ds['speed_dist_mol'], axis = 0)
            bins_atom = np.array(ds['bin_edges_atom'])
            bins_mol = np.array(ds['bin_edges_mol'])

        #Sources if neutrals are included
        if ((len(argv) > 0) and (argv[0] == 'fluid' or argv[0] == 'kinetic')):
            Sn_t_x = np.mean(data['Sn'], axis = 3)
            Sn_t_x = np.squeeze(Sn_t_x[t_ind_min:, :])*n0*oci
            Spe_t_x = np.mean(data['Spe'], axis = 3)
            Spe_t_x = np.squeeze(Spe_t_x[t_ind_min:, :])*n0*Te0*oci
            Spi_t_x = np.mean(data['Spi'], axis = 3)
            Spi_t_x = np.squeeze(Spi_t_x[t_ind_min:, :])*n0*Ti0*oci
            if argv[0] == 'kinetic':
                Sux_t_x = np.mean(data['Sux'], axis = 3)
                Sux_t_x = np.squeeze(Sux_t_x[t_ind_min:, :])*np.sqrt(Te0*EV*mi)*n0*oci
                Suz_t_x = np.mean(data['Suz'], axis = 3)
                Suz_t_x = np.squeeze(Suz_t_x[t_ind_min:, :])*np.sqrt(Te0*EV*mi)*n0*oci

        nc_dat = nc.Dataset(filename, 'w', 'NETCDF4')
        nc_dat.createDimension('x', xs.size)
        nc_dat.createDimension('x_ng', xs.size-4)
        nc_dat.createDimension('t', ts.size)
        nc_dat.createDimension('v_doms', speed_dist_atom.shape[0])
        nc_dat.createDimension('n_bins_atom', speed_dist_atom.shape[1])
        nc_dat.createDimension('n_bins_mol', speed_dist_mol.shape[1])
        nc_dat.createDimension('bin_edges_atom', bins_atom.size)
        nc_dat.createDimension('bin_edges_mol', bins_mol.size)
        t_var = nc_dat.createVariable('ts', 'float32', ('t'))
        t_var[:] = ts
        x_var = nc_dat.createVariable('xs', 'float32', ('x'))
        x_var[:] = xs
        n_var = nc_dat.createVariable('n_t_x', 'float32', ('t', 'x'))
        n_var[:, :] = n_t_x
        te_var = nc_dat.createVariable('te_t_x', 'float32', ('t', 'x'))
        te_var[:, :] = te_t_x
        ti_var = nc_dat.createVariable('ti_t_x', 'float32', ('t', 'x'))
        ti_var[:, :] = ti_t_x
        pe_var = nc_dat.createVariable('pe_t_x', 'float32', ('t', 'x'))
        pe_var[:, :] = pe_t_x
        pi_var = nc_dat.createVariable('pi_t_x', 'float32', ('t', 'x'))
        pi_var[:, :] = pi_t_x
        phi_var = nc_dat.createVariable('phi_t_x', 'float32', ('t', 'x'))
        phi_var[:, :] = phi_t_x
        Ex_var = nc_dat.createVariable('Ex_t_x', 'float32', ('t', 'x'))
        Ex_var[:, :] = Ex_t_x
        Ey_var = nc_dat.createVariable('Ey_t_x', 'float32', ('t', 'x'))
        Ey_var[:, :] = Ey_t_x
        B_var = nc_dat.createVariable('B', 'float32', ('x'))
        B_var[:] = B
        rad_flux_var = nc_dat.createVariable('rad_flux_t_x', 'float32', ('t', 'x'))
        rad_flux_var[: , :] = rad_flux_t_x
        rad_flux_Ee_var = nc_dat.createVariable('rad_flux_Ee_t_x', 'float32', ('t', 'x'))
        rad_flux_Ee_var[: , :] = rad_flux_Ee_t_x
        rad_flux_Ei_var = nc_dat.createVariable('rad_flux_Ei_t_x', 'float32', ('t', 'x'))
        rad_flux_Ei_var[: , :] = rad_flux_Ei_t_x
        if ((len(argv) > 0) and (argv[0] == 'fluid' or argv[0] == 'kinetic')):
            Ncold_var = nc_dat.createVariable('Ncold_t_x', 'float32', ('t', 'x_ng'))
            Ncold_var[:, :] = N_cold_t_x
            Nwarm_var = nc_dat.createVariable('Nwarm_t_x', 'float32', ('t', 'x_ng'))
            Nwarm_var[:, :] = N_warm_t_x
            Nhot_var = nc_dat.createVariable('Nhot_t_x', 'float32', ('t', 'x_ng'))
            Nhot_var[:, :] = N_hot_t_x
            Sn_var = nc_dat.createVariable('Sn_t_x', 'float32', ('t', 'x'))
            Sn_var[:, :] = Sn_t_x
            Spe_var = nc_dat.createVariable('Spe_t_x', 'float32', ('t', 'x'))
            Spe_var[:, :] = Spe_t_x
            Spi_var = nc_dat.createVariable('Spi_t_x', 'float32', ('t', 'x'))
            Spi_var[:, :] = Spi_t_x
        if ((len(argv) > 0) and (argv[0] == 'kinetic')):
            Sux_var = nc_dat.createVariable('Sux_t_x', 'float32', ('t', 'x'))
            Sux_var[:, :] = Sux_t_x
            Suz_var = nc_dat.createVariable('Suz_t_x', 'float32', ('t', 'x'))
            Suz_var[:, :] = Suz_t_x
            speed_dist_atom_var = nc_dat.createVariable('speed_dist_atom', 'float32', ('v_doms', 'n_bins_atom'))
            speed_dist_atom_var[:, :] = speed_dist_atom
            speed_dist_atom_cx_var = nc_dat.createVariable('speed_dist_atom_cx', 'float32', ('v_doms', 'n_bins_atom'))
            speed_dist_atom_cx_var[:, :] = speed_dist_atom_cx
            speed_dist_mol_var = nc_dat.createVariable('speed_dist_mol', 'float32', ('v_doms', 'n_bins_mol'))
            speed_dist_mol_var[:, :] = speed_dist_mol
            bins_atom_var = nc_dat.createVariable('bins_atom', 'float32', ('bin_edges_atom'))
            bins_atom_var[:] = bins_atom
            bins_mol_var = nc_dat.createVariable('bins_mol', 'float32', ('bin_edges_mol'))
            bins_mol_var[:] = bins_mol
        nc_dat.close()

if __name__ == "__main__":
   main(sys.argv[1:])
