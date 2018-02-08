
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from tqdm import tqdm

from lmfit import Parameters, minimize, report_fit, Minimizer
from yang_yang_1dbg import bethe_integrator

# Fundamental Constants
hbar = constants.codata.value('Planck constant over 2 pi')
a0 = constants.codata.value('Bohr radius')
uma = constants.codata.value('atomic mass constant')
kB = constants.codata.value('Boltzmann constant')
pi = np.pi  

mass = 86.909180527*uma

# Data
raw_data_h5 = 'raw_data.h5'
processed_data_h5 = 'processed_data.h5'
fit_outputs_h5 = 'fit_outputs.h5'

# Data load/save methods
def h5_save(group, name, array):
    """Set a h5 dataset to the given array, replacing existing values if it
    already exists. If the arrays do not have the same shape and datatype,
    then delete the existing dataset before replaing it. This will leave
    garbage in the h5 file that will need to be cleaned up if one wants to
    reclaim the disk space with h5repack. Having to do this is the main
    downside of using hdf5 for this instead of np.save/np.load/np.memmap"""
    try:
        group[name] = array
    except RuntimeError:
        try:
            # Try overwriting just the data if shape and dtype are compatible:
            group[name][:] = array
        except TypeError:
            del group[name]
            group[name] = array
            import sys
            msg = ('Warning: replacing existing h5 dataset, but disk space ' +
                   'has not been reclaimed, leaving the h5 file larger ' +
                   'than necessary. To reclaim disk space use the h5repack ' +
                   'tool or delete the whole h5 file regenerate from scratch\n')
            sys.stderr.write(msg)

def load_data():
    with h5py.File(processed_data_h5, 'r') as processed_data:
        data = np.array(processed_data['linear_density'])
        data[10, 646] = 0.
        uncertainty = np.array(processed_data['u_linear_density'])
        # The values of the realisation variables:
        final_dipole_realisations = np.array(processed_data['realisation_final_dipole'])
        short_TOF_realisations = np.array(processed_data['realisation_short_TOF'])
    return data, uncertainty, final_dipole_realisations, short_TOF_realisations

def bin_density_slice(density_slice, bin_size=2):
    return density_slice.reshape(-1, bin_size).mean(axis=1)

    #####################################################################################
    #####                                                                           #####
    #####                   ### ###  ###### #####   ###### ##                       #####
    #####                   ## # ##  ##  ## ##  ##  ##     ##                       #####
    #####                   ##   ##  ##  ## ##  ##  ###### ##                       #####
    #####                   ##   ##  ##  ## ##  ##  ##     ##                       #####
    #####                   ##   ##  ###### #####   ###### ######                   #####
    #####                                                                           #####
    #####################################################################################

def YY_thermodynamics(trans_freq, mass, temperature, chemical_potential, 
                      scatt_length, returned=None):
    YYsolver = bethe_integrator(trans_freq, mass, temperature, 
                                chemical_potential, scatt_length)
    n, g1D = YYsolver.density()
    try:
        if returned == 'epsilon':
            return YYsolver.get_k_space(10, 2**10), YYsolver.eps_solver(eps_tol=1e-10)
        elif returned == 'f_k':
            return YYsolver.get_k_space(10, 2**10), YYsolver.f_solver(f_tol=1e-10)
        elif returned == 'entropy':
            return YYsolver.entropy_per_particle()
        elif returned == 'pressure':
            return YYsolver.pressure()
        elif returned == 'g1D':
            return g1D
        else:
            return n
    except ValueError:
        import sys
        msg = ('Warning: returned argument is invalid, supported args are' +
               '"epsilon" for the dispersion, "f_k" for quasimomentum spectrum' +
               '"entropy" for entropy, "pressure" for pressure and None for density\n' )
        sys.stderr.write(msg)

def V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t, break_LDA=False):
    def anti_trap_model(x, A_a, x0_a, dx_a):
        return A_a/(1+((x-x0_a)**2/(dx_a/2)**2))
    def long_trap_model(x, A_t, x0_t, dx_t):
        return A_t*(np.exp(-(x-x0_t)**2/(2*dx_t**2)))
    V_a = anti_trap_model(x, A_a, x0_a, dx_a)
    V_t = long_trap_model(x, A_t, x0_t, dx_t)
    V = (V_a + V_t) - (V_a + V_t).min()
    if break_LDA:
        local_maxima = np.r_[True, V[1:]>V[:-1]] & np.r_[V[:-1]>V[1:], True]
        local_maxima_indices = np.array(np.where(local_maxima))[0, 0], np.array(np.where(local_maxima))[0,-1]
        htrap_a = 0.5*mass*(2*pi*9.09)**2*((x-x[local_maxima_indices[0]])*1e-6)**2/(hbar*2*pi)
        htrap_b = 0.5*mass*(2*pi*9.09)**2*((x-x[local_maxima_indices[1]])*1e-6)**2/(hbar*2*pi)
        V[0:local_maxima_indices[0]] += htrap_a[0:local_maxima_indices[0]]
        V[local_maxima_indices[1]::] += htrap_b[local_maxima_indices[1]::]
    return V_a, V_t, V

    #####################################################################################
    #####                                                                           #####
    #####                   ##       ######   ###### ######  ##                     #####
    #####                   ##       ##  ##   ##     ##  ##  ##                     #####
    #####                   ##       ##  ##   ##     ######  ##                     #####
    #####                   ##       ##  ##   ##     ##  ##  ##                     #####
    #####                   ######   ######   ###### ##  ##  ######                 #####
    #####                                                                           #####
    #####################################################################################

def compute_nx(x, mu0, T0, A_a, x0_a, dx_a, A_t, x0_t, dx_t):    
    V_anti, _, V_total = V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t, break_LDA=True)
    f_perp = V_anti
    mui = mu0 - V_total
    density_profile = []
    for j in (range(np.size(V_total))):
        density_profile.append(YY_thermodynamics(trans_freq=f_perp[j], 
                                                 mass=mass, 
                                                 temperature=T0, 
                                                 chemical_potential=mui[j], 
                                                 scatt_length=110*a0))
    return np.array(density_profile)

def lmfit_nx(xdata, ydata, dydata):
    params = Parameters()
    params.add('Global_chemical_potential', value=1.6e3, min=-1e3, max=3e3, vary=True)
    params.add('Temperature', value=180e-9, min=1e-10, max=1e-6, vary=True)
    params.add('Antitrap_height', value=19.91e3, min=8e3, max=25e3, vary=True)
    params.add('Antitrap_center', value=-1.34, min=-80, max=80, vary=True)
    params.add('Antitrap_width', value=150.84, min=120, max=620, vary=True)
    params.add('Trap_depth', value=-33325, min=-60e3, max=-8e3, vary=True)
    params.add('Trap_center', value=-9.75, min=-50, max=70, vary=False)
    params.add('Trap_width', value=69.87, min=60, max=180, vary=True)
    # Residuals
    def residuals_nx(pars, xdata, ydata, epsdata):
        mu0 = pars['Global_chemical_potential']
        T0 = pars['Temperature']
        A_anti = pars['Antitrap_height']
        x0_anti = pars['Antitrap_center']
        dx_anti = pars['Antitrap_width']
        A_trap = pars['Trap_depth']
        x0_trap = pars['Trap_center']
        dx_trap = pars['Trap_width']
        model = compute_nx(xdata, mu0, T0, A_anti, x0_anti, 
                           dx_anti, A_trap, x0_trap, dx_trap)
        return (ydata-model)/epsdata
    # Callback function
    def nx_callback(params, iteration, resid, *fcn_args, **fcn_kws):
        print(f'Iteration: {iteration:2d}')
        return None
    return minimize(residuals_nx, params, args=(xdata, ydata, dydata), 
                    method='leastsq', iter_cb=nx_callback, nan_policy='omit')

    #####################################################################################
    #####                                                                           #####
    #####                   ######   ##     ######  ###### ######  ##               #####
    #####                   ##       ##     ##  ##  ##  ## ##  ##  ##               #####
    #####                   ##  ##   ##     ##  ##  ###### ######  ##               #####
    #####                   ##  ##   ##     ##  ##  ##  ## ##  ##  ##               #####
    #####                   ######   ###### ######  ###### ##  ##  ######           #####
    #####                                                                           #####
    #####################################################################################

def compute_nxT(x, mu0, mu1, mu2, mu3, mu4, mu5, 
                mu6, mu7, mu8, mu9, mu10, mu11, 
                mu12, mu13, mu14, mu15, mu16, 
                mu17, mu18, mu19, mu20, mu21, 
                m22, m23, T0, T1, T2, 
                T3, T4, T5, T6, T7, 
                T8, T9, T10, T11, T12, 
                T13, T14, T15, T16, T17, 
                T18, T19, T20, T21, T22, 
                T23, A_a, x0_a, dx_a,
                A_t, x0_t, dx_t):
    mus = ([mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, 
            mu8, mu9, mu10, mu11, mu12, mu13, 
            mu14, mu15, mu16, mu17, mu18,
            mu19, mu20, mu21, m22, m23])
    Ts =  ([T0, T1, T2, T3, T4, T5, T6,
            T7, T8, T9, T10, T11, T12, T13,
            T14, T15, T16, T17, T18,
            T19, T20, T21, T22, T23])
    V_anti, _, V_total = V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t, break_LDA=True)
    f_perp = V_anti
    n = np.zeros((24, int(648/4)))
    for T_index in range(24):
        glob_mu, glob_T = mus[T_index], Ts[T_index]
        mu_index = 0
        for mui in tqdm(glob_mu - V_total):
            n[T_index, mu_index] = YY_thermodynamics(trans_freq=f_perp[mu_index], 
                                                     mass=mass, 
                                                     temperature=glob_T, 
                                                     chemical_potential=mui, 
                                                     scatt_length=110*a0)
            mu_index += 1
    return n

def lmfit_nxT(xdata, ydata, dydata):
    params = Parameters()
    params.add('mu0', value=1.8e3, min=-1e3, max=3e3, vary=True)
    params.add('mu1', value=1.9e3, min=-1e3, max=3e3, vary=True)
    params.add('mu2', value=1.7e3, min=-1e3, max=3e3, vary=True)
    params.add('mu3', value=1.6e3, min=-1e3, max=3e3, vary=True)
    params.add('mu4', value=1.5e3, min=-1e3, max=3e3, vary=True)
    params.add('mu5', value=1.4e3, min=-1e3, max=3e3, vary=True)
    params.add('mu6', value=1.2e3, min=-1e3, max=3e3, vary=True)
    params.add('mu7', value=1.2e3, min=-1e3, max=3e3, vary=True)
    params.add('mu8', value=1.1e3, min=-1e3, max=2.5e3, vary=True)
    params.add('mu9', value=1.1e3, min=-1e3, max=2.5e3, vary=True)
    params.add('mu10', value=1.0e3, min=-1e3, max=2.5e3, vary=True)
    params.add('mu11', value=0.9e3, min=-1e3, max=2e3, vary=True)
    params.add('mu12', value=0.8e3, min=-1e3, max=2e3, vary=True)
    params.add('mu13', value=0.7e3, min=-1e3, max=2e3, vary=True)
    params.add('mu14', value=0.6e3, min=-1e3, max=2e3, vary=True)
    params.add('mu15', value=0.5e3, min=-1e3, max=2e3, vary=True)
    params.add('mu16', value=0.4e3, min=-1e3, max=2e3, vary=True)
    params.add('mu17', value=0.3e3, min=-1e3, max=2e3, vary=True)
    params.add('mu18', value=0.2e3, min=-1e3, max=2e3, vary=True)
    params.add('mu19', value=0.5e3, min=-1e3, max=2e3, vary=True)
    params.add('mu20', value=0.4e3, min=-1e3, max=2e3, vary=True)
    params.add('mu21', value=0.3e3, min=-1e3, max=2e3, vary=True)
    params.add('mu22', value=0.2e3, min=-1e3, max=2e3, vary=True)
    params.add('mu23', value=0.1e3, min=-1e3, max=2e3, vary=True)
    params.add('T0', value=190e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T1', value=180e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T2', value=170e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T3', value=160e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T4', value=150e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T5', value=140e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T6', value=130e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T7', value=120e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T8', value=110e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T9', value=100e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T10', value=90e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T11', value=80e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T12', value=70e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T13', value=60e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T14', value=50e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T15', value=40e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T16', value=30e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T17', value=20e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T18', value=15e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T19', value=50e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T20', value=50e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T21', value=50e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T22', value=50e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('T23', value=50e-9, min=1e-10, max=1.5e-6, vary=True)
    params.add('Antitrap_height', value=16.13e3, min=8e3, max=22e3, vary=True)
    params.add('Antitrap_center', value=-7.21, min=-70, max=70, vary=True)
    params.add('Antitrap_width', value=2*185.2, min=300, max=450, vary=True)
    params.add('Trap_depth', value=-25e3, min=-50e3, max=-8e3, vary=True)
    params.add('Trap_center', value=-9.95, min=-70, max=70, vary=False)
    params.add('Trap_width', value=139.92, min=70, max=250, vary=True)
    # Residuals
    def residuals_nxT(pars, xdata, ydata, epsdata):
        mu0, T0, mu1, T1 = pars['mu0'], pars['T0'], pars['mu1'], pars['T1'] 
        mu2, T2, mu3, T3 = pars['mu2'], pars['T2'], pars['mu3'], pars['T3'] 
        mu4, T4, mu5, T5 = pars['mu4'], pars['T4'], pars['mu5'], pars['T5']
        mu6, T6, mu7, T7 = pars['mu6'], pars['T6'], pars['mu7'], pars['T7']
        mu8, T8, mu9, T9 = pars['mu8'], pars['T8'], pars['mu9'], pars['T9']
        mu10, T10, mu11, T11 = pars['mu10'], pars['T10'], pars['mu11'], pars['T11']
        mu12, T12, mu13, T13 = pars['mu12'], pars['T12'], pars['mu13'], pars['T13']
        mu14, T14, mu15, T15 = pars['mu14'], pars['T14'], pars['mu15'], pars['T15']
        mu16, T16, mu17, T17 = pars['mu16'], pars['T16'], pars['mu17'], pars['T17']
        mu18, T18, mu19, T19 = pars['mu18'], pars['T18'], pars['mu19'], pars['T19'] 
        mu20, T20, mu21, T21 = pars['mu20'], pars['T20'], pars['mu21'], pars['T21']
        mu22, T22, mu23, T23 = pars['mu22'], pars['T22'], pars['mu23'], pars['T23']
        A_anti, x0_anti, dx_anti = pars['Antitrap_height'], pars['Antitrap_center'], pars['Antitrap_width']
        A_trap,  x0_trap, dx_trap = pars['Trap_depth'], pars['Trap_center'], pars['Trap_width']
        model = compute_nxT(xdata, mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, 
                                   mu8, mu9, mu10, mu11, mu12, mu13, mu14,
                                   mu15, mu16, mu17, mu18, mu19, mu20, mu21, 
                                   mu22, mu23, T0, T1, T2, T3, T4, T5, 
                                   T6, T7, T8, T9, T10, T11, T12, T13, 
                                   T14, T15, T16, T17, T18, T19, T20, 
                                   T21, T22, T23, A_anti, x0_anti, dx_anti, 
                                   A_trap, x0_trap, dx_trap)
        return (ydata-model).flatten()/epsdata.flatten()
    # Callback function
    def nxT_callback(params, iteration, resid, *fcn_args, **fcn_kws):
        with h5py.File(fit_outputs_h5) as fit_outputs:
            if iteration == 1:
                del fit_outputs['partial_residuals']
                del fit_outputs['partial_sigmas']
            if iteration > 1:
                partial_residuals = list(fit_outputs['partial_residuals'])
                partial_sigmas = list(fit_outputs['partial_sigmas'])
            else:
                partial_residuals = []
                partial_sigmas = []
            sigma_res = np.std(resid)
            partial_residuals.append(resid)
            partial_sigmas.append(sigma_res)
            if (iteration % 100 == 0) or (iteration == 1):
                h5_save(fit_outputs, 'partial_residuals', partial_residuals)        
            h5_save(fit_outputs, 'partial_sigmas', partial_sigmas)
        print(f'Iteration: {iteration:2d}')
        return None
    minimizer = Minimizer(residuals_nxT, params, fcn_args=(xdata, ydata, dydata), 
                          iter_cb=nxT_callback, nan_policy='omit')
    return minimizer.minimize()

    #####################################################################################
    #####                                                                           #####
    #####                   ##       ######   ###### ######  ##                     #####
    #####                   ##       ##  ##   ##     ##  ##  ##                     #####
    #####                   ##       ##  ##   ##     ######  ##                     #####
    #####                   ##       ##  ##   ##     ##  ##  ##                     #####
    #####                   ######   ######   ###### ##  ##  ######                 #####
    #####                                                                           #####
    #####################################################################################

def local_fit(slice_index=None):
    eos_data, u_eos_data, _, _ = load_data()
    binned_n_data = np.array([bin_density_slice(nj, bin_size=4) for nj in eos_data])
    binned_u_n_data = np.array([bin_density_slice(nj, bin_size=4) for nj in u_eos_data])
    x_data = np.linspace(-np.size(binned_n_data[0,:])/2, 
                          np.size(binned_n_data[0,:])/2, 
                          np.size(binned_n_data[0,:]))
    fit_output, pot_output, temp_set, mu0_set, pot_set, redchi = [], [], [], [], [], []
    u_mu0_set, u_temps_set, u_pot_set, cov_mat_set = [], [], [], []
    def fit_indexed_slice(slice_index):
        result = lmfit_nx(x_data, binned_n_data[slice_index], binned_u_n_data[slice_index])
        report_fit(result)
        cov_mat_set.append((result.covar))
        fit_pars = np.array([result.params[key].value for key in result.params.keys()])
        fit_pars_err = np.sqrt(np.diag(result.covar))
        mu0_set.append(fit_pars[0])
        temp_set.append(fit_pars[1])
        pot_set.append(fit_pars[2::])
        u_mu0_set.append(fit_pars_err[0])
        u_temps_set.append(fit_pars_err[1])
        u_pot_set.append(fit_pars_err[2::])
        redchi.append(result.redchi)
        fit_output.append(compute_nx(x_data, *fit_pars))
        pot_output.append(V_potential_model(x_data, *fit_pars[2::], break_LDA=True)[2])
    if slice_index is not None:
        fit_indexed_slice(slice_index)
    else:
        for index in tqdm(list(range(24))):
            fit_indexed_slice(index)
    with h5py.File(fit_outputs_h5) as fit_outputs:
        h5_save(fit_outputs, 'local_fit_mu0_set', np.array(mu0_set))
        h5_save(fit_outputs, 'local_fit_temp_set', np.array(temp_set))
        h5_save(fit_outputs, 'local_fit_pot_set', np.array(pot_set))
        h5_save(fit_outputs, 'local_fit_redchi_set', np.array(redchi))
        h5_save(fit_outputs, 'local_fit_u_mu0_set', np.array(u_mu0_set))
        h5_save(fit_outputs, 'local_fit_u_temp_set', np.array(u_temps_set))
        h5_save(fit_outputs, 'local_fit_u_pot_set', np.array(u_pot_set))
        h5_save(fit_outputs, 'local_fit_covariance_matrix', np.array(cov_mat_set))
        h5_save(fit_outputs, 'local_fit_density_output', np.array(fit_output))
        h5_save(fit_outputs, 'local_fit_pot_output', np.array(pot_output))

    #####################################################################################
    #####                                                                           #####
    #####                   ######   ##     ######  ###### ######  ##               #####
    #####                   ##       ##     ##  ##  ##  ## ##  ##  ##               #####
    #####                   ##  ##   ##     ##  ##  ###### ######  ##               #####
    #####                   ##  ##   ##     ##  ##  ##  ## ##  ##  ##               #####
    #####                   ######   ###### ######  ###### ##  ##  ######           #####
    #####                                                                           #####
    #####################################################################################

def global_fit():
    eos_data, u_eos_data, _, _ = load_data()
    binned_n_data = np.array([bin_density_slice(nj, bin_size=4) for nj in eos_data])
    binned_u_n_data = np.array([bin_density_slice(nj, bin_size=4) for nj in u_eos_data])
    x_data = np.linspace(-np.size(binned_n_data[0,:])/2, 
                          np.size(binned_n_data[0,:])/2, 
                          np.size(binned_n_data[0,:]))
    glob_fit_result = lmfit_nxT(x_data, binned_n_data, binned_u_n_data)
    glob_cov_matrix = glob_fit_result.covar
    glob_fit_pars = np.array([glob_fit_result.params[key].value for key in glob_fit_result.params.keys()])
    glob_fit_pars_err = np.sqrt(np.diag(glob_cov_matrix))
    glob_fit_density = compute_nxT(x_data, *glob_fit_pars)
    glob_fit_potential = V_potential_model(x_data, *glob_fit_pars[-6::], break_LDA=False)[2]
    with h5py.File(fit_outputs_h5) as fit_outputs:
        h5_save(fit_outputs, 'global_fit_mu0_set', glob_fit_pars[0:24])
        h5_save(fit_outputs, 'global_fit_temp_set', glob_fit_pars[24:48])
        h5_save(fit_outputs, 'global_fit_pot_set', glob_fit_pars[-6::])
        h5_save(fit_outputs, 'global_fit_redchi', glob_fit_result.redchi)
        h5_save(fit_outputs, 'global_fit_u_mu0_set', glob_fit_pars_err[0:24])
        h5_save(fit_outputs, 'global_fit_u_temp_set', glob_fit_pars_err[24:48])
        h5_save(fit_outputs, 'global_fit_u_pot_set', glob_fit_pars_err[-6::])
        h5_save(fit_outputs, 'global_fit_covariance_matrix', np.array(glob_cov_matrix))
        h5_save(fit_outputs, 'global_fit_density_output', glob_fit_density)
        h5_save(fit_outputs, 'global_fit_pot_output', glob_fit_potential)
    import IPython
    IPython.embed()

if __name__ == '__main__':
    #local_fit(slice_index=0)
    global_fit()