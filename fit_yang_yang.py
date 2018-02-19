
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
        uncertainty[10, 646] = 0.
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

def compute_nxT(x, mus, Ts, A_a, x0_a, dx_a, A_t, x0_t, dx_t):
    V_anti, _, V_total = V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t, break_LDA=True)
    f_perp = V_anti
    n = np.zeros((len(mus), np.size(x)))
    for T_index in range(len(Ts)):
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

def lmfit_nxT(xdata, ydata, dydata, add_to_fit, mu_guess=None, T_guess=None):
    params = Parameters()
    if mu_guess is None:
        mu_guess = 1e3*np.ones_like(add_to_fit)
    if T_guess is None:
        T_guess = 50e-9*np.ones_like(add_to_fit)
    def add_mu_parameter(i, fix=False):
        params.add('mu'+str(i), value=mu_guess[i], min=-1e3, max=3e3, vary = not fix)
    def add_T_parameter(i, fix=False):
        params.add('T'+str(i), value=T_guess[i], min=1e-10, max=1.5e-6, vary = not fix)
    for slice_index in add_to_fit:
        add_mu_parameter(slice_index, fix=False)
    for slice_index in add_to_fit:
        add_T_parameter(slice_index, fix=False)   
    params.add('Antitrap_height', value=16.13e3, min=12e3, max=22e3, vary=True)
    params.add('Antitrap_center', value=-7.21, min=-20, max=20, vary=True)
    params.add('Antitrap_width', value=2*185.2, min=300, max=450, vary=True)
    params.add('Trap_depth', value=-25e3, min=-50e3, max=-1e3, vary=True)
    params.add('Trap_center', value=-9.95, min=-20, max=20, vary=True)
    params.add('Trap_width', value=139.92, min=90, max=250, vary=True)
    # Residuals
    def residuals_nxT(pars, xdata, ydata, epsdata):
        mus ,Ts = [], []
        for slice_index in add_to_fit:
            mus.append(pars['mu'+str(slice_index)])
            Ts.append(pars['T'+str(slice_index)])
        A_anti, x0_anti, dx_anti = pars['Antitrap_height'], pars['Antitrap_center'], pars['Antitrap_width']
        A_trap,  x0_trap, dx_trap = pars['Trap_depth'], pars['Trap_center'], pars['Trap_width']
        model = compute_nxT(xdata, mus, Ts, A_anti, x0_anti, dx_anti, A_trap, x0_trap, dx_trap)
        return (ydata-model).flatten()/epsdata.flatten()
    # Callback function
    def nxT_callback(params, iteration, resid, *fcn_args, **fcn_kws):
        with h5py.File(fit_outputs_h5) as fit_outputs:
            if iteration == 1:
                try:
                    del fit_outputs['partial_residuals']
                    del fit_outputs['partial_sigmas']
                except KeyError:
                    print("New file")
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
    data_subset = np.array([ydata[j, :] for j in add_to_fit])
    u_data_subset = np.array([dydata[j, :] for j in add_to_fit])
    minimizer = Minimizer(residuals_nxT, params, fcn_args=(xdata, data_subset, u_data_subset), 
                          iter_cb=nxT_callback, nan_policy='omit')
    return minimizer.minimize(method='leastsq')

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
    x_data = np.linspace(-np.size(eos_data[0,:])/2, 
                          np.size(eos_data[0,:])/2, 
                          np.size(binned_n_data[0,:]))
    subset = list(range(10, 24))
    mu_guess = np.linspace(1e3, 100, 24).tolist()
    T_guess = np.linspace(100e-9, 50e-9, 24).tolist()
    glob_fit_result = lmfit_nxT(x_data, binned_n_data, binned_u_n_data,
                                add_to_fit=subset, mu_guess=mu_guess, T_guess=T_guess)
    report_fit(glob_fit_result)
    glob_cov_matrix = np.array(glob_fit_result.covar)
    glob_fit_pars = np.array([glob_fit_result.params[key].value for key in glob_fit_result.params.keys()])
    glob_fit_pars_err = np.sqrt(np.diag(glob_cov_matrix))
    x_data = np.linspace(-np.size(eos_data[0,:])/2, 
                          np.size(eos_data[0,:])/2, 
                          np.size(eos_data[0,:]))
    mus, u_mus = glob_fit_pars[0:len(subset)], glob_fit_pars_err[0:len(subset)]
    Ts, u_Ts = glob_fit_pars[len(subset):2*len(subset)], glob_fit_pars_err[len(subset):2*len(subset)]
    glob_fit_density = compute_nxT(x_data, mus, Ts, *glob_fit_pars[-6::])
    _, _, glob_fit_potential = V_potential_model(x_data, *glob_fit_pars[-6::], break_LDA=True)
    with h5py.File(fit_outputs_h5) as fit_outputs:
        h5_save(fit_outputs, 'global_fit_mu0_set', mus)
        h5_save(fit_outputs, 'global_fit_temp_set', Ts)
        h5_save(fit_outputs, 'global_fit_pot_set', glob_fit_pars[-6::])
        h5_save(fit_outputs, 'global_fit_redchi', glob_fit_result.redchi)
        h5_save(fit_outputs, 'global_fit_u_mu0_set', u_mus)
        h5_save(fit_outputs, 'global_fit_u_temp_set', u_Ts)
        h5_save(fit_outputs, 'global_fit_u_pot_set', glob_fit_pars_err[-6::])
        h5_save(fit_outputs, 'global_fit_covariance_matrix', glob_cov_matrix)
        h5_save(fit_outputs, 'global_fit_density_output', glob_fit_density)
        h5_save(fit_outputs, 'global_fit_pot_output', glob_fit_potential)
    import IPython
    IPython.embed()

if __name__ == '__main__':
    #local_fit(slice_index=None)
    global_fit()