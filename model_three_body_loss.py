import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import odeint
from tqdm import tqdm

from lmfit import Parameters, minimize, report_fit, Minimizer
from yang_yang_1dbg import bethe_integrator

s, um, nm, ms = 1.0, 1e-6, 1e-9, 1e-3
nK, kHz = 1e-9, 1e3

# Fundamental Constants
hbar = constants.codata.value('Planck constant over 2 pi')
a0 = constants.codata.value('Bohr radius')
uma = constants.codata.value('atomic mass constant')
kB = constants.codata.value('Boltzmann constant')
pi = np.pi  

# Scattering cross section
mass = 86.909180527*uma
sigma_0 = 3*(780.24e-9)**2/(2*pi) 

# Pixel size in the object plane
pix_size = 5.6e-6/6.66

# Data
raw_data_h5 = 'raw_data.h5'
processed_data_h5 = 'processed_data.h5'
fit_outputs_h5 = 'fit_outputs.h5'
sfit_outputs_h5 = 'fit_outputs_subset.h5'
three_body_h5 = 'three_body.h5'

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

def load_data(dataset_h5, dataset_id):
    with h5py.File(dataset_h5, 'r') as dataset_h5:
        pulled_data = np.array(dataset_h5[dataset_id])
    return pulled_data

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

def model_pars_in_time(n_points, show_plots=False):
    full_density_dataset = load_data(processed_data_h5, 'linear_density')
    full_density_dataset[10, 646] = 0.
    mu0_set = load_data(fit_outputs_h5, 'global_fit_mu0_set')
    u_mu0_set = load_data(fit_outputs_h5, 'global_fit_u_mu0_set')
    temp_set = load_data(fit_outputs_h5, 'global_fit_temp_set')
    u_temp_set = load_data(fit_outputs_h5, 'global_fit_u_temp_set')
    time_realization = load_data(processed_data_h5, 'realisation_short_TOF')
    number_set = np.array([np.sum(n*pix_size) for n in full_density_dataset])

    # SUBSET 
    smu0_set = load_data(sfit_outputs_h5, 'global_fit_mu0_set')
    su_mu0_set = load_data(sfit_outputs_h5, 'global_fit_u_mu0_set')
    stemp_set = load_data(sfit_outputs_h5, 'global_fit_temp_set')
    su_temp_set = load_data(sfit_outputs_h5, 'global_fit_u_temp_set')
    sdensity_dataset = load_data(sfit_outputs_h5, 'global_fit_density_output')
    snumber_set = np.array([np.sum(n*pix_size) for n in sdensity_dataset])

    hold_times = np.array([time_realization[0], *time_realization[19::].tolist()])
    mu0s = np.array([mu0_set[0], *smu0_set[0::].tolist()])
    u_mu0s = np.array([u_mu0_set[0], *su_mu0_set[0::].tolist()])
    T0s = np.array([temp_set[0], *stemp_set[0::].tolist()])
    u_T0s = np.array([u_temp_set[0], *su_temp_set[0::].tolist()])
    number = np.array([number_set[0], *snumber_set[0::].tolist()])
    u_number = 200*np.ones_like(number)
    
    def exponential_decay(t, t0, tau_1, tau_2, yi, yf, how_fast_percent):
        fast_span = (yi-yf)*how_fast_percent
        slow_span = (yi-yf)*(1-how_fast_percent)
        return slow_span*np.exp(-(t-t0)/tau_2)+fast_span*np.exp(-(t-t0)/tau_1) + yf

    def exponential_lin_decay(t, t0, tau_1, yi, yf, lin_slope):
        return lin_slope*(t-t0) + (yi-yf)*np.exp(-(t-t0)/tau_1) + yf

    def lmfit_exponential_decay(xdata, ydata, dydata):
        pars = Parameters()
        # Educated guesses
        y0, yf = ydata[0], ydata[-1]
        pars.add('Start_time', value=0.0, min=-1*ms, max=1*ms, vary=False)
        pars.add('Decay_constant_1', value=2.0, min=50*ms, max=5*s, vary=True)
        pars.add('Decay_constant_2', value=4.0, min=50*ms, max=50*s, vary=True)
        pars.add('Initial_value', value=y0, min=-5*y0, max=5*y0, vary=False)
        pars.add('Final_value', value=0.0, max=yf, vary=True)
        pars.add('Fast_percentage', value=0.7, min=0.0, max=1.0, vary=True)
        # Residuals
        def residuals_exponential_decay(parameters, xdata, ydata, dydata):
            t0 = parameters['Start_time']
            tau1 = parameters['Decay_constant_1']
            tau2 = parameters['Decay_constant_2']
            yi = parameters['Initial_value']
            yf = parameters['Final_value']
            how_fast = parameters['Fast_percentage']
            model = exponential_decay(xdata, t0, tau1, tau2, yi, yf, how_fast)
            return (ydata-model)/dydata
        def exp_fit_callback(parameters, iteration, resid, *fargs, **fkwargs):
            #print(f'Iteration-if you care- : {iteration:2d}')
            return None
        return minimize(residuals_exponential_decay, pars, args=(xdata, ydata, dydata),
                        method='leastsq', iter_cb=exp_fit_callback, nan_policy='omit')

    def lmfit_lin_exponential_decay(xdata, ydata, dydata):
        pars = Parameters()
        # Educated guesses
        y0, yf = ydata[0], ydata[-1]
        pars.add('Start_time', value=0.0, min=-1*ms, max=1*ms, vary=False)
        pars.add('Decay_constant_1', value=0.75, min=10*ms, vary=True)
        pars.add('Initial_value', value=y0, min=0.01*y0, max=5*y0, vary=True)
        pars.add('Final_value', value=-1e6, vary=True)
        pars.add('Slope', value=-500, max=0, vary=True)
        # Residuals
        def residuals_exponential_decay(parameters, xdata, ydata, dydata):
            t0 = parameters['Start_time']
            tau1 = parameters['Decay_constant_1']
            yi = parameters['Initial_value']
            yf = parameters['Final_value']
            slope = parameters['Slope']
            model = exponential_lin_decay(xdata, t0, tau1, yi, yf, slope)
            return (ydata-model)/dydata
        def exp_fit_callback(parameters, iteration, resid, *fargs, **fkwargs):
            #print(f'Iteration-if you care- : {iteration:2d}')
            return None
        return minimize(residuals_exponential_decay, pars, args=(xdata, ydata, dydata),
                        method='leastsq', iter_cb=exp_fit_callback, nan_policy='omit')

    result_number = lmfit_exponential_decay(xdata=hold_times, ydata=number, dydata=u_number)
    result_mu0s = lmfit_lin_exponential_decay(xdata=hold_times, ydata=mu0s, dydata=u_mu0s)
    result_T0s = lmfit_exponential_decay(xdata=hold_times, ydata=T0s, dydata=u_T0s)
    #report_fit(result_number)
    #report_fit(result_mu0s)
    #report_fit(result_T0s)
    number_pars = np.array([result_number.params[key].value for key in result_number.params.keys()])
    mu0s_pars = np.array([result_mu0s.params[key].value for key in result_mu0s.params.keys()])
    T0s_pars = np.array([result_T0s.params[key].value for key in result_T0s.params.keys()])

    time_interpolation = np.linspace(0, 5, n_points)
    model_number = exponential_decay(time_interpolation, *number_pars)
    model_mu0s = exponential_lin_decay(time_interpolation, *mu0s_pars)
    model_T0s = exponential_decay(time_interpolation, *T0s_pars)

    if show_plots:
        # Plot results
        __fig__ = plt.figure(figsize=(14, 4))
        ax1 = plt.subplot(131)
        ax1.scatter(hold_times, number, c='C0', edgecolor='k')
        ax1.plot(time_interpolation, model_number, c='C0', lw=1.0)
        plt.xlim([-0.5, 5.5])
        plt.ylim([-0., 1.6e3])

        ax2 = plt.subplot(132)
        ax2.scatter(hold_times, mu0s, c='C1', edgecolor='k')
        ax2.plot(time_interpolation, model_mu0s, c='C1', lw=1.0)
        plt.xlim([-0.5, 5.5])
        plt.ylim([-0.5e3, 1.7e3])

        ax3 = plt.subplot(133)
        ax3.scatter(hold_times, T0s, c='C2', edgecolor='k')
        ax3.plot(time_interpolation, model_T0s, c='C2', lw=1.0)
        plt.xlim([-0.5, 5.5])
        plt.ylim([-0., 0.35e-6])
        
        plt.tight_layout()
        plt.show()
    return model_number, model_mu0s, model_T0s

def interpolate_yang_yang_density_and_integrate(n_points):
    _, mus, Ts = model_pars_in_time(n_points, show_plots=False)
    V_pars = load_data(fit_outputs_h5, 'global_fit_pot_set')
    x_space = np.linspace(-324, 324, 2**9)
    dx = x_space[1]-x_space[0]
    V_a, _, _ = V_potential_model(x_space, *V_pars)
    a_osc = np.sqrt(hbar/(2*pi*V_a*mass))
    integral_kernel = 1/(3*pi**2*a_osc**4)
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
    integral, number, i = [], [], -1
    for t in tqdm(np.linspace(0, 5, n_points)):
        i += 1
        density = compute_nx(x_space, mus[i], Ts[i], *V_pars)
        integral.append(np.sum(integral_kernel*density**3*dx*pix_size))
        number.append(np.sum(density*dx*pix_size))
    with h5py.File(three_body_h5) as three_body:
        h5_save(three_body, 'Number', np.array(number))
        h5_save(three_body, 'Integral', np.array(integral))
    return None

def integrate_number_ODE(t, Ni, Nf, K1, K3_1D):    
    dt = t[1] - t[0]
    integrated_number = np.zeros_like(t)
    K3_1D = np.abs(K3_1D)
    K1 = np.abs(K1)
    for i, time_lim in enumerate(t):
        integrated_number[i] = (Ni - K1*np.sum(dt*yy_interpolated_numbers[0:i]) -  
                                K3_1D*np.sum(dt*yy_interpolated_integrals[0:i]))
    integrated_number += Nf
    return integrated_number

def get_initial_condition():
    zero_time_density = load_data(processed_data_h5, 'linear_density')[0]
    return np.sum(zero_time_density*pix_size)

def lmfit_integrate_number_ODE(tdata, Ndata, dNdata):
    N_init = get_initial_condition()    
    pars = Parameters()
    pars.add('Nf', value=1, min=0., max=0.75*N_init, vary=True)
    pars.add('N0', value=N_init, min=0.75*N_init, max=1.25*N_init, vary=True)
    pars.add('k1', value=0.5, min=0., vary=True)
    pars.add('k3_1D', value=6e-40, vary=True)
    # Residuals
    def residuals_integrate_number_ODE(pars, tdata, Ndata, dNdata):
        Nf = pars['Nf']
        N0 = pars['N0']
        k1 = pars['k1']
        k3_1D = pars['k3_1D']
        model = integrate_number_ODE(tdata, N0, Nf, k1, k3_1D)
        return (Ndata-model)/dNdata
    return minimize(residuals_integrate_number_ODE, pars, args=(tdata, Ndata, dNdata),
                    method='leastsq', nan_policy='omit')

def fit_number_decay(n_points, save_fig=False):
    time = np.linspace(0., 5., n_points)
    # N_init = get_initial_condition()
    result_decay = lmfit_integrate_number_ODE(time, yy_interpolated_numbers, dNdata=140)
    report_fit(result_decay)
    decay_pars = np.array([result_decay.params[key].value for key in result_decay.params.keys()])
    fit_decay = integrate_number_ODE(time, *decay_pars)  

    full_density_dataset = load_data(processed_data_h5, 'linear_density')
    full_density_dataset[10, 646] = 0.
    time_realization = load_data(processed_data_h5, 'realisation_short_TOF')
    number_set = np.array([np.sum(n*pix_size) for n in full_density_dataset])

    hold_times = np.array([time_realization[0], *time_realization[19::].tolist()])
    number = np.array([number_set[0], *number_set[19::].tolist()])
    u_number = 200*np.ones_like(number)

    m_number, _, _ = model_pars_in_time(n_points, show_plots=False)

    _fig_ = plt.figure()
    ax1 = plt.subplot(111)
    ax1.plot(time, m_number, c='C0', ls='--', lw=2.0, label='Double exponential model')
    ax1.plot(time, fit_decay, c='k', lw=2.0, label='One-Three Body model')
    ax1.scatter(time, yy_interpolated_numbers, s=20, c='C2', edgecolor='k', label='Interpolated YY')
    ax1.scatter(hold_times, number, s=70, c='C3', edgecolor='k', label='Data')

    plt.legend()
    plt.tight_layout()
    plt.show()
    with h5py.File(three_body_h5) as three_body:
        h5_save(three_body, 'Fit_OneThree_Body_Decay', np.array(fit_decay))
        h5_save(three_body, 'OneThree_Body_Decay', np.array(m_number))
        h5_save(three_body, 'decay_parameters', decay_pars)
    return None


if __name__ == '__main__':
    n_points = 2**8
    # model_pars_in_time(n_points=n_points, show_plots=True)
    try:
        yy_interpolated_numbers = load_data(three_body_h5, 'Number')
        yy_interpolated_integrals = load_data(three_body_h5, 'Integral')
        try:
            if n_points != np.size(yy_interpolated_integrals):
                raise ValueError
        except ValueError:
            interpolate_yang_yang_density_and_integrate(n_points=n_points)
            yy_interpolated_numbers = load_data(three_body_h5, 'Number')
            yy_interpolated_integrals = load_data(three_body_h5, 'Integral')
    except:
        interpolate_yang_yang_density_and_integrate(n_points=n_points)
        yy_interpolated_numbers = load_data(three_body_h5, 'Number')
        yy_interpolated_integrals = load_data(three_body_h5, 'Integral')
    fit_number_decay(n_points=n_points)
    

