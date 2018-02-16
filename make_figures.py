import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.stats import norm
from tqdm import tqdm
from lmfit import Parameters, minimize, report_fit, Minimizer
from yang_yang_1dbg import bethe_integrator

um, nm, ms = 1e-6, 1e-9, 1e-3
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

def load_data(dataset_h5, dataset_id):
    with h5py.File(dataset_h5, 'r') as dataset_h5:
        pulled_data = np.array(dataset_h5[dataset_id])
    return pulled_data

# Figure convenience functions
def setup_figure(figsize=None):
    # Call figure instance, set latex & fonts.
    __fig__ = plt.figure(figsize=figsize)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    return __fig__

def label_current_ax(fig, xlabel='', ylabel=''):
    __ax__ = fig.gca()
    __ax__.set_xlabel(xlabel, fontsize=14)
    __ax__.set_ylabel(ylabel, fontsize=14)

    #####################################################################################
    #####                                                                           #####
    #####                   ### ###  ###### #####   ###### ##                       #####
    #####                   ## # ##  ##  ## ##  ##  ##     ##                       #####
    #####       #########   ##   ##  ##  ## ##  ##  ###### ##      #########        #####
    #####                   ##   ##  ##  ## ##  ##  ##     ##                       #####
    #####                   ##   ##  ###### #####   ###### ######                   #####
    #####                                                                           #####
    #####################################################################################

def V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t):
    
    def anti_trap_model(x, A_a, x0_a, dx_a):
        return A_a/(1+((x-x0_a)**2/(dx_a/2)**2))
    
    def long_trap_model(x, A_t, x0_t, dx_t):
        return A_t*(np.exp(-(x-x0_t)**2/(2*dx_t**2)))

    V_a = anti_trap_model(x, A_a, x0_a, dx_a)
    V_t = long_trap_model(x, A_t, x0_t, dx_t)
    V = V_a + V_t
    return V_a-V_a.min(), V_t-V_t.min(), V-V.min()

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

    #####################################################################################
    #####                                                                           #####
    #####           #####    ######   ##  ## ######  ###### ######  ##  ##          #####
    #####           ##  ##   ##       ### ## ##        ##     ##    ##  ##          #####
    #####           ##  ##   ####     ## ### ######    ##     ##    ######          #####
    #####           ##  ##   ##       ## ###     ##    ##     ##      ##            #####
    #####           #####    ######   ##  ## ######  ######   ##      ##            #####
    #####                                                                           #####
    #####################################################################################
def compute_nx(x, mu0, T0, A_a, x0_a, dx_a, A_t, x0_t, dx_t):  
    _, _, V_long = V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t)
    f_perp = A_a/(1+((x-x0_a)**2/(dx_a/2)**2))
    mui = mu0 - V_long
    density_profile = []
    for j in tqdm(range(np.size(V_long))):
        density_profile.append(YY_thermodynamics(trans_freq=f_perp[j], 
                                     mass=mass, 
                                     temperature=T0, 
                                     chemical_potential=mui[j], 
                                     scatt_length=110*a0))
    return np.array(density_profile)
    #####################################################################################
    #####                                                                           #####
    #####           ######   ##  ## ######   ######  ###### ######  ##  ##          #####
    #####           ##       ### ##   ##     ##  ##  ##  ## ##  ##  ##  ##          #####
    #####           ####     ######   ##     #####   ##  ## ######  ######          #####
    #####           ##       ## ###   ##     ##  ##  ##  ## ##        ##            #####
    #####           ######   ##  ##   ##     ##  ##  ###### ##        ##            #####
    #####                                                                           #####
    #####################################################################################
def compute_Sx(x, mu0, T0, A_a, x0_a, dx_a, A_t, x0_t, dx_t):      
    _, _, V_long = V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t)
    f_perp = A_a/(1+((x-x0_a)**2/(dx_a/2)**2))
    mui = mu0 - V_long
    entropy_profile = []
    for j in tqdm(range(np.size(V_long))):
        entropy_profile.append(YY_thermodynamics(trans_freq=f_perp[j], 
                                     mass=mass, 
                                     temperature=T0, 
                                     chemical_potential=mui[j], 
                                     scatt_length=110*a0,
                                     returned='entropy'))
    return np.array(entropy_profile)
    #####################################################################################
    #####                                                                           #####
    #####      ######   ######   ###### ######  ###### ##  ##  ######  ######       #####
    #####      ##  ##   ##  ##   ##     ##      ##     ##  ##  ##  ##  ##           #####
    #####      ######   #####    ####   ######  ###### ##  ##  #####   ####         #####
    #####      ##       ##  ##   ##         ##      ## ##  ##  ##  ##  ##           #####
    #####      ##       ##  ##   ###### ######  ###### ######  ##  ##  ######       #####
    #####                                                                           #####
    #####################################################################################
def compute_Px(x, mu0, T0, A_a, x0_a, dx_a, A_t, x0_t, dx_t):   
    _, _, V_long = V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t)
    f_perp = A_a/(1+((x-x0_a)**2/(dx_a/2)**2))
    mui = mu0 - V_long
    pressure_profile = []
    for j in tqdm(range(np.size(V_long))):
        pressure_profile.append(YY_thermodynamics(trans_freq=f_perp[j], 
                                     mass=mass, 
                                     temperature=T0, 
                                     chemical_potential=mui[j], 
                                     scatt_length=110*a0,
                                     returned='pressure'))
    return np.array(pressure_profile)
    #####################################################################################
    #####                                                                           #####
    #####                       ######            ####      ######                  #####
    #####                       ##                  ##      ##  ##                  #####
    #####                       ##  ##              ##  ### ##  ##                  #####
    #####                       ##  ##              ##      ##  ##                  #####
    #####                       ######            ######    ######                  #####
    #####                                                                           #####
    #####################################################################################
def compute_g1Dx(x, mu0, T0, A_a, x0_a, dx_a, A_t, x0_t, dx_t):   
    _, _, V_long = V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t)
    f_perp = A_a/(1+((x-x0_a)**2/(dx_a/2)**2))
    mui = mu0 - V_long
    g1D_profile = []
    for j in tqdm(range(np.size(V_long))):
        g1D_profile.append(YY_thermodynamics(trans_freq=f_perp[j], 
                                     mass=mass, 
                                     temperature=T0, 
                                     chemical_potential=mui[j], 
                                     scatt_length=110*a0,
                                     returned='g1D'))
    return np.array(g1D_profile)

def compute_kth(T0): 
    return  np.sqrt(2*mass*kB*T0/hbar**2)

    #####################################################################################
    #####                                                                           #####
    #####           ######   ######   ###### ######  ###### ######  ######          #####
    #####           ##  ##   ##  ##     ##   ##  ##  ##     ##        ##            #####
    #####   ######  ##  ##   ######     ##   ######  ###### ####      ##   ######   #####
    #####           ##  ##   ##  ##     ##   ##  ##      ## ##        ##            #####
    #####           ######   ##  ##     ##   ##  ##  ###### ######    ##            #####
    #####                                                                           #####
    #####################################################################################

def plot_density_data(save_plots=False):
    # Load all data & fits
    full_density_dataset = load_data(processed_data_h5, 'linear_density')
    full_density_dataset[10, 646] = 0.
    fit_density_dataset = load_data(fit_outputs_h5, 'global_fit_density_output')
    x_axis = np.linspace(-np.size(full_density_dataset[0,:])/2, 
                          np.size(full_density_dataset[0,:])/2, 
                          np.size(full_density_dataset[0,:]))
    realization = np.linspace(0, np.size(full_density_dataset[:,0]), 
                                 np.size(full_density_dataset[:,0]))
    # Output folder
    outdir = 'density_dataset'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # Density dataset.
    __fig__ = setup_figure()
    ax1 = plt.subplot(211)
    im1 = ax1.imshow(full_density_dataset/(1/um), cmap='seismic', aspect='auto',
                vmin=-10.0, vmax=10., origin='lower',
                extent=((x_axis*pix_size/um).min(), (x_axis*pix_size/um).max(),
                realization.min(), realization.max()))
    label_current_ax(fig=__fig__, ylabel='Realization')

    ax2 = plt.subplot(212)
    im2 = ax2.imshow(fit_density_dataset/(1/um), cmap='seismic', aspect='auto',
                vmin=-10.0, vmax=10., origin='lower',
                extent=((x_axis*pix_size/um).min(), (x_axis*pix_size/um).max(),
                realization.min(), realization.max()))
    label_current_ax(fig=__fig__, xlabel='$x \,[\mu m]$', ylabel='Realization')

    all_axes = np.array([ax1, ax2])
    cbar1 = __fig__.colorbar(mappable=im1, ax=all_axes.ravel().tolist(),
                             orientation='vertical')
    cbar1.set_label('$n_{1D} \,[\mu m ^{-1}]$', fontsize=14)
    cbar1.set_clim(-10.0, 10.0)

    if save_plots:
        plt.savefig(os.path.join(outdir, 'all_density_data_and_fit.png'))
        plt.clf()
            
    __fig__= setup_figure()
    slice_index = -1
    for n_1D, fit_n_1D in list(zip(full_density_dataset, fit_density_dataset)):
        slice_index += 1
        _ax_ = plt.scatter(x_axis*pix_size/um, n_1D/(1/um), s=3.0, c='C0', alpha=0.75)
        _ax_ = plt.plot(x_axis*pix_size/um, fit_n_1D/(1/um), 'k', linewidth=0.75)
        plt.ylim([-5., 10.])
        label_current_ax(fig=__fig__, xlabel='$x \,[\mu m]$', ylabel='$n_{1D} \,[\mu m ^{-1}]$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(outdir, f'{slice_index:04d}_density_slice_and_fit.png'))
            plt.clf()

    #####################################################################################
    #####                                                                           #####
    #####                    ######   ######   ######         ####                  #####
    #####                    ##         ##     ##             # ##                  #####
    #####      ########      ####       ##     ## ###           ##    ########      #####
    #####                    ##         ##     ##  ## ###       ##                  #####
    #####                    ##       ######   ###### ###     ######                #####
    #####                                                                           #####
    #####################################################################################

def plot_figure_1_draft(save_plots=False):

    def partial_derivatives(function, x, params, u_params):
        _, _, model_at_center = function(x, *params)
        partial_derivatives = []
        for i, (param, u_param) in enumerate(zip(params, u_params)):
            d_param = u_param/1e6
            params_with_partial_differential = np.zeros(len(params))
            params_with_partial_differential[:] = params[:]
            params_with_partial_differential[i] = param + d_param
            _, _, model_at_partial_differential = function(x, *params_with_partial_differential)
            partial_derivative = (model_at_partial_differential - model_at_center)/d_param
            partial_derivatives.append(partial_derivative)
        return partial_derivatives

    def model_uncertainty(function, x, params, covariance):
        u_params = [np.sqrt(abs(covariance[i,i]))  for i in range(len(params))]
        model_partial_derivatives = partial_derivatives(function, x, params, u_params)
        try:
            squared_model_uncertainty = np.zeros(x.shape)
        except TypeError:
            squared_model_uncertainty = 0
        for i in range(len(params)):
            for j in range(len(params)):
                squared_model_uncertainty += model_partial_derivatives[i]*model_partial_derivatives[j]*covariance[i,j]
        return np.sqrt(squared_model_uncertainty)

    def model_shaded_uncertainty(function, x, params, covariance, yrange=None, resolution=1024, 
                                columns_normalised=True):
        _, _, model_mean = function(x, *params)
        model_stddev = model_uncertainty(function, x, params, covariance)
        if yrange is None:
            yrange = [(model_mean - 10*model_stddev).min(), (model_mean + 10*model_stddev).max()]
        y = np.linspace(yrange[0], yrange[1], resolution)
        Model_Mean, Y = np.meshgrid(model_mean, y)
        Model_Stddev, Y = np.meshgrid(model_stddev, y)
        if columns_normalised:
            probability = np.exp(-(Y - Model_Mean)**2/(2*Model_Stddev**2))
        else:
            probability = norm.pdf(Y, Model_Mean, Model_Stddev)
        return probability, [x.min(), x.max(), y.min(), y.max()]

    schematic_setup_image = plt.imread('Fig1_Schematic.jpg')
    in_situ_single_shot_OD = load_data(processed_data_h5, 'naive_OD')[409, 180:255, :]
    integrated_single_shot_density = in_situ_single_shot_OD.sum(0)*pix_size/sigma_0
    average_naive_linear_density = load_data(processed_data_h5, 'naive_linear_density')[1]
    fit_linear_density = load_data(fit_outputs_h5, 'global_fit_density_output')[1]
    #fit_linear_density[-1] *= 0 # temporary
    final_dipole_realization = 0.525 #load_data(raw_data_h5, 'final_dipole')[409]
    short_TOF_realization = 0.0 #load_data(raw_data_h5, 'short_TOF')[409]
    potential_parameters = load_data(fit_outputs_h5, 'global_fit_pot_set')
    full_cov_matrix = load_data(fit_outputs_h5, 'global_fit_covariance_matrix')
    V_cov_matrix = np.ones((6, 6))
    V_cov_matrix[0:4, 0:4] = full_cov_matrix[48:52, 48:52]
    V_cov_matrix[5, :] = full_cov_matrix[-1, -6::]
    V_cov_matrix[:, 5] = full_cov_matrix[-6::, -1]
    ODx_axis = np.linspace(-np.size(in_situ_single_shot_OD[0,:])/2, 
                          np.size(in_situ_single_shot_OD[0,:])/2, 
                          np.size(in_situ_single_shot_OD[0,:]))
    ODy_axis = np.linspace(-np.size(in_situ_single_shot_OD[:,0])/2, 
                          np.size(in_situ_single_shot_OD[:,0])/2, 
                          np.size(in_situ_single_shot_OD[:,0]))
    antitrap, longtrap, V_total = V_potential_model(ODx_axis, *potential_parameters)
    probability, extent = model_shaded_uncertainty(V_potential_model, ODx_axis, potential_parameters, V_cov_matrix)

    import IPython
    IPython.embed()

    __fig__ = setup_figure(figsize=(6, 12))
    ax1 = plt.subplot(411)
    im1 = ax1.imshow(schematic_setup_image, aspect='equal')
    ax1.axis('off')
    plt.title('(a) Schematic', loc='left')

    # **** HAVE TO ADD SHADED UNCERTAINTY ****
    ax2 = plt.subplot(412)
    im2 = ax2.imshow(1-probability, origin='lower-left', aspect='auto',  extent=extent, 
               cmap='pink_r', vmin=1e-2, vmax=1)
    ax2.plot(ODx_axis*pix_size/um, V_total, linewidth=1.0, color='k', label='Total')
    ax2.plot(ODx_axis*pix_size/um, antitrap, linewidth=2.0, linestyle='--',
             color='C2', alpha=0.8, label='Anti-trap')
    ax2.plot(ODx_axis*pix_size/um, longtrap, linewidth=2.0, linestyle='--',
             color='r', alpha=0.8, label='Long Trap')
    ax2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.5, which='major')
    plt.ylim([-1e3, 25e3])
    label_current_ax(fig=__fig__, xlabel='$z\,[\mu m]$', ylabel='$V\,[\,\mathrm{Hz}\,]$')
    plt.title('(b) Longitudinal potential', loc='left')
    plt.legend()
    plt.tight_layout()

    ax3 = plt.subplot(413)
    im3 = ax3.imshow(in_situ_single_shot_OD, cmap='Blues', aspect='equal',
                vmin=-0.2, vmax=1.2, origin='lower',
                extent=((ODx_axis*pix_size/um).min(), (ODx_axis*pix_size/um).max(),
                        (ODy_axis*pix_size/um).min(), (ODy_axis*pix_size/um).max()))
    label_current_ax(fig=__fig__, xlabel='$z\,[\mu m]$', ylabel='$y\,[\mu m]$')
    plt.title('(c) Absorption image', loc='left')

    ax4 = plt.subplot(414)
    ax4.step(ODx_axis*pix_size/um, integrated_single_shot_density/(1/um),
                c='C0', linewidth=0.5, alpha=0.75, 
                label='Single shot distribution')
    ax4.scatter(ODx_axis*pix_size/um, average_naive_linear_density/(1/um),
                c='b', s=10.0, edgecolor='k', linewidth=1.0, 
                label='Mean distribution', alpha=0.5)
    ax4.plot(ODx_axis*pix_size/um, fit_linear_density/(1/um), # **************
             color='r', linewidth=1.0, linestyle='-', alpha=0.75, label='Fit')
    ax4.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
    plt.ylim([-10., 50.])
    label_current_ax(fig=__fig__, xlabel='$z\,[\mu m]$', ylabel='$n \,[\mu m ^{-1}]$')
    plt.title('(d) Sample data ', loc='left')
    plt.legend()
    plt.tight_layout()

    if save_plots:
        plt.savefig(f'Fig1_all.pdf', dpi=2000)
        plt.clf()

    #####################################################################################
    #####                                                                           #####
    #####                    ######   ######   ######         ######                #####
    #####                    ##         ##     ##                 ##                #####
    #####     ########       ####       ##     ## ###         ######    ########    #####
    #####                    ##         ##     ##  ## ###     ##                    #####
    #####                    ##       ######   ###### ###     ######                #####
    #####                                                                           #####
    #####################################################################################

def plot_figure_2_draft(save_plots=False):

    W, mW, ms = 1.0, 1e-3, 1e-3

    def half_gauss(t, duration, yi, yf):
        return ((yf-yi)*(1-norm.sf(t, loc=t[0]+duration/2, scale=0.15*duration))+yi).tolist()

    def hold_value(t, y0):
        return (y0*np.ones_like(t)).tolist()

    array_ize = lambda some_list: np.array(some_list).flatten()

    n_points = 2**5
    green_ramp, long_ramp, cross_ramp = [], [], []

    t_segment_0 = np.linspace(0, 250*ms, n_points)
    green_ramp.append(half_gauss(t_segment_0, 250*ms, 0*W, 50*mW))
    long_ramp.append(hold_value(t_segment_0, 0*W))
    cross_ramp.append(hold_value(t_segment_0, 550*mW))

    t_segment_1 = np.linspace(t_segment_0[-1], t_segment_0[-1]+250*ms, n_points)
    green_ramp.append(hold_value(t_segment_1, 50*mW))
    long_ramp.append(hold_value(t_segment_1, 0*W))
    cross_ramp.append(half_gauss(t_segment_1, 250*ms, 550*mW, 200*mW))

    t_segment_2 = np.linspace(t_segment_1[-1], t_segment_1[-1]+250*ms, n_points)
    green_ramp.append(hold_value(t_segment_2, 50*mW))
    long_ramp.append(half_gauss(t_segment_2, 250*ms, 0*W, 500*mW))
    cross_ramp.append(hold_value(t_segment_2, 200*mW))

    t_segment_3 = np.linspace(t_segment_2[-1], t_segment_2[-1]+250*ms, n_points)
    green_ramp.append(half_gauss(t_segment_3, 250*ms, 50*mW, 1*W))
    long_ramp.append(hold_value(t_segment_3, 500*mW))
    cross_ramp.append(half_gauss(t_segment_3, 250*ms, 200*mW, 0*W))

    time = np.concatenate((t_segment_0, t_segment_1, t_segment_2, t_segment_3))
    green_ramp, long_ramp, cross_ramp = array_ize(green_ramp), array_ize(long_ramp), array_ize(cross_ramp)

    __fig__ = setup_figure()
    ax1 = plt.subplot(111)
    ax1.plot(time/ms, green_ramp, marker='o', markeredgecolor='k', c='limegreen', 
             markeredgewidth=1.0, lw=1.0, label='Blue-detuned')
    ax1.plot(time/ms, long_ramp, marker='o', markeredgecolor='k', c='maroon', 
             markeredgewidth=1.0, lw=1.0, label='Red-detuned')
    ax1.plot(time/ms, cross_ramp, marker='o', markeredgecolor='k', c='cornflowerblue', 
             markeredgewidth=1.0,lw=1.0, label='BEC-cross dipole trap')
    ax1.axvline(250., c='k', ls='--', lw=1.5, alpha=0.5)
    ax1.axvline(500., c='k', ls='--', lw=1.5, alpha=0.5)
    ax1.axvline(750., c='k', ls='--', lw=1.5, alpha=0.5)
    ax1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
    plt.ylim([-0.1, 1.0])
    plt.xlim([-0., 1e3])
    label_current_ax(__fig__, xlabel='$t \,[ms]$', ylabel='Power [W]')
    #plt.title('Loading ramps', loc='left')
    plt.tight_layout()
    plt.legend()

    if save_plots:
        plt.savefig(f'Fig_2.pdf')
        plt.clf()

    #####################################################################################
    #####                                                                           #####
    #####                    ######   ######   ######         ######                #####
    #####                    ##         ##     ##                 ##                #####
    #####     ########       ####       ##     ## ###         ######    ########    #####
    #####                    ##         ##     ##  ## ###         ##                #####
    #####                    ##       ######   ###### ###     ######                #####
    #####                                                                           #####
    #####################################################################################

def plot_figure_3_draft(save_plots=False):
    A_ix, B_ix = 4, 20
    density_A = load_data(processed_data_h5, 'naive_linear_density')[A_ix]
    density_B = load_data(processed_data_h5, 'naive_linear_density')[B_ix]
    potential_parameters = load_data(fit_outputs_h5, 'global_fit_pot_set')
    x_axis = np.linspace(-np.size(density_A)/2, 
                          np.size(density_A)/2, 
                          np.size(density_A))
    _, _, potential = V_potential_model(x_axis, *potential_parameters)
    fit_density_A = load_data(fit_outputs_h5, 'global_fit_density_output')[A_ix]
    fit_density_B = load_data(fit_outputs_h5, 'global_fit_density_output')[B_ix]
    mu0_A = load_data(fit_outputs_h5, 'global_fit_mu0_set')[A_ix]
    mu0_B = load_data(fit_outputs_h5, 'global_fit_mu0_set')[B_ix]
    T_A = load_data(fit_outputs_h5, 'global_fit_temp_set')[A_ix]
    T_B = load_data(fit_outputs_h5, 'global_fit_temp_set')[B_ix]
    mu_A, mu_B = mu0_A-potential, mu0_B-potential
    g1D_A = compute_g1Dx(x_axis, mu0_A, T_A, *potential_parameters)
    g1D_B = compute_g1Dx(x_axis, mu0_B, T_B, *potential_parameters)
    tf_density_A, tf_density_B = hbar*2*pi*mu_A/g1D_A, hbar*2*pi*mu_B/g1D_B
    tf_density_A[tf_density_A<0], tf_density_B[tf_density_B<0] = 0, 0
    entropy_A = compute_Sx(x_axis, mu_A, T_A, *potential_parameters)
    entropy_B = compute_Sx(x_axis, mu_B, T_B, *potential_parameters)
    kth_A, kth_B = compute_kth(T_A), compute_kth(T_B)
    
    __fig__ = setup_figure()
    ax1 = plt.subplot(111)
    density_A_units, density_B_units = 1/(hbar**2/(2*mass*g1D_A)), 1/(hbar**2/(2*mass*g1D_B))
    mu_A_units, mu_B_units = 2*mass*g1D_A**2/hbar**2, 2*mass*g1D_B**2/hbar**2
    ax1.scatter(hbar*2*pi*mu_A/mu_A_units, density_A/density_A_units, c='lightskyblue', s=14.0, alpha=0.75, 
            label=f'$T_A/T_d(0) =$ {(kB*T_A/(hbar**2*density_A.max()**2/(2*mass))):.2f}')
    ax1.scatter(hbar*2*pi*mu_B/mu_B_units, density_B/density_B_units, c='midnightblue', s=14.0, alpha=0.75, 
            label=f'$T_B/T_d(0) =$ {(kB*T_B/(hbar**2*density_B.max()**2/(2*mass))):.2f}')
    ax1.step(hbar*2*pi*mu_A/mu_A_units, fit_density_A/(density_A_units), c='cornflowerblue', lw=1.5, 
            label='Yang-Yang A')
    ax1.step(hbar*2*pi*mu_B/mu_B_units, fit_density_B/(density_B_units), c='navy', lw=1.5, 
            label='Yang-Yang B')
    ax1.plot(hbar*2*pi*mu_A/mu_A_units, tf_density_A/(density_A_units), c='dimgray', lw=2.0, alpha=0.7,
            label='Mean Field A')
    ax1.plot(hbar*2*pi*mu_B/mu_B_units, tf_density_B/(density_B_units), c='k', lw=2.0, alpha=0.7, 
            label='Mean Field B')
    ax1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
    ax1.set_xscale('symlog')
    plt.ylim([-1., 3.])
    plt.xlim([-200., 3.])
    label_current_ax(__fig__, xlabel='$\mu \, [2mg_{1D}^2/\hbar^2]$', ylabel='$n\,[\hbar^2/2mg_{1D}] $')
    plt.title('(a) $\gamma^{-1}$ vs local chemical potential', loc='left')
    plt.tight_layout()
    plt.legend()

    if save_plots:
        plt.savefig(f'Fig_3a.pdf')
        plt.clf()

    # Temperature calibration
    def temperature(final_dipole):
        return 519.916*nK*final_dipole - 247.7625*nK

    full_density_dataset = load_data(processed_data_h5, 'linear_density')
    full_density_dataset[10, 646] = 0.
    peak_degeneracy_temps = np.array([hbar**2*n0.max()**2/(2*mass*kB*nK) for n0 in full_density_dataset])
    mu0_set = load_data(fit_outputs_h5, 'global_fit_mu0_set')
    u_mu0_set = load_data(fit_outputs_h5, 'global_fit_u_mu0_set')
    temp_set = load_data(fit_outputs_h5, 'global_fit_temp_set')
    u_temp_set = load_data(fit_outputs_h5, 'global_fit_u_temps_set') # Change in new version
    temp_realization = temperature(load_data(processed_data_h5, 'realisation_final_dipole'))
    time_realization = load_data(processed_data_h5, 'realisation_short_TOF')

    __fig__ = setup_figure(figsize=(8, 6))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222, sharey = ax1)
    ax3 = plt.subplot(223, sharex = ax1)
    ax4 = plt.subplot(224, sharex = ax2, sharey = ax3)

    ax1_x, ax1_y = temp_realization[0:19], mu0_set[0:19]
    ax1_u_x, ax1_u_y = 4*nK, u_mu0_set[0:19]
    ax1_shading = (temp_realization[0:19]/nK)/peak_degeneracy_temps[0:19]
    sc1 = ax1.scatter(ax1_x/nK, ax1_y, c=ax1_shading, cmap='Blues', edgecolor='k')
    _, _ ,ax1errorlinecollection = ax1.errorbar(ax1_x/nK, ax1_y, xerr=ax1_u_x/nK, yerr=ax1_u_y, 
                                                marker='', ls='', zorder=0)
    ax1errorlinecollection[0].set_color('k')
    ax1errorlinecollection[1].set_color('k')
    ax1.set_ylabel('$\mu_0 \, [\mathrm{Hz}]$', fontsize=14)
    ax1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.5, which='major')

    ax2_x = np.array([time_realization[0], *time_realization[19::].tolist()])
    ax2_y = np.array([mu0_set[0], *mu0_set[19::].tolist()])
    ax2_u_x, ax2_u_y = 0.001*ms, np.array([u_mu0_set[0], *u_mu0_set[19::].tolist()])
    ax2_peak_degeneracy_temps = np.array([peak_degeneracy_temps[0], *peak_degeneracy_temps[19::].tolist()])
    ax2_temps = np.array([temp_set[0], *temp_set[19::].tolist()])
    ax2_shading = (ax2_temps/nK)/ax2_peak_degeneracy_temps
    ax2.scatter(ax2_x, ax2_y, c=ax2_shading, cmap='Blues', edgecolor='k')
    _, _, ax2errorlinecollection = ax2.errorbar(ax2_x, ax2_y, xerr=ax2_u_x, yerr=ax2_u_y,
                                                marker='', ls='', zorder=0)
    ax2errorlinecollection[0].set_color('k'), ax2errorlinecollection[1].set_color('k')
    ax2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.5, which='major')

    ax3_x, ax3_y = ax1_x, temp_set[0:19]
    ax3_u_x, ax3_u_y = ax1_u_x, u_temp_set[0:19]
    ax3_shading = ax1_shading
    ax3.scatter(ax3_x/nK, ax3_y/nK, c=ax3_shading, cmap='Blues', edgecolor='k')
    _, _, ax3errorlinecollection = ax3.errorbar(ax3_x/nK, ax3_y/nK, xerr=ax3_u_x/nK, yerr=ax3_u_y/nK,
                                                marker='', ls='', zorder=0)
    ax3errorlinecollection[0].set_color('k'), ax3errorlinecollection[1].set_color('k')
    ax3.set_xlabel('$\mathrm{T_{3D}}\, [\mathrm{nK}]$', fontsize=14)
    ax3.set_ylabel('$\mathrm{T_{YY}}\, [\mathrm{nK}]$', fontsize=14)
    ax3.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.5, which='major')

    ax4_x, ax4_y = ax2_x, np.array([temp_set[0], *temp_set[19::].tolist()])
    ax4_u_x, ax4_u_y = ax2_u_x, np.array([u_temp_set[0], *u_temp_set[19::].tolist()])
    ax4_shading = ax2_shading
    ax4.scatter(ax4_x, ax4_y/nK, c=ax4_shading, cmap='Blues', edgecolor='k')
    _, _, ax4errorlinecollection = ax4.errorbar(ax4_x, ax4_y/nK, xerr=ax4_u_x, yerr=ax4_u_y/nK,
                                                marker='', ls='', zorder=0)
    ax4errorlinecollection[0].set_color('k'), ax4errorlinecollection[1].set_color('k')
    ax4.set_xlabel('$t [s]$', fontsize=14)
    ax4.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.5, which='major')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.tight_layout()

    all_axes = np.array([ax1, ax2, ax3, ax4])
    cbar = __fig__.colorbar(mappable=sc1, ax=all_axes.ravel().tolist(), orientation='vertical')
    cbar.set_label('$T_{YY}/T_d^{(0)}$')
    cbar.set_clim(0., ax1_shading.max())
    
    if save_plots:
        plt.savefig(f'Fig_3b.pdf')
        plt.clf()

    __fig__ = setup_figure()
    ax2 = plt.subplot(111)
    ax2.scatter(hbar*2*pi*mu_A/mu_A_units, entropy_A, c='lightskyblue', s=14.0, alpha=0.75, 
            label=f'$T_a/T_d(0) =$ {(kB*T_A/(hbar**2*density_A.max()**2/(2*mass))):.2f}')
    ax2.scatter(hbar*2*pi*mu_B/mu_B_units, entropy_B, c='midnightblue', s=14.0, alpha=0.75, 
            label=f'$T_b/T_d(0) =$ {(kB*T_B/(hbar**2*density_B.max()**2/(2*mass))):.2f}')
    ax2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
    plt.ylim([-0., 8.])
    plt.xlim([-10., 3.])
    label_current_ax(__fig__, xlabel='$\mu \, [2mg_{1D}^2/\hbar^2]$', ylabel='$S/N \,[k_B]$')
    plt.title('(c) Entropy vs local chemical potential', loc='left')
    plt.tight_layout()
    plt.legend()

    if save_plots:
        plt.savefig(f'Fig_3c.pdf')
        plt.clf()


#def _V(save_plots=False):
    
    # # Potential model, expansion and sample     
    # def quadratic_residuals(pars, x_data, y_data, eps_ydata):
    #     x0 = pars['Center']
    #     V0 = pars['Offset']
    #     V2 = pars['Quadratic']
    #     y_model = quadratic_expansion(x_data, x0, V0, V2)
    #     return (y_data-y_model).flatten()/eps_ydata
    
    # def quadratic_expansion(x, x0, V0, V2):
    #     return V0 + 0.5*87*uma*(2*pi*V2)**2*(x-x0)**2*pix_size**2
    
    # def lmfit_quadratic(x_data, y_data, dy_data):
    #     params = Parameters()
    #     params.add('Center', value=16.14, min=-20, max=20, vary=True)
    #     params.add('Offset', value=0.0, min=-2.0, max=2.0, vary=False)
    #     params.add('Quadratic', value=9.4, min=0., max=30, vary=True)
    #     return minimize(quadratic_residuals, params, args=(x_data, y_data, dy_data), method='leastsq')
    
    # _, _, V_sam = V_potential_model(x_axis, *potential_parameters)
    # _, _, V_ext = V_potential_model(extended_x_axis, *potential_parameters)
    # V_red = 2*pi*hbar*V_sam[np.abs(x_axis-15)<70]
    # reduced_x_axis = x_axis[np.abs(x_axis-15)<70]
    
    # V_anti_trap, V_long_trap, _ = V_potential_model(extended_x_axis, *potential_parameters)
    
    # result = lmfit_quadratic(reduced_x_axis, V_red, np.sqrt(np.std(V_red)))
    # #report_fit(result)
    # V_quadratic_fit_params = np.array([result.params[key].value for key in result.params.keys()])
    # harmonic_frequency = V_quadratic_fit_params[2]
    
    # __fig2__ = plt.figure()
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # plt.plot(extended_x_axis*pix_size/1e-6, 2*pi*hbar*V_ext/(kB*1e-9), 'k', label='Extended potential')
    # plt.plot(extended_x_axis*pix_size/1e-6, 2*pi*hbar*V_anti_trap/(kB*1e-9), 'g--', label='Anti-trap')
    # plt.plot(extended_x_axis*pix_size/1e-6, 2*pi*hbar*V_long_trap/(kB*1e-9), 'r--', label='Long-trap')
    # plt.plot(extended_x_axis*pix_size/1e-6, quadratic_expansion(extended_x_axis, *V_quadratic_fit_params)/(kB*1e-9), 
    #          'C9-', alpha=0.85, label=('Harmonic $f = %.2f \, \mathrm{Hz}$' %harmonic_frequency)) 
    # plt.scatter(x_axis*pix_size/1e-6, 2*pi*hbar*V_sam/(kB*1e-9), color='C8', alpha=0.5, label='Sampled potential')
    # plt.xlabel('$x \,[\mu m]$', fontsize=14)
    # plt.ylabel('$ V(x)\, [\mathrm{nK}]$', fontsize=14)
    # plt.ylim([-1.2e3, 1.2e3])
    # plt.legend(loc=4)
    # plt.grid()
    # plt.tight_layout()

    return None

if __name__ == '__main__':
    #plot_density_data(save_plots=True)
    #plot_figure_1_draft(save_plots=True)
    #plot_figure_2_draft(save_plots=True)
    #plot_figure_3_draft(save_plots=True)