import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as grsp
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
sfit_outputs_h5 = 'fit_outputs_subset.h5'
three_body_h5 = 'three_body.h5'
calibrations_h5 = 'calibrations.h5'

def load_data(dataset_h5, dataset_id):
    with h5py.File(dataset_h5, 'r') as dataset_h5:
        pulled_data = np.array(dataset_h5[dataset_id])
    return pulled_data

# Figure convenience functions
def setup_figure(figsize=None, dpi=None):
    # Call figure instance, set latex & fonts.
    __fig__ = plt.figure(figsize=figsize, dpi=dpi)
    font = {'family' : 'serif',
            'size'   : 14}
    plt.rc('text', usetex=True)
    plt.rc('font', **font)
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
    #A_a = 17.64e3
    def anti_trap_model(x, A_a, x0_a, dx_a):
        return A_a/(1+((x-x0_a)**2/(dx_a/2)**2))
    
    def long_trap_model(x, A_t, x0_t, dx_t):
        return A_t*(np.exp(-(x-x0_t)**2/(2*dx_t**2)))

    V_a = anti_trap_model(x, A_a, x0_a, dx_a)
    V_t = long_trap_model(x, A_t, x0_t, dx_t)
    V = V_a + V_t
    return V_a-V_a.min(), V_t-V_t.min(), V-V.min()

def bin_density_slice(density_slice, bin_size=2):
    return density_slice.reshape(-1, bin_size).mean(axis=1)

def bin_uncertainty_slice(uncertainty_slice, bin_size=2):
    return np.sqrt((uncertainty_slice**2).reshape(-1, bin_size).sum(axis=1))

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
def compute_nx(x, mu0, T0, ss, rs, A_a, x0_a, dx_a, A_t, x0_t, dx_t):  
    V_anti, _, V_total = V_potential_model(x, A_a, x0_a, dx_a, A_t, x0_t, dx_t)
    f_perp = V_anti
    mui = mu0 - V_total
    density_profile = []
    for j in tqdm(range(np.size(V_total))):
        density_profile.append(YY_thermodynamics(trans_freq=f_perp[j], 
                                     mass=mass, 
                                     temperature=T0, 
                                     chemical_potential=mui[j], 
                                     scatt_length=100*a0))
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
    systematic_density_wavelet = load_data(fit_outputs_h5, 'global_fit_systematic_density_wavelet')
    binned_syst_density_wave = bin_density_slice(systematic_density_wavelet, bin_size=4)
    x_axis = np.linspace(-np.size(full_density_dataset[0,:])/2, 
                          np.size(full_density_dataset[0,:])/2, 
                          np.size(full_density_dataset[0,:]))
    fit_density_numbers = [np.trapz(full_density_dataset[j, :], dx=(x_axis[1]-x_axis[0])*pix_size) 
                           for j in range(24)]
    systamatics_scale_par = load_data(fit_outputs_h5, 'global_fit_scale_set')
    weighted_systematics = np.array([(Nyy**0.5)*systamatics_scale_par*binned_syst_density_wave 
                                     for Nyy in fit_density_numbers])
    fit_density_dataset_only_YY = fit_density_dataset - weighted_systematics
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
    im2 = ax2.imshow(fit_density_dataset_only_YY/(1/um), cmap='seismic', aspect='auto',
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
    
    u_eos_data = load_data(processed_data_h5, 'u_linear_density')
    binned_n_data = np.array([bin_density_slice(nj, bin_size=4) for nj in full_density_dataset])
    binned_fit_n_data = fit_density_dataset_only_YY#np.array([bin_density_slice(nj, bin_size=4) for nj in fit_density_dataset])
    binned_u_n_data = np.array([bin_uncertainty_slice(nj, bin_size=4) for nj in u_eos_data])
    x_axis = np.linspace(-np.size(full_density_dataset[0,:])/2, 
                          np.size(full_density_dataset[0,:])/2, 
                          np.size(binned_n_data[0,:]))   
    __fig__= setup_figure()
    slice_index = -1
    for n_1D, fit_n_1D in list(zip(binned_n_data, binned_fit_n_data)):
        slice_index += 1
        _ax_ = plt.subplot(111)
        _ax_.plot(x_axis*pix_size/um, fit_n_1D/(1/um), 'r', linewidth=1.5)
        _ax_.scatter(x_axis*pix_size/um, n_1D/(1/um), s=10.0, c='w', alpha=0.9,
                           edgecolor='k')
        _, _, axerrorlinecollection = _ax_.errorbar(x_axis*pix_size/um, n_1D/(1/um), 
                                                    xerr=1.0, yerr=binned_u_n_data[slice_index]/(1/um),
                                                    marker='', ls='', zorder=0)
        axerrorlinecollection[0].set_color('k'), axerrorlinecollection[1].set_color('k')
        plt.ylim([-5., 12.])
        label_current_ax(fig=__fig__, xlabel='$x \,[\mu m]$', ylabel='$n_{1D} \,[\mu m ^{-1}]$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(outdir, f'{slice_index:04d}_density_slice_and_fit.png'),
                dpi=500)
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
    in_situ_single_shot_OD = load_data(processed_data_h5, 'naive_OD')[313, 180:255, :] #409
    integrated_single_shot_density = in_situ_single_shot_OD[31:39].sum(0)*pix_size/sigma_0
    bin_single_shot_density = bin_density_slice(integrated_single_shot_density, bin_size=4)
    average_linear_density = load_data(processed_data_h5, 'linear_density')[2]
    bin_avg_linear_density = bin_density_slice(average_linear_density, bin_size=4)
    fit_linear_density = load_data(fit_outputs_h5, 'global_fit_density_output')
    #fit_linear_density[-1] *= 0 # temporary
    final_dipole_realization = 0.545 #load_data(raw_data_h5, 'final_dipole')[409] #0.525
    short_TOF_realization = 0.0 #load_data(raw_data_h5, 'short_TOF')[409]
    potential_parameters = load_data(fit_outputs_h5, 'global_fit_pot_set')[1::]
    sfull_cov_matrix = load_data(sfit_outputs_h5, 'global_fit_covariance_matrix')
    full_cov_matrix = load_data(fit_outputs_h5, 'global_fit_covariance_matrix')
    V_cov_matrix = np.zeros((5, 5))
    # V_cov_matrix[0 , :] = sfull_cov_matrix[10, 10::]
    # V_cov_matrix[:, 0] = sfull_cov_matrix[10::, 10]
    # V_cov_matrix[3, 0], V_cov_matrix[0, 3] = 0., 0. 
    V_cov_matrix = full_cov_matrix[-5::, -5::]

    ODx_axis = np.linspace(-np.size(in_situ_single_shot_OD[0,:])/2, 
                            np.size(in_situ_single_shot_OD[0,:])/2, 
                            np.size(in_situ_single_shot_OD[0,:])) + potential_parameters[3] + 35
    ODy_axis = np.linspace(-np.size(in_situ_single_shot_OD[:,0])/2, 
                            np.size(in_situ_single_shot_OD[:,0])/2, 
                            np.size(in_situ_single_shot_OD[:,0]))
    bin_x_axis = np.linspace(-np.size(integrated_single_shot_density)/2, 
                              np.size(integrated_single_shot_density)/2, 
                              np.size(bin_single_shot_density)) + potential_parameters[3] + 35
    atom_number = np.trapz(fit_linear_density, dx=(bin_x_axis[1]-bin_x_axis[0])*pix_size, axis=1)
    systematic_wavelet = load_data(fit_outputs_h5, 'global_fit_systematic_density_wavelet')
    binned_systematic_density_wave = bin_density_slice(systematic_wavelet, bin_size=4)
    systamatics_scale_par = load_data(fit_outputs_h5, 'global_fit_scale_set')
    systematic_density_correction = np.array([(N**0.5)*binned_systematic_density_wave*systamatics_scale_par 
                                              for N in atom_number])
    from scipy import stats
    print(stats.kstest(systematic_density_correction.flatten(), 'norm'))
    # import IPython
    # IPython.embed()
    # assert False, "BREAK"
    fit_linear_density_no_systematics = fit_linear_density[2] - systematic_density_correction[2]
    antitrap, longtrap, V_total = V_potential_model(ODx_axis*pix_size/um, *potential_parameters)
    probability, extent = model_shaded_uncertainty(V_potential_model, ODx_axis*pix_size/um, 
                                                    potential_parameters, V_cov_matrix)
    def build_fig_1_a():
        __fig__ = setup_figure()
        ax1 = plt.subplot(111)
        im1 = ax1.imshow(schematic_setup_image, aspect='equal')
        ax1.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)

        if save_plots:
            plt.savefig(f'Fig_1a.pdf', dpi=1000)
            plt.clf()

    def build_fig_1_b():
        in_nK = lambda EinHz: EinHz*hbar*2*pi/(kB*1e-9)

        __fig__ = setup_figure()
        ax2 = plt.subplot(111)
        im2 = ax2.imshow(probability, origin='lower-left', aspect='auto',  extent=[extent[0], extent[1],
                         in_nK(extent[2])/1e3, in_nK(extent[3])/1e3], cmap='bone_r', vmin=0.0, vmax=1.0)
        #cbar2 = __fig__.colorbar(mappable=im2, ax=ax2, orientation='vertical')
        #cbar2.set_clim(-0., 1.0)
        ax2.plot(ODx_axis*pix_size/um, in_nK(V_total)/1e3, linewidth=1.0, color='k', label='Total')
        ax2.plot(ODx_axis*pix_size/um, in_nK(antitrap)/1e3, linewidth=2.0, linestyle='-',
                 color='C2', alpha=0.8, label='Anti-trap')
        ax2.plot(ODx_axis*pix_size/um, in_nK(longtrap)/1e3, linewidth=2.0, linestyle='-',
                 color='r', alpha=0.8, label='Long Trap')
        ax2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.5, which='major')
        plt.ylim([0.0, 1.25])
        label_current_ax(fig=__fig__, xlabel='$z\,(\mu m)$', ylabel='$V\,(\mu K)$')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)

        if save_plots:
            plt.savefig(f'Fig_1b.pdf')
            plt.clf()

    def build_fig_1_a_c_d():
        __fig__ = setup_figure(figsize=[5, 7], dpi=1000)
        gs = grsp.GridSpec(3, 1, height_ratios=[2, 1, 4])
        
        ax1 = plt.subplot(gs[0])
        im1 = ax1.imshow(schematic_setup_image, aspect='equal')
        ax1.axis('off')
        ax1.text(x=0, y=0, s='(a)')

        ax3 = plt.subplot(gs[1])
        im3 = ax3.imshow(in_situ_single_shot_OD, cmap='Greens', aspect='equal',
                    vmin=-0.2, vmax=1.2, origin='lower',
                    extent=((ODx_axis*pix_size/um).min(), (ODx_axis*pix_size/um).max(),
                            (ODy_axis*pix_size/um).min(), (ODy_axis*pix_size/um).max()))
        label_current_ax(fig=__fig__, ylabel='$y\,(\mathrm{\mu m})$')
        ax3.text(x=(ODx_axis*pix_size/um).min(), y=8, s='(b)')
        
        ax4 = plt.subplot(gs[2], sharex=ax3)
        ax4.scatter(bin_x_axis*pix_size/um, bin_single_shot_density/(1/um),
                    c='white', s=15.0, linewidth=0.35, alpha=1.0, edgecolor='k', 
                    label='Raw shot')
        ax4.scatter(bin_x_axis*pix_size/um, bin_avg_linear_density/(1/um),
                    c='forestgreen', s=15.0, edgecolor='k', linewidth=0.35, 
                    label='Processed average shot', alpha=1.0)
        ax4.scatter(bin_x_axis*pix_size/um, systematic_density_correction[1]/(1/um),
                    c='gray', s=5.0, alpha=1.0, edgecolor='k', linewidth=0.4,
                    label='Residuals', marker=',')
        ax4.plot(bin_x_axis*pix_size/um, fit_linear_density_no_systematics/(1/um), # **************
                 color='k', linewidth=2.5, linestyle='-', alpha=1.0, label='YY Fit')
        ax4.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
        plt.xlim([bin_x_axis[0]*pix_size/um, bin_x_axis[-1]*pix_size/um])
        plt.ylim([-3., 12.])
        label_current_ax(fig=__fig__, xlabel='$z\,(\mathrm{\mu m})$', ylabel='$n \,(\mathrm{\mu m}^{-1})$')
        ax4.text(x=bin_x_axis[0]*pix_size/um, y=11.2, s='(c)')
        
        plt.subplots_adjust(left=0.4, bottom=0.01, right=None, top=None,
                            wspace=0.3, hspace=0.01)
        plt.setp(ax3.get_xticklabels(), visible=False)
        ax3.set_adjustable('box-forced')
        ax1.set_adjustable('box-forced')
        plt.tight_layout()

        if save_plots:
            plt.savefig(f'Fig_1acd.pdf')
            plt.clf()

    def build_fig_1_d():
        __fig__ = setup_figure()
        ax4 = plt.subplot(111)
        ax4.scatter(bin_x_axis*pix_size/um, bin_single_shot_density/(1/um),
                    c='midnightblue', s=25.0, linewidth=0.25, alpha=0.85, edgecolor='k', 
                    label='Raw shot')
        ax4.scatter(bin_x_axis*pix_size/um, bin_avg_linear_density/(1/um),
                    c='dodgerblue', s=35.0, edgecolor='k', linewidth=0.25, 
                    label='Processed average shot', alpha=0.85)
        ax4.scatter(bin_x_axis*pix_size/um, systematic_density_correction[1]/(1/um),
                    c='grey', s=15.0, edgecolor='k',  linewidth=0.25, alpha=1.0, label='Residuals')
        ax4.plot(bin_x_axis*pix_size/um, fit_linear_density_no_systematics/(1/um), # **************
                 color='k', linewidth=2.5, linestyle='-', alpha=1.0, label='YY Fit')
        ax4.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
        plt.xlim([bin_x_axis[0]*pix_size/um, bin_x_axis[-1]*pix_size/um])
        plt.ylim([-3., 17.])
        label_current_ax(fig=__fig__, xlabel='$z\,[\mu m]$', ylabel='$n \,[\mu m ^{-1}]$')
        plt.title('(c)', loc='left')
        plt.legend()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)

        if save_plots:
            plt.savefig(f'Fig_1d.pdf')
            plt.clf()

    #build_fig_1_b()
    #build_fig_1_a_c_d()

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

    AO_command = np.array([-0.5, -0.45, -0.40, -0.35, -0.30, -0.25, -0.535, -0.525, -0.510, -0.200, 
                    -0.150, -0.100, -0.050, -0.000, 0.050, -0.540, -0.550, -0.560, 0.500, -0.57, -0.58, -0.59, -0.6])
    AO_command += 0.5
    calibrated_P = np.array([9*mW, 26*mW, 50*mW, 82*mW, 122*mW, 167*mW, 3*mW, 4*mW, 7*mW, 217*mW, 275*mW, 332*mW,
                            396*mW, 461*mW, 524*mW, 2*mW, 1*mW, 0.4*mW, 1*W, 0.05*mW, 0.04*mW, 0.36*mW, 1.07*mW])

    def lmfit_AOMresponse(xdata, ydata, u_ydata):
        pars = Parameters()
        pars.add('Norm', value=1.0, vary=True)

        def residuals(pars, xdata, ydata, u_ydata):
            pm = pars['Norm']
            ymodel = sine_squared(xdata, pm)
            return (ydata-ymodel)/u_ydata

        return minimize(residuals, pars, args=(xdata, ydata, u_ydata),
                          method='leastsq', nan_policy='omit')

    def sine_squared(p_in, p_max):
        return (np.sin((np.pi/2)*np.sqrt(p_in/p_max)))**2

    result = lmfit_AOMresponse(AO_command, calibrated_P, 0.01*calibrated_P)
    report_fit(result)
    parameters = np.array([result.params[key].value for key in result.params.keys()])

    voltages = np.linspace(-0.0, 1.0, 2**8)
    powers = sine_squared(voltages, *parameters)
    #plt.scatter(AO_command, calibrated_P, c='k')
    #plt.plot(voltages, powers)

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    load_inset_0 = plt.imread('load_step_0.png')
    load_inset_1 = plt.imread('load_step_1.png')
    load_inset_2 = plt.imread('load_step_2.png')
    load_inset_3 = plt.imread('load_step_3.png')

    __fig__ = setup_figure()
    ax1 = plt.subplot(111)
    ax1.plot(time, green_ramp, c='limegreen', ls='-', lw=3.0, label='Blue-detuned')
    ax1.plot(time, long_ramp, c='maroon', ls='--', lw=3.0, label='Red-detuned')
    ax1.plot(time, cross_ramp, c='cornflowerblue', ls='-.',lw=3.0, label='BEC-cross dipole trap')

    inset_0 = OffsetImage(load_inset_0, zoom=0.012)
    inset_0.image.axes = ax1
    ab_0 = AnnotationBbox(inset_0, [0.125, 1.2])
    ax1.add_artist(ab_0)
    
    inset_1 = OffsetImage(load_inset_1, zoom=0.012)
    inset_1.image.axes = ax1
    ab_1 = AnnotationBbox(inset_1, [0.375, 1.2])
    ax1.add_artist(ab_1)

    inset_2 = OffsetImage(load_inset_2, zoom=0.012)
    inset_2.image.axes = ax1
    ab_2 = AnnotationBbox(inset_2, [0.625, 1.2])
    ax1.add_artist(ab_2)

    inset_3 = OffsetImage(load_inset_3, zoom=0.012)
    inset_3.image.axes = ax1
    ab_3 = AnnotationBbox(inset_3, [0.875, 1.2])
    ax1.add_artist(ab_3)

    ax1.axvline(250*ms, c='k', ls='--', lw=1.0, alpha=0.5)
    ax1.axvline(500*ms, c='k', ls='--', lw=1.0, alpha=0.5)
    ax1.axvline(750*ms, c='k', ls='--', lw=1.0, alpha=0.5)
    ax1.text(100*ms, 0.8, s='(i)')
    ax1.text(350*ms, 0.8, s='(ii)')
    ax1.text(600*ms, 0.8, s='(iii)')
    ax1.text(825*ms, 0.8, s='(iv)')
    #ax1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
    plt.ylim([0, 1.4])
    plt.xlim([0., 1.0])
    label_current_ax(__fig__, xlabel='$t \,(s)$', ylabel='Intensity (a .u.)')
    #plt.title('Loading ramps', loc='left')
    plt.tight_layout()
    #plt.legend()

    if save_plots:
        plt.savefig(f'Fig_2.pdf', dpi=500)
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
    A_ix, B_ix = 0, 17
    density_A = bin_density_slice(load_data(processed_data_h5, 'linear_density')[A_ix], bin_size=4)
    density_B = bin_density_slice(load_data(processed_data_h5, 'linear_density')[B_ix], bin_size=4)
    u_density_A = bin_uncertainty_slice(load_data(processed_data_h5, 'u_linear_density')[A_ix], bin_size=4)
    u_density_B = bin_uncertainty_slice(load_data(processed_data_h5, 'u_linear_density')[B_ix], bin_size=4)
    potential_parameters = load_data(fit_outputs_h5, 'global_fit_pot_set')
    x_axis = np.linspace(-4*np.size(density_A)/2, 
                          4*np.size(density_A)/2, 
                          np.size(density_A))
    potential = bin_density_slice(load_data(fit_outputs_h5, 'global_fit_pot_output'), bin_size=4)
    systematic_density_wavelet = load_data(fit_outputs_h5, 'global_fit_systematic_density_wavelet')
    binned_syst_density_wave = bin_density_slice(systematic_density_wavelet, bin_size=4)
    density_numbers = [np.trapz(d, dx=(x_axis[1]-x_axis[0])*pix_size) for d in [density_A, density_B]] 
    systamatics_scale_par = load_data(fit_outputs_h5, 'global_fit_scale_set')
    weighted_systematics = np.array([(Nyy**0.5)*systamatics_scale_par*binned_syst_density_wave 
                                     for Nyy in density_numbers])
    mu0_A = load_data(fit_outputs_h5, 'global_fit_mu0_set')[A_ix]
    mu0_B = load_data(fit_outputs_h5, 'global_fit_mu0_set')[B_ix]
    u_mu0_A = load_data(fit_outputs_h5, 'global_fit_u_mu0_set')[A_ix]
    u_mu0_B = load_data(fit_outputs_h5, 'global_fit_u_mu0_set')[B_ix]
    T_A = load_data(fit_outputs_h5, 'global_fit_temp_set')[A_ix]
    T_B = load_data(fit_outputs_h5, 'global_fit_temp_set')[B_ix]
    fit_density_A = load_data(fit_outputs_h5, 'global_fit_density_output')[A_ix] - weighted_systematics[0]
    fit_density_B = load_data(fit_outputs_h5, 'global_fit_density_output')[B_ix] - weighted_systematics[1]
    density_A = density_A - weighted_systematics[0]
    density_B = density_B - weighted_systematics[1]
    mu_A, mu_B = mu0_A-potential, mu0_B-potential
    g1D_A = compute_g1Dx(x_axis, mu0_A, T_A, *potential_parameters)
    g1D_B = compute_g1Dx(x_axis, mu0_B, T_B, *potential_parameters)
    tf_density_A, tf_density_B = hbar*2*pi*mu_A/g1D_A, hbar*2*pi*mu_B/g1D_B
    tf_density_A[tf_density_A<0], tf_density_B[tf_density_B<0] = 0, 0
    # entropy_A = compute_Sx(x_axis, mu0_A, T_A, *potential_parameters)
    # entropy_B = compute_Sx(x_axis, mu0_B, T_B, *potential_parameters)
    # kth_A, kth_B = compute_kth(T_A), compute_kth(T_B)
    # import IPython
    # IPython.embed()

    def build_fig_1_e():
        density_A_units = 1/(hbar**2/(2*mass*g1D_A))
        gamma_A = density_A_units/density_A
        t_A = T_A/(hbar**2*density_A**2/(2*mass*kB))

        density_B_units = 1/(hbar**2/(2*mass*g1D_B))
        gamma_B = density_B_units/density_B
        t_B = T_B/(hbar**2*density_B**2/(2*mass*kB))

        gamma = np.linspace(1e-4, 1e2, 2**12)
        t = np.linspace(1e-4, 1e2, 2**12)

        border_1 = np.sqrt(gamma[gamma<=1e0])**2
        border_2 = gamma[gamma>=1e0]**0

        __fig__ = setup_figure()
        ax = plt.subplot(111)
        ax.plot((gamma[gamma<1e0]), (border_1), 'k',
                 (gamma[gamma>1e0]), (border_2), 'k',
                  np.ones(t.shape), t, 'k', gamma, np.ones(gamma.shape), 'k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.fill_between(gamma[gamma<1e0], 1e-3, border_1, where=border_1>1e-6, 
                        facecolor='cornflowerblue', alpha=0.75, label='Quasi-condensate')
        ax.fill_between(gamma[gamma>1e0], 1e-3, border_2, where=border_2>1e-3,
                        facecolor='red', alpha =0.5, label='Strong Coupling')
        ax.fill_between(gamma[gamma>1e0], border_2, 1e4, where=border_2<1e4, 
                        facecolor='green', alpha=0.5, label='Weak Coupling')
        ax.fill_between(gamma[gamma<1e0], border_1, 1e4, where=border_1<1e4, 
                        facecolor='green', alpha=0.5, label='Weak Coupling')
        ax.scatter(gamma_A[30:130], t_A[30:130], c='white', edgecolor='k')
        ax.scatter(gamma_B[30:130], t_B[30:130], c='white', edgecolor='k')
        label_current_ax(fig=__fig__, xlabel='$\gamma$', ylabel='$T/T_d$')
        plt.xlim([1e-3, 1e2])
        plt.ylim([1e-3, 1e2])

        #ax.text(4e-3, 2e2, 'Weak Coupling', style='normal',
        #        bbox={'facecolor':'yellow', 'alpha':1.0, 'pad':10})
        #ax.text(0.35e1, 2e-1, 'Strong Coupling', style='normal',
        #        bbox={'facecolor':'yellow', 'alpha':1.0, 'pad':10})
        #ax.text(1e-2, 1e-2, 'Quasi-Condensate', style='normal',
        #        bbox={'facecolor':'yellow', 'alpha':1.0, 'pad':10})
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)

        if save_plots:
            plt.savefig(f'Fig_1e.pdf')
            plt.clf()
    
    def build_fig_3_a():
        __fig__ = setup_figure()
        ax1 = plt.subplot(111)
        density_A_units, density_B_units = 1/(hbar**2/(2*mass*g1D_A)), 1/(hbar**2/(2*mass*g1D_B))
        mu_A_units, mu_B_units = 2*mass*g1D_A**2/hbar**2, 2*mass*g1D_B**2/hbar**2

        ax1_x, ax1_y = hbar*2*pi*mu_A/mu_A_units, density_A/density_A_units
        ax1_u_x, ax1_u_y = hbar*2*pi*u_mu0_A/mu_A_units, u_density_A/density_A_units
        sc1 = ax1.scatter(ax1_x, ax1_y, c='lightskyblue', s=14.0, alpha=1.0)
        _, _, sc1_errorcollection = ax1.errorbar(ax1_x, ax1_y, xerr=ax1_u_x, yerr=ax1_u_y, 
                                                    marker='', ls='', zorder=0)
        sc1_errorcollection[0].set_color('k')
        sc1_errorcollection[1].set_color('k')
        ax1.annotate(('$T_A/T_d^{(0)} =\,$'+f'{(kB*T_A/(hbar**2*density_A.max()**2/(2*mass))):.2f}'),
                    xy=(3.0, 2.7), xytext=(-25, 2.5), arrowprops=dict(facecolor='black', shrink=0.05,
                        width=0.5, headwidth=2))
        
        ax1_x, ax1_y = hbar*2*pi*mu_B/mu_B_units, density_B/density_B_units
        ax1_u_x, ax1_u_y = hbar*2*pi*u_mu0_B/mu_B_units, u_density_B/density_B_units
        sc2 = ax1.scatter(ax1_x, ax1_y, c='midnightblue', s=14.0, alpha=1.0)
        _, _, sc2_errorcollection = ax1.errorbar(ax1_x, ax1_y, xerr=ax1_u_x, yerr=ax1_u_y,
                                                    marker='', ls='', zorder=0)
        sc2_errorcollection[0].set_color('k')
        sc2_errorcollection[1].set_color('k')
        ax1.annotate(('$T_B/T_d^{(0)} =\,$'+f'{(kB*T_B/(hbar**2*density_B.max()**2/(2*mass))):.2f}'),
                    xy=(0.1, 0.8), xytext=(-30, 1.5), arrowprops=dict(facecolor='black', shrink=0.05,
                        width=0.5, headwidth=2))

        ax1.plot(hbar*2*pi*mu_A/mu_A_units, fit_density_A/(density_A_units), c='cornflowerblue', lw=1.5, 
                label='Yang-Yang A')
        ax1.plot(hbar*2*pi*mu_B/mu_B_units, fit_density_B/(density_B_units), c='navy', lw=1.5, 
                label='Yang-Yang B')
        ax1.plot(hbar*2*pi*mu_A/mu_A_units, tf_density_A/(density_A_units), c='k', lw=2.0, alpha=0.7,
                label='Mean Field A')
        #ax1.plot(hbar*2*pi*mu_B/mu_B_units, tf_density_B/(density_B_units), c='k', lw=2.0, alpha=0.7, 
                #label='Mean Field B')
        #ax1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
        #ax1.set_xscale('symlog')
        plt.ylim([-0.5, 3.0])
        plt.xlim([-50., 5])
        label_current_ax(__fig__, xlabel='$\hbar^2\,\mu/2 m g_{1 \mathrm{D} }^2$', 
                                  ylabel='$\gamma^{-1}$')
        plt.title('(a)',loc='left')
        plt.tight_layout()
        #plt.legend()

        if save_plots:
            plt.savefig(f'Fig_3a.pdf')
            plt.clf()

    # Temperature calibration
    def temperature(final_dipole):
        slope = load_data(calibrations_h5, 'T3D/slope')
        intercept = load_data(calibrations_h5, 'T3D/intercept')
        # Old calibration < 2*(519.916*nK*final_dipole - 247.7625*nK)
        u_slope = load_data(calibrations_h5, 'T3D/u_slope')
        u_intercept = load_data(calibrations_h5, 'T3D/u_intercept')
        temps = (slope*final_dipole + intercept)*nK
        u_temps = np.sqrt(slope**2*0.01**2+final_dipole**2*u_slope**2+u_intercept**2)*nK
        return temps, u_temps

    full_density_dataset = load_data(processed_data_h5, 'linear_density')
    full_density_dataset[10, 646] = 0.
    u_density_dataset = load_data(processed_data_h5, 'u_linear_density')
    mu0_set = load_data(fit_outputs_h5, 'global_fit_mu0_set')
    u_mu0_set = load_data(fit_outputs_h5, 'global_fit_u_mu0_set')
    temp_set = load_data(fit_outputs_h5, 'global_fit_temp_set')
    u_temp_set = load_data(fit_outputs_h5, 'global_fit_u_temp_set')
    n_max = full_density_dataset[:, 285:300].mean(axis=1)
    u_n_max = u_density_dataset[:, 285:300].mean(axis=1)
    peak_degeneracy_temps = hbar**2*n_max**2/(2*mass*kB)
    u_Td0 = hbar**2*n_max*u_n_max/(2*mass*kB)
    u_T_Td0 = np.sqrt(u_temp_set**2/peak_degeneracy_temps**2 + temp_set**2*u_Td0**2/peak_degeneracy_temps**4)

    temp_realization, u_temp_realization = temperature(load_data(processed_data_h5, 'realisation_final_dipole'))
    time_realization = load_data(processed_data_h5, 'realisation_short_TOF')

    def build_fig_3_b():
        __fig__ = setup_figure()
        
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222, sharey = ax1)
        ax3 = plt.subplot(223, sharex = ax1)
        ax4 = plt.subplot(224, sharex = ax2, sharey = ax3)

        ax1_x, ax1_y = temp_realization[0:19], temp_set[0:19]
        ax1_u_x, ax1_u_y = u_temp_realization[0:19], u_temp_set[0:19]
        ax1_shading = (temp_set[0:19])/peak_degeneracy_temps[0:19]
        sc1 = ax1.scatter(ax1_x/nK, ax1_y/nK, c='cornflowerblue', edgecolor='k')
        _, _ ,ax1errorlinecollection = ax1.errorbar(ax1_x/nK, ax1_y/nK, xerr=ax1_u_x/nK, yerr=ax1_u_y/nK, 
                                                    marker='', ls='', zorder=0)
        ax1errorlinecollection[0].set_color('k'), ax1errorlinecollection[1].set_color('k')
        ax1.set_ylabel('${T \, (\mathrm{nK})}$', fontsize=14)
        ax1.set_ylim(0.0, 200.0)
        ax1.set_xlim(0.0, 400.0)

        ax2_x = np.array([time_realization[0], *time_realization[19::].tolist()])
        ax2_y = np.array([temp_set[0], *temp_set[19::].tolist()])
        ax2_u_x, ax2_u_y = 0.001*ms, np.array([u_temp_set[0], *u_temp_set[19::].tolist()])
        ax2_peak_degeneracy_temps = np.array([peak_degeneracy_temps[0], *peak_degeneracy_temps[19::].tolist()])
        ax2_temps = np.array([temp_set[0], *temp_set[19::].tolist()])
        ax2_shading = (ax2_temps)/ax2_peak_degeneracy_temps
        ax2.scatter(ax2_x, ax2_y/nK, c='cornflowerblue', edgecolor='k')
        _, _, ax2errorlinecollection = ax2.errorbar(ax2_x, ax2_y/nK, xerr=ax2_u_x, yerr=ax2_u_y/nK,
                                                    marker='', ls='', zorder=0)
        ax2errorlinecollection[0].set_color('k'), ax2errorlinecollection[1].set_color('k')
        ax2.set_ylim(0.0, 200.0)
        ax2.set_xlim(-0.1, 5.1)

        ax3_x, ax3_y = ax1_x, temp_set[0:19]
        ax3_u_x, ax3_u_y = ax1_u_x, u_T_Td0[0:19]
        ax3_shading = ax1_shading
        ax3.scatter(ax3_x/nK, ax3_shading, c='cornflowerblue', edgecolor='k')
        _, _, ax3errorlinecollection = ax3.errorbar(ax3_x/nK, ax3_shading, xerr=ax3_u_x/nK, yerr=ax3_u_y,
                                                    marker='', ls='', zorder=0)
        ax3errorlinecollection[0].set_color('k'), ax3errorlinecollection[1].set_color('k')
        ax3.set_xlabel('${T_{\mathrm{3D}}\, (\mathrm{nK})}$', fontsize=14)
        ax3.set_ylabel('$T/T_d^{(0)}$', fontsize=14)
        ax3.set_ylim(0.0, 6.1)
        ax3.set_xlim(0.0, 400.0)


        ax4_x, ax4_y = ax2_x, np.array([temp_set[0], *temp_set[19::].tolist()])
        ax4_u_x, ax4_u_y = ax2_u_x, np.array([u_T_Td0[0], *u_T_Td0[19::].tolist()])
        ax4_shading = ax2_shading
        ax4.scatter(ax4_x, ax4_shading, c='cornflowerblue', edgecolor='k')
        _, _, ax4errorlinecollection = ax4.errorbar(ax4_x, ax4_shading, xerr=ax4_u_x, yerr=ax4_u_y,
                                                   marker='', ls='', zorder=0)
        ax4errorlinecollection[0].set_color('k'), ax4errorlinecollection[1].set_color('k')
        ax4.set_xlabel('$t (s)$', fontsize=14)
        ax4.set_ylim(0.0, 6.1)
        ax4.set_xlim(-0.1, 5.1)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)
        ax1.set_title('(b)', loc='left')
        plt.tight_layout()

        all_axes = np.array([ax1, ax2, ax3, ax4])
        #cbar = plt.colorbar(mappable=sc1, ax=all_axes.ravel().tolist(), orientation='vertical',
        #                    fraction=0.05, pad=0.05, shrink=0.8)
        #cbar.set_label('$T/T_d^{(0)}$')
        #cbar.set_clim(0., ax1_shading.max())
    
        if save_plots:
            plt.savefig(f'Fig_3b.pdf')
            plt.clf()

    # __fig__ = setup_figure()
    # ax2 = plt.subplot(111)
    # ax2.scatter(hbar*2*pi*mu_A/mu_A_units, entropy_A, c='lightskyblue', s=14.0, alpha=0.75, 
    #         label=f'$T_a/T_d(0) =$ {(kB*T_A/(hbar**2*density_A.max()**2/(2*mass))):.2f}')
    # ax2.scatter(hbar*2*pi*mu_B/mu_B_units, entropy_B, c='midnightblue', s=14.0, alpha=0.75, 
    #         label=f'$T_b/T_d(0) =$ {(kB*T_B/(hbar**2*density_B.max()**2/(2*mass))):.2f}')
    # ax2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
    # plt.ylim([-0., 4.])
    # plt.xlim([-10., 3.])
    # label_current_ax(__fig__, xlabel='$\mu \, [2mg_{1D}^2/\hbar^2]$', ylabel='$S/N \,[k_B]$')
    # plt.title('(c) Entropy vs local chemical potential', loc='left')
    # plt.tight_layout()
    # plt.legend()
    
    number = load_data(three_body_h5, 'Number_decay')
    u_number_decay = load_data(three_body_h5, 'u_number_decay')
    fit_number_decay = load_data(three_body_h5, 'Fit_One_Two_Three_Body_Decay')
    time = np.linspace(0., 5., np.size(number))
    fit_time = np.linspace(0.0, 5.1, np.size(fit_number_decay))
    
    def build_fig_3_c():
        __fig__ = setup_figure()
        ax3 = plt.subplot(111)
        #ax3.set_xscale('log')
        ax3.set_yscale('log')
        sc3 = ax3.scatter(time, number, c='cornflowerblue', edgecolor='k', s=30.0, 
                    label='Number data', marker='D')
        _, _, sc3errorcollection = ax3.errorbar(time, number, xerr=0., yerr=u_number_decay, 
                                             marker='', ls='', zorder=0)
        sc3errorcollection[0].set_color('k'), sc3errorcollection[1].set_color('k')
        ax3.plot(fit_time, fit_number_decay, lw=2.0, c='midnightblue', label='Decay model')
        #ax3.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.25, which='major')
        plt.ylim([-0., 2000.])
        plt.xlim([-0.1, 5.1])
        label_current_ax(__fig__, xlabel='$t \, (s)$', ylabel='${N}$')
        plt.title('(c)', loc='left')
        plt.tight_layout()

        if save_plots:
            plt.savefig(f'Fig_3c.pdf')
            plt.clf()
    
    #build_fig_1_e()
    #build_fig_3_a()
    #build_fig_3_b()
    build_fig_3_c()


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
    plot_figure_3_draft(save_plots=True)