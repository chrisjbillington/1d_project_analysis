import h5py
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from tqdm import tqdm
from lmfit import Parameters, minimize, report_fit, Minimizer

# Fundamental Constants
hbar = constants.codata.value('Planck constant over 2 pi')
a0 = constants.codata.value('Bohr radius')
uma = constants.codata.value('atomic mass constant')
kB = constants.codata.value('Boltzmann constant')
pi = np.pi  
pix_size = 5.6e-6
mass = 86.909180527*uma

ms, us = 1e-3, 1e-6
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
nK = 1e-9

# Data
calibration_shots_h5 = 'calibrations_shots.h5'
calibrations_h5 = 'calibrations.h5'

# Data load/save methods
def pack_calibration_data():
    
    root_fl = 'L:/internal/Spielman-group/RbChip/Data/Experiments/rb_chip/1D_with_field_lock/2017/'
    wG_profile_path = 'C:/Users/fns5/Desktop/long_06_01.png'

    def get_h5_names(root, sequences, clip_shot=None):
        all_h5f = {}
        for fpath in list([root+sequence for sequence in sequences]):
            all_files = list([f for f in listdir(fpath) if isfile(join(fpath, f)) 
                        if f.endswith(".h5")])
            if clip_shot is not None:
                del all_files[clip_shot::]
            all_h5f[fpath] = all_files
        return all_h5f

    magnification_h5s = get_h5_names(root_fl, ['06/06/0001', '06/06/0002'], clip_shot=None)
    isat_h5s = get_h5_names(root_fl, ['05/05/0009'], clip_shot=None)
    f_perp_h5s = get_h5_names(root_fl, ['05/31/0001', '06/02/0001', '06/06/0003'], clip_shot=None)
    zR_h5s = get_h5_names(root_fl, ['04/27/0000', '04/27/0001'], clip_shot=None)
    T3D_h5s = get_h5_names(root_fl, ['06/02/0002'], clip_shot=None)
    f_long_h5s = get_h5_names(root_fl, ['05/18/0004'], clip_shot=None)

    def get_globals(h5_dic, target_global):
        g_arr = []
        for path in list(h5_dic.keys()):
            for shot in tqdm(range(len(list(h5_dic[path])))):
                with h5py.File(join(path, h5_dic[path][shot])) as h5_file:
                    g_arr.append(h5_file['globals'].attrs[target_global])
        return np.array(g_arr)

    def get_raws_array(h5_dic, TOF_flag=False):
        raws_arr = []
        for path in list(h5_dic.keys()):
            for shot in tqdm(range(len(list(h5_dic[path])))):
                with h5py.File(join(path, h5_dic[path][shot])) as h5_file:
                    if h5_file['globals'].attrs['insitu_only']:
                        image_group = 'imagesXY_2_Flea3'
                    elif TOF_flag:
                        image_group = 'imagesXY_1_Flea3'
                    try:
                        raws_arr.append(h5_file['data/'+image_group+'/Raw'][:])
                    except KeyError:
                        raws_arr.append([np.ones([488, 648]), np.ones([488, 648]), np.zeros([488, 648])])
        return np.array(raws_arr)

    magnification_raw = get_raws_array(magnification_h5s)
    magnification_times = get_globals(magnification_h5s, 'short_TOF')
    isat_raw = get_raws_array(isat_h5s)
    isat_probeints = get_globals(isat_h5s, 'ProbeInt')
    f_perp_raw = get_raws_array(f_perp_h5s)
    zR_raw = get_raws_array(zR_h5s)
    T3D_raw = get_raws_array(T3D_h5s, TOF_flag=True)
    T3D_final_dipole = get_globals(T3D_h5s, 'FinalDipole')
    f_long_raw = get_raws_array(f_long_h5s)
    f_long_times = get_globals(f_long_h5s, 'short_TOF')

    from scipy.misc import imread
    wG_profile = imread(wG_profile_path, flatten=True)

    with h5py.File(calibration_shots_h5) as calibrations_raw:
        h5_save(calibrations_raw, 'magnification/magnification_raw', magnification_raw)
        h5_save(calibrations_raw, 'magnification/magnification_short_TOF', magnification_times)
        h5_save(calibrations_raw, 'Isat/Isat_raw', isat_raw)
        h5_save(calibrations_raw, 'Isat/Isat_ProbeInt', isat_probeints)
        h5_save(calibrations_raw, 'f_perp/f_perp_raw', f_perp_raw)
        h5_save(calibrations_raw, 'zR/zR_raw', zR_raw)
        h5_save(calibrations_raw, 'T3D/T3D_raw', T3D_raw)
        h5_save(calibrations_raw, 'T3D/T3D_final_dipole', T3D_final_dipole)
        h5_save(calibrations_raw, 'f_long/f_long_raw', f_long_raw)
        h5_save(calibrations_raw, 'f_long/f_long_times', f_long_times)
        h5_save(calibrations_raw, 'wG/wG_profile', wG_profile)

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

def calibrate_insitu_magnification():
    raw_images = load_data(calibration_shots_h5, 'magnification/magnification_raw')
    short_TOF = load_data(calibration_shots_h5, 'magnification/magnification_short_TOF')
    bonus_raw_4ms = load_data(calibration_shots_h5, 'f_perp/f_perp_raw')[::-1][0:285]

    time_series_1 = short_TOF[0:105]
    time_series_2 = short_TOF[201:-1]
    joint_time = np.concatenate((time_series_1, time_series_2))

    time_series_1 = np.append(time_series_1, 4*ms)
    time_series_2 = np.append(time_series_2, 4*ms)
    joint_time = np.append(joint_time, 4*ms)

    def compute_y_int_OD(shot):
        with np.errstate(divide='ignore'):
            div = np.ma.masked_invalid((shot[0]-shot[2])/(shot[1]-shot[2]))
            od = -np.log(div)
        return np.sum(od[150:400, 150:450], axis=1)

    bonus_y_slice = np.array([compute_y_int_OD(shot) for shot in bonus_raw_4ms]).mean(axis=0)
    y_slices = np.array([compute_y_int_OD(shot) for shot in raw_images] + [bonus_y_slice.tolist()])
    y_axis = np.linspace(0., y_slices.shape[1], y_slices.shape[1])
       
    def lmfit_gaussian_1d(xdata, ydata, u_ydata):
        pars = Parameters()
        pars.add('Amplitude', value=20.0, min=-0.5, max=200, vary=True)
        pars.add('Center', value=80., min=20, max=220, vary=True)
        pars.add('Stdev', value=8., min=2, max=90, vary=True)
        pars.add('Offset', value=-5, min=-10., max=30., vary=True)

        def residuals(pars, xdata, ydata, u_ydata):
            A = pars['Amplitude']
            x0 = pars['Center']
            dx = pars['Stdev']
            y0 = pars['Offset']
            ymodel = gaussian_1d(xdata, A, x0, dx, y0)
            return (ydata-ymodel)/u_ydata

        return minimize(residuals, pars, args=(xdata, ydata, u_ydata),
                          method='leastsq', nan_policy='omit')

    def gaussian_1d(x, amp, x0, sigma, offset):
        return amp*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

    def get_peak_locations():
        peak_locations = []
        for ny in tqdm(y_slices):
            result = lmfit_gaussian_1d(y_axis, ny, 3)
            parameters = np.array([result.params[key].value for key in result.params.keys()])
            peak_locations.append(parameters[1])
        return pix_size*np.array(peak_locations)

    peak_locations = get_peak_locations()
    peak_locations[-1] = pix_size*158.8
    peak_positions_series_1 = np.append(peak_locations[0:105], peak_locations[-1])
    peak_positions_series_2 = peak_locations[105::]

    def lmfit_free_fall(tdata, ydata, u_ydata):
        pars = Parameters()
        pars.add('t0', value=-2*us, min=-60*us, max=500*us, vary=False)
        pars.add('y0', value=370*um, min=350*um, max=380*um, vary=True)
        pars.add('v0', value=0.0, vary=False)
        pars.add('g', value=9.81, vary=False)
        pars.add('Magnification', value=5.33, min=0.0, max=10.0, vary=True)

        def residuals(pars, tdata, ydata, u_ydata):
            t0 = pars['t0']
            y0 = pars['y0']
            v0 = pars['v0']
            g = pars['g']
            mag = pars['Magnification']
            ymodel = free_fall(tdata, t0, y0, v0, g, mag)
            return (ymodel-ydata)/u_ydata

        return minimize(residuals, pars, args=(tdata, ydata, u_ydata),
                          method='leastsq', nan_policy='omit')

    def free_fall(t, t0, y0, v0, g, M):
        return y0 + (v0*(t-t0) + 0.5*g*(t-t0)**2)*M

    joint_res = lmfit_free_fall(joint_time, peak_locations, 0.5)
    result_1 = lmfit_free_fall(time_series_1, peak_positions_series_1, 0.5)
    result_2 = lmfit_free_fall(time_series_2, peak_positions_series_2, 0.5)

    joint_pars = np.array([joint_res.params[key].value for key in joint_res.params.keys()])
    free_fall_pars_1 = np.array([result_1.params[key].value for key in result_1.params.keys()])
    free_fall_pars_2 = np.array([result_2.params[key].value for key in result_2.params.keys()])

    M_bar = joint_pars[-1]
    M_1 = free_fall_pars_1[-1]
    M_2 = free_fall_pars_2[-1]

    import matplotlib.pyplot as plt
    tt = np.linspace(0, 6*ms, 2**10)
    plt.figure()
    plt.scatter(time_series_1/us, peak_positions_series_1/um)
    plt.scatter(time_series_2/us, peak_positions_series_2/um)
    plt.plot(tt/us, free_fall(tt, *free_fall_pars_1)/um, 'b-')
    plt.plot(tt/us, free_fall(tt, *free_fall_pars_2)/um, 'r-')
    plt.plot(tt/us, free_fall(tt, *joint_pars)/um,'k-')
    
    with h5py.File(calibrations_h5) as calibrations:
        h5_save(calibrations, 'magnification/M_1', M_1)
        h5_save(calibrations, 'magnification/M_2', M_2)
        h5_save(calibrations, 'magnification/M_bar', M_bar) 

def calibrate_insitu_Isat():
    raw_images = load_data(calibration_shots_h5, 'Isat/Isat_raw')
    probe_int = load_data(calibration_shots_h5, 'Isat/Isat_ProbeInt')

    magnification = 5.33 # Precalibrated for this scan
    
    def lowpass_filter(shot, order, rel_cutoff_freq, ftype):
        from scipy.signal import butter, lfilter
        if ftype == 'butter':
            b, a = butter(N=order, Wn=rel_cutoff_freq, btype='lowpass')
            return lfilter(b, a, shot)
        elif ftype == 'notch':
            r = int(rel_cutoff_freq*max(shot.shape)/2)
            a, b = shot.shape
            y, x = np.ogrid[-int(a/2):int(a/2), -int(b/2):int(b/2)]
            mask = x*x + y*y <= r*r
            mask = mask.astype(int)
            fraction = np.sum(mask)/(4*a*b)
            fourier =  np.fft.fftshift(np.fft.fft2(shot))         
            return fraction, np.abs(np.fft.ifft2(np.fft.ifftshift(fourier*mask)))

    def highpass_filter(shot, order, rel_cutoff_freq, ftype):
        from scipy.signal import butter, lfilter
        if ftype == 'butter':
            b, a = butter(N=order, Wn=rel_cutoff_freq, btype='highpass')
            return lfilter(b, a, shot)
        elif ftype == 'notch':
            r = int(rel_cutoff_freq*max(shot.shape)/2)
            a, b = shot.shape
            y, x = np.ogrid[-int(a/2):int(a/2), -int(b/2):int(b/2)]
            mask = x*x + y*y > r*r
            mask = mask.astype(int)
            fraction = np.sum(mask)/(4*a*b)
            fourier =  np.fft.fftshift(np.fft.fft2(shot))          
            return fraction, np.abs(np.fft.ifft2(np.fft.ifftshift(fourier*mask))) 

    def filtered_shot_noise_characteristics(show_fits=False):
        
        def parametrize_noise(filter_pars):
            I0 = np.zeros((raw_images.shape[0], raw_images.shape[2], raw_images.shape[3]))
            D0 = np.zeros_like(I0)
            I_lp, I_hp = np.zeros_like(I0), np.zeros_like(I0)
            D_lp, D_hp = np.zeros_like(D0), np.zeros_like(D0)
            for i, shot in list(enumerate(raw_images)):
                I0[i] = np.array(shot[1], dtype=float)
                D0[i] = np.array(shot[2], dtype=float)
                frac_lp, I_lp[i] = lowpass_filter(I0[i]-D0[i], *filter_pars) 
                I_hp[i] = (I0[i]-D0[i]) - I_lp[i]

            # ROIs >> [194:206, 308:320], [215:230, 300:340]
            mean_I0_counts = np.mean(I_lp, axis=(1, 2))
            I0_count_var = np.var(I_hp, axis=(1, 2))

            def lmfit_parabola(tdata, ydata, u_ydata):
                pars = Parameters()
                pars.add('a', value=1e-2, vary=True)
                pars.add('b', value=0.01, min=0., vary=True)
                pars.add('c', value=3.75**2, vary=False)

                def residuals(pars, tdata, ydata, u_ydata):
                    a = pars['a']
                    b = pars['b']
                    c = pars['c']
                    ymodel = second_order_poly(tdata, a, b, c)
                    return (ymodel-ydata)/u_ydata

                return minimize(residuals, pars, args=(tdata, ydata, u_ydata),
                                  method='leastsq', nan_policy='omit')

            def second_order_poly(x, a, b, c):
                return a*x**2 + b*x + c

            par_result = lmfit_parabola(mean_I0_counts, I0_count_var, 2)
            parameters = np.array([par_result.params[key].value for key in par_result.params.keys()])
            #report_fit(par_result)
                    
            # The following is more accurate, fixed after a preliminary run
            parameters[2] = np.amin(I0_count_var)
            if show_fits:
                plt.figure()
                cc = np.linspace(0, mean_I0_counts.max(), 2**8)
                plt.scatter(mean_I0_counts, I0_count_var)
                plt.plot(cc, second_order_poly(cc, *parameters))

            return parameters, frac_lp.mean()

        filtered_frac = np.linspace(1e-2, 0.8, 2**4)
        readout_noise, shot_noise, flat_field_noise, filtered_area = [], [], [], []
        """ Scan filter cutoff and find the offset and lin. coefficients as 
        a function of the filter cutoff. This separates de flat field regime 
        from the shot noise regime and we can get better characteristics."""
        for fc in tqdm(filtered_frac):
            noise_coefficients, lp_filter_area = parametrize_noise(filter_pars=[6, fc, 'notch'])
            readout_noise.append(noise_coefficients[2])
            shot_noise.append(noise_coefficients[1])
            flat_field_noise.append(noise_coefficients[0])
            filtered_area.append(lp_filter_area)
        return (np.array(readout_noise), np.array(shot_noise), 
                np.array(flat_field_noise), np.array(filtered_area))  

    readout_noise, shot, flat_field, filtered_area = filtered_shot_noise_characteristics()

    def lmfit_line(xdata, ydata, u_ydata):
        pars = Parameters()
        pars.add('Slope', value=1e-2, vary=True)
        pars.add('Intercept', value=0.01, min=0., vary=True)

        def residuals(pars, tdata, ydata, u_ydata):
            m = pars['Slope']
            y0 = pars['Intercept']
            ymodel = linear(xdata, m, y0)
            return (ymodel-ydata)/u_ydata

        return minimize(residuals, pars, args=(xdata, ydata, u_ydata),
                        method='leastsq', nan_policy='omit')

    def linear(x, m, y0):
        return m*x + y0

    readout_nominal = readout_noise[np.where(flat_field<0.0005)]
    shot_nominal = shot[np.where(flat_field<0.0005)]
    filter_nominal = filtered_area[np.where(flat_field<0.0005)]
    
    readout_res = lmfit_line(filter_nominal, readout_nominal, 1.)
    shot_res = lmfit_line(filter_nominal, shot_nominal, 1.)

    readout_pars = np.array([readout_res.params[key].value for key in readout_res.params.keys()])
    shot_pars = np.array([shot_res.params[key].value for key in shot_res.params.keys()])

    pe_gain_coefficient = shot_pars[1]
    readout_counts = np.sqrt(readout_pars[1])/pe_gain_coefficient
    quantum_efficiency = 0.35
    photons_per_count = 1/(pe_gain_coefficient*quantum_efficiency)
    pixel_area = ((pix_size/magnification)/cm)**2
    pulse_time = 20*us
    single_photon_energy = hbar*(3e8*2*np.pi/(780*nm))
    Isat = (1.67e-3)*pixel_area*pulse_time/(single_photon_energy*photons_per_count)
    
    # For alpha, the dataset is >
    # *** 1D_with_field_lock.py, seq 0003 index 0000-0026 ***
    # Calibration is logged in One Note \\1D Gas :: 2017-05-16//
    alpha = 0.92

    with h5py.File(calibrations_h5) as calibrations:
        h5_save(calibrations, 'Isat/Isat', Isat)
        h5_save(calibrations, 'Isat/alpha', alpha)

def calibrate_f_perp():
    raw_images = load_data(calibration_shots_h5, 'f_perp/f_perp_raw')
    magnification = load_data(calibrations_h5, 'magnification/M_bar')
    Isat = load_data(calibrations_h5, 'Isat/Isat')
    alpha = load_data(calibrations_h5, 'Isat/alpha')
    three_ms_dataset = raw_images[0:49]
    two_ms_dataset = raw_images[50:149]
    four_ms_dataset = raw_images[150::]

    def compute_shot_averaged_OD(dataset):
        for shot in dataset:
            dataset_od = []
            with np.errstate(divide='ignore'):
                div = np.ma.masked_invalid((shot[0]-shot[2])/(shot[1]-shot[2]))
                dataset_od.append(-alpha*np.log(div) + (shot[1]-shot[0])/Isat)
        return np.mean(np.array(dataset_od), axis=0)

    three_ms_avg_OD = compute_shot_averaged_OD(three_ms_dataset)
    two_ms_avg_OD = compute_shot_averaged_OD(two_ms_dataset)
    four_ms_avg_OD = compute_shot_averaged_OD(four_ms_dataset)

    def lowpass_filter(shot, rel_cutoff_freq):
        r = int(rel_cutoff_freq*max(shot.shape)/2)
        a, b = shot.shape
        y, x = np.ogrid[-int(a/2):int(a/2), -int(b/2):int(b/2)]
        mask = x*x + y*y <= r*r
        mask = mask.astype(int)
        fraction = np.sum(mask)/(4*a*b)
        fourier =  np.fft.fftshift(np.fft.fft2(shot))         
        return np.abs(np.fft.ifft2(np.fft.ifftshift(fourier*mask)))
    
    gaussian_blur=False
    if gaussian_blur:
        from scipy.ndimage.filters import gaussian_filter
        lp_3ms_data = gaussian_filter(three_ms_avg_OD, 0.6)
        lp_2ms_data = gaussian_filter(two_ms_avg_OD, 0.6)
        lp_4ms_data = gaussian_filter(four_ms_avg_OD, 0.6)
    else:
        lp_3ms_data = lowpass_filter(three_ms_avg_OD, 0.2)
        lp_2ms_data = lowpass_filter(two_ms_avg_OD, 0.2)
        lp_4ms_data = lowpass_filter(four_ms_avg_OD, 0.2)


    def lmfit_gaussian_1d(xdata, ydata, u_ydata, guesses=None):
        pars = Parameters()
        if guesses is None:
            pars.add('Amplitude', value=0.2, min=0.01, max=0.5, vary=True)
            pars.add('Center', value=172, min=160, max=270, vary=True)
            pars.add('Stdev', value=20, min=5, max=25, vary=True)
            pars.add('Offset', value=0.01, min=-0.3, max=0.2, vary=True)
            bckg_gain = 0.0
        else:
            pars.add('Amplitude', value=guesses[0], min=3, max=22, vary=True)
            pars.add('Center', value=guesses[1], min=75, max=147, vary=True)
            pars.add('Stdev', value=guesses[2], min=13, max=32, vary=True)
            pars.add('Offset', value=-guesses[3], min=-25, max=-14, vary=True)
            bckg_gain = 0.2

        def residuals(pars, xdata, ydata, u_ydata):
            A = pars['Amplitude']
            x0 = pars['Center']
            dx = pars['Stdev']
            y0 = pars['Offset']
            ymodel = gaussian_1d(xdata, A, x0, dx, y0) + bckg_gain*xdata
            return (ydata-ymodel)/u_ydata

        return minimize(residuals, pars, args=(xdata, ydata, u_ydata),
                          method='leastsq', nan_policy='omit')

    def gaussian_1d(x, amp, x0, sigma, offset):
        return amp*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

    def get_peak_widths(avgOD):
        peak_widths, u_peak_widths, redchis = [], [], []
        y_axis = np.linspace(0, avgOD.shape[0], avgOD.shape[0])
        dummy_counter = -1
        for ny in (avgOD.transpose()[120:250]):
            result = lmfit_gaussian_1d(y_axis, ny, 0.15)
            parameters = np.array([result.params[key].value for key in result.params.keys()])
            try: 
                covar_d = np.diag(np.array(result.covar))
            except ValueError:
                covar_d = np.zeros_like(parameters)     
            peak_widths.append((pix_size/magnification)*parameters[2])
            u_peak_widths.append((pix_size/magnification)*np.sqrt(covar_d)[2])
            redchis.append(result.redchi)
            dummy_counter += 1
            if dummy_counter % 50 == 0 and False:
                plt.figure()
                plt.step(y_axis, ny)
                plt.plot(y_axis, gaussian_1d(y_axis, *parameters))
        return np.array(peak_widths), np.array(u_peak_widths), np.array(redchis)

    y_width_2ms, u_y_width_2ms, redchis_2ms = get_peak_widths(two_ms_avg_OD[70:370, 124:424])
    y_width_3ms, u_y_width_3ms, redchis_3ms = get_peak_widths(three_ms_avg_OD[70:370, 124:424])
    y_width_4ms, u_y_width_4ms, redchis_4ms = get_peak_widths(four_ms_avg_OD[70:370, 124:424])

    def get_mean_width(avgOD):
        ny = avgOD.sum(axis=1)[100:290]
        y_axis = np.linspace(0, ny.shape[0], ny.shape[0])
        result = lmfit_gaussian_1d(y_axis, ny, 5, guesses=[15, 100, 25, -20])
        parameters = np.array([result.params[key].value for key in result.params.keys()])
        try: 
            covar_d = np.diag(np.array(result.covar))
        except ValueError:
            covar_d = np.zeros_like(parameters)     
        mean_width = (pix_size/magnification)*parameters[2]
        u_mean_width = (pix_size/magnification)*np.sqrt(covar_d)[2]
        redchi = result.redchi
        plt.figure()
        plt.step(y_axis, ny)
        plt.plot(y_axis, gaussian_1d(y_axis, *parameters)+0.02*y_axis)
        import IPython
        IPython.embed()
        return mean_width, u_mean_width, redchi

    y_width_2ms, u_y_width_2ms, redchis_2ms = get_peak_widths(two_ms_avg_OD[70:370, 124:424])
    y_width_3ms, u_y_width_3ms, redchis_3ms = get_peak_widths(three_ms_avg_OD[70:370, 124:424])
    y_width_4ms, u_y_width_4ms, redchis_4ms = get_peak_widths(four_ms_avg_OD[70:370, 124:424])

    # 05/25/2018 : Optimized fits by 'hand' because of poor SNR... Results are below 
    m_y_width_2ms, u_m_y_width_2ms, m_redchi_2ms = 15*(pix_size/magnification), pix_size/magnification, 1.
    #get_mean_width(two_ms_avg_OD[70:370, 124:424])
    m_y_width_3ms, u_m_y_width_3ms, m_redchi_3ms = 24*(pix_size/magnification), pix_size/magnification, 1.
    #get_mean_width(three_ms_avg_OD[70:370, 124:424])
    m_y_width_4ms, u_m_y_width_4ms, m_redchi_4ms = 28*(pix_size/magnification), pix_size/magnification, 1.
    #get_mean_width(four_ms_avg_OD[70:370, 124:424])

    def width_to_f_perp(gauss_width, u_gauss_width, TOF):
        frequencies = (2*mass*gauss_width**2)/(2*np.pi*hbar*TOF**2) 
        u_frequencies = (4*mass*gauss_width*u_gauss_width*2)/(2*np.pi*hbar*TOF**2)
        return frequencies, u_frequencies

    f_perp_2ms, u_f_perp_2ms = width_to_f_perp(y_width_2ms, u_y_width_2ms, TOF=2*ms)
    f_perp_3ms, u_f_perp_3ms = width_to_f_perp(y_width_3ms, u_y_width_3ms, TOF=3*ms)
    f_perp_4ms, u_f_perp_4ms = width_to_f_perp(y_width_4ms, u_y_width_4ms, TOF=4*ms)

    mean_f_perp_2ms, u_mean_f_perp_2ms = width_to_f_perp(m_y_width_2ms, u_m_y_width_2ms, TOF=2*ms)
    mean_f_perp_3ms, u_mean_f_perp_3ms = width_to_f_perp(m_y_width_3ms, u_m_y_width_3ms, TOF=3*ms)
    mean_f_perp_4ms, u_mean_f_perp_4ms = width_to_f_perp(m_y_width_4ms, u_m_y_width_4ms, TOF=4*ms)
    
    def clip_f_perp(freq, u_freq):
        freq[freq > 30e3] = 1.
        u_freq[freq > 30e3] = 1.
        return freq, u_freq

    f_perp_2ms, u_f_perp_2ms = clip_f_perp(f_perp_2ms, u_f_perp_2ms)
    f_perp_3ms, u_f_perp_3ms = clip_f_perp(f_perp_3ms, u_f_perp_3ms)
    f_perp_4ms, u_f_perp_4ms = clip_f_perp(f_perp_4ms, u_f_perp_4ms)

    fig = plt.figure()
    ax1 = plt.subplot(111)
    x_axis = np.linspace(0, f_perp_2ms.shape[0], f_perp_2ms.shape[0])
    sc1 = ax1.scatter(x_axis, f_perp_2ms, c='k', s=20)
    _, _, sc1_errorcollection = ax1.errorbar(x_axis, f_perp_2ms, xerr=0.5, yerr=u_f_perp_2ms, 
                                             marker='', ls='', zorder=0)
    sc1_errorcollection[0].set_color('r'), sc1_errorcollection[1].set_color('r')
    sc2 = ax1.scatter(x_axis, f_perp_3ms, c='k', s=20)
    _, _, sc2_errorcollection = ax1.errorbar(x_axis, f_perp_3ms, xerr=0.5, yerr=u_f_perp_3ms, 
                                             marker='', ls='', zorder=0)
    sc2_errorcollection[0].set_color('g'), sc2_errorcollection[1].set_color('g')
    sc3 = ax1.scatter(x_axis, f_perp_4ms, c='k', s=20)
    _, _, sc3_errorcollection = ax1.errorbar(x_axis, f_perp_4ms, xerr=0.5, yerr=u_f_perp_4ms, 
                                             marker='', ls='', zorder=0)
    sc3_errorcollection[0].set_color('b'), sc3_errorcollection[1].set_color('b')
    plt.show()

    with h5py.File(calibrations_h5) as calibrations:
        h5_save(calibrations, 'f_perp/2_ms', mean_f_perp_2ms)
        h5_save(calibrations, 'f_perp/3ms', mean_f_perp_3ms)
        h5_save(calibrations, 'f_perp/4ms', mean_f_perp_4ms)
        h5_save(calibrations, 'f_perp/u_2ms', u_mean_f_perp_2ms)
        h5_save(calibrations, 'f_perp/u_3ms', u_mean_f_perp_3ms)
        h5_save(calibrations, 'f_perp/u_4ms', u_mean_f_perp_4ms)

def calibrate_LG_zR():
    raw_images = load_data(calibration_shots_h5, 'zR/zR_raw')
    Isat = 127 #load_data(calibrations_h5, 'Isat/Isat')
    alpha = load_data(calibrations_h5, 'Isat/alpha')
    magnification = 5.33 # Previous to final magnification

    tube_on_dataset = raw_images[0:20]
    tube_off_dataset = raw_images[20::]

    def compute_dataset_OD(dataset):
        dataset_od = []
        for shot in tqdm(dataset):
            atoms, probe, dark = shot
            div = np.ma.masked_less_equal(np.ma.masked_invalid((atoms-dark)/(probe-dark)), 0.)
            Isat_term = (probe-atoms)/Isat
            corrected_OD = -alpha*np.log(div) + Isat_term
            dataset_od.append(corrected_OD)
        return np.array(dataset_od)

    on_OD = compute_dataset_OD(tube_on_dataset)
    off_OD = compute_dataset_OD(tube_off_dataset)

    difference = on_OD.mean(axis=0)-off_OD.mean(axis=0)
    contrast = difference/(on_OD.mean(axis=0)+off_OD.mean(axis=0))
    wavelet = np.nan_to_num(-np.log(contrast))

    clean_contrast = np.zeros_like(contrast)
    for i, row in list(enumerate(wavelet)):
        for j, col in list(enumerate(row)):
            if col == 0:
                clean_contrast[i, j] = difference[i, j]
            else:
                clean_contrast[i, j] = np.random.uniform(-0.02, 0.02)

    def lmfit_gaussian_beam_depletion(xdata, ydata, zdata, u_zdata):
        pars = Parameters()
        pars.add('Tilt', value=-0.02, min=-0.1, max=0.1, vary=False)
        pars.add('Depletion', value=-0.2, vary=True)
        pars.add('Waist', value=5.6, min=2, max=10, vary=True)
        pars.add('x0', value=320, min=300, max=350, vary=True)
        pars.add('y0', value=264, min=250, max=300, vary=True)
        pars.add('offset', value=0.15, min=-0.2, max=0.2, vary=True)
        pars.add('quad_offset', value=-0.01, vary=True)
        pars.add('cube_offset', value=0.1, vary=True)

        def residuals(pars, xdata, ydata, zdata, u_zdata):
            theta = pars['Tilt']
            amp = pars['Depletion']
            w0 = pars['Waist']
            x0 = pars['x0']
            y0 = pars['y0']
            z0 = pars['offset']
            q0 = pars['quad_offset']
            c0 = pars['cube_offset']
            zmodel = waist2d(xdata, ydata, theta, amp, w0, x0, y0, z0, q0, c0)
            return (zmodel-zdata)/u_zdata

        return minimize(residuals, pars, args=(xdata, ydata, zdata, u_zdata),
                        method='leastsq', nan_policy='omit')

    def waist2d(x, y, theta, amp, w0, x0, y0, offset, quad_offset, cube_offset):
        x = x[np.newaxis, :]
        y = y[:, np.newaxis]
        sigma_v, sigma_u = w0, np.pi*w0**2/(0.532*magnification/5.6)
        u, v = ((x-x0)*np.cos(theta)-(y-y0)*np.sin(theta)), ((x-x0)*np.sin(theta)+(y-y0)*np.cos(theta))
        I = amp*np.exp(-(v**2/(sigma_v**2*(1+u**2/sigma_u**2))))/(sigma_v**2*(1+u**2/sigma_u**2)) + offset + quad_offset*(u**2 + v**2)
        return I + cube_offset*(u**3+v**3)

    x_coords, y_coords = np.linspace(0, difference.shape[1], difference.shape[1]), np.linspace(0, difference.shape[0], difference.shape[0])
    result = lmfit_gaussian_beam_depletion(x_coords, y_coords, difference, 0.01)
    parameters = np.array([result.params[key].value for key in result.params.keys()])
    u_params = np.sqrt(np.diag(result.covar))
    report_fit(result)

    min_waist = parameters[2]*pix_size/magnification
    zR = pi*(min_waist)**2/(0.532*um)
    u_min_waist = u_params[2]*pix_size/magnification
    u_zR = pi*min_waist*u_min_waist/(0.532*um)

    with h5py.File(calibrations_h5) as calibrations:
        h5_save(calibrations, 'zR/w0', min_waist)
        h5_save(calibrations, 'zR/zR', zR)
        h5_save(calibrations, 'zR/u_w0', u_min_waist)
        h5_save(calibrations, 'zR/u_zR', u_zR)


def calibrate_T3D_TOF():
    raw_shots = load_data(calibration_shots_h5, 'T3D/T3D_raw')
    final_dipole_command = load_data(calibration_shots_h5, 'T3D/T3D_final_dipole')
    Isat = 297
    alpha = 1.0

    # TOF magnification gets "corrected", unfortunately this is the best that can be done.
    magnification = 3.44*load_data(calibrations_h5, 'magnification/M_bar')/5.33

    def compute_dataset_OD(dataset):
        dataset_od = []
        for shot in tqdm(dataset):
            atoms, probe, dark = shot
            div = np.ma.masked_less_equal(np.ma.masked_invalid((atoms-dark)/(probe-dark)), 0.)
            Isat_term = (probe-atoms)/Isat
            corrected_OD = -alpha*np.log(div) + Isat_term
            dataset_od.append(corrected_OD)
        return np.array(dataset_od)
    
    tof_OD = compute_dataset_OD(raw_shots)
    #log_OD = np.nan_to_num(np.log(tof_OD))

    x_slice_od = np.array([shot[234:244, :].mean(axis=0) for shot in tof_OD])
    y_slice_od = np.array([shot[:, 363:373].mean(axis=1) for shot in tof_OD])
    #log_x_slice_od = np.array([shot[234:244, :].mean(axis=0) for shot in log_OD])
    #log_y_slice_od = np.array([shot[:, 363:373].mean(axis=1) for shot in log_OD])

    def lmfit_bimodal1D(xdata, ydata, u_ydata):
        pars = Parameters()
        pars.add('g_amp', value = 0.5, min = 0., max = 5, vary = True)
        pars.add('g_x0', value = 360, min = 200, max = 400, vary = True)
        pars.add('g_stdev', value = 40., min = 10, max = 250, vary = True)
        pars.add('g_off', value = 0.0, vary = False)
        pars.add('tf_amp', value = 0.5, vary = True)
        pars.add('tf_x0', value = 364.24, min = 200, max = 400, vary = False)
        pars.add('tf_Rx', value= 50., min=0, max=100, vary = True)
        pars.add('tf_off', value = 0.0, vary = True)

        def residuals(pars, xdata, ydata, u_ydata):
            g_amp, g_x0 = pars['g_amp'], pars['g_x0']
            g_stdev, g_off  = pars['g_stdev'], pars['g_off'] 
            tf_amp, tf_x0 = pars['tf_amp'], pars['tf_x0']
            tf_Rx, tf_off =  pars['tf_Rx'], pars['tf_off']
            ymodel = bimodal(xdata, g_amp, g_x0, g_stdev, g_off, tf_amp, tf_x0, tf_Rx, tf_off)
            return (ymodel-ydata)/u_ydata
        
        return minimize(residuals, pars, args=(xdata, ydata, u_ydata),
                        method='leastsq', nan_policy='omit')

    def bimodal(x, gaussamp, gauss_center, gauss_width, gauss_offset, tf_amp, tf_center, tf_radius, tf_offset):
        return (gauss_1d(x, gaussamp, gauss_center, gauss_width, gauss_offset) +  
                thomas_fermi(x, tf_amp, tf_center, tf_radius, tf_offset))

    def thomas_fermi(x, amp, x0, xTF, off):
        return amp*np.maximum(1-(x-x0)**2/(xTF**2), 0)**(3/2) + off

    def gauss_1d(x, amp, x0, sigma_x, off):
        return amp*np.exp(-((x-x0)**2)/(2*sigma_x**2)) + off

    def get_thermal_widths(od_slices):
        widths, u_widths, redchis = [], [], []
        for od_slice in od_slices:
            x_coords = np.linspace(0, od_slice.shape[0], od_slice.shape[0])
            result = lmfit_bimodal1D(x_coords, od_slice, 0.1)
            pars = np.array([result.params[key].value for key in result.params.keys()])
            widths.append(pars[2])
            try:
                u_pars = np.sqrt(np.diag(result.covar))
            except ValueError:
                u_pars = np.zeros_like(pars)
            u_widths.append(u_pars[2])
            redchis.append(result.redchi)
            #plt.plot(x_coords, od_slice)
            #plt.plot(x_coords, bimodal(x_coords, *pars), 'r')
            #report_fit(result)

        return pix_size*np.array(widths)/magnification, pix_size*np.array(u_widths)/magnification, np.array(redchis)

    dx, u_dx, xredchis = get_thermal_widths(x_slice_od)
    dy, u_dy, yredchis = get_thermal_widths(y_slice_od)

    def width_to_temperature(TOF_widths, u_TOF_widths):
        TOF = 24.7*ms
        temperatures = TOF_widths**2*mass/(1.38e-23*TOF**2*nK)
        u_temperatures = 2*mass*TOF_widths*u_TOF_widths/(1.38e-23*TOF**2*nK)
        return temperatures, u_temperatures

    Tx, u_Tx = width_to_temperature(dx, u_dx)
    Ty, u_Ty = width_to_temperature(dy, u_dy)

    def lmfit_line(xdata, ydata, u_ydata):
        pars = Parameters()
        pars.add('Slope', value=500, vary=True)
        pars.add('Intercept', value=1.0, vary=True)

        def residuals(pars, tdata, ydata, u_ydata):
            m = pars['Slope']
            y0 = pars['Intercept']
            ymodel = linear(xdata, m, y0)
            return (ymodel-ydata)/u_ydata

        return minimize(residuals, pars, args=(xdata, ydata, u_ydata),
                        method='leastsq', nan_policy='omit')

    def linear(x, m, y0):
        return m*x + y0

    Tr = np.sqrt(Tx**2 + Ty**2)
    u_Tr = np.sqrt(u_Tx**2 + u_Ty**2)

    T_calibration_result = lmfit_line(final_dipole_command, Tr, u_Tr)
    pars = np.array([T_calibration_result.params[key].value for key in T_calibration_result.params.keys()])
    u_pars = np.sqrt(np.diag(T_calibration_result.covar))

    with h5py.File(calibrations_h5) as calibrations:
        h5_save(calibrations, 'T3D/slope', pars[0])
        h5_save(calibrations, 'T3D/intercept', pars[1])
        h5_save(calibrations, 'T3D/u_slope', u_pars[0])
        h5_save(calibrations, 'T3D/u_intercept', u_pars[1])    

def calibrate_f_long():
    raw_shots = load_data(calibration_shots_h5, 'f_long/f_long_raw')[13::]
    time = load_data(calibration_shots_h5, 'f_long/f_long_times')[13::]
    Isat = load_data(calibrations_h5, 'Isat/Isat')
    alpha = load_data(calibrations_h5, 'Isat/alpha')

    magnification = 5.33 # For this scan

    def compute_dataset_OD(dataset):
        dataset_od = []
        for shot in tqdm(dataset):
            atoms, probe, dark = shot
            div = np.ma.masked_less_equal(np.ma.masked_invalid((atoms-dark)/(probe-dark)), 0.)
            Isat_term = (probe-atoms)/Isat
            corrected_OD = -alpha*np.log(div) + Isat_term
            dataset_od.append(corrected_OD)
        return np.array(dataset_od)
    
    f_long_OD = compute_dataset_OD(raw_shots)
    od_slices = np.array([shot[231:238, :].mean(axis=0) for shot in f_long_OD])

    def lmfit_gaussian_1d(xdata, ydata, u_ydata, guesses=None):
        pars = Parameters()
        pars.add('Amplitude', value=0.2, min=0.01, max=0.5, vary=True)
        pars.add('Center', value=350, min=70, max=500, vary=True)
        pars.add('Stdev', value=150, min=5, max=350, vary=True)
        pars.add('Offset', value=0.01, min=-0.3, max=0.2, vary=True)

        def residuals(pars, xdata, ydata, u_ydata):
            A = pars['Amplitude']
            x0 = pars['Center']
            dx = pars['Stdev']
            y0 = pars['Offset']
            ymodel = gaussian_1d(xdata, A, x0, dx, y0)
            return (ydata-ymodel)/u_ydata

        return minimize(residuals, pars, args=(xdata, ydata, u_ydata),
                          method='leastsq', nan_policy='omit')

    def gaussian_1d(x, amp, x0, sigma, offset):
        return amp*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

    def get_peak_COM(od_slices):
        peaks, u_peaks, redchis = [], [], []
        x_axis = np.linspace(0, od_slices.shape[1], od_slices.shape[1])
        for od_slice in od_slices:
            result = lmfit_gaussian_1d(x_axis, od_slice, 0.1)
            pars = np.array([result.params[key].value for key in result.params.keys()])
            try: 
                u_pars = np.sqrt(np.diag(result.covar))
            except ValueError:
                u_pars = np.zeros_like(pars)     
            peaks.append((pix_size/magnification)*pars[1])
            u_peaks.append((pix_size/magnification)*u_pars[1])
            redchis.append(result.redchi)
        return np.array(peaks), np.array(u_peaks), np.array(redchis)

    xCOM, u_xCOM, xredchis = get_peak_COM(od_slices)

    def lmfit_sine_exp(xdata, ydata, u_ydata):
        pars = Parameters()
        pars.add('Amplitude', value=500, vary=True)
        pars.add('Frequency', value=10, vary=True)
        pars.add('Phase', value=0.1, vary=True)
        pars.add('Tau', value=0.01, vary=True)
        pars.add('Offset', value=0.001, vary=True)

        def residuals(pars, tdata, ydata, u_ydata):
            amp = pars['Amplitude']
            f0 = pars['Frequency']
            phi = pars['Phase']
            tau = pars['Tau']
            off = pars['Offset']
            ymodel = sine_exp(xdata, amp, f0, phi, tau, off)
            return (ymodel-ydata)/u_ydata

        return minimize(residuals, pars, args=(xdata, ydata, u_ydata),
                        method='leastsq', nan_policy='omit')

    def sine_exp(t, amp, f0, phi, tau, off):
        return amp*np.sin(2*pi*f0*t + phi)*np.exp(-t/tau)+off

    # Bad shots are deleted by hand... 
    xCOM [xCOM < 350*um] = np.nan 
    xCOM [xCOM > 450*um] = np.nan

    result = lmfit_sine_exp(time, xCOM, u_xCOM)
    pars = np.array([result.params[key].value for key in result.params.keys()])
    u_pars = np.sqrt(np.diag(result.covar))

    with h5py.File(calibrations_h5) as calibrations:
        h5_save(calibrations, 'f_long/f0', pars[1])
        h5_save(calibrations, 'f_long/u_f0', u_pars[1])

def calibrate_long_wG():
    long_profile = load_data(calibration_shots_h5, 'wG/wG_profile')
    magnification = load_data(calibrations_h5, 'magnification/M_bar')

    def lmfit_gaussian_2d(xdata, ydata, zdata, u_zdata):
        pars = Parameters()
        pars.add('Tilt', value=-0.02, min=-0.1, max=0.1, vary=True)
        pars.add('Amplitude', value=4e4, vary=True)
        pars.add('wx', value=150, min=2, max=400, vary=True)
        pars.add('wy', value=150, min=2, max=400, vary=True)
        pars.add('x0', value=371, vary=True)
        pars.add('y0', value=181, vary=True)
        pars.add('offset', value=1e3, vary=True)

        def residuals(pars, xdata, ydata, zdata, u_zdata):
            theta = pars['Tilt']
            amp = pars['Amplitude']
            wx = pars['wx']
            wy = pars['wy']
            x0 = pars['x0']
            y0 = pars['y0']
            off = pars['offset']
            zmodel = gaussian_2d(xdata, ydata, theta, amp, wx, wy, x0, y0, off)
            return (zmodel-zdata)/u_zdata

        return minimize(residuals, pars, args=(xdata, ydata, zdata, u_zdata),
                        method='leastsq', nan_policy='omit')

    def gaussian_2d(x, y, theta, amp, wx, wy, x0, y0, offset):
        x = x[np.newaxis, :]
        y = y[:, np.newaxis]
        u, v = ((x-x0)*np.cos(theta)-(y-y0)*np.sin(theta)), ((x-x0)*np.sin(theta)+(y-y0)*np.cos(theta))
        return amp*np.exp(-(2*v**2/wy**2) - (2*u**2/wx**2)) + offset 

    x_coords, y_coords = np.linspace(0, long_profile.shape[1], long_profile.shape[1]), np.linspace(0, long_profile.shape[0], long_profile.shape[0])
    result = lmfit_gaussian_2d(x_coords, y_coords, long_profile, 5)
    parameters = np.array([result.params[key].value for key in result.params.keys()])
    try:
        u_params = np.sqrt(np.diag(result.covar))
    except ValueError:
        u_params = np.zeros_like(parameters)
    report_fit(result)

    x_waist = parameters[2]*pix_size/magnification
    zR = pi*(x_waist)**2/(1.064*um)
    u_x_waist = u_params[2]*pix_size/magnification
    u_zR = pi*x_waist*u_x_waist/(1.064*um)

    with h5py.File(calibrations_h5) as calibrations:
        h5_save(calibrations, 'wG/wx0', x_waist)
        h5_save(calibrations, 'wG/u_wx0', u_x_waist)

def calibrate_long_trap_depth():
    wx = load_data(calibrations_h5, 'wG/wx0')
    u_wx = load_data(calibrations_h5, 'wG/u_wx0')

    pwr = 0.800

    # This is not really a calibration more than it is an estimate.
    from calculations.optics_and_fields.optical_dipole_traps import optical_dipole_trap

    long_beam_pars = {'wavelength': 1064*nm,
                    'power': pwr,
                    'mode' : 'gaussian',
                    'mode_pars': {'wx0':wx, 'wy0':101*um}}
    long_geometry_pars =  {'center':     {'x0':0*um, 'y0':0*um, 'z0':0*um}, 
                         'grid_size':  {'Nx':2**3+1, 'Ny':2**3+1, 'Nz':2**3+1},
                         'grid_range': {'xlim':500*um, 'ylim':500*um, 'zlim':1000*um}}
    long_trap = optical_dipole_trap(long_beam_pars, long_geometry_pars, atom='Rb')
    omega_long = long_trap.get_harmonic_curvature(in_x=True, in_Hz=False)
    long_trap_depth = long_trap.get_potential(offset_to_zero=True).max()

    with h5py.File(calibrations_h5) as calibrations:
        h5_save(calibrations, 'Vt/Vt', long_trap_depth)

if __name__ in '__main__':
    # pack_calibration_data()
    # calibrate_insitu_magnification()
    # calibrate_insitu_Isat()
    # calibrate_f_perp()
    # calibrate_LG_zR()
    # calibrate_T3D_TOF()
    # calibrate_f_long()
    # calibrate_long_wG()
    calibrate_long_trap_depth()
