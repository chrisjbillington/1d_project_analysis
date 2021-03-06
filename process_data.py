import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage.filters import uniform_filter1d
from tqdm import tqdm
from image_reconstruction.cpu_reconstructor import CPUReconstructor

pi = np.pi

PROCESS_DATA_OUTDIR = 'process_data_output'
if not os.path.exists(PROCESS_DATA_OUTDIR):
    os.mkdir(PROCESS_DATA_OUTDIR)

raw_data_h5 = 'raw_data.h5'
processed_data_h5 = 'processed_data.h5'

# The number of images and the dimensions of each image:
with h5py.File(raw_data_h5, 'r') as raw_data:
    n_shots = raw_data['atoms'].shape[0]
    image_shape = raw_data['atoms'].shape[1:]
    # The number of values of the final_dipole variable:
    realisations = sorted(set(zip(raw_data['short_TOF'], raw_data['final_dipole'])))
    n_realisations = len(realisations)

# How many principal components we use in the slice-by-slice reconstructions:
n_principal_components = 4

# The ROI where the atoms are:
ROI_x_start = 0
ROI_x_stop = image_shape[1]
ROI_y_start = 180
ROI_y_stop = 255

# A mask equal to zero in the ROI and ones elsewhere:
ROI_mask = np.ones(image_shape)
ROI_mask[ROI_y_start:ROI_y_stop, ROI_x_start:ROI_x_stop] = 0

# Imaging parameters:
with h5py.File('calibrations.h5') as f:
    Isat = f['Isat/Isat'].value
    alpha = f['Isat/alpha'].value
    magnification = f['magnification/M_bar'].value

pixel_size = 5.6e-6
dy_pixel = pixel_size / magnification
tau = 20e-6 # imaging pulse duration

# Physical constants
hbar = 1.054571628e-34
lambda_Rb = 780.241209686e-9
gamma_D2 = 1/26.2348e-9
m_Rb = 1.443160648e-25
sigma_0 = 3*lambda_Rb**2 / (2*pi)
v_recoil = 2*pi*hbar/(m_Rb * lambda_Rb)

# The shot we use for comparison of non-averaged with averaged data:
SHOT_OF_INTEREST = 313


def gaussian_blur(image, px):
    """gaussian blur an image by given number of pixels"""
    from scipy.signal import convolve2d
    x = np.arange(-4*px, 4*px + 1, 1)[None, :]
    y = np.arange(-4*px, 4*px + 1, 1)[:, None]
    kernel = np.exp(-(x**2 + y**2)/(2*px**2))
    kernel /= kernel.sum()
    return convolve2d(image, kernel, mode='same')


def h5_save(group, name, data):
    """Set a h5 dataset to the given array, replacing existing values if it
    already exists. If the arrays do not have the same shape and datatype,
    then delete the existing dataset before replaing it. This will leave
    garbage in the h5 file that will need to be cleaned up if one wants to
    reclaim the disk space with h5repack. Having to do this is the main
    downside of using hdf5 for this instead of np.save/np.load/np.memmap"""
    try:
        group[name] = data
    except RuntimeError:
        try:
            # Try overwriting just the data if shape and dtype are compatible:
            group[name][...] = data
        except TypeError:
            del group[name]
            group[name] = data
            import sys
            msg = ('Warning: replacing existing h5 dataset, but disk space ' +
                   'has not been reclaimed, leaving the h5 file larger ' +
                   'than necessary. To reclaim disk space use the h5repack ' +
                   'tool or delete the whole h5 file and regenerate from scratch\n')
            sys.stderr.write(msg)


def save_pca_images():
    """Save images of principal components of all raw sets of images to disk
    to see what they look like"""

    def _save_pca_images(images, name, mask=None):
        outdir = os.path.join(PROCESS_DATA_OUTDIR, 'pca_images', name)
        reconstructor = CPUReconstructor(max_ref_images=n_shots)
        print(f'Saving {name} PCA images to {outdir}/')
        if mask is None:
            mask = np.ones(image_shape)
        for image in tqdm(images, desc='  Adding ref images'):
            reconstructor.add_ref_image(image * mask)
        print("  Doing PCA")
        mean_image, principal_components, evals = reconstructor.pca_images()
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        plt.semilogy(evals/evals.sum(), '-')
        plt.xlabel('principal component number')
        plt.ylabel('fractional variance explained')
        plt.grid(True)
        plt.savefig(os.path.join(outdir, 'explained_variance.png'))
        plt.clf()

        plt.plot(evals.cumsum()/evals.sum(), '-')
        plt.xlabel('principal component number')
        plt.ylabel('cumulative fractional variance explained')
        plt.grid(True)
        plt.savefig(os.path.join(outdir, 'explained_variance_cumulative.png'))
        plt.clf()

        plt.imsave(os.path.join(outdir, 'mean_image.png'), mean_image)
        for i, image in tqdm(enumerate(principal_components),
                             desc='  Saving images', total=len(principal_components)):
            plt.imsave(os.path.join(outdir, f'principal_component_{i:04d}.png'), image)

    # Do this for each set of raw images:
    with h5py.File(raw_data_h5, 'r') as raw_data:
        _save_pca_images(raw_data['probe'], 'probe')
        _save_pca_images(raw_data['dark'], 'dark')
        _save_pca_images(raw_data['atoms'], 'atoms', ROI_mask)


def compute_mean_raw_images():
    """Compute the mean atoms, probe and dark frame and save to file"""
    outdir = os.path.join(PROCESS_DATA_OUTDIR, 'mean_raw_images')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    print('Computing mean raw images')
    with h5py.File(raw_data_h5, 'r') as raw_data:
        with h5py.File(processed_data_h5) as processed_data:
            print('  Computing mean atoms frame')
            mean_raw_atoms = np.mean(raw_data['atoms'], axis=0)
            h5_save(processed_data, 'mean_raw_atoms', mean_raw_atoms)
            plt.imshow(mean_raw_atoms)
            plt.colorbar()
            plt.savefig(os.path.join(outdir, 'mean_raw_atoms.png'))
            plt.clf()

            print('  Computing mean probe frame')
            mean_raw_probe = np.mean(raw_data['probe'], axis=0)
            h5_save(processed_data, 'mean_raw_probe', mean_raw_probe)
            plt.imshow(mean_raw_probe)
            plt.colorbar()
            plt.savefig(os.path.join(outdir, 'mean_raw_probe.png'))
            plt.clf()

            print('  Computing mean dark frame')
            mean_raw_dark = np.mean(raw_data['dark'], axis=0)
            h5_save(processed_data, 'mean_raw_dark', mean_raw_dark)
            plt.imshow(mean_raw_dark)
            plt.colorbar()
            plt.savefig(os.path.join(outdir, 'mean_raw_dark.png'))
            plt.clf()


def compute_dark_systematic_offset():
    """Compute an array that models the systematic difference in counts
   between the mean atoms and probe images, which we attribute to a systematic
   difference in dark counts between the frames"""
    
    def model_inhomogeneous_dark(gamma, y0, x0, sigma_x, delta, epsilon):
        """Extra dark counts present in the mean probe frame but not in the
        mean atoms frame, as a fitted model"""
        y = np.linspace(0, image_shape[0]-1, image_shape[0])[:, None] + np.ones(image_shape[1])[None, :]
        x = np.linspace(0, image_shape[1]-1, image_shape[1])[None, :] + np.ones(image_shape[0])[:, None]
        y_structure = gamma**2 / (y - y0)**2
        x_structure =  (x - x0) * (delta * np.exp(-(x - x0)**2/(2*sigma_x**2)) - epsilon**2 *(x -x[0, -1])**2)
        return y_structure + x_structure 

    def model_homogeneous_dark(beta):
        """Extra dark counts present in the mean atoms frame but not in the
        mean probe frame, as a fraction of the dark frame"""
        return beta * mean_raw_dark

    def model_scaled_probe(alpha):
        """Extra probe counts obtained by adding a multiple of (mean_raw_probe -
        mean_raw_dark) to the mean probe frame."""
        return mean_raw_probes + alpha * (mean_raw_probes - mean_raw_dark)

    def errfunc(args):
        """Error function for least squares fitting"""
        alpha, beta, gamma, y0, x_0, delta, sigma_x, epsilon = args

        scaled_and_offset_probe = model_scaled_probe(alpha) + model_homogeneous_dark(beta)
        extra_inhomogeneous_dark = model_inhomogeneous_dark(gamma, y0, x_0, delta, sigma_x, epsilon)
        residual_inhomogeneous_dark = scaled_and_offset_probe - mean_raw_atoms

        model_probe = scaled_and_offset_probe - extra_inhomogeneous_dark

        return ((model_probe - mean_raw_atoms) * ROI_mask).flatten()

    print('Computing model for systematic offset in dark counts')
    outdir = os.path.join(PROCESS_DATA_OUTDIR, 'dark_systematic_offset')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    with h5py.File(processed_data_h5) as processed_data:
        mean_raw_probes = processed_data['mean_raw_probe'][:]
        mean_raw_atoms = processed_data['mean_raw_atoms'][:]
        mean_raw_dark = processed_data['mean_raw_dark'][:]

        raw_difference = mean_raw_probes - mean_raw_atoms

        from scipy.optimize import leastsq
        (alpha, beta, gamma, y0, x_0, delta, sigma_x, epsilon), res = leastsq(errfunc, [0.0273, 0.00198, 97.9, -81.1, 32.5, 92.7, -0.0016, 3.45e-5])

        # Compute the modelled systematic difference and save to file:
        scaled_difference = model_scaled_probe(alpha) - mean_raw_atoms
        scaled_and_offset_difference = model_scaled_probe(alpha) + model_homogeneous_dark(beta) - mean_raw_atoms
        corrected_difference = model_scaled_probe(alpha) + model_homogeneous_dark(beta) - model_inhomogeneous_dark(gamma, y0, x_0, delta, sigma_x, epsilon) - mean_raw_atoms
        dark_systematic_offset = model_homogeneous_dark(beta) - model_inhomogeneous_dark(gamma, y0, x_0, delta, sigma_x, epsilon)
        h5_save(processed_data, 'dark_systematic_offset', dark_systematic_offset)


    # Plot the inhomogeneous part of the fit:
    scaled_and_offset_probe = model_scaled_probe(alpha) + model_homogeneous_dark(beta)
    extra_inhomogeneous_dark = model_inhomogeneous_dark(gamma, y0, x_0, delta, sigma_x, epsilon)
    residual_inhomogeneous_dark = scaled_and_offset_probe - mean_raw_atoms

    plt.plot((residual_inhomogeneous_dark * ROI_mask).mean(1), label='averaged over x (data)')
    plt.plot((extra_inhomogeneous_dark * ROI_mask).mean(1), label='averaged over x (model)')
    plt.plot((residual_inhomogeneous_dark * ROI_mask).mean(0), label='averaged over y (data)')
    plt.plot((extra_inhomogeneous_dark * ROI_mask).mean(0), label='averaged over y (model)')
    plt.grid(True)
    plt.legend()
    plt.xlabel('x or y pixel number')
    plt.ylabel('counts')
    plt.savefig(os.path.join(outdir, 'inhomogeneous_fit.png'))
    plt.clf()


    # Plot the effect of this on the average differences over x and y:
    plt.plot(raw_difference.mean(axis=1), label='raw')
    plt.plot(scaled_difference.mean(axis=1), label=f'raw scaled')
    plt.plot(scaled_and_offset_difference.mean(axis=1), label=f'raw scaled + const*dark')
    plt.plot(corrected_difference.mean(axis=1), label=f'raw scaled + model')

    plt.axvline(ROI_y_start, linestyle='--', color='k')
    plt.axvline(ROI_y_stop, linestyle='--', color='k')

    plt.legend()
    plt.grid(True)
    plt.ylabel('mean counts difference along y')
    plt.xlabel('y pixel numer')
    plt.savefig(os.path.join(outdir, 'difference_y_lineouts.png'))
    plt.clf()

    plt.plot((raw_difference).mean(axis=0), label=f'raw')
    plt.plot((scaled_difference).mean(axis=0), label=f'raw scaled')
    plt.plot((scaled_and_offset_difference).mean(axis=0), label=f'raw scaled + const*dark')
    plt.plot((corrected_difference).mean(axis=0), label=f'raw scaled + model')

    plt.legend()
    plt.grid(True)
    plt.ylabel('mean counts difference along x')
    plt.xlabel('x pixel number')
    plt.savefig(os.path.join(outdir, 'difference_x_lineouts.png'))
    plt.clf()


    # Plot blurred and unblurred 2D images of the scaled + offset differences,
    # the difference after correcting with the model, and the inhomogeneous part
    # of the model:

    plt.imshow(scaled_and_offset_difference, vmin=-1, vmax=1, cmap='seismic')
    plt.colorbar()
    plt.savefig(os.path.join(outdir, 'scaled_and_offset_difference.png'))
    plt.clf()

    plt.imshow(gaussian_blur(scaled_and_offset_difference, 5), vmin=-1, vmax=1, cmap='seismic')
    plt.colorbar()
    plt.savefig(os.path.join(outdir, 'scaled_and_offset_difference_blurred.png'))
    plt.clf()

    plt.imshow(corrected_difference, vmin=-1, vmax=1, cmap='seismic')
    plt.colorbar()
    plt.savefig(os.path.join(outdir, 'corrected_difference.png'))
    plt.clf()

    plt.imshow(gaussian_blur(corrected_difference, 5), vmin=-1, vmax=1, cmap='seismic')
    plt.colorbar()
    plt.savefig(os.path.join(outdir, 'corrected_difference_blurred.png'))
    plt.clf()

    plt.imshow(dark_systematic_offset, vmin=-1, vmax=1, cmap='seismic')
    plt.colorbar()
    plt.savefig(os.path.join(outdir, 'dark_systematic_offset.png'))
    plt.clf()


def reconstruct_probe_frames():
    """Reconstruct a probe frame for each atoms frame by modelling each atoms
    frame to a linear sum of all probe frames in the unmasked region, plus a
    multiple of our modelled systematic difference in dark counts. Save the
    coefficient of the fit for the systematic dark counts for later use in
    reconstructing dark frames"""

    print('Reconstructing probe frames')
    with h5py.File(raw_data_h5, 'r') as raw_data:
        with h5py.File(processed_data_h5) as processed_data:
            raw_atoms_images = raw_data['atoms']
            raw_probe_images = raw_data['probe']
            dark_systematic_offset = processed_data['dark_systematic_offset'][:]

            reconstructor = CPUReconstructor(max_ref_images=n_shots+1)

            for raw_probe_image in tqdm(raw_probe_images, desc='  Adding ref images'):
                reconstructor.add_ref_image(raw_probe_image)

            # Allow the reconstructor to use any scalar multiple of the mean
            # systematic offset in the reconstruction. We will extract the coefficient
            # of the reconstruction so that this can be treated as dark counts when we
            # later reconstruct dark frames:
            reconstructor.add_ref_image(dark_systematic_offset[:])

            reconstructed_probe_frames = np.zeros(raw_atoms_images.shape)
            dark_systematic_offset_coeffs = np.zeros(n_shots)

            for i, frame in tqdm(enumerate(raw_atoms_images), 
                                 desc='  Reconstructing', total=len(raw_atoms_images)):
                reconstructed_probe, rchi2, coeffs = reconstructor.reconstruct(frame, mask=ROI_mask, return_coeffs=True)
                reconstructed_probe_frames[i] = reconstructed_probe
                # How much of the modelled dark counts systematic offset was in this frame? 
                dark_systematic_offset_coeffs[i] = coeffs[-1]

            h5_save(processed_data, 'reconstructed_probe', reconstructed_probe_frames)
            h5_save(processed_data, 'dark_systematic_offset_coeffs', dark_systematic_offset_coeffs)


def plot_reconstructed_probe_frames():
    """Save to disk images of the raw probes, raw atoms and reconstructed probes for comparison"""

    print("Saving reconstructed probe images")

    outdir = os.path.join(PROCESS_DATA_OUTDIR, 'reconstructed_probe')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    with h5py.File(raw_data_h5, 'r') as raw_data:
        with h5py.File(processed_data_h5) as processed_data:
            raw_atoms_images = raw_data['atoms']
            raw_probe_images = raw_data['probe']
            reconstructed_probe_frames = processed_data['reconstructed_probe']
            for i in tqdm(range(len(raw_atoms_images)),  desc='  Saving images'):
                raw_atoms = raw_atoms_images[i]
                raw_probe = raw_probe_images[i]
                recon_probe = reconstructed_probe_frames[i]
                vmin = min(raw_atoms.min(), raw_probe.min(), recon_probe.min())
                vmax = max(raw_atoms.max(), raw_probe.max(), recon_probe.max())
                plt.imsave(os.path.join(outdir, f'{i:04d}_1_raw_probe.png'), raw_probe, vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(outdir, f'{i:04d}_2_raw_atoms.png'), raw_atoms, vmin=vmin, vmax=vmax)
                plt.imsave(os.path.join(outdir, f'{i:04d}_3_recon_probe.png'), recon_probe, vmin=vmin, vmax=vmax)


def reconstruct_dark_frames():
    """Reconstruct a dark frame for each atoms frame. Each reconstructed dark
    frame is the mean dark frame, plus a multiple of the systematic dark
    offset with coefficient determined by the least squares reconstruction
    during the probe reconstruction step, plus a multiple of each of principal
    components 5 and 6 of the probe frames (which we are taking to be readout
    noise since they appear identically in the principal components of the
    dark frames), with coefficients determined by the projection of the atoms
    frame onto those principal components in the unmasked region"""

    print("Reconstructing dark frames")

    with h5py.File(raw_data_h5, 'r') as raw_data:
        with h5py.File(processed_data_h5) as processed_data:
            mean_raw_dark = processed_data['mean_raw_dark']
            dark_systematic_offset = processed_data['dark_systematic_offset']
            dark_systematic_offset_coeffs = processed_data['dark_systematic_offset_coeffs']

            raw_atoms_images = raw_data['atoms']
            raw_probe_images = raw_data['probe']

            reconstructor = CPUReconstructor(max_ref_images=n_shots)
            for probe in tqdm(raw_probe_images, desc='  Adding ref images'):
                reconstructor.add_ref_image(probe)

            print('  Doing PCA')

            # The principal components we wish to count as dark:
            mean_image, principal_components, evals = reconstructor.pca_images()
            pc5, pc6 = principal_components[5], principal_components[6]
            del principal_components

            reconstructed_dark_frames = np.zeros(raw_atoms_images.shape)

            for i, frame in tqdm(enumerate(raw_atoms_images), 
                                 desc='  Reconstructing', total=len(raw_atoms_images)):
                _, _, coeffs = reconstructor.reconstruct(frame, mask=ROI_mask, n_principal_components=10,
                                                                return_coeffs=True)
                dark_systematic_offset_coeff = dark_systematic_offset_coeffs[i]
                offset = dark_systematic_offset_coeff * dark_systematic_offset
                reconstructed_dark = mean_raw_dark + offset + coeffs[5] * pc5 + coeffs[6] * pc6

                reconstructed_dark_frames[i] = reconstructed_dark

            h5_save(processed_data, 'reconstructed_dark', reconstructed_dark_frames)


def plot_reconstructed_dark_frames():
    """Save to disk images of the reconstructed dark frames"""

    print("Saving reconstructed dark images")

    outdir = os.path.join(PROCESS_DATA_OUTDIR, 'reconstructed_dark')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    with h5py.File(processed_data_h5) as processed_data:
        reconstructed_dark_frames = processed_data['reconstructed_dark']
        for i in tqdm(range(len(reconstructed_dark_frames)),  desc='  Saving images'):
            frame = reconstructed_dark_frames[i]
            plt.imsave(os.path.join(outdir, f'{i:04d}.png'), frame)


def compute_OD_and_absorption_and_saturation_parameter():
    """Compute the OD corresponding to each image, as well as the absorbed
    fraction and saturation parameter. Note that due to imaging aberrations
    the OD is not particularly physically meaningful, but we compute it
    anyway, partly so we can compare with the more physically meaningful
    results after other processing."""

    print("Computing naive OD, absorbed fraction and saturation parameter")

    with h5py.File(raw_data_h5, 'r') as raw_data:
        with h5py.File(processed_data_h5) as processed_data:
            raw_atoms_frames = raw_data['atoms']
            reconstructed_dark_frames = processed_data['reconstructed_dark']
            reconstructed_probe_frames = processed_data['reconstructed_probe']

            OD = np.zeros(raw_atoms_frames.shape)
            absorbed_fraction = np.zeros(raw_atoms_frames.shape)
            saturation_parameter = np.zeros(raw_atoms_frames.shape)

            zipvars = zip(raw_atoms_frames, reconstructed_probe_frames, reconstructed_dark_frames)
            for i, (atoms, probe, dark) in tqdm(enumerate(zipvars), total=n_shots):
                absorbed_fraction_i = 1 - (atoms - dark)/(probe - dark)
                saturation_parameter_i = (probe - dark) / Isat

                OD[i] = -alpha * np.log(1 - absorbed_fraction_i) + saturation_parameter_i * absorbed_fraction_i
                absorbed_fraction[i] =  absorbed_fraction_i
                saturation_parameter[i] = saturation_parameter_i

            h5_save(processed_data, 'naive_OD', OD)
            h5_save(processed_data, 'absorbed_fraction', absorbed_fraction)
            h5_save(processed_data, 'saturation_parameter', saturation_parameter)


def plot_OD_and_absorption_and_saturation_parameter():
    """Save to disk images of the OD and saturation parameter"""
    print("Saving naive OD images, and absorption and saturation images")

    outdir_OD = os.path.join(PROCESS_DATA_OUTDIR, 'naive_OD')
    if not os.path.exists(outdir_OD):
        os.mkdir(outdir_OD)
    outdir_absorbed_fraction = os.path.join(PROCESS_DATA_OUTDIR, 'absorbed_fraction')
    if not os.path.exists(outdir_absorbed_fraction):
        os.mkdir(outdir_absorbed_fraction)
    outdir_saturation_parameter = os.path.join(PROCESS_DATA_OUTDIR, 'saturation_parameter')
    if not os.path.exists(outdir_saturation_parameter):
        os.mkdir(outdir_saturation_parameter)

    with h5py.File(processed_data_h5) as processed_data:
        OD = processed_data['naive_OD']
        absorbed_fraction = processed_data['absorbed_fraction']
        saturation_parameter = processed_data['saturation_parameter']
        for i in tqdm(range(len(OD)),  desc='  Saving images'):
            plt.imsave(os.path.join(outdir_OD, f'{i:04d}.png'), OD[i], vmin=-1, vmax=1, cmap='seismic')
            plt.imsave(os.path.join(outdir_absorbed_fraction, f'{i:04d}.png'), absorbed_fraction[i], vmin=-1, vmax=1, cmap='seismic')
            plt.imsave(os.path.join(outdir_saturation_parameter, f'{i:04d}.png'), saturation_parameter[i], vmin=-1, vmax=1, cmap='seismic')


def compute_averages():
    """Average the (naive) ODs of each set of shots that are from the same
    point in parameter space (realisation) to produce a mean OD image for each
    point in the parameter space. Average is computed as:
        -alpha * log (1 - <A>) + <S * A>
    where <A> is the mean of the absorbed fractions and <S * A> is the mean of
    the saturation parameter multiplied by the absorbed fraction. The mean of
    the absorbed fractions is taken before the log to avoid biasing the result
    toward higher ODs, since the log of a Gaussian random variable has
    asymmetric uncertainties and cannot be simply averaged together to obtain
    an unbiased estimate of the most likely value.

    Also and more importantly, compute the mean absorbed fraction for each
    realisation and mean saturation parameter"""

    print("Computing average OD, absorbed_fraction and saturation parameter for each final_dipole/short_TOF pair")

    outdir_OD = os.path.join(PROCESS_DATA_OUTDIR, 'average_naive_OD')
    if not os.path.exists(outdir_OD):
        os.mkdir(outdir_OD)
    outdir_absorbed_fraction = os.path.join(PROCESS_DATA_OUTDIR, 'average_absorbed_fraction')
    if not os.path.exists(outdir_absorbed_fraction):
        os.mkdir(outdir_absorbed_fraction)
    outdir_saturation_parameter = os.path.join(PROCESS_DATA_OUTDIR, 'average_saturation_parameter')
    if not os.path.exists(outdir_saturation_parameter):
        os.mkdir(outdir_saturation_parameter)

    with h5py.File(raw_data_h5, 'r') as raw_data:
        final_dipole = raw_data['final_dipole'][:]
        short_TOF = raw_data['short_TOF'][:]

    with h5py.File(processed_data_h5) as processed_data:
        absorbed_fraction = processed_data['absorbed_fraction']
        saturation_parameter = processed_data['saturation_parameter']

        average_OD = np.zeros((n_realisations, *image_shape))
        average_absorbed_fraction = np.zeros(average_OD.shape)
        average_saturation_parameter = np.zeros(average_OD.shape)

        for i, (tof_val, dipole_val) in tqdm(enumerate(realisations), total=n_realisations, desc='  Saving images'):
            matching_shots = (final_dipole == dipole_val) & (short_TOF == tof_val)

            mean_absorbed_fraction = absorbed_fraction[matching_shots, :, :].mean(axis=0)
            mean_saturation_parameter = saturation_parameter[matching_shots, :, :].mean(axis=0)
            mean_linear_term = (absorbed_fraction[matching_shots, :, :] * saturation_parameter[matching_shots, :, :]).mean(axis=0)
            average_OD_i = -alpha * np.log(1 - mean_absorbed_fraction) + mean_linear_term

            average_OD[i] = average_OD_i
            average_absorbed_fraction[i] = mean_absorbed_fraction
            average_saturation_parameter[i] = mean_saturation_parameter

            plt.imsave(os.path.join(outdir_OD, f'{i:02d}.png'), average_OD_i, vmin=-1, vmax=1, cmap='seismic')
            plt.imsave(os.path.join(outdir_absorbed_fraction, f'{i:02d}.png'), mean_absorbed_fraction, vmin=-1, vmax=1, cmap='seismic')
            plt.imsave(os.path.join(outdir_saturation_parameter, f'{i:02d}.png'), mean_saturation_parameter, vmin=-1, vmax=1, cmap='seismic')

        h5_save(processed_data, 'naive_average_OD', average_OD)
        h5_save(processed_data, 'average_absorbed_fraction', average_absorbed_fraction)
        h5_save(processed_data, 'average_saturation_parameter', average_saturation_parameter)
        h5_save(processed_data, 'realisation_final_dipole', np.array(realisations)[:, 1])
        h5_save(processed_data, 'realisation_short_TOF', np.array(realisations)[:, 0])
    

def reconstruct_absorbed_fraction():
    """Do dimensionality reduction on each vertical slice of the absorbed
    fractions in the ROI using a principal component basis based on that slice
    and four surrounding slices in all average absorbed fractions. The aim of
    this is to reduce the noise present in each slice so that we can sum over
    more pixels vertically before noise dominates the result. Integrate the result
    over y pixels in the ROI to get the y integrated absorbed fraction
    for each realisation. Do this for both
    1. The unaveraged absorbed fractions.
    2. The averaged absorbed fractions.

    Averaging then reconstructing returns numerically identical results to
    reconstructing then averaging, so we compute reconstructed average absorbed
    fractions directly from the average absorbed fractions - but we also
    reconstruct pre-averaged images so we can consider the results of single shots
    too, including using their statistical variation for uncertainty estimation."""

    def get_reference_slices(slice_x_index):

        # Get the reference images to be used for reconstructing a particular
        # slice. This will be the slice we're reconstructing plus two slices
        # from each side, unless we're too close to the left or right of the
        # ROI in which case it will just be the nearest four other slices.
        start_index = x_index - 2
        stop_index = x_index + 3
        # Where, relative to the start index, is the slice we are reconstructing?
        offset = 2
        while start_index < 0:
            start_index += 1
            stop_index += 1
            offset -= 1
        while stop_index >= ROI_x_stop - ROI_x_start:
            start_index -= 1
            stop_index -= 1
            offset += 1

        # The slices, shape (n_realisations, ROI_y_stop - ROI_y_start, 5) 
        slices = average_absorbed_fraction_ROI[:, :, start_index:stop_index]

        assert slices.shape[-1] == 5

        # Transpose to get the vertical dimension first.
        # Shape (ROI_y_stop - ROI_y_start, n_realisations, 5)
        slices = slices.transpose((1, 0, 2))

        # Flatten the last two dimensions so we have a
        # (ROI_y_stop - ROI_y_start, 5*n_realisations) array:
        slices = slices.reshape(ROI_y_stop - ROI_y_start, 5*n_realisations)

        # Transpose again so we have each slice as a row.
        # Shape (5*n_realisations, ROI_y_stop - ROI_y_start)
        slices = slices.transpose()

        # Delete the row corresponding to the slice we are going to reconstruct.
        # Resulting shape: (94, 29):
        # slices = np.delete(slices, image_index*5 + offset, axis=0)

        # Verify for sure that the slice we're reconstructing is not in there:
        # for i in range(94):
        #     assert not np.array_equal(slices[i], data[image_index, :, x_index])

        return slices

    print('Reconstructing absorbed fractions slice-by-slice in ROI')
    with h5py.File(processed_data_h5) as processed_data:
        absorbed_fraction = processed_data['absorbed_fraction']
        absorbed_fraction_ROI = absorbed_fraction[:, ROI_y_start:ROI_y_stop, :]
        average_absorbed_fraction = processed_data['average_absorbed_fraction']
        average_absorbed_fraction_ROI = average_absorbed_fraction[:, ROI_y_start:ROI_y_stop, :]

        n_principal_components = 4

        # Reconstruct slice by slice:
        reconstructed_absorbed_fraction_ROI = np.zeros(absorbed_fraction_ROI.shape)
        reconstructed_average_absorbed_fraction_ROI = np.zeros(average_absorbed_fraction_ROI.shape)
        integrated_reconstructed_average_absorbed_fraction = np.zeros((n_realisations, average_absorbed_fraction_ROI.shape[2]))
        integrated_reconstructed_absorbed_fraction = np.zeros((n_shots, average_absorbed_fraction_ROI.shape[2]))

        for x_index in tqdm(range(absorbed_fraction_ROI.shape[2]), desc='  Reconstructing slices'):
            reference_slices = get_reference_slices(x_index)

            # Make a reconstructor with this slice over all the shots as reference images:
            reconstructor = CPUReconstructor(len(reference_slices), centered_PCA=False)
            reconstructor.add_ref_images(reference_slices)

            # Reconstruct this slice in each averaged image:
            for i in range(n_realisations):
                target_slice = average_absorbed_fraction_ROI[i, :, x_index]
                reconstructed_slice, rchi2 = reconstructor.reconstruct(target_slice,
                                                 n_principal_components=n_principal_components)
                reconstructed_average_absorbed_fraction_ROI[i, :, x_index] = reconstructed_slice
                integrated_reconstructed_average_absorbed_fraction[i, x_index] = reconstructed_slice.sum(axis=0) * dy_pixel

            # Reconstruct this slice in each pre-averaged image:
            for i in range(n_shots):
                target_slice = absorbed_fraction_ROI[i, :, x_index]
                reconstructed_slice, rchi2 = reconstructor.reconstruct(target_slice,
                                                 n_principal_components=n_principal_components)
                reconstructed_absorbed_fraction_ROI[i, :, x_index] = reconstructed_slice
                integrated_reconstructed_absorbed_fraction[i, x_index] = reconstructed_slice.sum(axis=0) * dy_pixel

        # Infer uncertainty in integrated_reconstructed_average_absorbed_fraction
        # from standard error of the mean for
        # integrated_reconstructed_absorbed_fraction:

        with h5py.File(raw_data_h5, 'r') as raw_data:
            final_dipole = raw_data['final_dipole'][:]
            short_TOF = raw_data['short_TOF'][:]

        u_integrated_reconstructed_average_absorbed_fraction = np.zeros(integrated_reconstructed_average_absorbed_fraction.shape)
        for i, (tof_val, dipole_val) in tqdm(enumerate(realisations), total=n_realisations, desc='  computing standard deviations'):
            matching_shots = (final_dipole == dipole_val) & (short_TOF == tof_val)
            u_integrated_reconstructed_average_absorbed_fraction[i] = integrated_reconstructed_absorbed_fraction[matching_shots, :].std(axis=0) / np.sqrt(matching_shots.sum())
            # 10px smoothing on the uncertainty in integrated absorbed fraction:
            u_integrated_reconstructed_average_absorbed_fraction[i] = uniform_filter1d(u_integrated_reconstructed_average_absorbed_fraction[i], size=10)

        # Save everything
        h5_save(processed_data, 'reconstructed_absorbed_fraction_ROI', reconstructed_absorbed_fraction_ROI)
        h5_save(processed_data, 'reconstructed_average_absorbed_fraction_ROI', reconstructed_average_absorbed_fraction_ROI)
        h5_save(processed_data, 'integrated_reconstructed_average_absorbed_fraction', integrated_reconstructed_average_absorbed_fraction)
        h5_save(processed_data, 'integrated_reconstructed_absorbed_fraction', integrated_reconstructed_absorbed_fraction)
        h5_save(processed_data, 'u_integrated_reconstructed_average_absorbed_fraction', u_integrated_reconstructed_average_absorbed_fraction)

        # plot them:
        outdir_OD = os.path.join(PROCESS_DATA_OUTDIR, 'reconstructed_average_absorbed_fraction')
        if not os.path.exists(outdir_OD):
            os.mkdir(outdir_OD)
        outdir_colsums = os.path.join(PROCESS_DATA_OUTDIR, 'integrated_reconstructed_average_absorbed_fraction')
        if not os.path.exists(outdir_colsums):
            os.mkdir(outdir_colsums)
        for i in tqdm(range(n_realisations), desc='  Saving images'):
            image = average_absorbed_fraction_ROI[i]
            reconstructed_image = reconstructed_average_absorbed_fraction_ROI[i]
            both_images = np.concatenate((image, reconstructed_image, image - reconstructed_image), axis=0)
            plt.imsave(os.path.join(outdir_OD, f'{i:04d}.png'), both_images, vmin=-0.5, vmax=0.5, cmap='seismic')

            column_density_orig = image.sum(axis=0)
            plt.plot(column_density_orig, linewidth=1.0, label='original')

            column_density_recon = integrated_reconstructed_average_absorbed_fraction[i]
            plt.plot(column_density_recon, label=f'uPCA{n_principal_components}', linewidth=1.0)

            plt.axis([0, 648, -0.5, 3])
            plt.xlabel('x pixel')
            plt.ylabel('y integrated absorbed fraction in ROI')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(outdir_colsums, f'{i:04d}.png'))
            plt.clf()


def check_absorbed_fractions_alignment():
    """Compute the average integrated (reconstructed) absorbed fractions again, but this
    time offset the data being averaged horizontally in order to align their centres, as
    determined by a Gaussian fit. This is to check that the individual shots are aligned
    with each other, or if not, by how much they vary in alignment."""

    outdir_alignment = os.path.join(PROCESS_DATA_OUTDIR, 'alignment')
    if not os.path.exists(outdir_alignment):
        os.mkdir(outdir_alignment)

    outdir_alignment_all = os.path.join(PROCESS_DATA_OUTDIR, 'alignment_all')
    if not os.path.exists(outdir_alignment_all):
        os.mkdir(outdir_alignment_all)


    with h5py.File(raw_data_h5, 'r') as raw_data:
        final_dipole = raw_data['final_dipole'][:]
        short_TOF = raw_data['short_TOF'][:]

    with h5py.File(processed_data_h5) as processed_data:
        shots = processed_data['integrated_reconstructed_absorbed_fraction'][:]
        averages = processed_data['integrated_reconstructed_average_absorbed_fraction'][:]
        averages_uncertainty = processed_data['u_integrated_reconstructed_average_absorbed_fraction'][:]

        def gaussian(x, x0, A, sigma):
            return A*np.exp(-(x - x0)**2 / (2*sigma**2))

        def fit_gaussian(x, y, u_y):
            from scipy.optimize import curve_fit
            params, covariance = curve_fit(gaussian, x, y, [x.max()/2, y[int(x.max()/2)], x.max()/5], maxfev=10000)
            return params, covariance

        all_offsets = []
        all_offsets_in_uncertainty_units = []

        for i, (tof_val, dipole_val) in tqdm(enumerate(realisations), total=n_realisations, desc='  aligning single shots'):
            matching_shots = shots[(final_dipole == dipole_val) & (short_TOF == tof_val), :]
            matching_shot_indices = np.array(range(len(shots)))[(final_dipole == dipole_val) & (short_TOF == tof_val)]
            average_shot = averages[i]
            u_average_shot = averages_uncertainty[i]
            u_individual_shot = u_average_shot * np.sqrt(len(matching_shots))
            x = np.arange(len(average_shot))
            (average_x0, average_A, average_sigma), _ = fit_gaussian(x, average_shot, u_average_shot)

            aligned_shots = np.zeros_like(matching_shots)

            offsets = np.zeros(len(matching_shots))

            for j, shot in enumerate(matching_shots):

                (x0, A, sigma), cov = fit_gaussian(x, shot, u_individual_shot)

                u_x0 = np.sqrt(cov[0,0])

                displacement_in_units_of_uncertainty = (average_x0 - x0)/u_x0
                # print(displacement_in_units_of_uncertainty)
                all_offsets_in_uncertainty_units.append(displacement_in_units_of_uncertainty)

                offset_pixels = int(round(average_x0 - x0))
                # offset_pixels = int(round(np.random.randn()*40))
                offset_shot = np.roll(shot, offset_pixels)

                offsets[j] = offset_pixels
                all_offsets.append(offset_pixels)
                if offset_pixels > 0:
                    offset_shot[:offset_pixels] = 0
                elif offset_pixels < 0:
                    offset_shot[offset_pixels:] = 0

                aligned_shots[j] = offset_shot

                # print(offset_pixels)

                offset_x0, offset_A, offset_sigma = fit_gaussian(x, offset_shot, u_individual_shot)

                plt.plot(x, shot*1e6, 'bo', markersize=1, label='single shot')
                plt.plot(x, offset_shot*1e6, 'ro', markersize=1, label='offset single shot')
                plt.plot(x, average_shot*1e6, 'ko', markersize=1, label='average')
                

                plt.plot(x, gaussian(x, average_x0, average_A, average_sigma)*1e6, 'k-')
                plt.plot(x, gaussian(x, x0, A, sigma)*1e6, 'b-')
                plt.plot(x, gaussian(x, offset_x0, offset_A, offset_sigma)*1e6, 'r-')
                plt.axvline(average_x0, color='k')
                plt.axvline(x0, color='b')
                plt.axvline(offset_x0, color='r')
                plt.axis(ymin=-.5, ymax=1)
                plt.legend()
                plt.ylabel(r'integrated absorbed fraction ($\mu$m)')
                raw_shot_no = matching_shot_indices[j]
                plt.savefig(os.path.join(outdir_alignment_all, f'{raw_shot_no:04d}.png'))
                # plt.show()
                plt.clf()


            aligned_average = aligned_shots.mean(axis=0)
            u_aligned_average = aligned_shots.std(axis=0) / np.sqrt(len(matching_shots))

            # aligned_x0, aligned_A, aligned_sigma = fit_gaussian(x, aligned_average, u_aligned_average)

            plt.plot(x, average_shot*1e6, 'o', markersize=1, label='original')
            plt.plot(x, aligned_average*1e6, 'o', markersize=1, label='aligned')

            plt.axis(ymin=-.5, ymax=1)
            plt.legend()
            plt.ylabel(r'integrated absorbed fraction ($\mu$m)')
            plt.figtext(0.15, 0.8, f"final_dipole: {dipole_val}\n short_tof: {tof_val}")
            # plt.show()
            plt.savefig(os.path.join(outdir_alignment, f'absorbed_{i:04d}.png'))

            plt.clf()
            plt.hist(offsets, bins=np.arange(-20.5, 20.5))
            plt.figtext(0.15, 0.8, f"final_dipole: {dipole_val}\n short_tof: {tof_val}")
            plt.xlabel('offset (pixels)')
            plt.ylabel('count')
            plt.savefig(os.path.join(outdir_alignment, f'offsets_{i:04d}.png'))
            plt.clf()
            # plt.show()

        all_offsets = np.array(all_offsets)
        plt.hist(all_offsets, bins=np.arange(-20.5, 20.5))
        plt.figtext(0.15, 0.8, fR"$\mu={all_offsets[abs(all_offsets) < 50].mean():.2f}$" + '\n' + 
                               fR"$\sigma={all_offsets[abs(all_offsets) < 50].std():.2f}$")
        plt.xlabel('offset (pixels)')
        plt.ylabel('count')
        plt.savefig(os.path.join(outdir_alignment, f'all_offsets.png'))
        plt.clf()

        all_offsets_in_uncertainty_units = np.array(all_offsets_in_uncertainty_units)
        plt.hist(all_offsets_in_uncertainty_units, bins=np.arange(-3, 3, .1))
        plt.figtext(0.15, 0.8, fR"$\mu={all_offsets_in_uncertainty_units[abs(all_offsets_in_uncertainty_units) < 50].mean():.2f}$" + '\n' + 
                               fR"$\sigma={all_offsets_in_uncertainty_units[abs(all_offsets_in_uncertainty_units) < 50].std():.2f}$")
        plt.xlabel('offset/uncertainty in fit centre')
        plt.ylabel('count')
        plt.savefig(os.path.join(outdir_alignment, f'all_offsets_uncertainties.png'))
        plt.clf()


def plot_reconstructed_absorbed_fraction():
    """Plot the individual reconstructed absorbed fractions, pre-averaging. A separate function to 
    plotting the averaged ones since there are many images, so it is slower."""
    with h5py.File(processed_data_h5) as processed_data:
        reconstructed_absorbed_fraction_ROI = processed_data['reconstructed_absorbed_fraction_ROI']
        absorbed_fraction_ROI = processed_data['absorbed_fraction'][:, ROI_y_start:ROI_y_stop, :]
         # plot the averaged data:
        outdir_A = os.path.join(PROCESS_DATA_OUTDIR, 'reconstructed_absorbed_fraction')
        if not os.path.exists(outdir_A):
            os.mkdir(outdir_A)
        for i in tqdm(range(n_shots), desc='  Saving images'):
            orig = absorbed_fraction_ROI[i]
            recon = reconstructed_absorbed_fraction_ROI[i]
            resid = recon - orig
            concatenated = np.concatenate((orig, recon, resid), axis=0)
            plt.imsave(os.path.join(outdir_A, f'{i:04d}.png'), concatenated, vmin=-0.5, vmax=0.5, cmap='seismic')


def compute_max_absorption_saturation_parameter():
    """determine where the y position of maximum absorption of the
    reconstructed absorbed_fractions is at each x position, and interpolate
    the mean saturation parameter to that point. This is done by simply
    finding the quadratic curve that maximises the sum of all (interpolated)
    absorbed fractions it passes through at each pixel and each
    realisation."""
    with h5py.File(processed_data_h5) as processed_data:
        A = processed_data['reconstructed_average_absorbed_fraction_ROI'][:]
        x = np.indices((A.shape[2],))[0]
        y = np.indices((A.shape[1],))[0]
        S = processed_data['average_saturation_parameter'][:, ROI_y_start:ROI_y_stop, ROI_x_start:ROI_x_stop]

        from scipy.interpolate import interp1d
        
        # average over all realisations:
        mean_A = A.mean(0)

        def neg_sum_absorbed_fraction(args):
            """Find the (interpolated) absorbed fraction passed through by a
            curve y = a * x**2 + mx+c, summed over x and all realisations. Return it
            multiplied by negative one for use with fmin"""
            a, m, c = args
            y0 = a*x**2 + m * x + c
            total = 0
            for x_index in x:
                interpolator = interp1d(y, mean_A[:, x_index], fill_value=0, bounds_error=False)
                sum_A_interp = interpolator(y0[x_index])
                total += sum_A_interp
            return -total

        print('Interpolating saturation parameter to y positions of max absorption')

        from scipy.optimize import minimize
        print('  Fitting curve to find max absorption positions...')
        res = minimize(neg_sum_absorbed_fraction, [2.37e-5, -0.01728, 38.8663])
        a, m, c = res.x
        y0 = a * x**2 + m*x + c
        plt.imshow(mean_A, vmin=0, vmax=0.1)
        plt.plot(x, y0, 'r-', linewidth=0.5)
        plt.savefig(os.path.join(PROCESS_DATA_OUTDIR, 'max_absorption_pos.png'), dpi=300)
        plt.clf()

        # Interpolate the saturation parameter at these points for each
        # realisation:
        S0 = np.zeros((n_realisations, S.shape[2]))
        for i in tqdm(range(n_realisations), desc='  interpolating saturation parameter over average shots'):
            for x_index in x:
                interpolator = interp1d(y, S[i, :, x_index])
                S_interp = interpolator(y0[x_index])
                S0[i, x_index] = S_interp

        h5_save(processed_data, 'max_absorption_average_saturation_parameter', S0)

        # And for each shot:
        S_individual = processed_data['saturation_parameter'][:, ROI_y_start:ROI_y_stop, ROI_x_start:ROI_x_stop]
        S0_individual = np.zeros((n_shots, S_individual.shape[2]))
        for i in tqdm(range(n_shots), desc='  interpolating saturation parameter over all shots'):
            for x_index in x:
                interpolator = interp1d(y, S_individual[i, :, x_index])
                S_interp = interpolator(y0[x_index])
                S0_individual[i, x_index] = S_interp

        h5_save(processed_data, 'max_absorption_saturation_parameter', S0_individual)

def compute_reconstructed_naive_average_OD():
    """based on the reconstructed absorbed fractions and saturation parameter,
    compute naive OD of each realisation using -alpha * log (1 - A) + S0 * A where
    A is the reconstructed average absorbed fraction and S is the saturation
    parameter at the max absorption y position"""

    outdir = os.path.join(PROCESS_DATA_OUTDIR, 'reconstructed_naive_average_OD')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
 
    with h5py.File(processed_data_h5) as processed_data:
        S0 = processed_data['max_absorption_average_saturation_parameter']
        A = processed_data['reconstructed_average_absorbed_fraction_ROI']
        reconstructed_naive_average_OD = np.zeros(A.shape)
        for i in tqdm(range(n_realisations), desc='  computing naive recon. OD'):
            A_i = A[i]
            S0_i = S0[i]
            OD_i = -alpha * np.log(1 - A_i) + S0_i * A_i 
            reconstructed_naive_average_OD[i] = OD_i
            plt.imsave(os.path.join(outdir, f'{i:04d}.png'), OD_i, vmin=-0.5, vmax=0.5, cmap='seismic')
        h5_save(processed_data, 'reconstructed_naive_average_OD', reconstructed_naive_average_OD)


def compute_naive_linear_density():
    """Compute the naive linear density as the y integral of the naive ODs."""
    with h5py.File(processed_data_h5) as processed_data:
        reconstructed_naive_average_OD = processed_data['reconstructed_naive_average_OD']

        print('computing naive linear density')

        outdir_linear_density = os.path.join(PROCESS_DATA_OUTDIR, 'naive_reconstructed_average_linear_density')
        if not os.path.exists(outdir_linear_density):
            os.mkdir(outdir_linear_density)

        naive_linear_density = np.zeros((n_realisations, image_shape[1]))
        for i in tqdm(range(n_realisations), desc='  computing naive lin. dens.'):
            linear_density_i = reconstructed_naive_average_OD[i].sum(0) * dy_pixel / sigma_0 
            naive_linear_density[i, :] = linear_density_i

            plt.plot(linear_density_i/1e6)
            plt.ylabel('linear density (per um)')
            plt.xlabel('x pixel')
            plt.grid(True)
            plt.axis([0, 648, -0.5, 16])
            plt.savefig(os.path.join(outdir_linear_density, f'{i:02d}.png'))
            plt.clf()

        h5_save(processed_data, 'naive_linear_density', naive_linear_density)


def compute_linear_density():
    """From the y-integrated absorbed fractions, invert the absorption model to
    find the linear density at each x position"""

    from scipy.optimize import fsolve, brentq
    
    sigma_y_min = np.sqrt(sigma_0 / pi)

    def column_density_from_absorbed_fraction(A, S):
        """Compute the column density n given an absorbed fraction A and
        saturation parameter S"""
        OD = -alpha * np.log(1-A) + S*A
        return OD/sigma_0

    def absorbed_fraction_from_column_density(n, S):
        """Inverse of above. Compute the absorbed fraction A given a column
        density n and saturation parameter S"""
        return brentq(lambda A: column_density_from_absorbed_fraction(A, S) - n, -200 , 1 - 1e-15)

    def sigma2_y_of_t(t, S):
        """Return the modelled mean squared y position of the scattering
        target assuming diffusion with scattering rate for the saturation
        parameter S"""
        R_scat = gamma_D2 / 2 * S / (1 + S)
        return (sigma_y_min)**2 + 1 / (18 * pi) * R_scat * v_recoil**2 * t**3

    def column_density_model(y, t, n_1d, S):
        """Return the column density as a function of y at time t given the
        linear density n_1D and our diffusion model with scattering rate for
        saturation parameter S."""
        sigma2_y = sigma2_y_of_t(t, S)
        return n_1d / np.sqrt(2*pi*sigma2_y) * np.exp(-y**2 / (2*sigma2_y))

    def absorbed_fraction_from_linear_density(y, t, n_1d, S):
        """Return the absorbed fraction at position y and time t for the given
        linear density and saturation parameter"""
        n = column_density_model(y, t, n_1d, S)
        A = absorbed_fraction_from_column_density(n, S)
        return A

    def A_meas_from_linear_density(n_1d, S):
        """Model for the integrated absorbed fraction given a particular
        linear density n_1d"""

        def integrand(y, t):
            return absorbed_fraction_from_linear_density(y, t, n_1d, S)

        from scipy.integrate import dblquad

        # Double integral of A from t=0 to tau and y=0 to 5 standard deviations
        # of the modelled Gaussian distribution:
        result, err = dblquad(integrand,
                              0, tau,
                              lambda t: 0,
                              lambda t: 5*np.sqrt(sigma2_y_of_t(t, S)))
        # Divide by tau to get the time-averaged absorbed fraction and
        # multiply by 2 to get the integral over all space instead of
        # just y > 0:
        A_meas = 2 / tau * result
        return A_meas

    def linear_density_from_a_meas(A_meas, S):
        """Inverse of above. Compute the linear density given a t and y
        integrated absorbed fraction A_meas"""
        try:
            return brentq(lambda n_1d: A_meas_from_linear_density(n_1d, S) - A_meas, -20/1e-6, 20/1e-6)
        except ValueError:
            # Density higher than 20 per micron, numerics break here and no finite
            # density may solve the equation. Return NaN. Likely a result of shot
            # noise giving the appearance of greater than unity fractional absorption.
            if A_meas_from_linear_density(20/1e-6, S) - A_meas < 0:
                return np.nan
            raise

    print('inverting absorption model to obtain linear densities')

    with h5py.File(processed_data_h5) as processed_data:

        # Compute the linear density of a subset of shots for comparison with the averages
        # (it takes too long to compute for all shots):
        with h5py.File(raw_data_h5, 'r') as raw_data:
            final_dipole = raw_data['final_dipole'][:]
            short_TOF = raw_data['short_TOF'][:]

        realisation_final_dipole = processed_data['realisation_final_dipole']
        realisation_short_TOF = processed_data['realisation_short_TOF']
        matching_realisations = ((realisation_final_dipole[:] == final_dipole[SHOT_OF_INTEREST])
                          & (realisation_short_TOF[:] == short_TOF[SHOT_OF_INTEREST]))
        realisation_of_interest = np.nonzero(matching_realisations)[0][0]
        matching_shots = ((final_dipole[:] == final_dipole[SHOT_OF_INTEREST])
                          & (short_TOF[:] == short_TOF[SHOT_OF_INTEREST]))
        shots_of_interest = np.nonzero(matching_shots)[0]

        # Random but reproducable subset of ten:
        np.random.seed(0)
        shots_of_interest = np.random.choice(shots_of_interest, 10, replace=False)

        h5_save(processed_data, 'realisation_of_interest', realisation_of_interest)
        h5_save(processed_data, 'shots_of_interest', shots_of_interest)

        S0_individual = processed_data['max_absorption_saturation_parameter']
        A_meas_individual = processed_data['integrated_reconstructed_absorbed_fraction']
        linear_density_shots_of_interest = np.zeros((len(shots_of_interest), A_meas_individual.shape[1]))
        for i in tqdm(range(len(shots_of_interest)), desc='  for shots of interest'):
            shot = shots_of_interest[i]
            for j in tqdm(range(A_meas_individual.shape[1]), desc=f'    over pixels ...'):
                linear_density_shots_of_interest[i, j] = linear_density_from_a_meas(A_meas_individual[shot, j], S0_individual[shot, j])
        h5_save(processed_data, 'linear_density_shots_of_interest', linear_density_shots_of_interest)

        # Compute linear density of averaged shots:
        S0 = processed_data['max_absorption_average_saturation_parameter']
        A_meas = processed_data['integrated_reconstructed_average_absorbed_fraction']
        u_A_meas = processed_data['u_integrated_reconstructed_average_absorbed_fraction']

        linear_density = np.zeros(A_meas.shape)
        u_linear_density = np.zeros(A_meas.shape)
        for i in tqdm(range(n_realisations), desc='  over averages'):
            for j in tqdm(range(A_meas.shape[1]), desc='    over pixels ...'):
                linear_density[i, j] = linear_density_from_a_meas(A_meas[i, j], S0[i, j])
                # Propagate the uncertainty by numerically differentiating the
                # model using a small step size:
                dA_meas = u_A_meas[i, j] / 100.0
                n_plus_dn = linear_density_from_a_meas(A_meas[i, j] + dA_meas, S0[i, j])
                u_linear_density[i, j] = u_A_meas[i, j] * np.abs(n_plus_dn - linear_density[i, j]) / dA_meas
        h5_save(processed_data, 'linear_density', linear_density)
        h5_save(processed_data, 'u_linear_density', u_linear_density)


def compute_naive_linear_density_uncertainty():
    """Make approximate uncertainties for the linear densities. The general
    strategy here is to compute the statistical uncertainty of a linear
    density computed as the integral of a naive OD, itself computed from the
    mean of slice-reconstructed absorbed fractions and the peak saturation
    parameter. This statistical uncertainty will be based only on the standard
    error of the mean of slice-reconstructed absorbed fractions, which
    approximates that the probe image has no uncertainty."""

    print('Making naive linear density uncertainty map')

    with h5py.File(raw_data_h5, 'r') as raw_data:
        final_dipole = raw_data['final_dipole'][:]
        short_TOF = raw_data['short_TOF'][:]

    with h5py.File(processed_data_h5) as processed_data:
        reconstructed_absorbed_fraction_ROI = processed_data['reconstructed_absorbed_fraction_ROI']
        A = processed_data['reconstructed_average_absorbed_fraction_ROI'][:]
        S = processed_data['max_absorption_average_saturation_parameter'][:]

        # Compute the standard deviation of the reconstructed pre-averaged absorbed
        # fractions, and divide by sqrt(N) to get the uncertainty in their mean:
        u_A = np.zeros(A.shape)
        for i, (tof_val, dipole_val) in tqdm(enumerate(realisations), total=n_realisations, desc='  computing standard deviations'):
            matching_shots = (final_dipole == dipole_val) & (short_TOF == tof_val)
            u_A[i] = reconstructed_absorbed_fraction_ROI[matching_shots, :, :].std(axis=0) / np.sqrt(matching_shots.sum())
            # 10px smoothing on u_A in x direction:
            u_A[i] = uniform_filter1d(u_A[i], size=10, axis=1)

        outdir_uA = os.path.join(PROCESS_DATA_OUTDIR, 'uncertainty_absorbed_fraction')
        if not os.path.exists(outdir_uA):
            os.mkdir(outdir_uA)

        for i in tqdm(range(n_realisations), desc='  Saving images'):
            concatenated = np.concatenate((A[i], u_A[i]), axis=0)
            plt.imsave(os.path.join(outdir_uA, f'{i:04d}.png'), concatenated, vmin=-0.5, vmax=0.5, cmap='seismic')

        # propagate u_A through the OD expression:
        u_OD = (alpha/(1-A) + S[:, np.newaxis, :]) * u_A

        # Sum in quadrature to get the uncertainty of the sums over y:
        u_colsum_OD = np.sqrt((u_OD**2).sum(axis=1))

        # multiply by dy to get the uncertainty in the integral:
        u_naive_linear_density = u_colsum_OD * dy_pixel / sigma_0 

        # Save it:
        h5_save(processed_data, 'u_naive_linear_density', u_naive_linear_density)


def plot_linear_density():
    """Plot the linear densities and compare to "naive" linear densities"""
    with h5py.File(processed_data_h5) as processed_data:
        naive_linear_density = processed_data['naive_linear_density']
        u_naive_linear_density = processed_data['u_naive_linear_density']
        linear_density = processed_data['linear_density']
        u_linear_density = processed_data['u_linear_density']

        outdir_linear_density = os.path.join(PROCESS_DATA_OUTDIR, 'linear_density')
        outdir_uncertainties = os.path.join(PROCESS_DATA_OUTDIR, 'u_linear_density')
        outdir_shots_of_interest = os.path.join(PROCESS_DATA_OUTDIR, 'shots_of_interest')
        if not os.path.exists(outdir_linear_density):
            os.mkdir(outdir_linear_density)
        if not os.path.exists(outdir_uncertainties):
            os.mkdir(outdir_uncertainties)
        if not os.path.exists(outdir_shots_of_interest):
            os.mkdir(outdir_shots_of_interest)

        for i in tqdm(range(n_realisations), desc='plotting linear density'):
            plt.errorbar(range(len(linear_density[i])),
                         naive_linear_density[i]/1e6,
                         yerr=u_naive_linear_density[i]/1e6,
                         label='naive linear density',
                         fmt='o',  markersize=0.0, capsize=1, lw=0.5)

            plt.errorbar(range(len(linear_density[i])),
                         linear_density[i]/1e6,
                         yerr=u_linear_density[i]/1e6,
                         label='modelled linear density',
                         fmt='o',  markersize=0.0, capsize=1, lw=0.5)

            plt.ylabel('linear density (per um)')
            plt.xlabel('x pixel')
            plt.legend()
            plt.grid(True)
            plt.axis([0, 648, -2, 16])
            plt.savefig(os.path.join(outdir_linear_density, f'{i:02d}.png'), dpi=300)
            plt.clf()

        # Plot the linear density of the shots-of-interest:
        linear_density_shots_of_interest = processed_data['linear_density_shots_of_interest'][:]
        realisation_of_interest = processed_data['realisation_of_interest'].value
        shots_of_interest = processed_data['shots_of_interest']

        for i in tqdm(range(len(linear_density_shots_of_interest)), desc='shots of interest'):
            plt.plot(range(len(linear_density_shots_of_interest[i])),
                         linear_density_shots_of_interest[i]/1e6, 'o', markersize=1,
                         label='modelled single linear density')

            plt.errorbar(range(len(linear_density[realisation_of_interest])),
                         linear_density[realisation_of_interest]/1e6,
                         yerr=u_linear_density[realisation_of_interest]/1e6,
                         label='modelled average linear density',
                         fmt='o',  markersize=0.0, capsize=1, lw=0.5)

            # Compute imbalance metric: fraction of single-show points above the
            # average points minus fraction below:
            valid_points = ~np.isnan(linear_density_shots_of_interest[i]) # exclude NaNs
            av = linear_density[realisation_of_interest][valid_points]
            raw = linear_density_shots_of_interest[i][valid_points]
            N = valid_points.sum()
            imbalance = ((raw > av).sum() - (raw < av).sum())/N

            plt.figtext(0.13, 0.55, f'shot {shots_of_interest[i]}')
            plt.figtext(0.13, 0.5, r'$(N_> - N_<)/N = '+f'{imbalance:.03f}$')
            plt.ylabel('linear density (per um)')
            plt.xlabel('x pixel')
            plt.legend()
            plt.grid(True)
            plt.axis([0, 648, -2, 16])
            plt.savefig(os.path.join(outdir_shots_of_interest, f'shot_{shots_of_interest[i]}.png'), dpi=300)
            plt.clf()

        # Plot just the uncertainties:
        for i in tqdm(range(n_realisations), desc='plotting uncertainty in linear density'):
            plt.plot(range(len(linear_density[i])),
                         u_naive_linear_density[i]/1e6,
                         label='naive linear density', lw=0.5)

            plt.plot(range(len(linear_density[i])),
                         u_linear_density[i]/1e6,
                         label='modelled linear density', lw=0.5)

            plt.ylabel('uncertainty in linear density (per um)')
            plt.xlabel('x pixel')
            plt.legend()
            plt.grid(True)
            plt.axis([0, 648, 0, 2])
            plt.savefig(os.path.join(outdir_uncertainties, f'{i:02d}.png'))
            plt.clf()

        plt.plot([-5,12], [-5,12], 'k--')
        for i in tqdm(range(n_realisations), desc='comparing linear density model'):
            plt.plot(naive_linear_density[i]/1e6, linear_density[i]/1e6, 'ro', markersize=0.5, alpha=0.5)
        
        plt.xlabel('naive linear_density (per um)')
        plt.ylabel('modelled linear density (per um)')
        plt.grid(True)
        plt.axis([-5, 12, -5, 12])
        plt.savefig(os.path.join(PROCESS_DATA_OUTDIR, 'linear_density_comparison.png'), dpi=300)

if __name__ == '__main__':
    # save_pca_images()
    compute_mean_raw_images()
    compute_dark_systematic_offset()
    reconstruct_probe_frames()
    # plot_reconstructed_probe_frames()
    reconstruct_dark_frames()
    # plot_reconstructed_dark_frames()
    compute_OD_and_absorption_and_saturation_parameter()
    # plot_OD_and_absorption_and_saturation_parameter()
    compute_averages()
    reconstruct_absorbed_fraction()
    # check_absorbed_fractions_alignment()
    # plot_reconstructed_absorbed_fraction()
    compute_max_absorption_saturation_parameter()
    compute_reconstructed_naive_average_OD()
    compute_naive_linear_density()
    compute_naive_linear_density_uncertainty()
    compute_linear_density()
    plot_linear_density()

    
