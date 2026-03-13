#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VROOMM v04 — Image Quality Analysis Pipeline
==============================================

This script analyses the spectral image quality of the VROOMM spectrograph
(v04 optical design) for two fiber configurations:

  - **Rectangular fiber** (high-resolution mode)
  - **Octagonal fiber** (low-resolution mode)

It reads Zemax Image Analysis ASCII exports (80x80 pixel PSF maps), extracts
the Line Spread Function (LSF) in the dispersion direction, and computes the
resolving power R = c / FWHM using two complementary methods:

  1. **Gaussian equivalent FWHM** — Following Bouchy et al. (2001, A&A, 374,
     733), we match a Gaussian that has the same integral and the same
     sum-of-squared-gradient (proportional to RV information content Q) as
     the observed LSF. This is the physically relevant metric for radial-
     velocity precision.

  2. **Direct FWHM** — Simple half-maximum interpolation on the LSF profile.
     A robust geometric measurement, less sensitive to wing structure.

For the rectangular fiber, the PSF is elongated along the slit direction and
tilted on the detector. An optimal rotation angle is found before extracting
the LSF. For the octagonal fiber (circular symmetry), no rotation is needed.

The local spectral dispersion (km/s per detector pixel) is computed from the
Zemax _XY.txt companion file, which contains order number, wavelength, and
x/y position on the detector for 5 field positions across each order.

Outputs:
  - psf_rotation_example.png : side-by-side native vs rotated PSF
  - resolving_power.png      : R vs wavelength (3-panel: rect, oct, dispersion)
  - rotation_angle.png       : optimal rotation angle vs wavelength (rect)

Dependencies: numpy, matplotlib, scipy
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np                              # numerical arrays
import matplotlib.pyplot as plt                 # plotting
from scipy.ndimage import rotate                # image rotation (cubic interp)
from scipy.optimize import minimize_scalar, minimize  # 1D and N-D optimizers


# =============================================================================
# Core I/O function
# =============================================================================
def read_zemax_map(filename: str) -> np.ndarray:
    """
    Read a Zemax image analysis ASCII file and return the 80x80 pixel map
    as a numpy array.

    Note: the header contains accented characters (e.g. "Université"), so
    we must use latin-1 encoding rather than UTF-8.

    Parameters
    ----------
    filename : str
        Path to the Zemax ASCII file.

    Returns
    -------
    np.ndarray
        2D array of shape (80, 80) with flux values.
    """
    # Skip 17 header lines, use latin-1 for accented characters in the header
    data = np.loadtxt(filename, skiprows=17, encoding='latin-1')
    return data


# =============================================================================
# PSF rotation utilities
# =============================================================================
def rotate_psf(psf: np.ndarray, angle: float, reshape: bool = False) -> np.ndarray:
    """
    Rotate a PSF image by an arbitrary angle.

    Uses cubic (order=3) spline interpolation for smooth results. Pixels
    outside the original image are filled with zeros.

    Parameters
    ----------
    psf : np.ndarray
        2D PSF array.
    angle : float
        Rotation angle in degrees (counter-clockwise).
    reshape : bool, optional
        If True, the output array is resized to contain the full rotated image.
        If False (default), the output keeps the same shape as the input.

    Returns
    -------
    np.ndarray
        Rotated PSF array.
    """
    return rotate(psf, angle, reshape=reshape, order=3, mode='constant', cval=0.0)


def find_optimal_rotation(psf: np.ndarray) -> float:
    """
    Find the rotation angle that maximizes the sharpness of the column-summed
    profile (i.e., the LSF in the dispersion direction).

    The objective is to maximize:
        sum( gradient( sum(PSF_rotated, axis=0) )^2 )

    This quantity (sum of squared gradient) is maximized when the PSF is
    aligned so that the dispersion direction is exactly along the columns.
    When properly aligned, the column-summed LSF has the steepest possible
    edges, maximizing the gradient energy. This is closely related to the
    Bouchy et al. (2001) quality factor Q.

    We negate the objective and use scipy's bounded scalar minimizer over
    the range [-90°, +90°].

    Parameters
    ----------
    psf : np.ndarray
        2D PSF array.

    Returns
    -------
    float
        Optimal rotation angle in degrees.
    """
    def neg_objective(angle):
        # Rotate the PSF by the candidate angle
        rotated = rotate_psf(psf, angle)
        # Sum along rows → column profile (the LSF in dispersion direction)
        profile = np.sum(rotated, axis=0)
        # Return negative sum of squared gradient (we want to MAXIMIZE)
        return -np.sum(np.gradient(profile)**2)

    # Bounded minimization over [-90°, 90°] — covers all orientations
    result = minimize_scalar(neg_objective, bounds=(-90, 90), method='bounded')
    return result.x


# =============================================================================
# Demo: read one PSF, find rotation, display side-by-side
# =============================================================================

# Read a sample PSF from the high-resolution (rectangular) fiber
# R1554.txt = order 155, field position 4
psf1 = read_zemax_map('VROOMM_v04_rectangular_fiber/R1554.txt')

# Find the angle that best aligns the slit with the pixel columns
best_angle = find_optimal_rotation(psf1)
print(f'Optimal rotation angle: {best_angle:.2f} degrees')

# Apply the rotation
psf1_rot = rotate_psf(psf1, best_angle)


# --- Plot: native vs rotated PSF (saved as PNG for the README) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(psf1)
axes[0].set_title('Native PSF')
axes[0].set_xlabel('Pixel X')
axes[0].set_ylabel('Pixel Y')

im = axes[1].imshow(psf1_rot)
axes[1].set_title(f'Optimal rotation ({best_angle:.1f}°)')
axes[1].set_xlabel('Pixel X')
axes[1].set_ylabel('Pixel Y')

#fig.colorbar(im, ax=axes, shrink=0.8)
plt.tight_layout()
plt.savefig('psf_rotation_example.png', dpi=150, bbox_inches='tight')
plt.show()

# Extract the LSF by summing along the spatial (slit) direction
# After rotation, axis=0 is the spatial direction, axis=1 is dispersion
# Summing over axis=0 collapses the spatial direction → 1D LSF
lsf1 = np.sum(psf1_rot, axis=0)


# =============================================================================
# Gaussian matching — Bouchy et al. (2001) approach
# =============================================================================
def gaussian(x, amplitude, center, sigma):
    """Simple 1D Gaussian function: A * exp(-0.5 * ((x-c)/σ)²)."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def match_gaussian_to_lsf(lsf: np.ndarray) -> dict:
    """
    Find the Gaussian equivalent of an observed LSF profile.

    Following Bouchy, Pepe & Queloz (2001, A&A, 374, 733), we find a
    Gaussian that simultaneously matches:

      1. The integral (total flux) of the LSF
      2. The sum of squared gradients — proportional to the radial-velocity
         information content Q

    This is NOT a least-squares fit to the LSF shape. Instead, it finds the
    Gaussian whose RV information content equals that of the actual (possibly
    non-Gaussian) LSF. The FWHM of this equivalent Gaussian is the relevant
    width for RV precision estimates.

    The optimization uses Nelder-Mead with very tight tolerances to ensure
    both constraints are satisfied to machine precision.

    Parameters
    ----------
    lsf : np.ndarray
        1D LSF profile.

    Returns
    -------
    dict
        Dictionary with keys:
        - amplitude : peak amplitude of the matched Gaussian
        - center    : center position (pixels)
        - sigma     : Gaussian sigma (pixels)
        - fwhm      : FWHM = 2√(2 ln 2) × σ (pixels)
        - profile   : the matched Gaussian evaluated on the same pixel grid
    """
    x = np.arange(len(lsf))

    # Target quantities from the observed LSF
    target_integral = np.sum(lsf)                       # total flux
    target_grad2 = np.sum(np.gradient(lsf) ** 2)        # RV info content ∝ Q

    def objective(params):
        amplitude, center, sigma = params
        g = gaussian(x, amplitude, center, sigma)
        integral = np.sum(g)
        grad2 = np.sum(np.gradient(g) ** 2)

        # Cost = squared relative error on integral + squared relative error
        # on gradient² — both must match simultaneously
        err_integral = ((integral - target_integral) / target_integral) ** 2
        err_grad2 = ((grad2 - target_grad2) / target_grad2) ** 2
        return err_integral + err_grad2

    # Initial guesses: peak of LSF, position of peak, and sigma from
    # the relation integral = amplitude × σ × √(2π)
    amp0 = np.max(lsf)
    cen0 = np.argmax(lsf)
    sig0 = target_integral / (amp0 * np.sqrt(2 * np.pi))

    # Nelder-Mead with very tight tolerances for high-precision matching
    result = minimize(objective, [amp0, cen0, sig0],
                      method='Nelder-Mead',
                      options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 100000})

    amp_fit, cen_fit, sig_fit = result.x
    sig_fit = abs(sig_fit)  # sigma must be positive

    # Convert sigma to FWHM: FWHM = 2 × √(2 ln 2) × σ ≈ 2.3548 × σ
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sig_fit

    # Evaluate the matched Gaussian on the pixel grid
    g_fit = gaussian(x, amp_fit, cen_fit, sig_fit)

    return {
        'amplitude': amp_fit,
        'center': cen_fit,
        'sigma': sig_fit,
        'fwhm': fwhm,
        'profile': g_fit,
    }


# --- Demo: match Gaussian to the sample LSF ---
gauss_result = match_gaussian_to_lsf(lsf1)
print(f"FWHM of matched Gaussian: {gauss_result['fwhm']:.4f} pixels")

# Plot LSF vs matched Gaussian
plt.figure()
plt.plot(lsf1, label='LSF')
plt.plot(gauss_result['profile'], '--', label=f"Gaussian (FWHM={gauss_result['fwhm']:.2f} px)")
plt.legend()
plt.xlabel('Pixel')
plt.ylabel('Flux')
plt.title('LSF vs matched Gaussian')
plt.show()


# =============================================================================
# Direct FWHM measurement
# =============================================================================
def measure_fwhm(lsf: np.ndarray) -> float:
    """
    Measure the FWHM of a 1D profile by half-maximum interpolation.

    Finds the peak value, then locates where the profile first rises above
    and last falls below half the peak. Linear interpolation is used at
    both edges for sub-pixel precision.

    This is a simple geometric measurement — it does not account for the
    shape of the wings or the RV information content. For RV science, use
    the Gaussian equivalent FWHM from match_gaussian_to_lsf() instead.

    Parameters
    ----------
    lsf : np.ndarray
        1D profile (e.g., column-summed PSF).

    Returns
    -------
    float
        FWHM in pixels. Returns NaN if the profile has fewer than 2 pixels
        above half-maximum.
    """
    peak = np.max(lsf)
    half_max = peak / 2.0
    # Find all pixels above half-maximum
    above = lsf >= half_max
    indices = np.where(above)[0]
    if len(indices) < 2:
        return np.nan

    # --- Left edge: interpolate between the last pixel below and first above ---
    i_left = indices[0]
    if i_left > 0:
        # Linear interpolation: find fractional pixel where profile = half_max
        left = i_left - 1 + (half_max - lsf[i_left - 1]) / (lsf[i_left] - lsf[i_left - 1])
    else:
        left = 0.0  # profile starts above half-max (edge case)

    # --- Right edge: interpolate between the last pixel above and first below ---
    i_right = indices[-1]
    if i_right < len(lsf) - 1:
        right = i_right + (half_max - lsf[i_right]) / (lsf[i_right + 1] - lsf[i_right])
    else:
        right = float(len(lsf) - 1)  # profile ends above half-max (edge case)

    # FWHM = distance between right and left half-maximum crossings
    return right - left


# =============================================================================
# Batch processing: loop over all PSFs for a given fiber type
# =============================================================================
def compute_fwhm_for_fiber(base_dir: str, optimize_rotation: bool = True):
    """
    Process all 25 PSFs (5 orders × 5 field positions) for one fiber type
    and compute the resolving power at each position.

    Steps for each PSF:
      1. Read the 80×80 Zemax ASCII map
      2. (Optional) Find and apply the optimal rotation angle
      3. Sum along the spatial direction to extract the 1D LSF
      4. Match a Gaussian equivalent (Bouchy 2001 method) → R_gauss
      5. Measure the direct FWHM → R_direct
      6. Convert from simulation pixels to km/s using local dispersion

    The local dispersion is derived from the _XY.txt companion file:
      - Column 0: diffraction order number (e.g., 67, 89, 111, 133, 155)
      - Column 1: wavelength in µm
      - Column 2: x position on detector in mm
      - Column 3: y position on detector in mm

    Within each order, dλ/dx is computed via np.gradient, then converted to
    velocity dispersion: disp = c × |dλ/dx| / λ (km/s per real pixel),
    where real pixels are 12 µm.

    The Zemax simulation uses 4× oversampled pixels, so:
      FWHM_real_pixels = FWHM_sim_pixels / 4
      FWHM_kms = FWHM_real_pixels × dispersion
      R = c / FWHM_kms

    Parameters
    ----------
    base_dir : str
        Directory containing the R*.txt PSF files and the _XY.txt file.
    optimize_rotation : bool
        If True, find the optimal rotation angle for each PSF (use for
        rectangular fiber). If False, skip rotation (use for octagonal fiber
        where the PSF is circularly symmetric).

    Returns
    -------
    orders : ndarray
        Diffraction order number for each of the 25 positions.
    wavelengths : ndarray
        Wavelength in µm for each position.
    R_gauss : ndarray
        Resolving power from the Gaussian equivalent FWHM (Bouchy 2001).
    R_direct : ndarray
        Resolving power from the direct half-maximum FWHM.
    dispersions : ndarray
        Local dispersion in km/s per real pixel at each position.
    angles : ndarray
        Optimal rotation angle in degrees for each PSF (0 if not optimized).
    """
    # --- Locate and read the _XY.txt companion file ---
    # This file maps each PSF to its order, wavelength, and detector position
    xy_file = [f for f in [f'{base_dir}/{f}' for f in __import__('os').listdir(base_dir)] if '_XY.txt' in f][0]
    xy_data = np.loadtxt(xy_file, skiprows=1)  # skip 1 header line

    orders = xy_data[:, 0].astype(int)   # diffraction order (67, 89, 111, 133, 155)
    wavelengths = xy_data[:, 1]           # wavelength in µm
    x_mm = xy_data[:, 2]                 # x position on detector in mm

    # --- Compute local dispersion for each field position ---
    # Speed of light in km/s
    c_kms = 299792.458

    # Convert detector positions from mm to real pixel units
    # The VROOMM detector has 12 µm pixels = 0.012 mm
    pixel_size_mm = 0.012
    x_pix = x_mm / pixel_size_mm

    # Dispersion array: one value per field position (km/s per real pixel)
    dispersions = np.zeros(len(wavelengths))

    # Process each order independently (5 field positions per order)
    unique_orders = np.unique(orders)
    for order in unique_orders:
        mask = orders == order
        idx = np.where(mask)[0]

        wl_order = wavelengths[idx]   # wavelengths within this order (µm)
        xp_order = x_pix[idx]         # x positions within this order (pixels)

        # dλ/dx using centered finite differences (np.gradient handles edges)
        # Units: µm per real pixel
        dlambda_dx = np.gradient(wl_order, xp_order)

        # Convert to velocity dispersion: v = c × dλ/λ
        # Units: km/s per real pixel
        for j, ii in enumerate(idx):
            dispersions[ii] = c_kms * abs(dlambda_dx[j]) / wl_order[j]

    # --- Build the list of PSF filenames ---
    # Naming convention: R{order}{1-5}.txt
    # e.g., order 155, position 3 → R1553.txt
    filenames = []
    for i, order in enumerate(orders):
        idx_in_order = (i % 5) + 1  # position index 1..5 within the order
        filenames.append(f'{base_dir}/R{order}{idx_in_order}.txt')

    # --- Main loop: process each PSF ---
    fwhm_gauss_values = []
    fwhm_direct_values = []
    angles = []
    for i, fname in enumerate(filenames):
        print(f'Processing {fname} (λ={wavelengths[i]:.5f} µm, disp={dispersions[i]:.3f} km/s/pix)...')

        # Step 1: Read the 80×80 PSF map
        psf = read_zemax_map(fname)

        # Step 2: Find optimal rotation (rectangular fiber) or skip (octagonal)
        if optimize_rotation:
            angle = find_optimal_rotation(psf)
        else:
            angle = 0.0
        angles.append(angle)

        # Step 3: Apply rotation and extract the 1D LSF
        psf_rot = rotate_psf(psf, angle)
        lsf = np.sum(psf_rot, axis=0)  # sum along spatial → dispersion profile

        # Step 4: Match Gaussian equivalent (Bouchy 2001)
        result = match_gaussian_to_lsf(lsf)

        # Step 5: Convert FWHM from simulation pixels to km/s
        # The Zemax simulation uses 4× oversampled pixels relative to the
        # real 12 µm detector pixels, so divide by 4 to get real pixels,
        # then multiply by local dispersion to get km/s
        disp = dispersions[i]  # km/s per real pixel

        # Gaussian equivalent FWHM → resolving power
        fwhm_gauss_kms = (result['fwhm'] / 4.0) * disp
        R_gauss = c_kms / fwhm_gauss_kms
        fwhm_gauss_values.append(R_gauss)

        # Direct FWHM → resolving power
        fwhm_direct_kms = (measure_fwhm(lsf) / 4.0) * disp
        R_direct = c_kms / fwhm_direct_kms
        fwhm_direct_values.append(R_direct)
        print(f'  angle={angle:.1f}°, R_gauss={R_gauss:.0f}, R_direct={R_direct:.0f}')

    return orders, wavelengths, np.array(fwhm_gauss_values), np.array(fwhm_direct_values), dispersions, np.array(angles)


# =============================================================================
# Run the analysis for both fiber types
# =============================================================================

# --- High-resolution mode: rectangular fiber ---
# The rectangular slit produces an elongated PSF that is tilted on the
# detector. We optimise the rotation angle to align it with the pixel grid
# before extracting the LSF.
orders_rect, wl_rect, R_gauss_rect, R_direct_rect, disp_rect, angles_rect = compute_fwhm_for_fiber('VROOMM_v04_rectangular_fiber', optimize_rotation=True)

# --- Low-resolution mode: octagonal fiber ---
# The octagonal fiber produces a nearly circular PSF, so no rotation
# optimisation is needed (angle fixed to 0°).
orders_oct, wl_oct, R_gauss_oct, R_direct_oct, disp_oct, angles_oct = compute_fwhm_for_fiber('VROOMM_v04_octogonal_fiber', optimize_rotation=False)


# =============================================================================
# Plot 1: Resolving power + dispersion (3-panel figure)
# =============================================================================

# Custom blue-to-red colormap — vivid at all wavelengths (no grey midpoint!)
from matplotlib.colors import LinearSegmentedColormap
cmap_br = LinearSegmentedColormap.from_list('blue_red', ['#0000FF', '#FF0000'])

# Normalise wavelengths across both fibers for consistent coloring
all_wl = np.concatenate([wl_rect, wl_oct])
wl_min, wl_max = all_wl.min(), all_wl.max()
norm = plt.Normalize(vmin=wl_min, vmax=wl_max)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

# --- Panel 1: Rectangular fiber (high-resolution mode) ---
for i in range(len(wl_rect)):
    c = cmap_br(norm(wl_rect[i]))
    # Filled circles = Gaussian equivalent (Bouchy 2001)
    ax1.plot(wl_rect[i], R_gauss_rect[i], 'o', color=c, markersize=8)
    # Open diamonds = direct FWHM
    ax1.plot(wl_rect[i], R_direct_rect[i], 'D', color=c, markersize=6, markerfacecolor='none', markeredgewidth=1.5)

# Dummy entries for the legend (black markers, no wavelength color)
ax1.plot([], [], 'ok', markersize=8, label='Gaussian equivalent FWHM')
ax1.plot([], [], 'Dk', markersize=6, markerfacecolor='none', markeredgewidth=1.5, label='Direct FWHM')
ax1.set_ylabel('Resolving power R')
ax1.set_title('Rectangular fiber (high-resolution mode)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Octagonal fiber (low-resolution mode) ---
for i in range(len(wl_oct)):
    c = cmap_br(norm(wl_oct[i]))
    # Filled squares = Gaussian equivalent
    ax2.plot(wl_oct[i], R_gauss_oct[i], 's', color=c, markersize=8)
    # Open triangles = direct FWHM
    ax2.plot(wl_oct[i], R_direct_oct[i], '^', color=c, markersize=6, markerfacecolor='none', markeredgewidth=1.5)

ax2.plot([], [], 'sk', markersize=8, label='Gaussian equivalent FWHM')
ax2.plot([], [], '^k', markersize=6, markerfacecolor='none', markeredgewidth=1.5, label='Direct FWHM')
ax2.set_ylabel('Resolving power R')
ax2.set_title('Octagonal fiber (low-resolution mode)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Local dispersion (both fibers overlaid) ---
for i in range(len(wl_rect)):
    c = cmap_br(norm(wl_rect[i]))
    ax3.plot(wl_rect[i], disp_rect[i], 'o', color=c, markersize=8)
for i in range(len(wl_oct)):
    c = cmap_br(norm(wl_oct[i]))
    ax3.plot(wl_oct[i], disp_oct[i], 's', color=c, markersize=6, markerfacecolor='none', markeredgewidth=1.5)

ax3.plot([], [], 'ok', markersize=8, label='Rectangular')
ax3.plot([], [], 'sk', markersize=6, markerfacecolor='none', markeredgewidth=1.5, label='Octogonal')
ax3.set_xlabel('Wavelength (µm)')
ax3.set_ylabel('Dispersion (km/s/pixel)')
ax3.set_title('Local dispersion')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

fig.suptitle('VROOMM v04 — Image quality analysis', fontsize=13)
plt.tight_layout()
plt.savefig('resolving_power.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# Plot 2: Rotation angle vs wavelength (rectangular fiber only)
# =============================================================================
# The rectangular slit image is tilted on the detector due to the spectrograph
# anamorphism and grating geometry. This tilt varies with wavelength and order.
# Tracking this angle is important for:
#   - Validating the optical model
#   - Understanding cross-dispersion contamination
#   - Ensuring the LSF extraction is properly aligned

fig_rot, ax_rot = plt.subplots(figsize=(10, 4))
for i in range(len(wl_rect)):
    c = cmap_br(norm(wl_rect[i]))
    ax_rot.plot(wl_rect[i], angles_rect[i], 'o', color=c, markersize=8)

ax_rot.set_xlabel('Wavelength (µm)')
ax_rot.set_ylabel('Optimal rotation angle (°)')
ax_rot.set_title('PSF rotation angle — Rectangular fiber (high-resolution mode)')
ax_rot.grid(True, alpha=0.3)
ax_rot.axhline(0, color='grey', linewidth=0.5, linestyle='--')
plt.tight_layout()
plt.savefig('rotation_angle.png', dpi=150, bbox_inches='tight')
plt.show()