#!/usr/bin/env python3
"""
lsf_nyquist.py
==============

Analyse the spectral Line Spread Function (LSF) of a Zemax-simulated
spectrograph and quantify *aliased power*: the fraction of LSF energy at
spatial frequencies above the detector Nyquist limit.

Key idea
--------
A pixel detector samples the focal-plane signal at 1 sample per real pixel.
The Nyquist–Shannon theorem says that any spatial frequency above

    f_N = 0.5 cycles / real_pixel

cannot be faithfully represented; its power gets *aliased* (folded back into
lower frequencies, corrupting the measured line profile).

Three reference cases:
  • sinc LSF         — band-limited by construction → 0 % aliased power
  • Gaussian LSF     — infinite bandwidth → always some aliased power
  • Real (Zemax) LSF — sits somewhere between the two

The Zemax simulation is 4× oversampled (1 sim_pixel = 0.25 real_pixel), so
the accessible frequency range is 0–2 cycles/real_pixel and the detector
Nyquist sits at f = 0.5 cycles/real_pixel = 1/4 of the simulation bandwidth.

Outputs
-------
  fig_01_psf_rotation.png  — raw PSF → rotated PSF → extracted 1-D LSF
  fig_02_lsf_profiles.png  — LSF vs matched Gaussian in real space
  fig_03_power_spectra.png — power spectrum + cumulative aliased-power curve
  fig_04_summary.png       — aliased fraction for all 25 LSFs vs wavelength
  fig_05_aliasing_vs_fwhm.png — aliased power vs LSF FWHM (real pixels)

Usage
-----
  python lsf_nyquist.py

All user-adjustable settings (data paths, pixel sizes, FFT length, …) are
read from  config.yaml  in the same directory.  Edit that file — do not
touch the Python code — to run on your own data.

Dependencies: numpy, matplotlib, scipy, pyyaml
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as ndimage_rotate
from scipy.optimize import minimize_scalar

# =============================================================================
# Load configuration
# =============================================================================
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
if not os.path.isfile(_CONFIG_FILE):
    sys.exit(f'ERROR: config.yaml not found at {_CONFIG_FILE}\n'
             'Make sure config.yaml is in the same directory as lsf_nyquist.py.')

with open(_CONFIG_FILE, 'r') as _f:
    _cfg = yaml.safe_load(_f)

# --- Data paths ---
EXAMPLE_FILE = _cfg['example_file']
DATA_DIR     = _cfg['data_dir']

# --- Geometry ---
DETECTOR_PIX_UM = float(_cfg['detector_pixel_um'])
SIM_PIX_UM      = float(_cfg['sim_pixel_um'])
OVERSAMPLE      = DETECTOR_PIX_UM / SIM_PIX_UM   # e.g. 12 / 3 = 4
HEADER_LINES    = int(_cfg['zemax_header_lines'])

if abs(OVERSAMPLE - round(OVERSAMPLE)) > 0.01:
    print(f'WARNING: oversampling factor is not a whole number '
          f'({OVERSAMPLE:.4f}).  Check detector_pixel_um / sim_pixel_um.')
OVERSAMPLE_INT = int(round(OVERSAMPLE))   # used where an integer is needed

# --- FFT ---
N_FFT = int(_cfg['n_fft'])

# --- Output ---
OUTPUT_DIR = _cfg['output_dir']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Derived constant (always 0.5 by Nyquist theorem) ---
F_NYQUIST = 0.5   # cycles / real pixel

print('Configuration loaded from config.yaml:')
print(f'  example_file      : {EXAMPLE_FILE}')
print(f'  data_dir          : {DATA_DIR}')
print(f'  detector_pixel_um : {DETECTOR_PIX_UM} µm')
print(f'  sim_pixel_um      : {SIM_PIX_UM} µm')
print(f'  oversample        : {OVERSAMPLE:.2f}×  ({OVERSAMPLE_INT}× integer)')
print(f'  n_fft             : {N_FFT}')
print(f'  output_dir        : {os.path.abspath(OUTPUT_DIR)}')
print()

# =============================================================================
# Colours (consistent across all figures)
# =============================================================================
C_LSF   = '#1f77b4'   # blue  — observed LSF
C_GAUSS = '#d62728'   # red   — matched Gaussian
C_NYQ   = '#2ca02c'   # green — Nyquist line


# =============================================================================
# Analytic Gaussian aliased fraction
# =============================================================================
from scipy.special import erfc as _erfc

def gaussian_aliased_fraction_analytic(fwhm_real_px: float,
                                       f_cut: float = F_NYQUIST) -> float:
    """
    Exact fraction of power above f_cut for a Gaussian LSF.

    For a unit-area Gaussian with sigma σ (real pixels), the one-sided
    power spectrum is |G(f)|² = exp(-4π²σ²f²), and the fraction of total
    power at |f| > f_cut is exactly erfc(2πσ f_cut).

    Parameters
    ----------
    fwhm_real_px : FWHM in real detector pixels
    f_cut        : cutoff frequency in cycles / real_pixel
    """
    sigma_real = fwhm_real_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return float(_erfc(2.0 * np.pi * sigma_real * f_cut))

# =============================================================================
# I/O
# =============================================================================
def read_zemax_map(filename: str) -> np.ndarray:
    """
    Read a Zemax Image Analysis ASCII export and return the flux map.
    Header contains accented characters, so latin-1 encoding is required.
    The number of header lines to skip is read from config.yaml.
    """
    return np.loadtxt(filename, skiprows=HEADER_LINES, encoding='latin-1')


# =============================================================================
# PSF rotation
# =============================================================================
def rotate_psf(psf: np.ndarray, angle: float) -> np.ndarray:
    """Rotate PSF by *angle* degrees (CCW), cubic interpolation, zero-fill."""
    return ndimage_rotate(psf, angle, reshape=False, order=3,
                          mode='constant', cval=0.0)


def find_optimal_rotation(psf: np.ndarray) -> float:
    """
    Find the angle that maximises the sharpness of the column-summed LSF.
    Sharpness = sum of squared gradients of the 1-D profile, which is
    maximised when the dispersion direction is perfectly aligned with the
    pixel columns.
    """
    def neg_sharpness(angle: float) -> float:
        profile = np.sum(rotate_psf(psf, angle), axis=0)
        return -np.sum(np.gradient(profile) ** 2)

    result = minimize_scalar(neg_sharpness, bounds=(-90, 90), method='bounded')
    return float(result.x)


def extract_lsf(psf: np.ndarray, angle: float | None = None):
    """
    Optimally rotate the PSF and collapse along the spatial (slit) axis.

    Returns
    -------
    lsf       : 1-D flux profile (dispersion direction)
    angle     : rotation angle used (degrees)
    psf_rot   : the rotated 2-D PSF
    """
    if angle is None:
        angle = find_optimal_rotation(psf)
    psf_rot = rotate_psf(psf, angle)
    lsf = np.sum(psf_rot, axis=0)
    return lsf, angle, psf_rot


# =============================================================================
# LSF helpers
# =============================================================================
def measure_fwhm(lsf: np.ndarray) -> float:
    """
    Half-maximum FWHM in pixels (sub-pixel precision via linear interpolation).
    """
    peak = np.max(lsf)
    half = peak / 2.0
    above = np.where(lsf >= half)[0]
    if len(above) < 2:
        return np.nan
    i_l = above[0]
    left = (i_l - 1 + (half - lsf[i_l - 1]) / (lsf[i_l] - lsf[i_l - 1])
            if i_l > 0 else 0.0)
    i_r = above[-1]
    right = (i_r + (half - lsf[i_r]) / (lsf[i_r + 1] - lsf[i_r])
             if i_r < len(lsf) - 1 else float(len(lsf) - 1))
    return right - left


def make_gaussian_lsf(n: int, fwhm_pix: float, center: int) -> np.ndarray:
    """
    Gaussian with the given FWHM (sim pixels), centred at *center*, unit sum.
    """
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    x = np.arange(n) - center
    g = np.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()


# =============================================================================
# Power spectrum and aliasing
# =============================================================================
def power_spectrum(lsf: np.ndarray, n_fft: int | None = None):
    """
    Compute the one-sided, Parseval-correct power spectral density of the LSF.

    Parameters
    ----------
    lsf   : 1-D array (any normalisation; internally forced to unit sum)
    n_fft : FFT length (zero-pads if > len(lsf)); defaults to len(lsf)

    Returns
    -------
    freq  : spatial frequency in cycles / real_pixel
    psd   : power at each bin, normalised so that sum(psd) == 1.
            Bin k contributes weight 2 (both ±freq); DC and sim-Nyquist
            weight 1 (correct one-sided Parseval form).
    """
    if n_fft is None:
        n_fft = len(lsf)
    lsf_norm = lsf / lsf.sum()
    fft = np.fft.rfft(lsf_norm, n=n_fft)
    raw = np.abs(fft) ** 2

    # Parseval weights: DC and sim-Nyquist count once, all others twice
    w = np.full(len(raw), 2.0)
    w[0] = 1.0
    w[-1] = 1.0
    psd = raw * w
    psd /= psd.sum()   # normalise to unit total power

    freq_sim  = np.fft.rfftfreq(n_fft, d=1.0)   # cycles / sim_pixel
    freq_real = freq_sim * OVERSAMPLE             # cycles / real_pixel
    return freq_real, psd


def aliased_fraction(lsf: np.ndarray, f_cut: float = F_NYQUIST) -> float:
    """
    Fraction of total LSF power at spatial frequencies strictly above f_cut
    (in cycles / real_pixel).  Default = detector Nyquist = 0.5 cyc/px.
    """
    freq, psd = power_spectrum(lsf)
    return float(psd[freq > f_cut].sum())


def cumulative_above(lsf: np.ndarray, n_fft: int = 512):
    """
    For each frequency f in the (zero-padded) spectrum, compute the fraction
    of total power at frequencies >= f.  Returns (freq, cum_above).

    Zero-padding to n_fft=512 gives a smooth curve for plotting; the total
    power (and hence the fraction at any boundary) is unchanged.
    """
    freq, psd = power_spectrum(lsf, n_fft=n_fft)
    # Reverse cumulative sum: psd is already normalised, so cumsum from right
    cum = np.cumsum(psd[::-1])[::-1]
    return freq, cum


# =============================================================================
# ── SINGLE-PSF DEMO ──────────────────────────────────────────────────────────
# =============================================================================
print(f'Loading example PSF: {EXAMPLE_FILE}')
psf_raw = read_zemax_map(EXAMPLE_FILE)

print('Finding optimal rotation angle ...')
lsf_ex, angle_ex, psf_rot_ex = extract_lsf(psf_raw)
cen_ex    = int(np.argmax(lsf_ex))
fwhm_sim  = measure_fwhm(lsf_ex)
fwhm_real = fwhm_sim / OVERSAMPLE_INT

print(f'  Optimal rotation : {angle_ex:.2f} °')
print(f'  LSF FWHM         : {fwhm_sim:.2f} sim px  =  {fwhm_real:.2f} real px')

# Normalised profiles
lsf_norm = lsf_ex / lsf_ex.sum()
g_norm   = make_gaussian_lsf(len(lsf_ex), fwhm_sim, cen_ex)

# Aliased-power fractions
frac_lsf        = aliased_fraction(lsf_ex)
frac_gauss      = aliased_fraction(g_norm)   # numerical (may hit FP floor)
frac_gauss_anal = gaussian_aliased_fraction_analytic(fwhm_real)

print(f'  Aliased power — real LSF       : {100 * frac_lsf:.4f} %')
print(f'  Aliased power — Gaussian (FFT) : {100 * frac_gauss:.4g} %')
print(f'  Aliased power — Gaussian (analytic, erfc): {100 * frac_gauss_anal:.3e} %')
print(f'  Aliased power — sinc           : 0.0000 %  (band-limited by definition)')

# High-resolution spectra for plotting (n_fft from config)
freq_hi, psd_lsf_hi   = power_spectrum(lsf_ex,  n_fft=N_FFT)
_,       psd_gauss_hi  = power_spectrum(g_norm,  n_fft=N_FFT)
freq_cum, cum_lsf      = cumulative_above(lsf_ex, n_fft=N_FFT)
_,        cum_gauss    = cumulative_above(g_norm,  n_fft=N_FFT)

# Sinc reference: ideal band-limited signal has a step-function cumulative
# that drops to exactly 0 at f_Nyquist.  We represent it as a step for clarity.
cum_sinc = np.where(freq_cum <= F_NYQUIST, 1.0, 0.0)
# Normalise so the cumulative starts at 1 (consistent with others)
# (the "1" means "all power is at frequencies <= Nyquist")
# We shift the step to align with the other curves' starting point
cum_sinc_start = cum_lsf[0]          # both LSF and Gaussian start near 1
cum_sinc = np.where(freq_cum <= F_NYQUIST, cum_sinc_start, 0.0)

# =============================================================================
# Figure 1 — PSF rotation + LSF extraction
# =============================================================================
fig1, axes = plt.subplots(1, 3, figsize=(15, 5),
                           gridspec_kw={'width_ratios': [1, 1, 1.3]})

for ax, img, title in zip(axes[:2],
                           [psf_raw, psf_rot_ex],
                           ['(a) Raw PSF (native orientation)',
                            f'(b) Rotated PSF  ({angle_ex:.1f}°)']):
    im = ax.imshow(img, origin='lower', cmap='inferno', aspect='equal')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Detector column (sim px)')
    ax.set_ylabel('Detector row (sim px)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='W mm⁻²')

ax = axes[2]
x_real = np.arange(len(lsf_ex)) / OVERSAMPLE_INT
ax.plot(x_real, lsf_norm, color=C_LSF, lw=2)
ax.axhline(0.5 * lsf_norm.max(), color='grey', lw=0.8, ls='--')
# FWHM arrow
cen_r = cen_ex / OVERSAMPLE
ax.annotate('', xy=(cen_r + fwhm_real / 2, 0.5 * lsf_norm.max()),
            xytext=(cen_r - fwhm_real / 2, 0.5 * lsf_norm.max()),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
ax.text(cen_r, 0.5 * lsf_norm.max() * 1.06,
        f'FWHM = {fwhm_real:.1f} px', ha='center', fontsize=10)
ax.set_xlabel('Detector column (real pixel)', fontsize=11)
ax.set_ylabel('Normalised intensity', fontsize=11)
ax.set_title('(c) Extracted 1-D LSF\n(collapsed along slit axis)', fontsize=11)
ax.set_xlim([0, len(lsf_ex) / OVERSAMPLE_INT])
ax.grid(True, alpha=0.3)

fig1.suptitle(f'Step 1 — PSF rotation and LSF extraction\n({EXAMPLE_FILE})',
              fontsize=12)
plt.tight_layout()
_out = os.path.join(OUTPUT_DIR, 'fig_01_psf_rotation.png')
plt.savefig(_out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved  {_out}')


# =============================================================================
# Figure 2 — LSF vs Gaussian in real space
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(9, 5))

x_cen = (np.arange(len(lsf_ex)) - cen_ex) / OVERSAMPLE_INT   # centred, real px
ax2.plot(x_cen, lsf_norm,  color=C_LSF,   lw=2.5, label='Observed LSF (Zemax)')
ax2.plot(x_cen, g_norm,    color=C_GAUSS, lw=2,   ls='--',
         label=f'Gaussian, same FWHM = {fwhm_real:.2f} px')
ax2.axhline(0, color='grey', lw=0.5)

# FWHM bracket
hm = 0.5 * lsf_norm.max()
ax2.axhline(hm, color='grey', lw=0.8, ls=':')
ax2.annotate('', xy=(fwhm_real / 2, hm),  xytext=(-fwhm_real / 2, hm),
             arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
ax2.text(0, hm * 1.07, f'FWHM = {fwhm_real:.2f} px', ha='center', fontsize=11)

ax2.set_xlabel('Position relative to LSF centre  (real pixels)', fontsize=12)
ax2.set_ylabel('Normalised intensity  (unit-sum profile)', fontsize=12)
ax2.set_title('Step 2 — LSF profile vs matched Gaussian\n'
              'Both profiles have the same FWHM; the Gaussian has broader wings',
              fontsize=12)
ax2.legend(fontsize=11)
ax2.set_xlim([-12, 12])
ax2.grid(True, alpha=0.3)
plt.tight_layout()
_out = os.path.join(OUTPUT_DIR, 'fig_02_lsf_profiles.png')
plt.savefig(_out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved  {_out}')


# =============================================================================
# Figure 3 — Power spectrum  +  cumulative aliased power          (2 panels)
# =============================================================================
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# ── top panel: log power spectral density ── ─────────────────────────────────
ax3a.semilogy(freq_hi, np.maximum(psd_lsf_hi,  1e-12),
              color=C_LSF,   lw=2.5, label='Observed LSF')
ax3a.semilogy(freq_hi, np.maximum(psd_gauss_hi, 1e-12),
              color=C_GAUSS, lw=2,   ls='--',
              label=f'Gaussian  (same FWHM = {fwhm_real:.1f} px)')
ax3a.axvline(F_NYQUIST, color=C_NYQ, lw=2, ls=':', zorder=5,
             label=f'Detector Nyquist  f_N = {F_NYQUIST} cyc/px')
ax3a.axvspan(F_NYQUIST, freq_hi[-1], alpha=0.08, color='red',
             label='Aliased zone  (f > f_N)')
ax3a.set_ylabel('Normalised power spectral density', fontsize=11)
ax3a.set_title('Power spectrum in spatial frequency space\n'
               'Power to the right of the green line is aliased by the detector',
               fontsize=12)
ax3a.legend(fontsize=10)
ax3a.grid(True, which='both', alpha=0.3)

# ── bottom panel: fraction of total power above frequency f ─────────────────
ax3b.semilogy(freq_cum, np.maximum(cum_lsf,   1e-12),
              color=C_LSF,   lw=2.5,
              label=f'Observed LSF      (aliased = {100*frac_lsf:.3f} %)')
ax3b.semilogy(freq_cum, np.maximum(cum_gauss, 1e-12),
              color=C_GAUSS, lw=2,   ls='--',
              label=f'Gaussian          (aliased = {100*frac_gauss_anal:.2e} %\n'
                    f'                   analytic erfc formula)')

# Sinc reference: a vertical drop to zero exactly at Nyquist
ax3b.axvline(F_NYQUIST, color=C_NYQ, lw=2, ls=':', zorder=5,
             label='Detector Nyquist  f_N = 0.5 cyc/px\n'
                   '(sinc would reach 0 here — 0.000 % aliased)')
ax3b.axvspan(F_NYQUIST, freq_cum[-1], alpha=0.08, color='red')

# Horizontal guide lines at the two aliased fractions
ax3b.axhline(frac_lsf,   color=C_LSF,   lw=1, ls='-.', alpha=0.7)
ax3b.axhline(frac_gauss, color=C_GAUSS, lw=1, ls='-.', alpha=0.7)

ax3b.set_xlabel('Spatial frequency  (cycles / real pixel)', fontsize=12)
ax3b.set_ylabel('Fraction of total power ABOVE this frequency', fontsize=11)
ax3b.set_title('Cumulative aliased power vs spatial-frequency cutoff\n'
               'Reading at f_N gives the aliased fraction for the real detector\n'
               'A perfect sinc (band-limited) profile would drop to zero '
               'exactly at the green line',
               fontsize=12)
ax3b.legend(fontsize=10)
ax3b.set_xlim([0, freq_cum[-1]])
ax3b.set_ylim([1e-7, 2.0])
ax3b.grid(True, which='both', alpha=0.3)

plt.tight_layout()
_out = os.path.join(OUTPUT_DIR, 'fig_03_power_spectra.png')
plt.savefig(_out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved  {_out}')


# =============================================================================
# ── BATCH: all 25 LSFs ───────────────────────────────────────────────────────
# =============================================================================
print(f'\nBatch processing all LSFs in {DATA_DIR}/ ...')

# Read wavelength map from the XY companion file
xy_file = os.path.join(DATA_DIR,
                       [f for f in os.listdir(DATA_DIR) if f.endswith('_XY.txt')][0])
xy_data = np.loadtxt(xy_file, skiprows=1)   # 25 rows: order wl xmm ymm

wl_map = {}          # (order, field_index_1to5) → wavelength µm
order_count: dict = {}
for row in xy_data:
    order = int(row[0])
    order_count[order] = order_count.get(order, 0) + 1
    field = order_count[order]
    wl_map[(order, field)] = float(row[1])

# Process each PSF
records = []   # list of dicts
orders_list = sorted({int(r[0]) for r in xy_data})

for order in orders_list:
    for field in range(1, 6):
        fname = os.path.join(DATA_DIR, f'R{order}{field}.txt')
        if not os.path.isfile(fname):
            print(f'  WARNING: {fname} not found — skipping.')
            continue
        psf = read_zemax_map(fname)
        lsf, ang, _ = extract_lsf(psf)
        cen = int(np.argmax(lsf))
        fwhm_s = measure_fwhm(lsf)
        fwhm_r = fwhm_s / OVERSAMPLE_INT
        g = make_gaussian_lsf(len(lsf), fwhm_s, cen)
        frac_l  = aliased_fraction(lsf)
        frac_g  = gaussian_aliased_fraction_analytic(fwhm_r)
        wl      = wl_map.get((order, field), np.nan)
        records.append(dict(order=order, field=field, wl=wl,
                            fwhm=fwhm_r, frac_lsf=frac_l, frac_gauss=frac_g,
                            angle=ang))
        print(f'  R{order}{field}  λ={wl:.4f} µm  FWHM={fwhm_r:.2f} real-px  '
              f'aliased: LSF={100*frac_l:.4f}%  Gauss={100*frac_g:.3e}%')

# =============================================================================
# Figure 4 — Summary: aliased fraction + FWHM for all 25 LSFs
# =============================================================================
orders_uniq = sorted({r['order'] for r in records})
cmap = plt.cm.plasma
colors = [cmap(i / max(len(orders_uniq) - 1, 1)) for i in range(len(orders_uniq))]

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

for i, ord_val in enumerate(orders_uniq):
    sub = [r for r in records if r['order'] == ord_val]
    wl_arr  = np.array([r['wl']         for r in sub])
    fl_arr  = np.array([r['frac_lsf']   for r in sub])
    fg_arr  = np.array([r['frac_gauss'] for r in sub])
    fw_arr  = np.array([r['fwhm']       for r in sub])
    c = colors[i]

    ax4a.plot(wl_arr, 100 * fl_arr, 'o-',   color=c, lw=1.5, ms=7,
              label=f'Order {ord_val}')
    ax4a.plot(wl_arr, 100 * fg_arr, 's--',  color=c, lw=1,   ms=5,
              markerfacecolor='none', markeredgewidth=1.5)
    ax4b.plot(wl_arr, fw_arr,       'o-',   color=c, lw=1.5, ms=7)

ax4a.set_ylabel('Power above Nyquist  (%)', fontsize=11)
ax4a.set_yscale('log')
ax4a.set_title('Aliased power fraction across all 25 PSFs\n'
               'Solid circles = real LSF  |  open squares = same-FWHM Gaussian (analytic erfc)\n'
               'A perfect sinc would give exactly 0 % (off the bottom of the log scale)',
               fontsize=11)
ax4a.legend(title='Circles = LSF / Squares = Gaussian',
            fontsize=9, loc='upper right')
ax4a.grid(True, which='both', alpha=0.3)

ax4b.set_xlabel('Wavelength  (µm)', fontsize=12)
ax4b.set_ylabel('LSF FWHM  (real pixels)', fontsize=11)
ax4b.set_title('LSF width across wavelengths and diffraction orders', fontsize=11)
ax4b.grid(True, alpha=0.3)

fig4.suptitle('VROOMM v04 rectangular fiber — Nyquist aliasing summary\n'
              f'Data: {DATA_DIR}/', fontsize=12)
plt.tight_layout()
_out = os.path.join(OUTPUT_DIR, 'fig_04_summary.png')
plt.savefig(_out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved  {_out}')

# =============================================================================
# Figure 5 — Aliased power vs LSF FWHM
# =============================================================================
# For all 25 real LSFs, plot the measured aliased fraction against FWHM.
# Overlay the exact analytic Gaussian curve: frac = erfc(2*pi*sigma*f_N)
# where sigma = FWHM / (2*sqrt(2*ln2)).
# This shows the clear monotonic trend and how far the real LSF deviates
# from the Gaussian prediction at any given FWHM.

fwhm_vals  = np.array([r['fwhm']       for r in records])   # real pixels
frac_lsf_v = np.array([r['frac_lsf']  for r in records])   # FFT-measured
frac_g_v   = np.array([r['frac_gauss'] for r in records])  # analytic erfc

# Smooth Gaussian reference curve over the FWHM range in the data
fwhm_smooth = np.linspace(0.5 * fwhm_vals.min(), 1.5 * fwhm_vals.max(), 500)
frac_gauss_smooth = np.array(
    [gaussian_aliased_fraction_analytic(fw) for fw in fwhm_smooth])

fig5, ax5 = plt.subplots(figsize=(9, 6))

# Analytic Gaussian curve (smooth line)
ax5.semilogy(fwhm_smooth, 100 * frac_gauss_smooth,
             color=C_GAUSS, lw=2, ls='--',
             label='Gaussian LSF  (analytic$\\,$erfc formula)')

# Real LSF measurements (one point per PSF, colour = order)
orders_arr = np.array([r['order'] for r in records])
for i, ord_val in enumerate(orders_uniq):
    mask = orders_arr == ord_val
    ax5.semilogy(fwhm_vals[mask], 100 * frac_lsf_v[mask],
                 'o', color=colors[i], ms=8, zorder=5,
                 label=f'Order {ord_val}  (real LSF)')

ax5.set_xlabel('LSF FWHM  (real pixels)', fontsize=12)
ax5.set_ylabel('Power above Nyquist  (%)', fontsize=12)
ax5.set_title(
    'Aliased power vs LSF width\n'
    'Circles = real (Zemax) LSF  |  dashed = same-FWHM Gaussian (analytic)\n'
    'The real LSF sits millions of times above the Gaussian at every FWHM',
    fontsize=11)
ax5.legend(fontsize=9, loc='upper right')
ax5.grid(True, which='both', alpha=0.3)
# Clip y-axis to a decade around the real LSF data — the Gaussian curve is
# so many orders of magnitude below that showing it would make the data invisible
y_lo = 10 ** (np.floor(np.log10(frac_lsf_v.min())) - 0.5)
y_hi = 10 ** (np.ceil( np.log10(frac_lsf_v.max())) + 0.5)
ax5.set_ylim([100 * y_lo, 100 * y_hi])
plt.tight_layout()
_out = os.path.join(OUTPUT_DIR, 'fig_05_aliasing_vs_fwhm.png')
plt.savefig(_out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved  {_out}')

print(f'\nAll done.  Output figures written to  {os.path.abspath(OUTPUT_DIR)}/')
for f in ['fig_01_psf_rotation.png', 'fig_02_lsf_profiles.png',
          'fig_03_power_spectra.png', 'fig_04_summary.png',
          'fig_05_aliasing_vs_fwhm.png']:
    print(f'  {os.path.join(OUTPUT_DIR, f)}')
