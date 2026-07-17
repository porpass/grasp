# Ionosphere

Methods for compensating ionospheric phase distortion in Mars-orbiting
radar sounder data.

GRaSP exposes two autofocus methods plus a wrapper:

- **Campbell** — single-parameter `E / |f|^b` power-law search.
  Tuned for SHARAD (`b ≈ 1.93`) but applicable to any single-band radar.
- **Contrast** — three-parameter polynomial in offset frequency
  (`a2·f² + a3·f³ + a4·f⁴`) following Cartacci 2013. `a3` and `a4` are
  derived from `a2`, so the grid search is effectively one-dimensional.
- **`ionospheric_compensation`** — high-level dispatcher that runs
  either method and returns a structured `CampbellResult` or
  `ContrastResult`.

Use Campbell when the dispersion is well-described by a single power
law (typical SHARAD). Use Contrast when you want explicit a2/a3/a4
coefficients or when the polynomial model fits your observation better. 
The dispatcher is the recommended entry point for new code.

## Examples

### 1. Use of the Campbell Method (SHARAD-tuned)

#### Model Set-up
```
import numpy as np
import matplotlib.pyplot as plt
import grasp

# Chirp parameters (SHARAD-ish)
nfft = 4096
fs   = (80/3) * 1e6
dt   = 1.0 / fs
tau  = 85.05e-6
bw   = 10e6
f_phys = 20e6        # The actual center frequency
freq = f_phys + np.fft.fftfreq(nfft, d=dt)

# Clean chirp + matched filter + spectral window
chirp, _, _ = grasp.create_complex_baseband_chirp(
    nfft, dt, tau, bw, chirp_direction=-1)
h_f, f = grasp.create_filter("INVERSE", chirp, dt, bw)
rc_wnd = grasp.form_window_bandlimited(
    np.fft.fftshift(f), bw, "BH", standard_order=True)

# Apply ionospheric dispersion to the chirp in the frequency domain
E_true = 2.013e16    # ~π of phase at the band edge (f = bw/2) for b=1.93
b      = 1.93        # SHARAD power-law exponent (Campbell et al. 2014)
sgn    = 1           # sign chosen so the positive Campbell grid can recover

phase_iono = np.zeros_like(freq)
nz = freq != 0
phase_iono[nz] = sgn * E_true / np.abs(freq[nz])**b

chirp_f       = np.fft.fft(chirp)
chirp_dist_f  = chirp_f * np.exp(1j * phase_iono)

# Range-compress both clean and distorted versions
clean_pulse   = grasp.range_compress(chirp_f,      h_f, rc_wnd,
                                     input_time=False, output_time=True)
smeared_pulse = grasp.range_compress(chirp_dist_f, h_f, rc_wnd,
                                     input_time=False, output_time=True)

# Tile across records so the autofocus has columns to work with
n_recs = 16
ts = np.tile(smeared_pulse[:, None], (1, n_recs))
```

`sgn=+1` on the distortion side paired with `sgn=-1` on the Campbell
call is required because Campbell's E-grid is positive-only; flipping
the distortion sign lets the positive grid find the optimum. Contrast
(Section 2) scans a symmetric grid and therefore uses the physical sign
directly — the asymmetry between the two sections is convention, not
physics.

#### Application
```
ts_camp, e_vals, _ = grasp.ionosphere_campbell(
    ts, dt=dt, bw=bw, f0=0.0, n_take=4, b=b,
    n_phase=400, delta=0.0125, sgn=-1, wnd="hanning",
    f_phys=20e6)

E_est = float(e_vals[0])
pct   = 100.0 * (E_est - E_true) / E_true
print(f"Campbell — E_true = {E_true:.3e},  E_est = {E_est:.3e},  Δ = {pct:+.2f}%")

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
ax[0].plot(np.roll(np.abs(smeared_pulse), nfft//2), label="distorted")
ax[0].plot(np.roll(np.abs(clean_pulse), nfft//2), '--', label="clean (reference)")
ax[0].set_title("Before: ionospheric smear"); ax[0].legend()
ax[1].plot(np.roll(np.abs(ts_camp[:, 0]), nfft//2), label="Campbell-corrected")
ax[1].plot(np.roll(np.abs(clean_pulse), nfft//2), '--', label="clean (reference)")
ax[1].set_title("After: Campbell autofocus"); ax[1].legend()
for a in ax:
    a.set_xlim(nfft//2 - 100, nfft//2 + 100)
plt.tight_layout(); plt.show()
```

### 2. Use of the Contrast Method

Contrast fits a third-order polynomial in offset frequency. Use it
when you need the full `a2 / a3 / a4` coefficients or when Campbell's
single-power-law model is too rigid for your data.

#### Model Set-up
```
import numpy as np
import matplotlib.pyplot as plt
import grasp
from grasp.constants import C

# Chirp parameters (MARSIS-ish)
nfft = 1024
fs   = 2.8 * 1e6
dt   = 1.0 / fs
tau  = 250e-6
bw   = 1e6
f_phys = 4.0e6        # The actual center frequency
L_eq = 80e3
tau_0 = 2*L_eq / C
n_recs = 4

# Set up distortion (Cartacci 2013 polynomial coefficients)
a2_true  = 4.25e-11
a3_true  = -(a2_true / f_phys) * (1 - (a2_true * f_phys) / (np.pi * tau_0))
a4_true  =  (a2_true / f_phys**2) * (1 - (a2_true * f_phys) / (0.5 * np.pi * tau_0))

# Baseband offset frequency — same as Contrast's f_in when f0=0
f_in = np.fft.fftfreq(nfft, d=dt)
delta_phi_true = a2_true * f_in**2 + a3_true * f_in**3 + a4_true * f_in**4

# True bulk ionospheric delay (a linear phase ramp in the frequency
# domain). The Contrast autofocus does not recover this directly —
# the Level 1 calibration (Cartacci 2013 + Campbell 2014) infers
# delay from the recovered a2 and applies the matching un-shift.
true_delay_s = 25 * dt                              # ~25 cells of delay
delay_phase = np.exp(-2j * np.pi * f_in * true_delay_s)

# Clean chirp + matched filter + spectral window
chirp, _, _ = grasp.create_complex_baseband_chirp(
    nfft, dt, tau, bw, chirp_direction=-1)
h_f, f = grasp.create_filter("INVERSE", chirp, dt, bw)
rc_wnd = grasp.form_window_bandlimited(
    np.fft.fftshift(f), bw, "BH", standard_order=True)

chirp_f = np.fft.fft(chirp)

# Distort the chirp — polynomial smear plus the bulk delay.
chirp_dist_f  = chirp_f * np.exp(-1j * delta_phi_true) * delay_phase

# Range Compress
clean_pulse   = grasp.range_compress(chirp_f,      h_f, rc_wnd,
                                     input_time=False, output_time=True)

smeared_pulse = grasp.range_compress(chirp_dist_f, h_f, rc_wnd,
                                     input_time=False, output_time=True)

ts = np.tile(smeared_pulse[:, None], (1, n_recs))
```

#### Application
```
ts_con, a2_vals, a3_vals, a4_vals, delay_cells = grasp.ionosphere_contrast(
    ts, dt=dt, bw=bw, f0=0.0, n_take=4, n_grid=801, wnd="hanning",
    L_eq=L_eq, f_phys=f_phys)

a2_est        = float(a2_vals[0])
pct_a2        = 100.0 * (a2_est - a2_true) / a2_true
delay_est_s   = float(delay_cells[0]) * dt
pct_delay     = 100.0 * (delay_est_s - true_delay_s) / true_delay_s
print(f"Contrast — a2_true    = {a2_true:.3e},  a2_est    = {a2_est:.3e},  Δ = {pct_a2:+.2f}%")
print(f"           true_delay = {true_delay_s*1e6:.3f} μs, "
      f"est_delay = {delay_est_s*1e6:.3f} μs, Δ = {pct_delay:+.2f}%")

smear   = np.roll(np.abs(smeared_pulse),   nfft//2)
clean   = np.roll(np.abs(clean_pulse),     nfft//2)
corr    = np.roll(np.abs(ts_con[:, 0]),    nfft//2)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
ax[0].plot(np.abs(smear), label="distorted")
ax[0].plot(np.abs(clean), '--', label="clean (reference)")
ax[0].set_title("Before: ionospheric smear"); ax[0].legend()
ax[1].plot(np.abs(corr), label="Contrast-corrected")
ax[1].plot(np.abs(clean), '--', label="clean (reference)")
ax[1].set_title("After: Contrast method"); ax[1].legend()
for a in ax:
    a.set_xlim(nfft//2 - 30, nfft//2 + 30)
plt.tight_layout(); plt.show()
```

The corrected pulse could appear slightly broader than the clean reference
because the Contrast method applies its own spectral window during
autofocus, on top of the window already applied by `range_compress`.
Switching `wnd="RECTANGLE"` in the call makes the
recovered pulse match the reference exactly.

### 3. Use of the `ionospheric_compensation` dispatcher

`ionospheric_compensation` is the recommended entry point for new
code: it dispatches to Campbell or Contrast based on the `method`
kwarg and returns the result as a structured `CampbellResult` or
`ContrastResult`. The setup is identical to Section 2; only the
application block changes.

#### Application
```
ts_con, iono_result = grasp.ionospheric_compensation(ts,
                                                     dt=dt,
                                                     bw=bw,
                                                     f0=0.0,
                                                     method="CONTRAST",
                                                     n_take=n_recs,
                                                     wnd="BH",
                                                     L_eq=L_eq,
                                                     f_phys=f_phys)

a2_est = float(iono_result.a2[0])
pct    = 100.0 * (a2_est - a2_true) / a2_true
print(f"Contrast — a2_true = {a2_true:.3e},  a2_est = {a2_est:.3e},  Δ = {pct:+.2f}%")

smear   = np.roll(np.abs(smeared_pulse),   nfft//2)
clean   = np.roll(np.abs(clean_pulse),     nfft//2)
corr    = np.roll(np.abs(ts_con[:, 0]),    nfft//2)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
ax[0].plot(np.abs(smear), label="distorted")
ax[0].plot(np.abs(clean), '--', label="clean (reference)")
ax[0].set_title("Before: ionospheric smear"); ax[0].legend()
ax[1].plot(np.abs(corr), label="Contrast-corrected")
ax[1].plot(np.abs(clean), '--', label="clean (reference)")
ax[1].set_title("After: Contrast method"); ax[1].legend()
for a in ax:
    a.set_xlim(nfft//2 - 30, nfft//2 + 30)
plt.tight_layout(); plt.show()
```

`iono_result.a3` and `iono_result.a4` carry the corresponding
higher-order coefficients if you want to record or post-process the
full polynomial.

::: grasp.processing.ionosphere.campbell

::: grasp.processing.ionosphere.chapman

::: grasp.processing.ionosphere.contrast

::: grasp.processing.ionosphere.ionospheric_compensation
