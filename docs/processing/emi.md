# EMI

Electromagnetic interference suppression methods applied during range
processing.

## Examples

### 0. Set up
```
import numpy as np
import matplotlib.pyplot as plt
import grasp

# 1. Synthetic complex baseband chirp
nfft = 4096
fs   = (80/3)*1e6
dt   = 1.0 / fs
tau  = 85.05e-6           # pulse length
bw   = 10e6            # chirp bandwidth

chirp, _, _ = grasp.create_complex_baseband_chirp(
    nfft, dt, tau, bw, chirp_direction=-1)

# Replicate across a few records so the docs visual shows column-wise behaviour
n_recs = 4
data = np.tile(chirp[:, None], (1, n_recs))

# 2. Inject narrowband EMI tones in the time domain (must be in-band, |f| <= bw/2)
bin_hz = 1.0 / (nfft * dt)
emi_freqs = np.round(np.array([-3.2e6, -1.1e6, 2.4e6, 3.9e6]) / bin_hz) * bin_hz
emi_amps  = np.array([5.0e-2, 1.0e-2, 8.0e-2, 4.0e-2])
t = np.arange(nfft) * dt
for f0, a0 in zip(emi_freqs, emi_amps):
    data = data + a0 * np.exp(1j * 2 * np.pi * f0 * t)[:, None]
    
spec = np.fft.fft(data, axis=0)
freqs = np.fft.fftfreq(nfft, d=dt)
```

### 1. Suppressing EMI with the ADAPTIVE method
```
# Adaptive spectral notch (frequency domain)

clean_spec, emi_mask = grasp.adaptive_spectral_notch(spec, freqs, bw, f_cen=0.0,
                                                     window_size=65, k=3,
                                                     statistic="MAD", replace="INTERP",
                                                     interp_pad=5)

# Before / after (record 0)
order = np.argsort(freqs)
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
ax[0].plot(freqs[order] / 1e6, np.abs(spec[order, 0]))
ax[0].set_title("Before: chirp spectrum + EMI spikes")
ax[1].plot(freqs[order] / 1e6, np.abs(clean_spec[order, 0]))
ax[1].set_title("After: adaptive spectral notch (INTERP)")
ax[1].set_xlabel("Frequency (MHz)")
plt.tight_layout()
plt.show()
```

### 2. Suppressing EMI with the THRESHOLD method
```
# Threshold spectral notch (frequency domain)

clean_spec, emi_mask = grasp.threshold_emi(spec, freqs, bw,
                                           threshold_k=1.0,
                                           value_k=0.5,
                                           f_cen=0, window_size=128)
                                           
# Before / after (record 0)
order = np.argsort(freqs)
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
ax[0].plot(freqs[order] / 1e6, np.abs(spec[order, 0]))
ax[0].set_title("Before: chirp spectrum + EMI spikes")
ax[1].plot(freqs[order] / 1e6, np.abs(clean_spec[order, 0]))
ax[1].set_title("After: threshold notch")
ax[1].set_xlabel("Frequency (MHz)")
plt.tight_layout()
plt.show()
```

### 3. Using the suppress_emi wrapper
```
# suppress_emi handles the FFT/IFFT and dispatches to the chosen method.
# Pass time-domain data with input_time=True.

clean_spec, emi_mask = grasp.suppress_emi(data, nfft=nfft, dt=dt, bw=bw, f_cen=0.0,
                                          method="ADAPTIVE", statistic="MAD",
                                          k=3, window_size=129, replace="ZERO",
                                          input_time=True, output_time=False)

# Before / after (record 0)
order = np.argsort(freqs)
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
ax[0].plot(freqs[order] / 1e6, np.abs(spec[order, 0]))
ax[0].set_title("Before: chirp spectrum + EMI spikes")
ax[1].plot(freqs[order] / 1e6, np.abs(clean_spec[order, 0]))
ax[1].set_title("After: adaptive spectral notch (ZERO)")
ax[1].set_xlabel("Frequency (MHz)")
plt.tight_layout()
plt.show()
```


::: grasp.processing.emi.suppression

::: grasp.processing.emi.adaptive

::: grasp.processing.emi.threshold

::: grasp.processing.emi.utils
