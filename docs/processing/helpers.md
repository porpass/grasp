# Auxiliary Processing Functions

Shared signal processing utilities — windows and helpers used across
range, EMI, ionosphere, and SAR stages.

## Available Windows
- RECTANGLE | NONE
- HANN | HANNING 
- HAMMING
- BLACKMAN
- BLACKMAN-HARRIS | BLACKMAN_HARRIS | BH
- NUTTALL
- BARTLETT
- COSINE | SINE | RAISED_COSINE
- FLATTOP
- TUKEY

## Examples

### 1. Form a window, calculate the broadening factor, and plot the results
```
import matplotlib.pyplot as plt
import grasp
n = 2049
wnd = grasp.form_window(n, "HANNING")
brd = grasp.broadening_factor(wnd)
print(f"Broadening Factor: {brd}")
plt.plot(wnd)
plt.show()
```

### 2. Form a bandlimited window in natural order
```
import grasp
import numpy as np
file_info = grasp.identify_file("/path/to/file"))
params = grasp.grab_radar_params(file_info)
freq = np.fft.fftshift(np.fft.fftfreq(params.nfft, d=params.dt)
wnd = grasp.form_window_bandlimited(freq, params.bw, "TUKEY", alpha=0.25, f_cen=0.0, standard_order=False)
```

### 3. Shift an example data set to complex baseband
```
import numpy as np
import matplotlib.pyplot as plt
import grasp

n_samp, n_recs = 512, 4
nfft = 2048
dt    = 1.0 / 80e6   # 80 MS/s
f_cen = 20e6         # carrier at 20 MHz
bw    = 10e6         # keep-band after shift

# Real passband tone (spectrum has peaks at ±f_cen)
t = np.arange(n_samp) * dt
tone = np.cos(2 * np.pi * f_cen * t).astype(np.float32)
data = np.tile(tone[:, None], (1, n_recs))

bb, _ = grasp.to_complex_baseband(data, nfft=nfft, dt=dt, f_cen=f_cen, bw=bw)

# Match FFT lengths so both axes are in the same Hz.
freq     = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt))
in_spec  = np.fft.fftshift(np.abs(np.fft.fft(data[:, 0], n=nfft)))
out_spec = np.fft.fftshift(np.abs(np.fft.fft(bb[:, 0])))

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(freq / 1e6, in_spec)
ax[0].set_title("Input (real passband, peaks at ±20 MHz)")
ax[1].plot(freq / 1e6, out_spec)
ax[1].set_title("Output (complex baseband, peak at 0 Hz)")
ax[1].set_xlabel("Frequency (MHz)")
plt.tight_layout()
plt.show()
```

::: grasp.processing.utils

::: grasp.processing.windows
