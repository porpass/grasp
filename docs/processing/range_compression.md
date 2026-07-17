# Range Compression

Chirp generation, matched-filter construction, and range compression.

## Examples

### 1. Create a down chirp at complex baseband
```
import grasp
file_info = grasp.identify_file("/path/to/file")
params = grasp.grab_radar_params(file_info)
chirp, t_tau, f_tau = grasp.create_complex_baseband_chirp(params.nfft, params.dt,
                                                          params.tau, params.bw,
                                                          chirp_direction = -1)
```

### 2. Create a SHARAD calibrated chirp
```
import grasp
aux_data = grasp.read("/path/to/sharad/aux.dat")
chirp, f = grasp.create_sharad_calibrated_chirp(aux_data['TX_TEMP'][0], aux_data['RX_TEMP'][0])
```

### 3. Create an inverse filter for range compression
```
import grasp
file_info = grasp.identify_file("/path/to/sharad/aux.dat")
params = grasp.grab_radar_params(file_info)
chirp, t_tau, f_tau = grasp.create_complex_baseband_chirp(params.nfft, params.dt,
                                                          params.tau, params.bw,
                                                          chirp_direction = -1)
h_f, f = grasp.create_filter("INVERSE", chirp, params.dt, params.bw)
```

### 4. Range Compression Example (Self-compress a chirp (Klauder wavelet))
```
import grasp
import numpy as np
file_info = grasp.identify_file("/path/to/file")
params = grasp.grab_radar_params(file_info)
chirp_t, t_tau, f_tau = grasp.create_complex_baseband_chirp(
    params.nfft, params.dt, params.tau, params.bw, chirp_direction=-1)

# build the filter from the TIME-DOMAIN chirp
h_f, f = grasp.create_filter("INVERSE", chirp_t, params.dt, params.bw)

wnd = grasp.form_window_bandlimited(
    np.fft.fftshift(f), params.bw, "BH", standard_order=True)

# only FFT for the range_compress data argument
chirp_f = np.fft.fft(chirp_t)
data = grasp.range_compress(chirp_f, h_f, wnd, input_time=False, output_time=True)
```

::: grasp.processing.range_compression.chirps

::: grasp.processing.range_compression.compress

::: grasp.processing.range_compression.filters
