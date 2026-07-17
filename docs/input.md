# Input

The input module identifies, reads, and parses radar data files
distributed through the SHARAD, MARSIS, and LRS public archives. It
also handles loading pre-existing GRaSP state from HDF5 and parsing
TOML job files used to drive the high-level pipeline.

## Examples

### 1. Identify a File and Print Results
```
import grasp
file_info = grasp.identify_file("/path/to/science_file.dat")
file_info.print()
```

### 2. Load Instrument Parameters and Print Results
```
import grasp
file_info = grasp.identify_file("/path/to/science_file.dat")
parameters = grasp.grab_radar_params(file_info)
parameters.print()
```

### 3. Reading Files
```
import grasp
sci = grasp.read("/path/to/science_file.dat")
aux = grasp.read("/path/to/aux_file.dat")
lbl = grasp.read("/path/to/lbl_file.lbl")
```

### 4. Load a pre-existing GRaSP Radar Sounder State (HDF5 file)
```
import grasp
hdf5_file = "/path/to/hdf.h5"
rs = grasp.load(hdf5_file)
```

### 5. Read GRaSP-style output files
```
import grasp
grsp_data = grasp.read("../Testing/SHARAD/RD-retest/s_sharad_0436601_ss19_csim_csim.grsp")
grsp.print()
```

## API reference

::: grasp.input.read
::: grasp.input.load
::: grasp.input.job
::: grasp.input.input_support
::: grasp.input.utils
