# Core

Top-level package modules: the abstract radar sounder, the factory that
instantiates instrument-specific subclasses, the typed parameter and
metadata containers, and the high-level processing entrypoint.

## Examples

### 1. Instantiating a radar_sounder with files
```
import grasp
lbl_file = "/path/to/lbl"
aux_file = "/path/to/aux"
sci_file = "/path/to/sci"
rs = grasp.radar_sounder([lbl_file, aux_file, sci_file])
```
### 2. Instantiating a radar_sounder with a job file. Print job summary
```
import grasp
job_file = "/path/to/job.toml"
rs = grasp.radar_sounder(job_file)
grasp.print_job_summary(rs)
```

### 3. Instantiating a radar_sounder with a state (HDF5) file
```
import grasp
state_file = "/path/to/state.h5"
rs = grasp.radar_sounder(state_file)
```

### 4. Full Processing using grasp (job file required)
```
import grasp
job_file = "/path/to/job.toml"
rs = grasp.process(job_file)
```

::: grasp.radar_sounder

::: grasp.instantiator

::: grasp.grasp_types

::: grasp.grasp

::: grasp.constants
