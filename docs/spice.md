# SPICE


## Examples

### 1. Grab SPICE Parameters
```
import grasp
in_file = "/path/to/file"
file_info = grasp.identify_file(in_file)
sp_params = grasp.grab_spice_params(file_info.platform)
sp_params.print()
```

### 2. Find and load the appropriate MK File
```
import grasp
in_file = "/path/to/file"
file_info = grasp.identify_file(in_file)
data = grasp.read(in_file)
grasp.print_mk_paths(file_info.instrument)
# The KEY for the appropriate UTC Time Stamps is INSTRUMENT-DEPENDENT
# This example is for a LRS SA WF Observation 
years = grasp.determine_observation_years(data['OBSERVATION_TIME'])
mk_file = grasp.find_mk_files(file_info.instrument, years)
print(mk_file)
grasp.furnish(mk_file)
```

### 3. Get Radii
Furnish an MK file as show above, then
```
radii = grasp.get_radii(file_info.target)
```

### 4. Convert Ephemeris Time to UTC
Load an MK File like above. Note: not all datasets supported by GRaSP have an EPHEMERIS_TIME
key.
```
et = data['EPHEMERIS_TIME']
utc = grasp.et2utc(et)
```

### 5. Convert UTC to Ephemeris Time
Load an MK File like above. Note: datasets supported by GRaSP identify their UTC key differently.
```
utc = data['GEOMETRY_EPOCH']
et = grasp.utc2et(utc)
```

### 6. Compute State Vectors
With Ephemeris Time
```
state_vectors = grasp.compute_state_vectors(et, sp_params.observer_str,
                                            sp_params.target_str,
                                            sp_params.fix_ref,
                                            ab_corr=sp_params.ab_corr,
                                            utc=False) 
```
With UTC (note the utc = True argument)
```
state = grasp.compute_state_vectors(utc, sp_params.observer_str,
                                    sp_params.target_str,
                                    sp_params.fix_ref,
                                    ab_corr=sp_params.ab_corr,
                                    utc=True) 
```

### 7. Compute Geodetic Coordinates from state vectors
Will compute the latitude, longitude, altitude above the ellipsoid, spacecraft radius,
and planetary radius. 

Compute the state vectors as shown above, then:
```
geodetic = grasp.compute_geodetic_position(state.r_t,
                                           state.r_s,
                                           state.r_st)
```

### 8. Decompose the velocity vectors
Will compute tangential and radial velocities. 

Compute the state vectors as shown above, then
```
velocities = grasp.decompose_velocity(state.r_s, state.v_s)
```

### 9. Compute illumination angles
Using either ephemeris time (ets) or UTCs and the state vector result 
as shown above:
```
illum = grasp.compute_sza(utcs,
                          sp_params.observer_str,
                          sp_params.target_str,
                          state.r_t,
                          utc = True)
```

### 10. Compute all geometries
Using either ephemeris time (ets) or UTCs as shown above
```
geom = grasp.compute_geometry(ets,
                              sp_params.observer_str,
                              sp_params.target_str,
                              sp_params.fix_ref,
                              ab_corr=sp_params.ab_corr,
                              compute_illum = True,
                              utc = False)
```

### 11. Unload MK Files
After loading in an MK file and performing your desired processing
```
grasp.unload(mk_file)
```

::: grasp.spice.observation_geometry
::: grasp.spice.utils