# External Data

The take advantage of the full functionality of GRaSP, external data sources should
be downloaded and made accessible to GRaSP.

## Configuration File
Within `src`, there is a `config.example.toml` file. During installation, you should have copied
this file to `config.toml`. For GRaSP to automatically locate, load, and utilize the SPICE kernels
on your system, the following must be filled out in `config.toml`:

```
[spice]
mro_mk = "/path/to/mrosp_1000/extras/mk"
mex_mk = "/path/to/mexsp_2000/EXTRAS/MK"
selene_mk = "/path/to/slnsp_1000/extras/mk"
```

Note: The MK Files have pre-defined paths in them. Please ensure this path is correct or edit them
appropriately. A short script to change the default paths is quite easy to put together.

## SPICE
In the SAR processing routines, we opted to use the state vectors for various calculations
rather than relying on header information. This decision was made due to unpredictable errors
or missing data in the archived data headers. Therefore, in order to properly conduct SAR processing
using GRaSP, one must have access to the SPICE kernels for each instrument

### LRS
The SPICE kernels are available at:



