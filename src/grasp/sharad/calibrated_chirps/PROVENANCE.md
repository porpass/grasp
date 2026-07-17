# SHARAD Calibration Chirp Files

## Source

These reference chirp files were obtained from the NASA Planetary Data
System (PDS) Geosciences Node SHARAD archive:

- Archive: MRO/SHARAD Experiment Data Record (EDR) Archive
- Directory: `CALIB/`
- Curator: PDS Geosciences Node, Washington University in St. Louis
- URL: https://pds-geosciences.wustl.edu/missions/mro/sharad.htm

## Licensing

PDS data products are works of the US Government and are not subject
to copyright under 17 U.S.C. § 105. They are made available by the
Planetary Data System for unrestricted use. See:
https://pds.nasa.gov/datastandards/policies/

## Citation

When using GRaSP's SHARAD calibrated chirp processing, please cite
both this package and the original SHARAD calibration documentation
referenced in the SHARAD EDR Data Product SIS (available in the
archive's `DOCUMENT/` directory).

## Files

40 files following the naming convention
`reference_chirp_{tx}_{rx}.dat` where:

- `tx ∈ {m20tx, m15tx, m10tx, m05tx, p00tx, p20tx, p40tx, p60tx}`
- `rx ∈ {m20rx, p00rx, p20rx, p40rx, p60rx}`

Each file is a binary array of 4096 little-endian float32 values:
the first 2048 are the real part of the spectrum, the next 2048 the
imaginary part. Sample spacing dt = 0.0375 μs. See
`grasp.processing.range_compression.chirps.create_sharad_calibrated_chirp`.

## Verification

These files are bit-identical to those distributed by the PDS as of
the archive snapshot dated <date you downloaded>.