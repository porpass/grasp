# Quickstart

This guide takes you from installation to producing a radargram.
It assumes you already have access to a SHARAD,
MARSIS, or LRS data product and the matching SPICE kernels — if not,
see [External Data](external-data.md) first.

<!-- TODO: confirm the assumed audience above is accurate. -->

## 1. Install

=== "pip"

    ```bash
    pip install grasp
    ```

=== "conda + editable"

    ```bash
    git clone https://github.com/mr-perry/grasp.git
    cd grasp
    conda env create -f environment.yml
    conda activate grasp
    pip install -e .
    ```

<!-- TODO: keep both tabs or drop one — depends on PyPI publishing status. -->

Verify the install:

```bash
python -c "import grasp; print(grasp.__version__)"
```

## 2. Configure data paths

GRaSP needs to know where your SPICE kernels and reference DEMs live.
Copy the example config and edit the paths:

```bash
cp examples/config.example.toml ~/.config/grasp/config.toml
$EDITOR ~/.config/grasp/config.toml
```

<!-- TODO: confirm the actual config.toml resolution path GRaSP uses. -->

## 3. Write a minimal job file

Save the following as `my_first_job.toml`:

```toml
# TODO: replace with an actual minimal SHARAD EDR example
[input]
label_file    = "/path/to/your/edr.lbl"
science_file  = "/path/to/your/edr_s.dat"
auxiliary_file = "/path/to/your/edr_a.dat"

[range_compression]
enabled = true
method  = "MATCHED"

[output_parameters]
out_dir = "./output"
```

See the [TOML Configuration Reference](toml-reference.md) for every
section and key.

## 4. Run the job

```bash
python -m grasp.grasp my_first_job.toml
```

You should see an attribution banner followed by per-stage progress
messages. When the job finishes, look in `./output/` for:

<!-- TODO: list expected output filenames once a real run is documented. -->

- `<base>_rc.h5` — range-compressed data
- `<base>_rc.png` — quick-look radargram
- (other artifacts depending on which stages you enabled)

## 5. Next steps

- [TOML Configuration Reference](toml-reference.md) — every config option
- [Algorithms](processing/index.md) — what each pipeline stage actually does
- Instrument-specific guides:
  [SHARAD](sharad.md) · [MARSIS](marsis.md) · [LRS](lrs.md)
