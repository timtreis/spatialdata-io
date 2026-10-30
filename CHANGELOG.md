# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

Release notes for `v0.7.1` and earlier are available on the [Releases][] page.

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/
[releases]: https://github.com/scverse/spatialdata-io/releases

## [Unreleased]

### Added

- `imzml()` (extra `imzml`, needs `pyimzml`) reads 2-D imzML imaging mass spectrometry into pixels x m/z features
  (a deposited m/z list, or the top-K ppm bins by total intensity) with one square per pixel, in µm when the pixel size is known.
- `imc()` reads one IMC region of interest from an OME-NGFF zarr(.zip), a per-page OME-TIFF (MCD Viewer export) or a
  multi-page TIFF named by a steinbock `panel.csv`, with optional mask labels; pixels are µm (1 µm ablation).
- `dbit_fragments()` reads DBiT spatial-ATAC / spatial-CUT&Tag deposits: it streams a 10x fragments file into Tn5
  insertion counts per peak or genome tile (or reads a 10x peak matrix), with one square per DBiT pixel and the tissue image.
- `cosmx()` adds a `"microns"` coordinate system next to the pixel `"global"` one, scaled by the pixel size from
  the FOV positions file (when it lists px and mm) or the CosMx spec; `sdata.attrs["cosmx"]` records the value and its source.
- `mibi()` reads one MIBI field of view in the Angelo lab layout (one TIFF per channel, as deposited on the
  BioImage Archive) into an image named by its channels, with optional segmentation masks as labels.
- `cell_table` reads a cells x genes table of raw integer counts with centroids into a CSR table and
  `cells` circle shapes, from a wide CSV/TSV, a counts file (delimited or `.mtx`) plus a `coordinates`
  file, or an `.h5ad` file (`layer` picks the counts). `split_by` creates one shapes element per section.
  The coordinate `unit` is required and mapped to micrometres with a `Scale` transformation; unit,
  radius provenance (source, median, min and max in µm) and the integer dtype of `X` (`int32`, or `int64`
  if a count exceeds the `int32` range) are stored in `table.uns["spatialdata_io"]`.
- `stereoseq_gef` reads flat Stereo-seq GEF (HDF5) files as deposited on GEO. It reads the cells from
  `cellbin.gef` (table and polygons), the square bins from `tissue.gef` or a full `.gef` at a chosen `bin_size`,
  or both together as two tables, each annotating its own shapes.
- `spatialdata_io` ships a `py.typed` marker, so downstream type checkers use its annotations.
- `xenium` reads pre-1.3.0 (XOA < 1.3.0) and CSV-only bundles: when the parquet files and
  `cells.zarr.zip` are absent, the table, boundaries, and transcripts are read from the `cells.csv`,
  `cell_boundaries.csv`, `nucleus_boundaries.csv` and `transcripts.csv` outputs (`.gz` or plain),
  and the raster cell/nucleus labels are reconstructed by rasterizing the boundary polygons. The
  cell feature matrix falls back to a `cell_feature_matrix/` MatrixMarket directory or
  `cell_feature_matrix.tar.gz` when `cell_feature_matrix.h5` is missing.
- `phenocycler()` reader and CLI command for Akoya PhenoCycler-Fusion 1.x/2.x `.qptiff` / `.er.qptiff` scans
  (incl. Opal mode) and other PerkinElmer/Akoya QPTIFFs (Vectra Polaris, PhenoImager HT, inForm component data):
  lazy images with channel names from the page XML, scan metadata in `sdata.attrs["akoya"]`, optional `.xpd` /
  `*channels.csv` sidecars and segmentation masks (`labels`); `read_all_bit_depths=True` also reads the 8-bit scan.

### Changed

- Adopted the current `cookiecutter-scverse` template: `hatch`-managed test environments, `uv`-based CI and
  documentation builds, `mypy` type checking of `src` and `tests`, `biome`/`pyproject-fmt`/`zizmor` pre-commit hooks,
  and Dependabot updates.
- Requires `tifffile>=2026.5.2` and `zarr>=3.2`, the first versions whose zarr store the lazy Akoya TIFF reads work with.
- Requires `flowio>=1.4`, the first version with `FlowData.pnn_labels`, with which `codex()` detects CODEX FCS tables.
- `codex()` was rewritten to read CODEX Processor, CODEX Uploader and CODEX Instrument Manager (`.qptiff`)
  outputs, sharing the QPTIFF core with `phenocycler()`. Breaking changes:
  - Element names are `{region}_image`, `{region}_cells`, `{region}_labels` and `table` (were `images`, `{region}`).
  - All regions are read (was the first file found); `regions=[...]` subsets. Unrelated regions get one coordinate
    system each; regions of one slide are placed at their stage positions in `"global"`.
  - `table.var_names` and image channels carry marker names (FCS `$PnS`, CSV `cycCCC_chHHH:<marker>`,
    `channelNames.txt`), not `cycCCC_chHHH`.
  - New keyword-only parameters `regions`, `channel_names`, `cells_table`, `labels`, `*_models_kwargs`;
    `imread_kwargs` is ignored with a warning.

### Fixed

- `visium()` skips a `fullres_image_file` that holds fewer than 99% of the spots (e.g. a deposited crop or the
  CytAssist capture image), with a warning, instead of placing it misregistered at identity.
- `codex()` read the CSV table and the image relative to the current working directory instead of `path`, and an
  operator-precedence bug in the table check made it ignore `fcs`.

### Removed

- Support for Python 3.11.
