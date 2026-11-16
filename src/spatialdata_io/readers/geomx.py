from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from shapely.geometry import Point
from spatialdata import SpatialData, sanitize_name
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale

from spatialdata_io.readers._utils._features import FEATURE_TYPES, GENE_EXPRESSION
from spatialdata_io.readers._utils._utils import _set_observation_unit, _set_reader_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["geomx"]

_SHAPES = "aois"
_NEGATIVE = "Negative Control Probe"
_UNANNOTATED = "Unannotated Probe"
_PROTEIN = "Protein Expression"
# without a PKC (or deposited CodeClass) a target is a negative control only if its name starts exactly with this
_NEGPROBE_PREFIX = "NegProbe-"
_DCC_SECTIONS = ("Header", "Scan_Attributes", "NGS_Processing_Attributes")
# numeric NGS fields, as GeomxTools coerces them (readDccFile.R)
_DCC_NUMERIC = ("Raw", "Trimmed", "Stitched", "Aligned", "umiQ30", "rtsQ30", "DeduplicatedReads")
_SECTION = re.compile(r"^<(/?)(\w+)>$")


def geomx(
    dcc: str | Path | Sequence[str | Path] | None = None,
    pkc: str | Path | Sequence[str | Path] | None = None,
    annotation: str | Path | None = None,
    *,
    counts: str | Path | None = None,
    layers: Mapping[str, str | Path] | None = None,
    image: str | Path | None = None,
    annotation_sheet: str | int = 0,
    matrix_sheet: str | int = 0,
    matrix_sep: str | None = None,
    decimal: str = ".",
    probe_level: bool = False,
    target_column: str = "TargetName",
    feature_columns: Sequence[str] = (),
    aoi_id: str | None = None,
    x: str | None = None,
    y: str | None = None,
    area: str | None = None,
    aoi_radius: float | None = None,
    coordinates_in_image: bool = False,
    segment_map: Mapping[str, str] | None = None,
) -> SpatialData:
    """Read a *GeoMx DSP* (NanoString/Bruker) experiment: one table row per AOI (area of illumination).

    Two inputs are supported:

        - **DCC + PKC**: ``dcc`` is a folder of ``*.dcc`` files (optionally ``.gz``, one per AOI) or a list of
          them, and ``pkc`` the probe kit configuration file(s) (``*.pkc``, JSON). The table rows are named by the
          DCC file stem (GeomxTools' ``Sample_ID``).
        - **matrix**: ``counts`` is a features x AOIs table (``.csv``/``.tsv``/``.txt``, optionally ``.gz``, or an
          ``.xlsx`` sheet ``matrix_sheet``), its first column the feature names and the other columns the AOIs,
          except those named in ``feature_columns``, which go to ``var``. With ``probe_level`` the rows are probes and
          ``target_column`` names their target; otherwise the rows are targets.

    Elements:

        - ``table``: AOIs x targets. Probe counts are **summed** per target, so the counts stay integers and their
          total equals the probe total (GeomxTools' ``aggregateCounts`` takes the geometric mean instead).
          ``var["feature_types"]`` is ``"Negative Control Probe"`` where the PKC ``CodeClass`` contains
          ``"Negative"`` (GeomxTools' rule), with ``var["is_control"]``. A target that no PKC or deposited
          ``CodeClass`` types is a negative control if its name starts with ``"NegProbe-"`` (e.g. ``NegProbe-WTX``);
          the rule and the number of targets it typed are in ``uns["spatialdata_io"]["negprobe_rule"]``. Protein kits (PKC ``AnalyteType``
          ``"Protein"``) go to ``table_protein`` the same way.
        - ``table_probe`` (DCC or ``probe_level`` input): AOIs x probes, every ``RTS_ID`` in the PKCs or the DCCs,
          counts as deposited. A probe in no PKC is kept as ``"Unannotated Probe"``. A probe listed twice in one DCC
          is kept as ``<RTS_ID>_dup1``, ``_dup2``, ... (counted in ``uns["spatialdata_io"]["duplicate_probes"]``).
        - ``obs``: every ``annotation`` column, values verbatim; names that are not valid SpatialData names are
          made valid by :func:`spatialdata.sanitize_name` (pairs in ``uns["spatialdata_io"]["renamed_columns"]``), and every DCC ``Header``, ``Scan_Attributes`` and
          ``NGS_Processing_Attributes`` field as ``dcc_<field>``.
        - ``aois``: one circle per AOI at ``x``/``y``, radius ``sqrt(area / pi)`` with ``area``, else ``aoi_radius``,
          else 1. AOIs without coordinates (e.g. no-template controls) are kept in ``<table>_no_coordinates``.
          Without any ``x``/``y`` there is no shapes element and ``uns["spatialdata_io"]["no_coordinates"]``
          is ``True``. The coordinates are in pixels (``uns["spatialdata_io"]["coordinate_unit"]``), in coordinate
          system ``"global"`` when ``coordinates_in_image``, else in their own system ``"aoi_coordinates"``.
        - ``slide_image``: the ``image`` OME-TIFF, channels named from its OME-XML, in ``"global"``; scaled to µm
          when the OME-XML gives a physical pixel size (then the AOIs too, if ``coordinates_in_image``).
        - ``layers``: each other matrix (e.g. normalised values), as deposited, as a layer of ``table`` when it has
          the same AOIs and features, else as its own table ``table_<name>``. Without ``counts`` or DCCs the table
          has no ``X`` and ``uns["spatialdata_io"]["no_raw_counts"]`` is ``True``; values are never back-computed.
        - ``table_annotation_unmatched``: the annotation rows that match no AOI, all columns.

    Parameters
    ----------
    dcc
        DCC files, or a folder of them.
    pkc
        PKC file(s); required with ``dcc``, optional for a matrix (it then types the targets).
    annotation
        AOI annotation: a GeoMx lab worksheet (``*_LabWorksheet.txt``, read from the line after ``Annotations``),
        an ``.xlsx`` sheet ``annotation_sheet``, or a delimited table.
    counts
        Raw counts matrix (see above).
    layers
        Other matrices by layer name, read like ``counts``.
    image
        Slide OME-TIFF.
    annotation_sheet, matrix_sheet
        Sheet name or index of an ``.xlsx`` annotation and matrix.
    matrix_sep, decimal
        Delimiter (default: by suffix) and decimal mark of delimited matrices.
    probe_level, target_column, feature_columns
        Layout of a matrix (see above).
    aoi_id
        The annotation column that holds the AOI names (DCC stems, DCC ``Scan_Attributes`` ``ID`` or matrix columns),
        or a :meth:`str.format` template of annotation columns, e.g. ``"{Scan name} | {ROI (label):03d} | {Segment}"``.
        Default: ``Sample_ID`` if present, else the first column.
    x, y, area
        Annotation columns with the AOI centre and area.
    aoi_radius
        AOI radius in coordinate units, without ``area``.
    coordinates_in_image
        Whether ``x``/``y`` are pixels of ``image``.
    segment_map
        Segment names of the AOIs mapped to those of the annotation, e.g. ``{"Desmin": "Cardiomyocyte"}``, for AOI
        names in the DSP form ``<scan> | <ROI> | <segment>`` whose segment the annotation names differently. The map
        is applied to the AOI names only to join the annotation; the AOIs keep their deposited names, their deposited
        segment is in ``obs["segment_deposited"]`` and the map in ``uns["spatialdata_io"]["segment_map"]``.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    if (dcc is None) == (counts is None and not layers):
        raise ValueError("Pass `dcc` (with `pkc`), or `counts` and/or `layers`.")
    if dcc is not None and pkc is None:
        raise ValueError("`dcc` needs `pkc`.")
    meta: dict[str, Any] = {}
    pkc_var, pkc_headers = _read_pkcs(pkc) if pkc is not None else (None, [])
    if pkc_headers:
        meta["pkc"] = {h["file"]: {k: "" if v is None else str(v) for k, v in h.items()} for h in pkc_headers}

    probe: ad.AnnData | None = None
    if dcc is not None:
        assert pkc_var is not None
        probe, dcc_obs, dups = _read_dccs(dcc, pkc_var)
        meta["duplicate_probes"] = dups
        obs = dcc_obs
        target = _sum_targets(probe)
    elif counts is not None:
        df, fvar = _read_matrix(counts, matrix_sheet, matrix_sep, decimal, feature_columns)
        obs = pd.DataFrame(index=df.columns.astype(str))
        if probe_level:
            if target_column not in fvar.columns:
                raise ValueError(f"`target_column` {target_column!r} is not among `feature_columns`.")
            pvar = _annotate_probes(fvar.rename(columns={target_column: "Target"}), pkc_var)
            probe = ad.AnnData(sp.csr_matrix(df.T.to_numpy()), obs=obs.copy(), var=pvar)
            target = _sum_targets(probe)
        else:
            target = ad.AnnData(sp.csr_matrix(df.T.to_numpy()), obs=obs.copy(), var=_annotate_targets(fvar, pkc_var))
    else:
        assert layers is not None
        name, path = next(iter(layers.items()))
        df, fvar = _read_matrix(path, matrix_sheet, matrix_sep, decimal, feature_columns)
        obs = pd.DataFrame(index=df.columns.astype(str))
        target = ad.AnnData(X=None, obs=obs.copy(), var=_annotate_targets(fvar, pkc_var))
        meta["no_raw_counts"] = True
    tvar = cast(pd.DataFrame, target.var)
    if "CodeClass" not in tvar or tvar["CodeClass"].isna().any():
        n_neg = _negprobe_rule(tvar, pd.Series(tvar.index, index=tvar.index))
        meta["negprobe_rule"] = {
            "prefix": _NEGPROBE_PREFIX,
            "applies_to": "targets that no PKC or deposited CodeClass types",
            "n_targets": n_neg,
        }
        if probe is not None:
            pvar = cast(pd.DataFrame, probe.var)
            _negprobe_rule(pvar, pvar["Target"])
    if target.X is not None:
        values = cast(sp.csr_matrix, target.X).data if sp.issparse(target.X) else np.asarray(target.X)
        meta["counts_integer"] = bool(np.all(np.mod(values, 1) == 0))
        if not meta["counts_integer"]:
            logger.warning("The counts are not all integers; they are stored as deposited.")

    tables: dict[str, ad.AnnData] = {}
    for name, path in (layers or {}).items():
        df, fvar = _read_matrix(path, matrix_sheet, matrix_sep, decimal, feature_columns)
        same = set(df.columns.astype(str)) == set(target.obs_names) and set(df.index.astype(str)) == set(
            target.var_names
        )
        if same:
            target.layers[name] = df.loc[target.var_names, target.obs_names].T.to_numpy(dtype=float)
        else:
            logger.info(f"Layer {name!r} has other AOIs or features than the counts; stored as 'table_{name}'.")
            extra = ad.AnnData(
                df.T.to_numpy(dtype=float),
                obs=pd.DataFrame(index=df.columns.astype(str)),
                var=_annotate_targets(fvar, pkc_var),
            )
            evar = cast(pd.DataFrame, extra.var)
            _negprobe_rule(evar, pd.Series(evar.index, index=evar.index))
            extra.uns["spatialdata_io"] = {"values": name}
            tables[f"table_{name}"] = extra

    renamed: list[list[str]] = []
    if segment_map is not None and annotation is None:
        raise ValueError("`segment_map` needs `annotation`.")
    if annotation is not None:
        ann = _read_annotation(Path(annotation), annotation_sheet, meta)
        keys = None
        if segment_map is not None:
            keys, segments = _map_segments(obs.index, segment_map)
            obs = obs.assign(segment_deposited=np.asarray(segments, dtype=object))
            meta["segment_map"] = {"map": dict(segment_map), "n_aois": int((keys != obs.index).sum())}
        obs, unmatched, join = _join_annotation(obs, ann, aoi_id, keys)
        meta["annotation_join"] = join
        if len(unmatched):
            tables["table_annotation_unmatched"] = ad.AnnData(
                np.zeros((len(unmatched), 0)), obs=_writable(unmatched, renamed)
            )
    obs = _writable(obs, renamed)
    if renamed:
        meta["renamed_columns"] = renamed

    shapes = {}
    images = {}
    meta["coordinate_unit"] = "px"
    cs = "global" if coordinates_in_image else "aoi_coordinates"
    um_per_px = None
    if image is not None:
        images["slide_image"], um_per_px = _read_image(Path(image))
        if um_per_px is not None:
            meta["um_per_px"] = um_per_px
    located = np.zeros(len(obs), dtype=bool)
    if x is not None and y is not None:
        renamed_map = dict(renamed)
        xy = obs[[renamed_map.get(x, x), renamed_map.get(y, y)]].to_numpy(dtype=float)
        if area is not None:
            radius = np.sqrt(obs[renamed_map.get(area, area)].to_numpy(dtype=float) / np.pi)
            meta["radius_source"] = f"derived:{area}"
        else:
            radius = np.full(len(obs), aoi_radius if aoi_radius is not None else 1.0)
            meta["radius_source"] = "stated" if aoi_radius is not None else "default"
        located = np.isfinite(xy).all(axis=1) & np.isfinite(radius)
    if located.any():
        transform = Scale([um_per_px, um_per_px], axes=("x", "y")) if coordinates_in_image and um_per_px else Identity()
        circles = gpd.GeoDataFrame(
            {"geometry": [Point(px, py) for px, py in xy[located]], "radius": radius[located]},
            index=np.arange(located.sum()),
        )
        shapes[_SHAPES] = ShapesModel.parse(circles, transformations={cs: transform})
        meta["coordinate_system"] = cs
        if n := int((~located).sum()):  # e.g. no-template controls without an annotation row
            logger.warning(f"{n} AOI(s) have no coordinates; stored in '<table>_no_coordinates'.")
            meta["n_no_coordinates"] = n
    else:
        meta["no_coordinates"] = True

    main: dict[str, ad.AnnData] = {}
    prot = (target.var["analyte"] == "Protein").to_numpy()
    if not prot.all():
        main["table"] = target if not prot.any() else target[:, ~prot].copy()
    if prot.any():  # protein kits go to their own table, as in the other readers
        main["table_protein"] = target[:, prot].copy()
    if probe is not None:
        main["table_probe"] = probe
    pos = pd.Series(np.arange(len(obs)), index=obs.index)
    for key, table in main.items():
        table.obs = obs.loc[table.obs_names].copy()
        table.var = _writable(cast(pd.DataFrame, table.var), [])
        table.uns["spatialdata_io"] = dict(meta)
        _set_observation_unit(table, "region", None)
        if not shapes:
            tables[key] = TableModel.parse(table)
            continue
        here = located[pos.loc[table.obs_names].to_numpy()]
        if not here.all():
            tables[f"{key}_no_coordinates"] = TableModel.parse(table[~here].copy())
            table = table[here].copy()
        table.obs["region"] = pd.Categorical([_SHAPES] * table.n_obs)
        table.obs["instance_id"] = np.arange(table.n_obs)
        tables[key] = TableModel.parse(table, region=_SHAPES, region_key="region", instance_key="instance_id")
    sdata = SpatialData(shapes=shapes, images=images, tables=tables)
    return _set_reader_metadata(sdata, "geomx")


def _open_text(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as f:  # type: ignore[operator]
        return f.read().splitlines()


def _read_pkcs(paths: str | Path | Sequence[str | Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Probe table indexed by ``RTS_ID`` from one or more PKC files, plus their headers."""
    paths = [paths] if isinstance(paths, str | Path) else list(paths)
    rows, headers = [], []
    for p in map(Path, paths):
        d = json.loads("\n".join(_open_text(p)))
        module = re.sub(r"_[^_]+$", "", p.name.removesuffix(".gz").removesuffix(".pkc"))  # GeomxTools' PKCModule
        headers.append(
            {"file": p.name, "module": module, **{k: v for k, v in d.items() if not isinstance(v, list | dict)}}
        )
        analyte = d.get("AnalyteType", "RNA")
        for t in d.get("Targets", []):
            for prb in t.get("Probes", []):
                # protein kits carry the RTS_ID on the target (readPKCFile.R)
                src = t if analyte == "Protein" else prb
                rows.append(
                    {
                        "RTS_ID": str(src.get("RTS_ID", "")).replace("RNA", "RTS00"),
                        "Target": t["DisplayName"],
                        "Module": module,
                        "CodeClass": t.get("CodeClass", ""),
                        "ProbeID": prb.get("ProbeID"),
                        "GeneID": ", ".join(map(str, src.get("GeneID", []))) or None,
                        "SystematicName": ", ".join(map(str, src.get("SystematicName", []))) or None,
                        "analyte": t.get("AnalyteType", analyte),
                    }
                )
    df = pd.DataFrame(rows)
    # the same probe in several PKCs (e.g. kit versions): one row, all modules named
    modules = df.groupby("RTS_ID", sort=False)["Module"].agg(lambda m: ", ".join(dict.fromkeys(m)))
    df = df.drop_duplicates("RTS_ID").set_index("RTS_ID")
    df["Module"] = modules.loc[df.index]
    return df, headers


def _types(code_class: pd.Series, analyte: pd.Series) -> pd.DataFrame:
    negative = code_class.fillna("").astype(str).str.contains("Negative").to_numpy()
    base = np.where(analyte.fillna("RNA").to_numpy() == "Protein", _PROTEIN, GENE_EXPRESSION)
    return pd.DataFrame(
        {FEATURE_TYPES: np.where(negative, _NEGATIVE, base), "is_control": negative}, index=code_class.index
    )


def _negprobe_rule(var: pd.DataFrame, names: pd.Series) -> int:
    """Type the targets that nothing typed and whose name starts with ``NegProbe-`` as negative controls, in place.

    Returns how many it typed. A probe ``var`` passes its ``Target`` column as ``names``.
    """
    untyped = var["CodeClass"].isna() if "CodeClass" in var else pd.Series(True, index=var.index)
    hit = (untyped & names.astype("string").str.startswith(_NEGPROBE_PREFIX).fillna(False)).to_numpy(dtype=bool)
    var.loc[hit, FEATURE_TYPES] = _NEGATIVE
    var.loc[hit, "is_control"] = True
    return int(hit.sum())


def _map_segments(names: pd.Index, segment_map: Mapping[str, str]) -> tuple[pd.Index, list[str | None]]:
    """AOI names with their segment (after the last ``" | "``) mapped, and the deposited segments."""
    keys, segments = [], []
    for name in map(str, names):
        head, sep, seg = name.rpartition(" | ")
        segments.append(seg if sep else None)
        keys.append(f"{head}{sep}{segment_map[seg]}" if sep and seg in segment_map else name)
    if keys == list(map(str, names)):
        raise ValueError(
            f"`segment_map` keys {sorted(segment_map)[:5]} name no AOI segment (`<scan> | <ROI> | <segment>`)."
        )
    return pd.Index(keys), segments


def _annotate_probes(var: pd.DataFrame, pkc: pd.DataFrame | None) -> pd.DataFrame:
    """``var`` of a probe table: the PKC fields where known, else ``Unannotated Probe``."""
    var = var.copy()
    var.index = var.index.astype(str)
    base = var.index.str.replace(r"_dup\d+$", "", regex=True)
    if pkc is not None:
        known = pkc.reindex(base)
        known.index = var.index
        for col in pkc.columns:
            if col in var.columns:
                var[col] = var[col].where(var[col].notna(), known[col])
            else:
                var[col] = known[col]
    if "analyte" not in var.columns and "AnalyteType" in var.columns:
        var["analyte"] = var["AnalyteType"]
    for col in ("Target", "CodeClass", "analyte"):
        if col not in var.columns:
            var[col] = None
    var = var.join(_types(var["CodeClass"], var["analyte"]))
    var.loc[var["Target"].isna(), FEATURE_TYPES] = _UNANNOTATED
    return var


def _annotate_targets(var: pd.DataFrame, pkc: pd.DataFrame | None) -> pd.DataFrame:
    """``var`` of a target table, typed from the PKC (or a ``CodeClass``/``AnalyteType`` column if deposited)."""
    var = var.copy()
    var.index = var.index.astype(str)
    if pkc is not None:
        t = pkc.groupby("Target", sort=False).agg(
            CodeClass=("CodeClass", "first"), Module=("Module", "first"), analyte=("analyte", "first")
        )
        for col in t.columns:
            if col not in var.columns:
                var[col] = t[col].reindex(var.index).to_numpy()
    code = var["CodeClass"] if "CodeClass" in var else pd.Series(None, index=var.index, dtype=object)
    analyte = var.get("analyte", var.get("AnalyteType", pd.Series(None, index=var.index, dtype=object)))
    var["analyte"] = analyte
    return var.join(_types(code, analyte))


def _read_dccs(
    dcc: str | Path | Sequence[str | Path], pkc: pd.DataFrame
) -> tuple[ad.AnnData, pd.DataFrame, dict[str, int]]:
    if isinstance(dcc, str | Path) and Path(dcc).is_dir():
        files = sorted(p for p in Path(dcc).iterdir() if p.name.endswith((".dcc", ".dcc.gz")))
    else:
        files = [Path(dcc)] if isinstance(dcc, str | Path) else [Path(p) for p in dcc]
    if not files:
        raise ValueError(f"No DCC files in {dcc}.")
    names = [p.name.removesuffix(".gz").removesuffix(".dcc") for p in files]
    if len(set(names)) != len(names):
        raise ValueError("Two DCC files have the same name.")
    obs_rows, counts, dups = [], [], {}
    for name, p in zip(names, files, strict=True):
        fields, code, n_dup = _parse_dcc(p)
        obs_rows.append(fields)
        counts.append(code)
        if n_dup:
            dups[name] = n_dup
    # every probe of the PKCs, then those only seen in DCCs, in order of appearance
    ids = list(pkc.index) + [r for r in dict.fromkeys(r for c in counts for r in c) if r not in pkc.index]
    col = {r: i for i, r in enumerate(ids)}
    rows, cols, vals = [], [], []
    for i, c in enumerate(counts):
        for r, v in c.items():
            rows.append(i)
            cols.append(col[r])
            vals.append(v)
    values = np.asarray(vals, dtype=float)
    dtype = np.int64 if np.all(np.mod(values, 1) == 0) else np.float64
    X = sp.csr_matrix((values.astype(dtype), (rows, cols)), shape=(len(files), len(ids)))
    var = _annotate_probes(pd.DataFrame(index=pd.Index(ids, name=None)), pkc)
    obs = pd.DataFrame(obs_rows, index=names)
    for field in obs.columns:
        if field.removeprefix("dcc_") in _DCC_NUMERIC:
            obs[field] = pd.to_numeric(obs[field], errors="coerce")
    return ad.AnnData(X, obs=pd.DataFrame(index=names), var=var), obs, dups


def _parse_dcc(path: Path) -> tuple[dict[str, str], dict[str, float], int]:
    """Fields (``dcc_<field>``), probe counts (duplicates as ``_dupN``) and the number of duplicates of one DCC."""
    fields: dict[str, str] = {}
    counts: dict[str, float] = {}
    section, n_dup, other = None, 0, []
    for line in _open_text(path):
        line = line.strip()
        if m := _SECTION.match(line):
            section = None if m[1] else m[2]
            continue
        if not line:
            continue
        if section == "Code_Summary":
            rts, value = line.split(",", 1)
            key, k = rts, 0
            while key in counts:
                k += 1
                key = f"{rts}_dup{k}"
            n_dup += k > 0
            counts[key] = float(value)
        elif section in _DCC_SECTIONS:
            parts = next(csv.reader([line]))
            if len(parts) == 1 and fields:  # continuation of the previous value (as GeomxTools)
                last = next(reversed(fields))
                fields[last] += " " + parts[0]
            else:
                fields[f"dcc_{parts[0]}"] = ",".join(parts[1:])
        else:
            other.append(line)
    if other:
        fields["dcc_unsectioned"] = "\n".join(other)
    return fields, counts, n_dup


def _sum_targets(probe: ad.AnnData) -> ad.AnnData:
    """AOIs x targets: the sum of the probe counts of each target (unannotated probes have no target)."""
    var = cast(pd.DataFrame, probe.var)
    counts = cast(sp.csr_matrix, probe.X)
    has = var["Target"].notna().to_numpy()
    key = var["Target"].astype(str) + "\x00" + var["Module"].astype(str) if "Module" in var else var["Target"]
    codes, uniq = pd.factorize(key[has])
    agg = sp.csr_matrix((np.ones(has.sum()), (np.flatnonzero(has), codes)), shape=(probe.n_vars, len(uniq)))
    X = sp.csr_matrix(counts @ agg)
    if np.issubdtype(counts.dtype, np.integer):
        X = X.astype(counts.dtype)
    first = var[has].groupby(codes, sort=True).first()
    names = first["Target"].astype(str)
    if names.duplicated().any():  # the same target name in two kits
        dup = names.duplicated(keep=False)
        names[dup] = names[dup] + " (" + first.loc[dup, "Module"].astype(str) + ")"
    tvar = pd.DataFrame(
        {
            "CodeClass": first["CodeClass"].to_numpy(),
            "Module": first["Module"].to_numpy() if "Module" in first else None,
            "analyte": first["analyte"].to_numpy(),
            "n_probes": np.bincount(codes, minlength=len(uniq)),
        },
        index=pd.Index(names.to_numpy(), dtype=str),
    )
    tvar = tvar.join(_types(tvar["CodeClass"], tvar["analyte"]))
    return ad.AnnData(X, obs=pd.DataFrame(index=probe.obs_names), var=tvar)


def _read_table(path: Path, sheet: str | int, sep: str | None, decimal: str, **kwargs: Any) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet, **kwargs)
    if sep is None:
        sfx = [s.lower() for s in path.suffixes if s.lower() != ".gz"]
        sep = "," if sfx and sfx[-1] == ".csv" else "\t"
    return pd.read_csv(path, sep=sep, decimal=decimal, **kwargs)


def _read_matrix(
    path: str | Path, sheet: str | int, sep: str | None, decimal: str, feature_columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features x AOIs values and the ``var`` (feature metadata columns) of one matrix file."""
    df = _read_table(Path(path), sheet, sep, decimal, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    fcols = [c for c in feature_columns if c in df.columns]
    var = df[fcols].copy()
    values = df.drop(columns=fcols).apply(pd.to_numeric)
    if values.index.duplicated().any():
        raise ValueError(f"{Path(path).name}: feature names are not unique.")
    return values, var


def _read_annotation(path: Path, sheet: str | int, meta: dict[str, Any]) -> pd.DataFrame:
    name = path.name.lower().removesuffix(".gz")
    if name.endswith("_labworksheet.txt"):
        lines = _open_text(path)
        start = next(i for i, line in enumerate(lines) if line.startswith("Annotations"))
        meta["worksheet_header"] = [line for line in lines[:start] if line.strip()]
        return pd.read_csv(path, sep="\t", skiprows=start + 1, dtype=str, keep_default_na=False).replace("", np.nan)
    return _read_table(path, sheet, None, ".")


def _join_annotation(
    obs: pd.DataFrame, ann: pd.DataFrame, aoi_id: str | None, names: pd.Index | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """``obs`` with the annotation columns, the annotation rows that match no AOI, and how they were joined.

    ``names`` replaces the AOI names for matching (e.g. with mapped segments); ``obs`` keeps its index.
    """
    if aoi_id is None:
        aoi_id = "Sample_ID" if "Sample_ID" in ann.columns else str(ann.columns[0])
    if "{" in aoi_id:
        keys = pd.Series([aoi_id.format_map(r) for r in ann.to_dict("records")], index=ann.index)
    elif aoi_id in ann.columns:
        keys = ann[aoi_id]
    else:
        raise ValueError(f"`aoi_id` {aoi_id!r} is neither an annotation column nor a format template.")
    keys = keys.astype(str)
    if keys.duplicated().any():
        raise ValueError(f"AOI names from {aoi_id!r} are not unique: {sorted(set(keys[keys.duplicated()]))[:5]}")
    # match the DCC stems, or else the DCC Scan_Attributes ID (e.g. GEO files named GSM..._<ID>.dcc)
    candidates = {"name": pd.Series(obs.index if names is None else names, index=obs.index)}
    if "dcc_ID" in obs.columns:
        candidates["dcc_ID"] = obs["dcc_ID"].astype(str)
    by, ids = max(candidates.items(), key=lambda kv: kv[1].isin(set(keys)).sum())
    pos = pd.Series(np.arange(len(ann)), index=keys.to_numpy())
    matched = ids[ids.isin(pos.index)]
    rows = ann.iloc[pos.loc[matched.to_numpy()].to_numpy()].set_axis(matched.index)
    clash = [c for c in ann.columns if c in obs.columns]
    out = obs.join(rows.rename(columns={c: f"annotation_{c}" for c in clash}))
    taken = set(matched.to_numpy())
    unmatched = ann[~keys.isin(taken).to_numpy()].set_axis(keys[~keys.isin(taken)].to_numpy())
    join = {"aoi_id": aoi_id, "matched_on": by, "n_matched": len(matched), "n_unmatched": len(unmatched)}
    if len(matched) < len(obs):
        logger.warning(f"{len(obs) - len(matched)} AOI(s) have no annotation row.")
    return out, unmatched, join


def _writable(df: pd.DataFrame, renamed: list[list[str]]) -> pd.DataFrame:
    """``df`` with valid, distinct column names and object columns as strings, so it can be written.

    Names are made valid by :func:`spatialdata.sanitize_name`; a name that equals another one or ``region`` /
    ``instance_id`` (ignoring case) gets ``_`` appended.

    Each renamed column is appended to ``renamed`` as ``[deposited name, stored name]``.
    """
    df = df.copy()
    cols: list[str] = []
    taken = {"region", "instance_id"}  # the table keys; SpatialData compares names case-insensitively
    for c in df.columns:
        new = sanitize_name(str(c))
        while new.lower() in taken:
            new += "_"
        taken.add(new.lower())
        if new != str(c) and [str(c), new] not in renamed:
            renamed.append([str(c), new])
        cols.append(new)
    df.columns = cols
    df.index = df.index.astype(str)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
            if df[c].isna().all():
                df[c] = df[c].astype(float)
    return df


def _read_image(path: Path) -> tuple[Any, float | None]:
    """The slide image (c, y, x) with OME channel names, and its µm per pixel if the OME-XML gives it."""
    import dask.array as da
    import tifffile
    import zarr
    from ome_types import from_xml

    with tifffile.TiffFile(path) as tf:
        ome = tf.ome_metadata
        axes = tf.series[0].axes
    arr = da.from_zarr(zarr.open(tifffile.imread(path, aszarr=True, level=0), mode="r"))
    for i in reversed(range(len(axes))):  # drop singleton time/z axes
        if axes[i] not in "CSYX" and arr.shape[i] == 1:
            arr, axes = arr[(slice(None),) * i + (0,)], axes[:i] + axes[i + 1 :]
    axes = axes.replace("S", "C")
    if "C" not in axes:
        arr, axes = arr[None], "C" + axes
    if sorted(axes) != ["C", "X", "Y"]:
        raise ValueError(f"{path.name}: unsupported image axes {tf.series[0].axes!r}.")
    arr = arr.transpose(*(axes.index(a) for a in "CYX"))
    channels: list[str] | None = None
    um = None
    if ome:
        pixels = from_xml(ome).images[0].pixels
        names = [c.name for c in pixels.channels]
        if len(names) == arr.shape[0] and all(names) and len(set(names)) == len(names):
            channels = cast(list[str], names)
        if pixels.physical_size_x is not None and pixels.physical_size_x == pixels.physical_size_y:
            unit = getattr(pixels.physical_size_x_unit, "value", "µm")
            factor = {"µm": 1.0, "nm": 1e-3, "mm": 1e3}.get(unit)
            um = None if factor is None else float(pixels.physical_size_x) * factor
    transform = Scale([um, um], axes=("x", "y")) if um else Identity()
    img = Image2DModel.parse(arr, dims=("c", "y", "x"), c_coords=channels, transformations={"global": transform})
    return img, um
