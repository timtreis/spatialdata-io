"""FOV orientation of CosMx exports: placing mm-only FOV positions, and the seam test that checks a placement.

An mm-only ``fov_positions`` file gives each FOV's stage position, but not which way the local px run in a global
frame. The export's own global px do: per FOV and axis, ``CenterX/Y_global_px`` in the metadata (and ``x/y_global_px``
in the transcripts) are ``offset + sign * local px`` with ``sign`` +1 or -1, and a seam test shows that this frame is
the one where tissue continues across FOV edges (spatialdata-io-sddb
``tasks/2026-11-13-cosmx-frame-gsm9351707-report.md``, sections 1 and 4).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from spatialdata._logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ._discovery import CosMxDataset

# Largest half-range of ``global - sign * local`` px within one FOV and axis that still counts as a pure translation.
# Job 43237132 (_verify/sddbio/cosmx_orientation/tol_*.json, 16 exports) measured, on the right sign, 0.0 px for the
# metadata CenterX/Y in every export and up to 1.0 px for the transcripts (GSM9309756 FALD4; 0.51 px elsewhere), and on
# the wrong sign at least 146 px (GSM9351707 Leuven_4 transcripts, x) on every FOV that has two distinct local px.
ORIENTATION_TOL_PX = 2.0

_METADATA_COLS = ("CenterX_global_px", "CenterY_global_px", "CenterX_local_px", "CenterY_local_px")
_TX_COLS = ("x_global_px", "y_global_px", "x_local_px", "y_local_px")


def _positions_kind(csv_path: Path) -> str:
    """``"px+mm"``, ``"px"`` or ``"mm"``: the coordinate columns of a FOV positions file.

    By the rule of :func:`._io._read_fov_locs`: a column name containing ``px`` / ``mm``.
    """
    cols = [c.lower() for c in pd.read_csv(csv_path, nrows=0).columns]
    has_px, has_mm = any("px" in c for c in cols), any("mm" in c for c in cols)
    return "px+mm" if has_px and has_mm else "px" if has_px else "mm"


def _columns(path: Path, wanted: tuple[str, ...]) -> list[str] | None:
    """The file's fov column then *wanted*, matched case-insensitively, or ``None`` when one is missing."""
    cols = {c.lower(): c for c in pd.read_csv(path, nrows=0).columns}
    found = [cols.get(c.lower()) for c in ("fov", *wanted)]
    return None if any(c is None for c in found) else found  # type: ignore[return-value]


def _csv_chunks(path: Path, cols: list[str]) -> Iterator[pd.DataFrame]:
    """*cols* of a CSV (or .csv.gz), streamed in blocks; all but the first (the fov column) as float."""
    # floats throughout: a coordinate column can mix ints and "1e+05" (see _io._stream_csvgz_to_parquet)
    opts = pacsv.ConvertOptions(include_columns=cols, column_types={c: pa.float64() for c in cols[1:]})
    stream = pacsv.open_csv(pa.input_stream(str(path)), convert_options=opts)
    for batch in stream:
        yield batch.to_pandas()


def _offset_ranges(path: Path, cols: list[str], fovs: set[int] | None = None) -> pd.DataFrame:
    """Per FOV, the min and max of ``global - local`` (``p``) and ``global + local`` (``m``) on each axis."""
    fc, gx, gy, lx, ly = cols
    parts = []
    for ch in _csv_chunks(path, cols):
        ch = ch.dropna()
        if fovs is not None:
            ch = ch[ch[fc].isin(fovs)]
        f = ch[fc].astype(int).rename("fov")
        o = pd.DataFrame({"xp": ch[gx] - ch[lx], "xm": ch[gx] + ch[lx], "yp": ch[gy] - ch[ly], "ym": ch[gy] + ch[ly]})
        parts.append(o.groupby(f).agg(["min", "max"]))
    if not parts:
        return pd.DataFrame()
    r = pd.concat(parts)
    return r.groupby(level=0).agg({c: "min" if c[1] == "min" else "max" for c in r.columns})


def _signs(ranges: pd.DataFrame, source: str) -> dict[int, dict[str, tuple[int, float, float]]]:
    """Per FOV and axis ``(sign, offset for +1, offset for -1)``; sign 0 when both signs fit (one distinct local px)."""
    out: dict[int, dict[str, tuple[int, float, float]]] = {}
    bad = []
    for fov, row in ranges.iterrows():
        axes = {}
        for ax in ("x", "y"):
            lo_p, hi_p, lo_m, hi_m = (
                row[(f"{ax}p", "min")],
                row[(f"{ax}p", "max")],
                row[(f"{ax}m", "min")],
                row[(f"{ax}m", "max")],
            )
            ok_p, ok_m = (hi_p - lo_p) / 2 <= ORIENTATION_TOL_PX, (hi_m - lo_m) / 2 <= ORIENTATION_TOL_PX
            if not ok_p and not ok_m:
                bad.append(
                    f"FOV {fov} {ax}: global - local spans {hi_p - lo_p:.1f} px, global + local {hi_m - lo_m:.1f} px"
                )
            axes[ax] = (
                1 if ok_p and not ok_m else -1 if ok_m and not ok_p else 0,
                (lo_p + hi_p) / 2,
                (lo_m + hi_m) / 2,
            )
        out[int(fov)] = axes  # type: ignore[call-overload]
    if bad:
        raise ValueError(
            f"{source}: the global px are not a translation of the local px with a sign of +1 or -1 within "
            f"{ORIENTATION_TOL_PX} px in {len(bad)} FOV axis/axes ({'; '.join(bad[:3])}), so the FOV orientation "
            "cannot be taken from them."
        )
    return out


def _export_frame(
    fov_locs: pd.DataFrame, datasets: list[CosMxDataset]
) -> tuple[dict[int, dict[str, tuple[int, float, float]]], list[str]] | None:
    """Per FOV and axis ``(sign, offset +1, offset -1)``, and the columns used.

    From the metadata, then the transcripts for FOVs whose sign the metadata leaves open; ``None`` when neither has
    global and local px.
    """
    fovs = {int(f) for f in fov_locs.index}
    frame: dict[int, dict[str, tuple[int, float, float]]] = {}
    used: list[str] = []
    for attr, wanted in (("metadata_file", _METADATA_COLS), ("tx_file", _TX_COLS)):
        path = next((getattr(ds, attr) for ds in datasets if getattr(ds, attr) is not None), None)
        open_fovs = {f for f in fovs if f not in frame or any(frame[f][a][0] == 0 for a in ("x", "y"))}
        if path is None or not open_fovs or (cols := _columns(path, wanted)) is None:
            continue
        found = _signs(_offset_ranges(path, cols, open_fovs), path.name)
        for f, axes in found.items():
            if (prev := frame.get(f)) is None:
                frame[f] = axes
                continue
            # the metadata keeps its offsets (cell centres are exact); the transcripts only settle an open sign
            if clash := [a for a in ("x", "y") if prev[a][0] * axes[a][0] < 0]:
                raise ValueError(
                    f"CosMx FOV orientation: FOV {f} has opposite {clash} signs in {used[0]} and {path.name}."
                )
            frame[f] = {a: (prev[a][0] or axes[a][0], *prev[a][1:]) for a in ("x", "y")}
        used.append(f"{path.name} {'/'.join(cols[1:3])} - {'/'.join(cols[3:5])}")
    return (frame, used) if frame else None


def _orient_mm_only(
    fov_locs: pd.DataFrame, frame: dict[int, dict[str, tuple[int, float, float]]], fov_size: tuple[float, float]
) -> tuple[pd.DataFrame, list[str]]:
    """Place each FOV at its export offset and sign (``xmax`` / ``ymax`` are left to the caller).

    Returns the FOV table and notes on the FOVs that the export does not resolve by themselves.
    """
    notes = []
    sign: dict[str, dict[int, int]] = {}
    off: dict[str, dict[int, float]] = {}
    for ax in ("x", "y"):
        resolved = {f: a[ax][0] for f, a in frame.items() if a[ax][0] != 0}
        run = set(resolved.values())
        open_fovs = sorted(f for f in fov_locs.index if int(f) not in resolved)
        if open_fovs and len(run) != 1:
            raise ValueError(
                f"CosMx FOV orientation: FOV(s) {open_fovs[:5]} have no {ax} sign of their own (one distinct local px "
                f"or no cells/transcripts) and the other FOVs do not share one sign ({sorted(run)})."
            )
        s_run = next(iter(run)) if run else 0
        sign[ax] = {int(f): resolved.get(int(f), s_run) for f in fov_locs.index}
        off[ax] = {f: a[ax][1] if sign[ax][f] > 0 else a[ax][2] for f, a in frame.items() if f in sign[ax]}
        if one := [f for f in open_fovs if int(f) in frame]:
            notes.append(f"{len(one)} FOV(s) with one distinct local {ax} px take the run's {ax} sign {s_run:+d}")
        if none := [int(f) for f in open_fovs if int(f) not in frame]:
            off[ax].update(_fit_offsets(fov_locs, off[ax], none, ax))
            notes.append(
                f"{len(none)} FOV(s) without cells or transcripts are placed in {ax} from the fit of the other FOVs' "
                "offsets against fov_positions"
            )
    if mirrored := sorted(f for f, s in sign["x"].items() if s < 0):
        raise ValueError(
            f"CosMx FOV orientation: in FOV(s) {mirrored[:5]} the export's global x runs against the local x px; the "
            "reader places local x with a sign of +1 only."
        )
    h = fov_size[1]
    out = fov_locs.copy()
    out["xmin"] = [off["x"][int(f)] for f in out.index]
    # the px+mm placement (_io.py, flip_y=False): y = ymin + (height - y_local), so ymin = offset - height
    out["ymin"] = [off["y"][int(f)] - (0.0 if sign["y"][int(f)] > 0 else float(h)) for f in out.index]
    out["flip_y"] = [sign["y"][int(f)] > 0 for f in out.index]
    out["frame"] = "export_global_px"  # tells the CellStatsDir stitcher to place with this table, not the file's
    return out, notes


def _fit_offsets(fov_locs: pd.DataFrame, known: dict[int, float], todo: list[int], ax: str) -> dict[int, float]:
    """Offsets of FOVs without data: ``a + b * stage px`` with ``b`` +1 or -1, fit on the FOVs with data."""
    col = "xmin" if ax == "x" else "ymin"  # mm-only: X_mm / Y_mm * MM_TO_PX (_io.py, _read_fov_locs)
    stage = fov_locs[col].astype(float)
    k = np.array(sorted(known))
    if len(k) < 2:
        raise ValueError(f"CosMx FOV orientation: FOV(s) {todo[:5]} have no data and fewer than 2 FOVs fit their {ax}.")
    o, s = np.array([known[f] for f in k]), stage.loc[k].to_numpy()
    fits = {b: np.median(o - b * s) for b in (1, -1)}
    b = min(fits, key=lambda b: np.abs(o - b * s - fits[b]).max())
    resid = float(np.abs(o - b * s - fits[b]).max())
    if resid > ORIENTATION_TOL_PX:
        raise ValueError(
            f"CosMx FOV orientation: FOV(s) {todo[:5]} have no data, and the other FOVs' {ax} offsets are not "
            f"fov_positions +- a constant (residual {resid:.1f} px > {ORIENTATION_TOL_PX} px)."
        )
    return {f: float(fits[b] + b * stage.loc[f]) for f in todo}


def _orient_fov_locs(
    fov_locs: pd.DataFrame,
    positions_file: Path,
    datasets: list[CosMxDataset],
    fov_size: tuple[float, float] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The FOV table to place with, and ``orientation`` / ``orientation_reason`` for ``sdata.attrs["cosmx"]``."""
    kind = _positions_kind(positions_file)
    if kind == "px+mm":
        return fov_locs, {
            "orientation": "fov_positions_px_mm",
            "orientation_reason": f"{positions_file.name} gives px and mm; FOVs placed at its px, local y flipped",
        }
    if kind == "px":
        return fov_locs, {
            "orientation": "fov_positions_px",
            "orientation_reason": f"{positions_file.name} gives px only; FOVs placed at its px, local px unflipped",
        }
    found = _export_frame(fov_locs, datasets)
    if found is None:
        reason = (
            f"{positions_file.name} gives mm only, and neither the metadata (CenterX/Y_global_px, CenterX/Y_local_px) "
            "nor the transcripts (x/y_global_px, x/y_local_px) give global px; local px placed with sign +1 on both "
            "axes against the mm positions, which nothing in the export confirms"
        )
        logger.warning("CosMx FOV orientation unverified: %s.", reason)
        return fov_locs, {"orientation": "unverified", "orientation_reason": reason}
    if fov_size is None:
        raise ValueError("CosMx FOV orientation: placing mm-only FOVs from the export's global px needs the FOV size.")
    frame, used = found
    out, notes = _orient_mm_only(fov_locs, frame, fov_size)
    signs = sorted({(1, 1 if f else -1) for f in out["flip_y"]})
    reason = (
        f"{positions_file.name} gives mm only; per-FOV offsets and local px signs (x, y) {signs} from "
        f"{'; then '.join(used)}"
    )
    if notes:
        reason += "; " + "; ".join(notes)
    logger.info("CosMx FOV orientation: %s.", reason)
    return out, {"orientation": "from_export_global_px", "orientation_reason": reason}


# ---------------------------------------------------------------------------
# seam test
# ---------------------------------------------------------------------------


def _seam_scores(
    origins: pd.DataFrame,
    points: dict[int, tuple[np.ndarray, np.ndarray]],
    fov_size: tuple[float, float],
    *,
    strip_px: float = 50.0,
    bin_px: float = 40.0,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Seam scores per axis and relative sign, from FOV origins (``X0``, ``Y0``) and per-FOV local px points."""
    w, h = fov_size
    fovs = [int(f) for f in origins.index if int(f) in points]
    rows = []
    for sx, sy in itertools.product((1, -1), (1, -1)):
        rect = {
            f: (
                (origins.X0[f], origins.X0[f] + w) if sx > 0 else (origins.X0[f] - w, origins.X0[f]),
                (origins.Y0[f], origins.Y0[f] + h) if sy > 0 else (origins.Y0[f] - h, origins.Y0[f]),
            )
            for f in fovs
        }
        glob = {f: (origins.X0[f] + sx * points[f][0], origins.Y0[f] + sy * points[f][1]) for f in fovs}
        for a, b in itertools.permutations(fovs, 2):
            for axis in (0, 1):  # 0: b right of a (shared vertical edge), 1: b below a
                e_a, e_b, size = rect[a][axis][1], rect[b][axis][0], (w, h)[axis]
                if abs(e_a - e_b) > 0.05 * size:
                    continue
                lo = max(rect[a][1 - axis][0], rect[b][1 - axis][0])
                hi = min(rect[a][1 - axis][1], rect[b][1 - axis][1])
                if hi - lo < 0.25 * (w, h)[1 - axis]:
                    continue
                bins = np.arange(lo, hi + bin_px, bin_px)
                ga, gb = glob[a], glob[b]
                in_a = (ga[axis] >= e_a - strip_px) & (ga[axis] <= e_a)
                in_b = (gb[axis] >= e_b) & (gb[axis] <= e_b + strip_px)
                pa = np.histogram(ga[1 - axis][in_a], bins)[0]
                pb = np.histogram(gb[1 - axis][in_b], bins)[0]
                if pa.sum() < 50 or pb.sum() < 50 or pa.std() == 0 or pb.std() == 0:
                    continue
                rows.append((sx, sy, axis, float(np.corrcoef(np.log1p(pa), np.log1p(pb))[0, 1])))
    r = pd.DataFrame(rows, columns=["sx", "sy", "axis", "r"])
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for axis, name, key in ((0, "x", "sx"), (1, "y", "sy")):
        t = r[r["axis"] == axis]
        out[name] = {
            f"{s:+d}": {"median_r": float(u["r"].median()) if len(u) else None, "n_pairs": int(len(u))}
            for s in (1, -1)
            for u in (t[t[key] == s],)
        }
    return out


def check_fov_orientation(
    path: str | Path,
    dataset_id: str | None = None,
    *,
    fov_size_px: tuple[int, int] | None = None,
    strip_px: float = 50.0,
    bin_px: float = 40.0,
) -> dict[str, Any]:
    """Seam test: which way the local px of a CosMx export run within each FOV, from tissue continuity.

    The FOVs are placed at their ``fov_positions`` origins ``(X0, Y0)`` (px columns, else mm times
    ``1000 / 0.120280945``, as the reader does) and each point at ``(X0 + sx * x_local, Y0 + sy * y_local)``, for
    the four ``(sx, sy)``. For every pair of FOVs whose boxes share an edge (within 5% of the FOV size, overlapping
    by at least 25% along it), the points within ``strip_px`` of the edge on each side are binned along the edge in
    ``bin_px`` bins, and the Pearson correlation of the two ``log1p`` count profiles is the pair's score (pairs with
    fewer than 50 points on a side are skipped). Tissue continues across a seam only in the right orientation, so
    the right sign scores clearly higher. Horizontal neighbours test ``sx`` and vertical neighbours test ``sy``;
    each axis pools the pairs over the other axis's sign.

    The points are the transcripts' local px near the FOV edges, else the polygon vertices. Ported from
    ``_verify/sddbio/cosmx_frame/seam_probe.py`` and its axis-resolved form ``frame_class.py`` (job 43236829).

    Parameters
    ----------
    path
        The CosMx export directory, as for :func:`spatialdata_io.cosmx`.
    dataset_id
        The run to read when the directory holds several.
    fov_size_px
        The FOV ``(width, height)`` in px; by default resolved as :func:`spatialdata_io.cosmx` does.
    strip_px, bin_px
        Strip depth and bin width in px.

    Returns
    -------
    ``{"x": {"+1": {"median_r", "n_pairs"}, "-1": {...}}, "y": {...}, "winner": {"x": +1 | -1 | None, "y": ...}}``.
    A sign is relative: ``+1`` means local px grow in the direction in which the ``fov_positions`` coordinate grows.
    ``winner`` is ``None`` for an axis without scored pairs for both signs.
    """
    from pathlib import Path as _Path

    from ._discovery import _set_up_cosmx_dataset_for_conversion
    from ._io import _read_fov_locs
    from ._reader import _fov_size_px

    dataset = _set_up_cosmx_dataset_for_conversion(path=_Path(path).resolve(), dataset_id=dataset_id)
    if dataset.fov_positions_file is None:
        raise ValueError("check_fov_orientation needs a FOV positions file.")
    w, h = (float(v) for v in _fov_size_px(dataset, fov_size_px)["fov_size_px"])
    locs = _read_fov_locs(dataset.fov_positions_file)
    origins = pd.DataFrame({"X0": locs["xmin"].astype(float), "Y0": locs["ymin"].astype(float)})

    points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if dataset.tx_file is not None and (cols := _columns(dataset.tx_file, ("x_local_px", "y_local_px"))) is not None:
        parts = []
        for ch in _csv_chunks(dataset.tx_file, cols):
            x, y = ch[cols[1]], ch[cols[2]]
            parts.append(ch[(x <= strip_px) | (x >= w - strip_px) | (y <= strip_px) | (y >= h - strip_px)])
        pts, source = pd.concat(parts).set_axis(["fov", "x", "y"], axis=1), dataset.tx_file.name
    elif (
        dataset.polygons_file is not None
        and (cols := _columns(dataset.polygons_file, ("x_local_px", "y_local_px"))) is not None
    ):
        pts = pd.read_csv(dataset.polygons_file, usecols=cols).set_axis(["fov", "x", "y"], axis=1)
        source = dataset.polygons_file.name
    else:
        raise ValueError("check_fov_orientation needs x_local_px / y_local_px in the transcripts or polygons file.")
    for f, t in pts.groupby("fov"):
        points[int(f)] = (t["x"].to_numpy(float), t["y"].to_numpy(float))  # type: ignore[call-overload]

    scores: dict[str, Any] = dict(_seam_scores(origins, points, (w, h), strip_px=strip_px, bin_px=bin_px))
    winner = {}
    for ax in ("x", "y"):
        p, m = scores[ax]["+1"]["median_r"], scores[ax]["-1"]["median_r"]
        winner[ax] = None if p is None or m is None else (1 if p > m else -1)
    scores["winner"] = winner
    scores["points"] = source
    return scores
