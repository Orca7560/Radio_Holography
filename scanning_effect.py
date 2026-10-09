#!/usr/bin/env python3
"""Estimate and correct scan-lag offsets in raster beam measurements.

Use either an SKD trajectory or beam.txt offsets with a supplied scan speed.
In SKD mode the summary and schedule are joined by timestamp. A trial
time lag tau changes each sample coordinate as

    theta_corrected = theta_commanded - velocity * tau.

By default, one scan direction is held fixed while the other direction's lag
is varied. This is repeated with the fixed direction reversed; the alignment
with the largest Pearson correlation coefficient of the amplitude maps is selected. These are alternative
relative alignments, not two independent absolute lags. --bidirectional-lags
enables the simultaneous two-dimensional search using the same criterion.
The Airy score in that mode is diagnostic only; it does not select the lag.

Usage:
  python scanning_effect.py beam.txt schedule.skd
  python scanning_effect.py beam.txt --skd schedule.skd
  python scanning_effect.py beam.txt --scan-speed 3 --scan-axis el
  python scanning_effect.py beam.txt schedule.skd --outdir I26184Y/scanning_result
  python scanning_effect.py beam.txt schedule.skd --max-lag-ms 500 --lag-step-ms 2 \
      --min-snr 5

Inputs:
  measurements  Whitespace- or comma-separated correlation summary with at
                least the columns Epoch, Amp, Phase, SNR (e.g. beam.txt
                produced by group_up_txt_prd.py). Epoch must parse as
                "%Y/%j %H:%M:%S.%f".
  schedule      Optional in --scan-speed mode; otherwise a .skd file
                containing a $SKED block with per-scan
                Az/El offsets in arcmin (used only to reconstruct the
                commanded trajectory; the fitted lag is independent of the
                --prd based pointing already in beam.txt, if any).

Outputs (written under --outdir, default "scanning_result/"):
  <measurements stem>_hosei<suffix>  Copy with each scanning direction's
                                       Epoch shifted by its selected lag.
  matched_and_corrected.csv          Row-classified, lag-corrected samples.
  lag_search.csv                     Lag candidates and amplitude correlations.
  best_lags.csv                      Selected increasing/decreasing lag pair.
  scan_rows.csv                      One row per detected scan leg
                                      (direction, az/el rate, sample count).
  scanning_diagnostics.png           Lag scores plus amplitude/phase maps
                                      before and after correction.
"""

from __future__ import annotations

import argparse
import errno
import io
import os
import tempfile
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.special import j1


T = TypeVar("T")


def progress(items: Iterable[T], total: int, label: str = "Processing", width: int = 30) -> Iterator[T]:
    """Yield items while drawing a dependency-free progress bar on stderr."""
    started = time.monotonic()
    for completed, item in enumerate(items, start=1):
        yield item
        elapsed = time.monotonic() - started
        fraction = completed / total if total else 1.0
        filled = round(width * fraction)
        eta = elapsed * (total - completed) / completed
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r{label}: [{bar}] {completed}/{total} ({fraction:6.1%}) "
            f"elapsed {elapsed:6.1f}s, ETA {eta:6.1f}s",
            end="",
            file=sys.stderr,
            flush=True,
        )
    print(file=sys.stderr)


@dataclass
class ScanRow:
    row_id: int
    indices: np.ndarray
    az_rate: float
    el_rate: float
    direction: int


def parse_measurements(path: Path) -> pd.DataFrame:
    # gico/fringe-style summaries are often named *.txt.  Their extension is
    # irrelevant: detect either comma-separated or whitespace-separated text.
    df = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True, comment="#")
    required = {"Epoch", "Amp", "Phase", "SNR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"measurement file lacks columns: {sorted(missing)}")
    df["time"] = pd.to_datetime(df["Epoch"], format="%Y/%j %H:%M:%S.%f")
    for column in ("Amp", "Phase", "SNR"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["time", "Amp", "Phase", "SNR"]).sort_values("time")


def write_corrected_measurements(
    source: Path, destination: Path, lag_by_epoch: dict[str, float]
) -> None:
    """Shift scanning epochs by direction; keep ON and unmatched epochs intact."""
    epoch_pattern = re.compile(r"(?P<epoch>\d{4}/\d{3}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)")

    def shift_epoch(match: re.Match[str]) -> str:
        original = match.group("epoch")
        if original not in lag_by_epoch:
            return original
        fraction_digits = len(original.rsplit(".", 1)[1]) if "." in original else 0
        timestamp = pd.to_datetime(original, format="%Y/%j %H:%M:%S.%f")
        corrected = timestamp - pd.Timedelta(lag_by_epoch[original], unit="s")
        base = corrected.strftime("%Y/%j %H:%M:%S")
        if fraction_digits == 0:
            return base
        fraction = f"{corrected.microsecond:06d}"[:fraction_digits].ljust(fraction_digits, "0")
        return f"{base}.{fraction}"

    original_text = source.read_text(encoding="utf-8")
    destination.write_text(epoch_pattern.sub(shift_epoch, original_text), encoding="utf-8")


def parse_skd_time(token: str) -> pd.Timestamp:
    # YYDDDHHMMSS[.fraction], e.g. 26184082203.01
    match = re.fullmatch(r"(\d{2})(\d{3})(\d{2})(\d{2})(\d{2})(?:\.(\d+))?", token)
    if not match:
        raise ValueError(f"unrecognised SKD time: {token}")
    yy, doy, hh, mm, ss, frac = match.groups()
    year = 2000 + int(yy)
    base = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(doy) - 1)
    value = base + pd.Timedelta(hours=int(hh), minutes=int(mm), seconds=int(ss))
    return value + pd.Timedelta(float(f"0.{frac or '0'}"), unit="s")


def parse_skd(path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    in_sked = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("$"):
            in_sked = stripped.upper().startswith("$SKED")
            continue
        if not in_sked or not stripped or stripped.startswith("*"):
            continue
        fields = stripped.split()
        if len(fields) < 5:
            continue
        try:
            records.append(
                {
                    "source": fields[0],
                    "time": parse_skd_time(fields[1]),
                    "duration_s": float(fields[2]),
                    "az_arcmin": float(fields[3]),
                    "el_arcmin": float(fields[4]),
                }
            )
        except ValueError:
            continue
    if not records:
        raise ValueError("no valid $SKED rows found")
    return pd.DataFrame(records).sort_values("time").reset_index(drop=True)


def join_by_time(meas: pd.DataFrame, skd: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    # Pandas 3 can preserve different datetime resolutions (for example us
    # from the summary text and ns from a constructed SKD timestamp).  asof
    # joins require exactly the same dtype on both sides.
    meas = meas.copy()
    skd = skd.copy()
    meas["time"] = pd.to_datetime(meas["time"]).astype("datetime64[ns]")
    skd["time"] = pd.to_datetime(skd["time"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        meas.sort_values("time"), skd.sort_values("time"), on="time",
        direction="nearest", tolerance=pd.Timedelta(tolerance_s, unit="s"),
    )
    merged = merged.dropna(subset=["az_arcmin", "el_arcmin"]).copy()
    if merged.empty:
        raise ValueError("no measurement timestamps matched SKD timestamps")
    return merged.reset_index(drop=True)


def make_rows(df: pd.DataFrame, gap_s: float, rate_floor: float = 1e-5,
              scan_axis: str = "az") -> tuple[pd.DataFrame, list[ScanRow]]:
    """Split samples where time or the commanded scan velocity changes."""
    t = (df.time.astype("int64") / 1e9).to_numpy()
    x, y = df.az_arcmin.to_numpy(), df.el_arcmin.to_numpy()
    dt = np.diff(t)
    vx = np.divide(np.diff(x), dt, out=np.zeros_like(dt), where=dt > 0)
    vy = np.divide(np.diff(y), dt, out=np.zeros_like(dt), where=dt > 0)
    primary, cross = (vx, vy) if scan_axis == "az" else (vy, vx)
    sign = np.sign(primary)
    sign[np.abs(primary) < rate_floor] = 0
    new = np.zeros(len(df), dtype=bool)
    new[0] = True
    new[1:] = ((dt > gap_s) |
               ((sign != np.r_[sign[0], sign[:-1]]) &
                (np.r_[dt[0], dt[:-1]] <= gap_s)) | (np.abs(cross) > 0.2))
    row_id = np.cumsum(new) - 1
    df = df.copy()
    df["row_id"] = row_id
    rows: list[ScanRow] = []
    for rid in np.unique(row_id):
        ind = np.where(row_id == rid)[0]
        if len(ind) < 3:
            continue
        local_t, local_x, local_y = t[ind], x[ind], y[ind]
        duration = local_t[-1] - local_t[0]
        if duration <= 0:
            continue
        az_rate = (local_x[-1] - local_x[0]) / duration
        el_rate = (local_y[-1] - local_y[0]) / duration
        primary_rate = az_rate if scan_axis == "az" else el_rate
        cross_rate = el_rate if scan_axis == "az" else az_rate
        direction = int(np.sign(primary_rate)) if abs(primary_rate) > rate_floor and abs(cross_rate) < 0.2 else 0
        rows.append(ScanRow(int(rid), ind, az_rate, el_rate, direction))
    row_info = pd.DataFrame(
        [dict(row_id=r.row_id, n_samples=len(r.indices), az_rate_arcmin_s=r.az_rate,
              el_rate_arcmin_s=r.el_rate, direction=r.direction) for r in rows]
    )
    return df, rows


def rows_with_fixed_speed(
    measurements: pd.DataFrame, speed: float, gap_s: float, scan_axis: str,
) -> tuple[pd.DataFrame, list[ScanRow]]:
    """Use beam offsets and only the direction signs, never SKD or rate magnitude."""
    df = measurements.copy().reset_index(drop=True)
    for source, target in (("Az_Offset", "az_arcmin"), ("El_Offset", "el_arcmin")):
        if source not in df:
            raise ValueError("--scan-speed requires Az_Offset and El_Offset in beam.txt")
        df[target] = pd.to_numeric(df[source], errors="coerce")
    if not np.isfinite(df[["az_arcmin", "el_arcmin"]].to_numpy()).all():
        raise ValueError("beam.txt offsets must be finite in --scan-speed mode")
    primary_col = "Az_Rate_arcmin_s" if scan_axis == "az" else "El_Rate_arcmin_s"
    cross_col = "El_Rate_arcmin_s" if scan_axis == "az" else "Az_Rate_arcmin_s"
    if primary_col in df:
        primary = pd.to_numeric(df[primary_col], errors="coerce").to_numpy()
        cross = (pd.to_numeric(df[cross_col], errors="coerce").to_numpy()
                 if cross_col in df else np.zeros(len(df)))
        if not np.isfinite(primary).all() or not np.isfinite(cross).all():
            raise ValueError("beam.txt rate columns must be finite for direction classification")
        direction = np.where((np.abs(primary) > 1e-5) & (np.abs(cross) < 0.2),
                             np.sign(primary), 0).astype(int)
        dt = np.diff(df.time.astype("int64").to_numpy() / 1e9)
        new = np.r_[True, (dt > gap_s) | (direction[1:] != direction[:-1])]
        df["row_id"] = np.cumsum(new) - 1
        rows = []
        for rid, group in df.groupby("row_id"):
            indices = group.index.to_numpy()
            d = int(direction[indices[0]])
            rows.append(ScanRow(int(rid), indices,
                                d * speed if scan_axis == "az" else 0.0,
                                d * speed if scan_axis == "el" else 0.0, d))
        print("Direction: signs of beam.txt rate columns; zero/cross-axis rates excluded")
    else:
        df, inferred = make_rows(df, gap_s, scan_axis=scan_axis)
        rows = [ScanRow(r.row_id, r.indices,
                        r.direction * speed if scan_axis == "az" else 0.0,
                        r.direction * speed if scan_axis == "el" else 0.0,
                        r.direction) for r in inferred]
        print("Direction: inferred from beam.txt offset changes")
    return df, rows


def map_on_grid(df: pd.DataFrame, xcol: str, ycol: str, value: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    points = df[[xcol, ycol]].to_numpy()
    valid = np.isfinite(points).all(axis=1) & np.isfinite(value)
    if valid.sum() < 3:
        return np.full_like(grid_x, np.nan, dtype=float)
    return griddata(points[valid], value[valid], (grid_x, grid_y), method="linear")


def pair_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Pearson correlation of amplitudes on common useful pixels.

    Peak normalization is used only for the existing 5% signal mask.
    Mean-subtracted amplitudes determine the coefficient, not phase.
    """
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 20:
        return np.nan, 0
    a, b = a[valid], b[valid]
    amax, bmax = np.max(a), np.max(b)
    if amax <= 0 or bmax <= 0:
        return np.nan, 0
    a, b = a / amax, b / bmax
    useful = (a + b) * 0.5 > 0.05
    if useful.sum() < 10:
        return np.nan, 0
    a, b = a[useful], b[useful]
    a, b = a - np.mean(a), b - np.mean(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(norm) or norm <= 0:
        return np.nan, int(useful.sum())
    correlation = np.dot(a, b) / norm
    return float(np.clip(correlation, -1.0, 1.0)), int(useful.sum())


def airy_main_beam_score(
    a: np.ndarray, b: np.ndarray, models: list[np.ndarray],
    central_mask: np.ndarray,
) -> float:
    """Compare the combined field amplitude with an origin-centred Airy beam.

    This is a diagnostic for the selected bidirectional alignment only.
    Its origin-centred model does not influence the correlation maximum.
    """
    valid = central_mask & np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10:
        return np.inf
    av, bv = a[valid], b[valid]
    amax, bmax = np.max(av), np.max(bv)
    if amax <= 0 or bmax <= 0:
        return np.inf
    observed = 0.5 * (av / amax + bv / bmax)
    return min(float(np.mean((observed - model[valid]) ** 2)) for model in models)


def search_direction_lags(
    df: pd.DataFrame, grid_x: np.ndarray, grid_y: np.ndarray,
    max_lag_ms: float, step_ms: float, min_snr: float,
    model_el_deg: float, airy_radius_arcmin: float, scan_axis: str = "az",
    bidirectional: bool = False,
) -> tuple[float, float, pd.DataFrame]:
    """Search one moving direction at a time, or both together if requested."""
    plus = df[(df.direction > 0) & (df.SNR >= min_snr)].copy()
    minus = df[(df.direction < 0) & (df.SNR >= min_snr)].copy()
    if len(plus) < 3 or len(minus) < 3:
        raise ValueError(f"both +{scan_axis.title()} and -{scan_axis.title()} rows with enough SNR are required")
    cache: dict[tuple[int, float], np.ndarray] = {}

    def direction_map(direction: int, lag_ms: float) -> np.ndarray:
        key = (direction, round(float(lag_ms), 8))
        if key not in cache:
            source = plus if direction > 0 else minus
            lag_s = lag_ms / 1000.0
            x = source.az_arcmin.to_numpy() - source.az_rate_arcmin_s.to_numpy() * lag_s
            y = source.el_arcmin.to_numpy() - source.el_rate_arcmin_s.to_numpy() * lag_s
            points = np.column_stack((x, y))
            values = source.Amp.to_numpy()
            finite = np.isfinite(points).all(axis=1) & np.isfinite(values)
            cache[key] = griddata(points[finite], values[finite],
                                  (grid_x, grid_y), method="linear")
        return cache[key]

    radius = np.hypot(grid_x * np.cos(np.deg2rad(model_el_deg)), grid_y)
    central_mask = radius <= airy_radius_arcmin
    theta = radius * np.pi / (180.0 * 60.0)
    wavelength = 3e8 / 8.448e9
    models = []
    for width_factor in np.linspace(0.85, 1.15, 7):
        u = np.pi * 32.0 * theta / (wavelength * width_factor)
        models.append(np.where(u == 0, 1.0, 2.0 * j1(u) / np.where(u == 0, 1.0, u)))
    models = [np.abs(model) for model in models]

    records: dict[tuple[float, float], dict[str, float]] = {}

    def evaluate(lag_plus: float, lag_minus: float) -> None:
        key = (round(float(lag_plus), 8), round(float(lag_minus), 8))
        if key not in records:
            a, b = direction_map(1, key[0]), direction_map(-1, key[1])
            correlation, overlap = pair_metrics(a, b)
            records[key] = dict(lag_increasing_ms=key[0], lag_decreasing_ms=key[1],
                                correlation=correlation, overlap=overlap, airy_mse=np.nan)

    coarse_step = max(20.0, step_ms)
    coarse = np.unique(np.r_[np.arange(-max_lag_ms, max_lag_ms + coarse_step / 2,
                                      coarse_step), -max_lag_ms, 0.0, max_lag_ms])
    coarse = coarse[(coarse >= -max_lag_ms) & (coarse <= max_lag_ms)]
    if bidirectional:
        for lag_plus in progress(coarse, len(coarse), label=f"Coarse +{scan_axis.title()} lag"):
            for lag_minus in coarse:
                evaluate(lag_plus, lag_minus)
    else:
        for lag in progress(coarse, len(coarse), label=f"Coarse one-direction lag"):
            evaluate(0.0, lag)  # +axis fixed, -axis shifted
            evaluate(lag, 0.0)  # -axis fixed, +axis shifted

    def valid_records(fixed_direction: int | None = None) -> list[dict[str, float]]:
        valid = [record for record in records.values()
                 if np.isfinite(record["correlation"])
                 and (fixed_direction is None or
                      record["lag_increasing_ms" if fixed_direction > 0 else "lag_decreasing_ms"] == 0)]
        if not valid:
            raise ValueError("no valid increasing/decreasing lag alignment")
        # Interpolation edges can produce a deceptively high correlation when
        # little useful sky remains in common.
        max_overlap = max(record["overlap"] for record in valid)
        return [record for record in valid if record["overlap"] >= 0.8 * max_overlap]

    def choose() -> tuple[float, float]:
        valid = valid_records()
        # Maximize Pearson r in both modes. Resolve exact ties by preferring
        # the smaller total applied lag; the Airy model never overrides r.
        best = min(valid, key=lambda record: (-record["correlation"],
                                              abs(record["lag_increasing_ms"]) +
                                              abs(record["lag_decreasing_ms"])))
        if bidirectional:
            best["airy_mse"] = airy_main_beam_score(
                direction_map(1, best["lag_increasing_ms"]),
                direction_map(-1, best["lag_decreasing_ms"]), models, central_mask)
        return best["lag_increasing_ms"], best["lag_decreasing_ms"]

    if bidirectional:
        coarse_plus, coarse_minus = choose()
        fine_plus = np.arange(max(-max_lag_ms, coarse_plus - coarse_step),
                              min(max_lag_ms, coarse_plus + coarse_step) + step_ms / 2, step_ms)
        fine_minus = np.arange(max(-max_lag_ms, coarse_minus - coarse_step),
                               min(max_lag_ms, coarse_minus + coarse_step) + step_ms / 2, step_ms)
        for lag_plus in progress(fine_plus, len(fine_plus), label=f"Fine +{scan_axis.title()} lag"):
            for lag_minus in fine_minus:
                evaluate(lag_plus, lag_minus)
    else:
        for fixed_direction in (1, -1):
            best_on_axis = min(valid_records(fixed_direction),
                               key=lambda record: -record["correlation"])
            moving_lag = best_on_axis["lag_decreasing_ms" if fixed_direction > 0
                                      else "lag_increasing_ms"]
            fine = np.arange(max(-max_lag_ms, moving_lag - coarse_step),
                             min(max_lag_ms, moving_lag + coarse_step) + step_ms / 2,
                             step_ms)
            for lag in progress(fine, len(fine), label=f"Fine {'-' if fixed_direction > 0 else '+'}{scan_axis.title()} lag"):
                if fixed_direction > 0:
                    evaluate(0.0, lag)
                else:
                    evaluate(lag, 0.0)
    best_plus, best_minus = choose()
    return best_plus, best_minus, pd.DataFrame(records.values())


def make_maps(df: pd.DataFrame, xcol: str, ycol: str, grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amp = map_on_grid(df, xcol, ycol, df.Amp.to_numpy(), grid_x, grid_y)
    complex_value = df.Amp.to_numpy() * np.exp(1j * np.deg2rad(df.Phase.to_numpy()))
    real = map_on_grid(df, xcol, ycol, complex_value.real, grid_x, grid_y)
    imag = map_on_grid(df, xcol, ycol, complex_value.imag, grid_x, grid_y)
    return amp, np.rad2deg(np.angle(real + 1j * imag))


def save_diagnostic_figure(fig, path: Path, **kwargs) -> Path:
    """Render in memory, then replace the target without opening it for writing.

    Windows viewers may lock an existing PNG on a WSL-mounted drive. If
    replacement fails for a lock/access error, retain the unique new PNG
    beside it and report its actual path.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", **kwargs)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.stem + "_new_", suffix=".png",
        delete=False,
    ) as temporary:
        temporary.write(buffer.getvalue())
        new_path = Path(temporary.name)
    try:
        os.replace(new_path, path)
    except OSError as exc:
        if exc.errno in (errno.EINVAL, errno.EACCES, errno.EPERM, errno.EBUSY):
            print(f"[WARN] Could not replace {path}: {exc}. "
                  f"New diagnostic saved to {new_path}", file=sys.stderr)
            return new_path
        new_path.unlink(missing_ok=True)
        raise
    return path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("measurements", type=Path,
                    help="correlation summary text with columns Epoch, Amp, Phase, SNR "
                         "(e.g. beam.txt from group_up_txt_prd.py)")
    p.add_argument("schedule", type=Path, nargs="?",
                   help="SKD schedule file (legacy positional form; alternatively --skd)")
    p.add_argument("--skd", type=Path,
                   help="SKD trajectory: estimate speeds from coordinates and timestamps")
    p.add_argument("--scan-speed", "--scan-speed-arcmin-s", type=float, metavar="ARCMIN/S",
                   help="positive speed along --scan-axis in offset-coordinate arcmin/s; "
                        "use beam.txt Az_Offset/El_Offset without SKD. Rate-column signs "
                        "identify directions when present; otherwise infer from offsets.")
    p.add_argument("--outdir", type=Path, default=Path("scanning_result"),
                    help="directory for all output files (default: scanning_result/)")
    p.add_argument("--match-tolerance", type=float, default=0.006,
                    help="max time difference [s] allowed when joining a measurement to an "
                         "SKD row (default: 0.006)")
    p.add_argument("--row-gap", type=float, default=2.0,
                    help="time gap [s] that starts a new scan row/leg (default: 2.0)")
    p.add_argument("--scan-axis", choices=("az", "el"), default="az",
                   help="varying axis within each scan leg (default: az)")
    p.add_argument("--bidirectional-lags", action="store_true",
                   help="fit increasing/decreasing lags simultaneously; default: hold one "
                        "direction fixed and scan the other, then swap")
    p.add_argument("--min-snr", type=float, default=3.0,
                    help="minimum SNR for a sample to be used when fitting the lag "
                         "(default: 3.0)")
    p.add_argument("--max-lag-ms", type=float, default=1000.0,
                    help="search lag candidates over +/- this many ms (default: 1000.0)")
    p.add_argument("--lag-step-ms", type=float, default=5.0,
                    help="fine step [ms] for each direction (default: 5.0); coarse search uses at least 20 ms")
    p.add_argument("--grid-size", type=int, default=121,
                    help="number of grid points per axis used when comparing the forward/"
                         "reverse amplitude maps (default: 121)")
    p.add_argument("--model-el-deg", type=float, default=57.3,
                    help="elevation [deg] for the Az projection in the Airy comparison (default: 57.3)")
    p.add_argument("--airy-radius-arcmin", type=float, default=12.0,
                    help="central radius [arcmin] for the diagnostic Airy score; not used to select lags (default: 12)")
    if len(sys.argv) == 1:
        p.print_help()
        return 1
    args = p.parse_args()
    if args.schedule is not None and args.skd is not None:
        p.error("specify the SKD only once: positional schedule or --skd")
    schedule = args.skd if args.skd is not None else args.schedule
    if (schedule is None) == (args.scan_speed is None):
        p.error("choose exactly one mode: SKD schedule or --scan-speed")
    if args.scan_speed is not None and (not np.isfinite(args.scan_speed) or args.scan_speed <= 0):
        p.error("--scan-speed must be a finite positive number in arcmin/s")
    if args.max_lag_ms <= 0 or args.lag_step_ms <= 0 or args.grid_size < 5:
        p.error("--max-lag-ms and --lag-step-ms must be positive; --grid-size must be at least 5")
    if not (0 <= args.model_el_deg < 90) or args.airy_radius_arcmin <= 0:
        p.error("--model-el-deg must be in [0,90) and --airy-radius-arcmin must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Reading and matching input data...", file=sys.stderr, flush=True)
    measurements = parse_measurements(args.measurements)
    if schedule is not None:
        print(f"Trajectory: SKD {schedule}; speed estimated per scan row")
        joined = join_by_time(measurements, parse_skd(schedule), args.match_tolerance)
        joined, rows = make_rows(joined, args.row_gap, scan_axis=args.scan_axis)
    else:
        print(f"Trajectory: beam.txt offsets; specified {args.scan_axis.upper()} speed "
              f"{args.scan_speed:g} arcmin/s (offset-coordinate speed)")
        joined, rows = rows_with_fixed_speed(
            measurements, args.scan_speed, args.row_gap, args.scan_axis)
    if not rows:
        raise ValueError("could not identify scan rows; check timestamps and --row-gap")
    rate = {r.row_id: (r.az_rate, r.el_rate, r.direction) for r in rows}
    joined["az_rate_arcmin_s"] = joined.row_id.map(lambda k: rate.get(k, (0.0, 0.0, 0))[0])
    joined["el_rate_arcmin_s"] = joined.row_id.map(lambda k: rate.get(k, (0.0, 0.0, 0))[1])
    joined["direction"] = joined.row_id.map(lambda k: rate.get(k, (0.0, 0.0, 0))[2])

    qx = np.linspace(joined.az_arcmin.quantile(0.01), joined.az_arcmin.quantile(0.99), args.grid_size)
    qy = np.linspace(joined.el_arcmin.quantile(0.01), joined.el_arcmin.quantile(0.99), args.grid_size)
    grid_x, grid_y = np.meshgrid(qx, qy)
    search_label = ("increasing/decreasing lag pairs" if args.bidirectional_lags
                    else "one moving direction at a time")
    print(f"[2/4] Searching {search_label}...", file=sys.stderr, flush=True)
    lag_plus_ms, lag_minus_ms, search = search_direction_lags(
        joined, grid_x, grid_y, args.max_lag_ms, args.lag_step_ms,
        args.min_snr, args.model_el_deg, args.airy_radius_arcmin, args.scan_axis,
        args.bidirectional_lags)
    sample_lag_s = np.where(joined.direction > 0, lag_plus_ms / 1000.0,
                            np.where(joined.direction < 0, lag_minus_ms / 1000.0, 0.0))
    joined["lag_applied_ms"] = sample_lag_s * 1000.0
    joined["az_corrected_arcmin"] = joined.az_arcmin - joined.az_rate_arcmin_s * sample_lag_s
    joined["el_corrected_arcmin"] = joined.el_arcmin - joined.el_rate_arcmin_s * sample_lag_s
    joined["az_shift_arcmin"] = joined.az_corrected_arcmin - joined.az_arcmin
    joined["el_shift_arcmin"] = joined.el_corrected_arcmin - joined.el_arcmin
    print("[3/4] Writing result tables...", file=sys.stderr, flush=True)
    corrected_beam_path = args.outdir / f"{args.measurements.stem}_hosei{args.measurements.suffix}"
    lag_by_epoch = dict(zip(joined["Epoch"], sample_lag_s))
    write_corrected_measurements(args.measurements, corrected_beam_path, lag_by_epoch)
    joined.to_csv(args.outdir / "matched_and_corrected.csv", index=False)
    search.to_csv(args.outdir / "lag_search.csv", index=False)
    selected = search[(np.isclose(search.lag_increasing_ms, lag_plus_ms)) &
                      (np.isclose(search.lag_decreasing_ms, lag_minus_ms))].iloc[0]
    pd.DataFrame([selected]).to_csv(args.outdir / "best_lags.csv", index=False)
    pd.DataFrame([dict(row_id=r.row_id, n_samples=len(r.indices), az_rate_arcmin_s=r.az_rate,
                       el_rate_arcmin_s=r.el_rate, direction=r.direction) for r in rows]).to_csv(args.outdir / "scan_rows.csv", index=False)

    print("[4/4] Creating diagnostic maps...", file=sys.stderr, flush=True)
    amp0, phase0 = make_maps(joined, "az_arcmin", "el_arcmin", grid_x, grid_y)
    amp1, phase1 = make_maps(joined, "az_corrected_arcmin", "el_corrected_arcmin", grid_x, grid_y)
    extent = [qx.min(), qx.max(), qy.min(), qy.max()]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    finite = search[np.isfinite(search.correlation)]
    axis_name = args.scan_axis.title()
    if args.bidirectional_lags:
        points = ax[0, 0].scatter(finite.lag_increasing_ms, finite.lag_decreasing_ms,
                                  c=finite.correlation, s=8, cmap="viridis", rasterized=True)
        ax[0, 0].scatter([lag_plus_ms], [lag_minus_ms], marker="x", color="red",
                         s=100, label="selected")
        fig.colorbar(points, ax=ax[0, 0], label="Amplitude correlation coefficient r")
        ax[0, 0].set(xlabel=f"+{axis_name} lag [ms]", ylabel=f"-{axis_name} lag [ms]",
                     title="Lag-pair search")
    else:
        for fixed, moving_col, label in (
            ("lag_increasing_ms", "lag_decreasing_ms", f"+{axis_name} fixed / -{axis_name} shifted"),
            ("lag_decreasing_ms", "lag_increasing_ms", f"-{axis_name} fixed / +{axis_name} shifted"),
        ):
            curve = finite[np.isclose(finite[fixed], 0)].sort_values(moving_col)
            ax[0, 0].plot(curve[moving_col], curve.correlation, label=label)
        ax[0, 0].scatter([lag_minus_ms or lag_plus_ms], [selected.correlation],
                         marker="x", color="red", s=100, label="selected")
        ax[0, 0].set(xlabel="Moving direction lag [ms]", ylabel="Amplitude correlation coefficient r",
                     title="One-direction lag searches")
    ax[0, 0].legend()
    for axes, image, title, cmap, vmin, vmax in (
        (ax[0, 1], amp0, "Amplitude: commanded coordinates", "viridis", 0, np.nanmax([amp0, amp1])),
        (ax[0, 2], amp1, "Amplitude: scan-lag corrected", "viridis", 0, np.nanmax([amp0, amp1])),
        (ax[1, 1], phase0, "Phase: commanded coordinates", "twilight", -180, 180),
        (ax[1, 2], phase1, "Phase: scan-lag corrected", "twilight", -180, 180),
    ):
        im = axes.imshow(image, extent=extent, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        axes.set(title=title, xlabel="Az offset [arcmin]", ylabel="El offset [arcmin]")
        fig.colorbar(im, ax=axes, shrink=0.8)
    ax[1, 0].axis("off")
    speed = np.nanmedian(np.abs(joined.loc[joined.direction != 0, f"{args.scan_axis}_rate_arcmin_s"]))
    ax[1, 0].text(0.03, 0.85, "Scan-lag correction", fontsize=15, weight="bold")
    ax[1, 0].text(
        0.03, 0.60,
        f"+{axis_name} lag: {lag_plus_ms:.1f} ms\n-{axis_name} lag: {lag_minus_ms:.1f} ms\n"
        f"Median |{axis_name} rate|: {speed:.4g} arcmin/s\n"
        + (f"Airy score: {selected.airy_mse:.5g}" if args.bidirectional_lags
           else f"Fixed: {'+' if lag_plus_ms == 0 else '-'}{axis_name}"), fontsize=13)
    diagnostic_path = save_diagnostic_figure(
        fig, args.outdir / "scanning_diagnostics.png", dpi=180)
    plt.close(fig)
    print(f"Matched samples: {len(joined)}")
    print(f"Best amplitude correlation: r={selected.correlation:.6f}, pixels={int(selected.overlap)}")
    print(f"Best +{axis_name} lag: {lag_plus_ms:.1f} ms")
    print(f"Best -{axis_name} lag: {lag_minus_ms:.1f} ms")
    print(f"Typical |{axis_name} correction|: {abs(speed) * max(abs(lag_plus_ms), abs(lag_minus_ms)) / 1000:.5g} arcmin")
    print(f"Corrected beam: {corrected_beam_path}")
    print(f"Diagnostic figure: {diagnostic_path}")
    print(f"Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
