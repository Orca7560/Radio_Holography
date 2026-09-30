#!/usr/bin/env python3
"""Compare forward/reverse raster scans by normalized cross-correlation.

Run beside scanning_effect.py, using the same beam.txt and SKD inputs. This is
an independent diagnostic: it does not overwrite the existing lag estimate or
silently apply a correction to corr_fringe.

For each trial lag, the moving direction's sample coordinates are corrected by
    theta_trial = theta_SKD - velocity * lag_seconds.
Only that direction is regridded. Pearson's normalized cross-correlation is
computed against the fixed direction on common finite pixels near the beam.
Both choices of fixed direction are evaluated separately. Their best lags
describe the same relative displacement in different coordinate frames; they
are not independently determined absolute instrumental delays.

Example:
  python3 scanning_effect_xcorr.py test_data/beam_184.txt test_data/schedule_184.skd
      --outdir test_result/xcorr_184 --max-lag-ms 1000 --lag-step-ms 1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def measurements(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True,
                        comment="#")
    required = {"Epoch", "Amp", "Phase", "SNR"}
    if required - set(frame):
        raise ValueError(f"missing measurement columns: {sorted(required - set(frame))}")
    frame["time"] = pd.to_datetime(frame.Epoch, format="%Y/%j %H:%M:%S.%f")
    for name in ("Amp", "Phase", "SNR"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna(subset=["time", "Amp", "SNR"]).sort_values("time")


def schedule(path: Path) -> pd.DataFrame:
    records = []
    in_sked = False
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.strip().split()
        if not words:
            continue
        if words[0].startswith("$"):
            in_sked = words[0].upper() == "$SKED"
            continue
        if not in_sked or words[0].startswith("*") or len(words) < 5:
            continue
        match = re.fullmatch(r"(\d{2})(\d{3})(\d{2})(\d{2})(\d{2})(?:\.(\d+))?", words[1])
        if match is None:
            continue
        yy, doy, hh, mm, ss, fraction = match.groups()
        stamp = (pd.Timestamp(year=2000 + int(yy), month=1, day=1)
                 + pd.Timedelta(days=int(doy) - 1, hours=int(hh),
                                minutes=int(mm), seconds=int(ss))
                 + pd.Timedelta(float("0." + (fraction or "0")), unit="s"))
        records.append((stamp, float(words[3]), float(words[4])))
    if not records:
        raise ValueError("no valid $SKED rows")
    return pd.DataFrame(records, columns=["time", "az_arcmin", "el_arcmin"]).sort_values("time")


def classify(meas: pd.DataFrame, skd: pd.DataFrame, tolerance_s: float,
             row_gap_s: float, scan_axis: str = "az") -> pd.DataFrame:
    left, right = meas.copy(), skd.copy()
    left["time"] = left.time.astype("datetime64[ns]")
    right["time"] = right.time.astype("datetime64[ns]")
    data = pd.merge_asof(left, right, on="time", direction="nearest",
                         tolerance=pd.Timedelta(tolerance_s, unit="s"))
    data = data.dropna(subset=["az_arcmin", "el_arcmin"]).reset_index(drop=True)
    if len(data) < 10:
        raise ValueError("too few measurements matched the SKD; check --match-tolerance")
    seconds = data.time.astype("int64").to_numpy() / 1e9
    az = data.az_arcmin.to_numpy()
    el = data.el_arcmin.to_numpy()
    dt = np.diff(seconds)
    vx = np.divide(np.diff(az), dt, out=np.zeros_like(dt), where=dt > 0)
    vy = np.divide(np.diff(el), dt, out=np.zeros_like(dt), where=dt > 0)
    primary, cross = (vx, vy) if scan_axis == "az" else (vy, vx)
    sign = np.sign(primary)
    sign[np.abs(primary) < 1e-5] = 0
    boundary = np.zeros(len(data), dtype=bool)
    boundary[0] = True
    boundary[1:] = ((dt > row_gap_s) |
                    ((sign != np.r_[sign[0], sign[:-1]]) &
                     (np.r_[dt[0], dt[:-1]] <= row_gap_s)) |
                    (np.abs(cross) > 0.2))
    data["row_id"] = np.cumsum(boundary) - 1
    data["az_rate_arcmin_s"] = np.nan
    data["el_rate_arcmin_s"] = np.nan
    data["direction"] = 0
    for _, row in data.groupby("row_id", sort=False):
        if len(row) < 3:
            continue
        inds = row.index.to_numpy()
        duration = seconds[inds[-1]] - seconds[inds[0]]
        if duration <= 0:
            continue
        rate_x = (az[inds[-1]] - az[inds[0]]) / duration
        rate_y = (el[inds[-1]] - el[inds[0]]) / duration
        primary_rate, cross_rate = (rate_x, rate_y) if scan_axis == "az" else (rate_y, rate_x)
        direction = int(np.sign(primary_rate)) if abs(primary_rate) > 1e-5 else 0
        if direction and abs(cross_rate) < 0.2:
            data.loc[inds, "az_rate_arcmin_s"] = rate_x
            data.loc[inds, "el_rate_arcmin_s"] = rate_y
            data.loc[inds, "direction"] = direction
    return data


def amplitude_map(data: pd.DataFrame, lag_ms: float,
                  gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    lag_s = lag_ms / 1000.0
    x = data.az_arcmin.to_numpy() - data.az_rate_arcmin_s.to_numpy() * lag_s
    y = data.el_arcmin.to_numpy() - data.el_rate_arcmin_s.to_numpy() * lag_s
    a = data.Amp.to_numpy()
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(a)
    if finite.sum() < 3:
        return np.full_like(gx, np.nan)
    return griddata(np.column_stack((x[finite], y[finite])), a[finite],
                    (gx, gy), method="linear")


def normalized_xcorr(fixed: np.ndarray, moving: np.ndarray,
                     main_beam_fraction: float, min_pixels: int) -> tuple[float, int]:
    """Pearson cross-correlation at zero spatial separation for one trial shift."""
    mask = np.isfinite(fixed) & np.isfinite(moving)
    if mask.sum() < min_pixels:
        return np.nan, 0
    peak = max(float(np.nanmax(fixed)), float(np.nanmax(moving)))
    floor = min(float(np.nanmin(fixed)), float(np.nanmin(moving)))
    mask &= (fixed + moving) / 2 > floor + main_beam_fraction * (peak - floor)
    count = int(mask.sum())
    if count < min_pixels:
        return np.nan, count
    a, b = fixed[mask], moving[mask]
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return (float(np.dot(a, b) / denom), count) if denom > 0 else (np.nan, count)


def search(fixed: np.ndarray, moving: pd.DataFrame,
           gx: np.ndarray, gy: np.ndarray, lags: np.ndarray,
           fraction: float, min_pixels: int, min_overlap: float
           ) -> tuple[pd.DataFrame, float, np.ndarray]:
    rows = []
    for lag in lags:
        candidate = amplitude_map(moving, float(lag), gx, gy)
        corr, overlap = normalized_xcorr(fixed, candidate, fraction, min_pixels)
        rows.append((float(lag), corr, overlap))
    result = pd.DataFrame(rows, columns=["lag_ms", "xcorr", "overlap_pixels"])
    usable = result[np.isfinite(result.xcorr)].copy()
    if usable.empty:
        raise ValueError("no trial has enough common main-beam pixels; adjust grid/range/SNR")
    usable = usable[usable.overlap_pixels >= min_overlap * usable.overlap_pixels.max()]
    best = usable.sort_values(["xcorr", "overlap_pixels"], ascending=[False, False]).iloc[0]
    lag = float(best.lag_ms)
    return result, lag, amplitude_map(moving, lag, gx, gy)


def save_diagnostic(path: Path, results: dict, plus: pd.DataFrame,
                    minus: pd.DataFrame, width: float, grid_size: int,
                    scan_axis: str = "az") -> Path:
    """Plot a fine central grid; preserve any pre-existing diagnostic image."""
    axis = np.linspace(-width, width, grid_size)
    gx, gy = np.meshgrid(axis, axis)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    extent = [-width, width, -width, width]
    axis_name = scan_axis.title()
    for row_index, (name, fixed_data, moving_data) in enumerate((
            (f"+{axis_name} fixed / -{axis_name} shifted", plus, minus),
            (f"-{axis_name} fixed / +{axis_name} shifted", minus, plus))):
        table, lag, _ = results[name]
        fixed = amplitude_map(fixed_data, 0, gx, gy)
        moving_raw = amplitude_map(moving_data, 0, gx, gy)
        moved = amplitude_map(moving_data, lag, gx, gy)
        ax = axes[row_index]
        ax[0].plot(table.lag_ms, table.xcorr, color="navy")
        ax[0].axvline(lag, color="crimson", label=f"{lag:g} ms")
        ax[0].set(xlabel="Moving direction lag [ms]", ylabel="Normalized correlation",
                  title=name)
        ax[0].legend()
        vmax = max(np.nanmax(fixed), np.nanmax(moving_raw), np.nanmax(moved))
        for pane, image, title in ((ax[1], fixed, "Fixed direction"),
                                   (ax[2], moving_raw, "Moving: before"),
                                   (ax[3], moved, "Moving: selected lag")):
            im = pane.imshow(image, origin="lower", extent=extent, cmap="viridis",
                             vmin=0, vmax=vmax, aspect="equal")
            if pane is not ax[1]:
                pane.contour(gx, gy, fixed, levels=[0.5 * np.nanmax(fixed)],
                             colors="white", linewidths=1.2)
            pane.set(title=title, xlabel="Az [arcmin]",
                     ylabel="El [arcmin]")
            fig.colorbar(im, ax=pane, shrink=0.75)
    fig.suptitle("Central beam; white contour = fixed direction at 50% peak", fontsize=13)
    destination = path
    number = 1
    while destination.exists():
        destination = path.with_name(f"{path.stem}_{number:03d}{path.suffix}")
        number += 1
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path, help="beam.txt: Epoch, Amp, Phase, SNR")
    parser.add_argument("schedule", type=Path, help="SKD file with $SKED offsets")
    parser.add_argument("--outdir", type=Path, default=Path("scanning_xcorr_result"),
                        help="separate result directory (default: scanning_xcorr_result)")
    parser.add_argument("--max-lag-ms", type=float, default=1000,
                        help="search each moving direction from -MAX to +MAX ms (default: 1000)")
    parser.add_argument("--lag-step-ms", type=float, default=5,
                        help="lag spacing in ms (default: 5)")
    parser.add_argument("--match-tolerance", type=float, default=0.006,
                        help="SKD/beam timestamp matching tolerance in seconds (default: 0.006)")
    parser.add_argument("--row-gap", type=float, default=2.0,
                        help="time gap separating scan rows in seconds (default: 2)")
    parser.add_argument("--scan-axis", choices=("az", "el"), default="az",
                        help="varying axis within each scan leg (default: az)")
    parser.add_argument("--min-snr", type=float, default=3,
                        help="minimum SNR for lag search (default: 3)")
    parser.add_argument("--grid-size", type=int, default=121,
                        help="number of comparison pixels per map axis (default: 121)")
    parser.add_argument("--main-beam-fraction", type=float, default=0.05,
                        help="only pixels above this fraction of the peak (default: 0.05)")
    parser.add_argument("--min-pixels", type=int, default=20,
                        help="minimum common main-beam pixels (default: 20)")
    parser.add_argument("--min-overlap", type=float, default=0.8,
                        help="minimum useful overlap relative to the maximum (default: 0.8)")
    parser.add_argument("--plot-width-arcmin", type=float, default=12,
                        help="central diagnostic map half-width in arcmin (default: 12)")
    parser.add_argument("--plot-grid-size", type=int, default=241,
                        help="pixels per axis for diagnostic maps only (default: 241)")
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    args = parser.parse_args()
    if (args.max_lag_ms <= 0 or args.lag_step_ms <= 0 or args.grid_size < 5
            or args.min_pixels < 3 or not 0 <= args.main_beam_fraction < 1
            or not 0 < args.min_overlap <= 1 or args.plot_width_arcmin <= 0
            or args.plot_grid_size < 5):
        parser.error("invalid lag/grid/overlap/beam threshold")
    data = classify(measurements(args.measurements), schedule(args.schedule),
                    args.match_tolerance, args.row_gap, args.scan_axis)
    usable = data[data.SNR >= args.min_snr]
    plus, minus = usable[usable.direction == 1], usable[usable.direction == -1]
    if min(len(plus), len(minus)) < 3:
        raise ValueError(f"both +{args.scan_axis.title()} and -{args.scan_axis.title()} scans with enough SNR are required")
    qx = np.linspace(data.az_arcmin.quantile(0.01), data.az_arcmin.quantile(0.99),
                     args.grid_size)
    qy = np.linspace(data.el_arcmin.quantile(0.01), data.el_arcmin.quantile(0.99),
                     args.grid_size)
    gx, gy = np.meshgrid(qx, qy)
    plus_map, minus_map = (amplitude_map(plus, 0, gx, gy),
                           amplitude_map(minus, 0, gx, gy))
    lags = np.arange(-args.max_lag_ms, args.max_lag_ms + args.lag_step_ms / 2,
                     args.lag_step_ms)
    results = {}
    axis_name = args.scan_axis.title()
    for name, fixed, moving in ((f"+{axis_name} fixed / -{axis_name} shifted", plus_map, minus),
                                (f"-{axis_name} fixed / +{axis_name} shifted", minus_map, plus)):
        print(f"Searching {name}: {len(lags)} trials", file=sys.stderr, flush=True)
        results[name] = search(fixed, moving, gx, gy, lags,
                               args.main_beam_fraction, args.min_pixels,
                               args.min_overlap)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (table, lag, _) in results.items():
        table.to_csv(args.outdir / ("lag_search_minus_moving.csv" if name.startswith(f"+{axis_name}")
                                        else "lag_search_plus_moving.csv"), index=False)
        best = table.loc[np.isclose(table.lag_ms, lag)].iloc[0]
        rows.append({"fixed_direction": f"+{axis_name}" if name.startswith(f"+{axis_name}") else f"-{axis_name}",
                     "moving_direction": f"-{axis_name}" if name.startswith(f"+{axis_name}") else f"+{axis_name}",
                     "best_moving_lag_ms": lag,
                     "normalized_xcorr": float(best.xcorr),
                     "overlap_pixels": int(best.overlap_pixels)})
    pd.DataFrame(rows).to_csv(args.outdir / "best_lags_xcorr.csv", index=False)
    diagnostic_path = save_diagnostic(args.outdir / "scanning_xcorr_diagnostics.png",
                                      results, plus, minus,
                                      args.plot_width_arcmin, args.plot_grid_size, args.scan_axis)
    for result in rows:
        print(f"{result['fixed_direction']} fixed, {result['moving_direction']} shifted: "
              f"lag={result['best_moving_lag_ms']:g} ms, "
              f"r={result['normalized_xcorr']:.5f}, "
              f"pixels={result['overlap_pixels']}")
    print("These are alternative relative alignments, not two independent absolute lags.")
    print(f"Diagnostic image: {diagnostic_path}")
    print(f"Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
