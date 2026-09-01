#!/usr/bin/env python3
"""Estimate and correct scan-lag offsets in raster beam measurements.

The input correlation summary and SKD file are joined by timestamp.  A trial
time lag tau changes each sample coordinate as

    theta_corrected = theta_commanded - velocity * tau.

tau is selected by making the amplitude maps from the two opposite azimuth
scan directions agree as closely as possible.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


@dataclass
class ScanRow:
    row_id: int
    indices: np.ndarray
    az_rate: float
    el_rate: float
    direction: int


def parse_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    required = {"Epoch", "Amp", "Phase", "SNR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"measurement file lacks columns: {sorted(missing)}")
    df["time"] = pd.to_datetime(df["Epoch"], format="%Y/%j %H:%M:%S.%f")
    for column in ("Amp", "Phase", "SNR"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["time", "Amp", "Phase", "SNR"]).sort_values("time")


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
    merged = pd.merge_asof(
        meas.sort_values("time"), skd.sort_values("time"), on="time",
        direction="nearest", tolerance=pd.Timedelta(tolerance_s, unit="s"),
    )
    merged = merged.dropna(subset=["az_arcmin", "el_arcmin"]).copy()
    if merged.empty:
        raise ValueError("no measurement timestamps matched SKD timestamps")
    return merged.reset_index(drop=True)


def make_rows(df: pd.DataFrame, gap_s: float, rate_floor: float = 1e-5) -> tuple[pd.DataFrame, list[ScanRow]]:
    """Split samples where time or the commanded scan velocity changes."""
    t = (df.time.astype("int64") / 1e9).to_numpy()
    x, y = df.az_arcmin.to_numpy(), df.el_arcmin.to_numpy()
    dt = np.diff(t)
    vx = np.divide(np.diff(x), dt, out=np.zeros_like(dt), where=dt > 0)
    vy = np.divide(np.diff(y), dt, out=np.zeros_like(dt), where=dt > 0)
    sign = np.sign(vx)
    sign[np.abs(vx) < rate_floor] = 0
    new = np.zeros(len(df), dtype=bool)
    new[0] = True
    new[1:] = (dt > gap_s) | (sign != np.r_[sign[0], sign[:-1]]) | (np.abs(vy) > 0.2)
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
        direction = int(np.sign(az_rate))
        rows.append(ScanRow(int(rid), ind, az_rate, el_rate, direction))
    row_info = pd.DataFrame(
        [dict(row_id=r.row_id, n_samples=len(r.indices), az_rate_arcmin_s=r.az_rate,
              el_rate_arcmin_s=r.el_rate, direction=r.direction) for r in rows]
    )
    return df, rows


def map_on_grid(df: pd.DataFrame, xcol: str, ycol: str, value: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    points = df[[xcol, ycol]].to_numpy()
    valid = np.isfinite(points).all(axis=1) & np.isfinite(value)
    if valid.sum() < 3:
        return np.full_like(grid_x, np.nan, dtype=float)
    return griddata(points[valid], value[valid], (grid_x, grid_y), method="linear")


def score_lag(df: pd.DataFrame, rows: list[ScanRow], lag_s: float, grid_x: np.ndarray, grid_y: np.ndarray, min_snr: float) -> float:
    work = df.copy()
    rate = {r.row_id: (r.az_rate, r.el_rate, r.direction) for r in rows}
    work["vx"] = work.row_id.map(lambda k: rate.get(k, (np.nan, np.nan, 0))[0])
    work["vy"] = work.row_id.map(lambda k: rate.get(k, (np.nan, np.nan, 0))[1])
    work["direction"] = work.row_id.map(lambda k: rate.get(k, (np.nan, np.nan, 0))[2])
    work = work[(work.direction != 0) & (work.SNR >= min_snr)].copy()
    work["x_corr"] = work.az_arcmin - work.vx * lag_s
    work["y_corr"] = work.el_arcmin - work.vy * lag_s
    plus, minus = work[work.direction > 0], work[work.direction < 0]
    if len(plus) < 3 or len(minus) < 3:
        return np.nan
    a = map_on_grid(plus, "x_corr", "y_corr", plus.Amp.to_numpy(), grid_x, grid_y)
    b = map_on_grid(minus, "x_corr", "y_corr", minus.Amp.to_numpy(), grid_x, grid_y)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 20:
        return np.nan
    a, b = a[valid], b[valid]
    a, b = a / np.nanmax(a), b / np.nanmax(b)
    # Put emphasis on the main beam and useful sidelobes, not the noise floor.
    useful = (a + b) * 0.5 > 0.05
    return float(np.mean((a[useful] - b[useful]) ** 2)) if useful.sum() >= 10 else np.nan


def make_maps(df: pd.DataFrame, xcol: str, ycol: str, grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amp = map_on_grid(df, xcol, ycol, df.Amp.to_numpy(), grid_x, grid_y)
    complex_value = df.Amp.to_numpy() * np.exp(1j * np.deg2rad(df.Phase.to_numpy()))
    real = map_on_grid(df, xcol, ycol, complex_value.real, grid_x, grid_y)
    imag = map_on_grid(df, xcol, ycol, complex_value.imag, grid_x, grid_y)
    return amp, np.rad2deg(np.angle(real + 1j * imag))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("measurements", type=Path, help="CSV: Epoch, Amp, Phase, SNR")
    p.add_argument("schedule", type=Path, help="SKD schedule file")
    p.add_argument("--outdir", type=Path, default=Path("scanning_result"))
    p.add_argument("--match-tolerance", type=float, default=0.006, help="timestamp match tolerance [s]")
    p.add_argument("--row-gap", type=float, default=2.0, help="new-row time gap [s]")
    p.add_argument("--min-snr", type=float, default=3.0, help="minimum SNR for lag fit")
    p.add_argument("--max-lag-ms", type=float, default=1000.0)
    p.add_argument("--lag-step-ms", type=float, default=5.0)
    p.add_argument("--grid-size", type=int, default=121)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    joined = join_by_time(parse_measurements(args.measurements), parse_skd(args.schedule), args.match_tolerance)
    joined, rows = make_rows(joined, args.row_gap)
    if not rows:
        raise ValueError("could not identify scan rows; check timestamps and --row-gap")
    rate = {r.row_id: (r.az_rate, r.el_rate, r.direction) for r in rows}
    joined["az_rate_arcmin_s"] = joined.row_id.map(lambda k: rate.get(k, (np.nan, np.nan, 0))[0])
    joined["el_rate_arcmin_s"] = joined.row_id.map(lambda k: rate.get(k, (np.nan, np.nan, 0))[1])
    joined["direction"] = joined.row_id.map(lambda k: rate.get(k, (np.nan, np.nan, 0))[2])

    qx = np.linspace(joined.az_arcmin.quantile(0.01), joined.az_arcmin.quantile(0.99), args.grid_size)
    qy = np.linspace(joined.el_arcmin.quantile(0.01), joined.el_arcmin.quantile(0.99), args.grid_size)
    grid_x, grid_y = np.meshgrid(qx, qy)
    lags = np.arange(-args.max_lag_ms, args.max_lag_ms + args.lag_step_ms, args.lag_step_ms) / 1000.0
    scores = np.array([score_lag(joined, rows, lag, grid_x, grid_y, args.min_snr) for lag in lags])
    if not np.isfinite(scores).any():
        raise ValueError("lag cannot be fitted: both +Az and -Az rows with enough SNR are required")
    best_lag = float(lags[np.nanargmin(scores)])
    joined["az_corrected_arcmin"] = joined.az_arcmin - joined.az_rate_arcmin_s * best_lag
    joined["el_corrected_arcmin"] = joined.el_arcmin - joined.el_rate_arcmin_s * best_lag
    joined["az_shift_arcmin"] = joined.az_corrected_arcmin - joined.az_arcmin
    joined["el_shift_arcmin"] = joined.el_corrected_arcmin - joined.el_arcmin
    joined.to_csv(args.outdir / "matched_and_corrected.csv", index=False)
    pd.DataFrame({"lag_ms": lags * 1000, "mismatch": scores}).to_csv(args.outdir / "lag_search.csv", index=False)
    pd.DataFrame([dict(row_id=r.row_id, n_samples=len(r.indices), az_rate_arcmin_s=r.az_rate,
                       el_rate_arcmin_s=r.el_rate, direction=r.direction) for r in rows]).to_csv(args.outdir / "scan_rows.csv", index=False)

    amp0, phase0 = make_maps(joined, "az_arcmin", "el_arcmin", grid_x, grid_y)
    amp1, phase1 = make_maps(joined, "az_corrected_arcmin", "el_corrected_arcmin", grid_x, grid_y)
    extent = [qx.min(), qx.max(), qy.min(), qy.max()]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    ax[0, 0].plot(lags * 1000, scores, color="black")
    ax[0, 0].axvline(best_lag * 1000, color="crimson", label=f"best = {best_lag * 1000:.1f} ms")
    ax[0, 0].set(xlabel="Assumed lag [ms]", ylabel="Forward/reverse mismatch", title="Lag fit")
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
    speed = np.nanmedian(np.abs(joined.loc[joined.direction != 0, "az_rate_arcmin_s"]))
    ax[1, 0].text(0.03, 0.85, "Scan-lag correction", fontsize=15, weight="bold")
    ax[1, 0].text(0.03, 0.60, f"Best lag: {best_lag * 1000:.1f} ms\nMedian |Az rate|: {speed:.4g} arcmin/s\nTypical |Az shift|: {abs(speed * best_lag):.4g} arcmin", fontsize=13)
    fig.savefig(args.outdir / "scanning_diagnostics.png", dpi=180)
    plt.close(fig)
    print(f"Matched samples: {len(joined)}")
    print(f"Best scan lag: {best_lag * 1000:.1f} ms")
    print(f"Typical Az correction: {abs(speed * best_lag):.5g} arcmin ({abs(speed * best_lag) * 60:.5g} arcsec)")
    print(f"Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
