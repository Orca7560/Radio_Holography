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
             row_gap_s: float) -> pd.DataFrame:
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
    sign = np.sign(vx)
    sign[np.abs(vx) < 1e-5] = 0
    boundary = np.zeros(len(data), dtype=bool)
    boundary[0] = True
    boundary[1:] = ((dt > row_gap_s) | (sign != np.r_[sign[0], sign[:-1]])
                    | (np.abs(vy) > 0.2))
    data["row_id"] = np.cumsum(boundary) - 1
