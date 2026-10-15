#!/usr/bin/env python3
"""Re-integrate the central raster with frinZ and plot a small complex beam.

Examples:
  python3 beam_analysis.py --in I26287YA --antenna 32 --scan-axis az
  python3 beam_analysis.py --in I26287YA --antenna 34 --scan-axis el \
      --scan-lag-ms 100 --dry-run

Uses observation XML files to identify offset correlation files and their start
times/durations, and PRD pointing knots to select the +/-12 arcmin square.
frinZ is invoked once per integration with --length and --skip only.
--step-samples is this script's spacing, not an option passed to frinZ.
Sample counts are converted using XML stream/output (or --samples-per-second).
Positive lag shifts the selected direction's integration later, as in
corr_fringe.py; output Epoch/coordinates retain the nominal integration times.
This script does not run gico3 or apply ON phase calibration/34m phase inversion.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


@dataclass(frozen=True)
class Scan:
    path: Path
    start: datetime
    duration: float
    sample_rate: float


def read_prd(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 9 or not re.fullmatch(r"\d{13}", fields[0]):
            continue
        t = datetime.strptime(fields[0], "%Y%j%H%M%S")
        az, el = float(fields[7]), float(fields[8])
        if not np.isfinite([az, el]).all():
            raise ValueError(f"nonfinite PRD position: {line}")
        records.append((t, az, el))
    records.sort()
    if len(records) < 2 or any(b[0] <= a[0] for a, b in zip(records, records[1:])):
        raise ValueError("PRD needs at least two strictly increasing time knots")
    return records


def pointing(records, times, t):
    right = bisect_right(times, t)
    if right == 0 or right >= len(records):
        return None
    a, b = records[right - 1], records[right]
    dt = (b[0] - a[0]).total_seconds()
    vx, vy = (b[1] - a[1]) / dt, (b[2] - a[2]) / dt
    elapsed = (t - a[0]).total_seconds()
    return a[1] + vx * elapsed, a[2] + vy * elapsed, vx, vy


def scan_direction(position, axis):
    primary, cross = position[2:] if axis == "az" else position[2:][::-1]
    return int(np.sign(primary)) if abs(primary) > 1e-5 and abs(cross) < 1e-8 else 0


def read_scans(workdir, xml_paths, on_length, sample_rate_override=None):
    scans = {}
    for path in xml_paths:
        root = ET.parse(path).getroot()
        label = root.findtext("stream/label")
        if label is None:
            raise ValueError(f"XML lacks stream/label: {path}")
        stream = root.find("stream")
        rate_text = root.findtext("stream/output") or (stream.get("output") if stream is not None else None)
        if sample_rate_override is None and rate_text is None:
            raise ValueError(f"XML lacks stream/output: {path}; specify --samples-per-second")
        sample_rate = float(sample_rate_override if sample_rate_override is not None else rate_text)
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError(f"invalid correlation output rate in {path}")
        for process in root.findall("process"):
            start = datetime.strptime(process.findtext("epoch").strip(), "%Y/%j %H:%M:%S")
            skip = float(process.findtext("skip"))
            length = float(process.findtext("length"))
            if not np.isfinite([skip, length]).all() or not 0 <= skip < length:
                raise ValueError(f"invalid XML length/skip: {path}")
            duration = length - skip
            if math.isclose(duration, on_length, rel_tol=0, abs_tol=1e-9):
                continue
            start += timedelta(seconds=skip)
            stamp = start.strftime("%Y%j%H%M%S")
            for cor in sorted((workdir / "step_cor").glob(f"*_{stamp}_{label.strip()}.cor")):
                scan = Scan(cor.resolve(), start, duration, sample_rate)
                if cor in scans and scans[cor] != scan:
                    raise ValueError(f"conflicting XML metadata for {cor}")
                scans[cor] = scan
    return sorted(scans.values(), key=lambda scan: (scan.start, str(scan.path)))


def selected_scan(scan, records, times, axis, half_width):
    """Select real scan legs whose PRD trajectory intersects the square."""
    end = scan.start + timedelta(seconds=scan.duration)
    for a, b in zip(records, records[1:]):
        lo, hi = max(a[0], scan.start), min(b[0], end)
        if hi <= lo:
            continue
        p = pointing(records, times, lo + (hi - lo) / 2)
        if p is None or not scan_direction(p, axis):
            continue
        vx, vy = p[2:]
        az0 = p[0] - vx * (hi - lo).total_seconds() / 2
        el0 = p[1] - vy * (hi - lo).total_seconds() / 2
        az1 = az0 + vx * (hi - lo).total_seconds()
        el1 = el0 + vy * (hi - lo).total_seconds()
        if (min(az0, az1) <= half_width and max(az0, az1) >= -half_width
                and min(el0, el1) <= half_width and max(el0, el1) >= -half_width):
            return True
    return False


def make_tasks(scan, records, times, args):
    """Select nominal central windows, then shift their physical read positions."""
    rate = scan.sample_rate
    chosen = 1 if args.forward_direction == "increasing" else -1
    tasks = []
    total_samples = math.floor(scan.duration * rate + 1e-6)
    for nominal in range(0, total_samples - args.length_samples + 1, args.step_samples):
        epoch = scan.start + timedelta(seconds=nominal / rate)
        p = pointing(records, times, epoch + timedelta(seconds=args.length_samples / rate / 2))
        if p is None:
            continue
        direction = scan_direction(p, args.scan_axis)
        if not direction or abs(p[0]) > args.half_width or abs(p[1]) > args.half_width:
            continue
        requested = args.scan_lag_ms if direction == chosen else 0.0
        units = requested * rate / 1000
        # Round to the nearest actual output sample, ties away from zero.
        lag_units = int(math.copysign(math.floor(abs(units) + 0.5), units))
        actual = nominal + lag_units
        if actual < 0 or actual + args.length_samples > total_samples:
            continue
        command = [args.frinz, "--in", str(scan.path),
                   "--length", str(args.length_samples), "--skip", str(actual)]
        tasks.append((command, epoch, p, lag_units * 1000 / rate))
    return tasks


def parse_result(line):
    if not re.match(r"^\s*\d{4}/\d{3}\s", line):
        return None
    fields = line.split()
    if len(fields) < 8:
        raise ValueError(f"incomplete frinZ result: {line.rstrip()}")
    # Follow corr_fringe's conventions: 20-column frinZ output has fractional
    # amplitude, whereas legacy 17-column fringe output uses percent.
    amp = float(fields[5]) if len(fields) >= 20 else float(fields[5]) / 100
    length, snr, phase = float(fields[4]), float(fields[6]), float(fields[7])
    if not np.isfinite([amp, length, snr, phase]).all() or length <= 0:
        raise ValueError(f"invalid frinZ values: {line.rstrip()}")
    return length, amp, phase, snr


def process_scan(scan, tasks, args, outdir):
    raw_path = outdir / (scan.path.stem + ".stdout.txt")
    err_path = outdir / (scan.path.stem + ".stderr.txt")
    result = []
    print(f"[INFO] {scan.path.name}: {len(tasks)} central integrations; "
          f"output={scan.sample_rate:g} Hz; "
          f"length={args.length_samples / scan.sample_rate * 1000:g} ms", flush=True)
    requested_units = args.scan_lag_ms * scan.sample_rate / 1000
    quantized = math.copysign(math.floor(abs(requested_units) + 0.5), requested_units)
    if not math.isclose(requested_units, quantized, abs_tol=1e-8):
        print(f"[WARN] requested lag={args.scan_lag_ms:g} ms; applied lag="
              f"{quantized * 1000 / scan.sample_rate:g} ms on selected direction "
              f"(sample spacing={1000 / scan.sample_rate:g} ms)", file=sys.stderr)
    with raw_path.open("w") as raw, err_path.open("w") as errors:
        for index, (command, epoch, p, applied_lag) in enumerate(tasks, start=1):
            command_text = shlex.join(command)
            raw.write("# " + command_text + "\n")
            errors.write("# " + command_text + "\n")
            completed = subprocess.run(command, cwd=args.workdir.resolve(),
                                       capture_output=True, text=True)
            raw.write(completed.stdout)
            errors.write(completed.stderr)
            raw.flush()
            errors.flush()
            if completed.returncode:
                raise RuntimeError(f"frinZ failed ({completed.returncode}): {command_text}; see {err_path}")
            parsed = [value for line in completed.stdout.splitlines()
                      if (value := parse_result(line)) is not None]
            if len(parsed) != 1:
                raise ValueError(f"expected one result per --length/--skip call, got {len(parsed)}; see {raw_path}")
            length, amp, phase, snr = parsed[0]
            expected = args.length_samples / scan.sample_rate
            if not math.isclose(length, expected, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{scan.path}: expected {expected:g} s, got {length:g} s; check XML output rate; see {raw_path}")
            az, el, vx, vy = p
            result.append(dict(Epoch=epoch.strftime("%Y/%j %H:%M:%S.%f"),
                               Length=length, Amp=amp, Phase=phase, SNR=snr,
                               Az_Offset=az, El_Offset=el,
                               Az_Rate_arcmin_s=vx, El_Rate_arcmin_s=vy,
                               Applied_Lag_ms=applied_lag))
            if index == 1 or index % 50 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} integrations", flush=True)
    return result


def plot_map(rows, args, outdir):
    # Interpolate the complex field, not wrapped phase. Leave outside the
    # sampled convex hull empty; the text file always contains raw samples.
    points = np.array([(r["Az_Offset"], r["El_Offset"]) for r in rows])
    if len(points) < 3 or np.linalg.matrix_rank(points - points[0]) < 2:
        print("[WARN] fewer than two distinct raster rows: ANA_beam.txt saved, no 2D map", file=sys.stderr)
        return
    field = np.array([r["Amp"] * np.exp(1j * np.deg2rad(r["Phase"])) for r in rows])
    axis = np.linspace(-args.half_width, args.half_width, 241)
    xx, yy = np.meshgrid(axis, axis)
    grid = griddata(points, field.real, (xx, yy), method="linear") + 1j * griddata(
        points, field.imag, (xx, yy), method="linear")
    peak = np.nanmax(np.abs(grid))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("no positive beam amplitude for plotting")
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.maximum(20 * np.log10(np.abs(grid) / peak), args.db_min)
    plt.rcParams.update({"axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    for ax, values, cmap, low, high, label in (
            (axes[0], db, "inferno", args.db_min, 0, "Normalized Amplitude [dB]"),
            (axes[1], np.angle(grid, deg=True), "twilight", -180, 180, "Raw Phase [deg]")):
        im = ax.pcolormesh(xx, yy, values, shading="nearest", cmap=cmap, vmin=low, vmax=high)
        ax.set(xlabel="Az offset [arcmin]", xlim=(args.half_width, -args.half_width),
               ylim=(-args.half_width, args.half_width))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.linspace(-args.half_width, args.half_width, 5))
        ax.set_yticks(np.linspace(-args.half_width, args.half_width, 5))
        fig.colorbar(im, ax=ax, label=label)
    axes[0].set_ylabel("El offset [arcmin]")
    fig.savefig(outdir / "ANA_beam_map.png", dpi=150)
    pdf = outdir / "pdf"
    pdf.mkdir(exist_ok=True)
    fig.savefig(pdf / "ANA_beam_map.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", "--input", "--workdir", dest="workdir", type=Path, required=True,
                        metavar="DIR", help="作業ディレクトリ（step_cor・XML・PRDを含む）。--workdirも互換用に使用可。")
    parser.add_argument("--antenna", type=int, choices=(32, 34), default=32, help="measurement antenna; selects PRD, does not flip input phases")
    parser.add_argument("--scan-axis", choices=("az", "el"), default="az")
    parser.add_argument("--prd", type=Path, help="explicit PRD path (relative to invocation directory)")
    parser.add_argument("--xml", type=Path, action="append", help="explicit observation XML; repeatable")
    parser.add_argument("--output", type=Path, help="summary path (default: WORKDIR/ANA_beam.txt)")
    parser.add_argument("--outdir", type=Path, help="raw output and map directory (default: WORKDIR/beam_analysis)")
    parser.add_argument("--on-length", type=float, default=10, help="XML length-skip identifying ON scans [s]")
    parser.add_argument("--scan-lag-ms", type=float, default=0, help="integration shift [ms] for --forward-direction; positive=later, negative=earlier")
    parser.add_argument("--forward-direction", choices=("increasing", "decreasing"), default="increasing")
    parser.add_argument("--length-samples", type=int, default=10, help="frinZ --length (default: 10)")
    parser.add_argument("--step-samples", type=int, default=10, help="this script's start spacing in samples; implemented with repeated frinZ --skip (default: 10)")
    parser.add_argument("--samples-per-second", type=float, default=None, help="override correlation output rate [Hz]; default: XML stream/output")
    parser.add_argument("--half-width", type=float, default=12, help="Az/El half-width [arcmin] (default: 12)")
    parser.add_argument("--db-min", type=float, default=-45)
    parser.add_argument("--frinz", default="frinZ", help="frinZ executable")
    parser.add_argument("--dry-run", action="store_true", help="select files and print commands without executing frinZ")
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 1
    args = parser.parse_args(argv)
    if "/" in args.frinz:
        args.frinz = str(Path(args.frinz).resolve())
    for name in ("half_width", "on_length"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.samples_per_second is not None and (not math.isfinite(args.samples_per_second) or args.samples_per_second <= 0):
        parser.error("--samples-per-second must be finite and positive")
    if args.length_samples <= 0 or args.step_samples <= 0 or not math.isfinite(args.scan_lag_ms):
        parser.error("length/step must be positive and lag finite")
    if not math.isfinite(args.db_min) or args.db_min >= 0:
        parser.error("--db-min must be finite and negative")
    workdir = args.workdir.resolve()
    prd = args.prd or workdir / f"{workdir.name}{args.antenna}.prd"
    xmls = args.xml or sorted(workdir.glob("*.xml"))
    if not xmls:
        raise ValueError(f"no XML files in {workdir}; specify --xml")
    records = read_prd(prd)
    times = [r[0] for r in records]
    scans = [s for s in read_scans(workdir, xmls, args.on_length, args.samples_per_second)
             if selected_scan(s, records, times, args.scan_axis, args.half_width)]
    if not scans:
        raise ValueError("no offset .cor files intersect the selected raster square; check PRD/XML/scan axis")
    print(f"Selected {len(scans)} offset .cor files; PRD={prd}")
    plans = [(scan, make_tasks(scan, records, times, args)) for scan in scans]
    plans = [(scan, tasks) for scan, tasks in plans if tasks]
    if not plans:
        raise ValueError("no full integration windows inside the selected square")
    for scan, tasks in plans:
        print(f"{scan.path.name}: {len(tasks)} calls; output={scan.sample_rate:g} Hz; "
              f"integration={args.length_samples / scan.sample_rate * 1000:g} ms")
        if args.dry_run:
            for command, _, _, _ in tasks:
                print("+ " + shlex.join(command))
    if args.dry_run:
        return 0
    outdir = args.outdir or workdir / "beam_analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scan, tasks in plans:
        rows.extend(process_scan(scan, tasks, args, outdir))
    if not rows:
        raise ValueError("no frinZ samples retained inside the selected square")
    rows.sort(key=lambda r: r["Epoch"])
    output = args.output or workdir / "ANA_beam.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_map(rows, args, outdir)
    print(f"Saved {len(rows)} raw samples: {output}")
    print(f"Map/raw frinZ output: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, ET.ParseError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)

