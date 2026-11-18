#!/usr/bin/env python3
"""ON較正後の複素電場を同じ指向格子で積分し、Holography用CSVを出力する。"""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from holography_mc import calibrate_amplitude, calibrate_phase


def read_and_calibrate(path, args):
    data = pd.read_csv(path, encoding="utf-8-sig")
    data.columns = data.columns.str.strip()
    required = ["Epoch", "Length", "Amp", "Phase", "SNR", "Az_Offset", "El_Offset"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{path}: 必須列がありません: {', '.join(sorted(missing))}")
    if "Beam_State" in data:
        raise ValueError(f"{path}: 積分済み入力は使用できません。元のbeam.txtを指定してください。")
    epoch = data["Epoch"]
    data["Epoch"] = pd.to_datetime(epoch, format="%Y/%j %H:%M:%S.%f", errors="coerce").fillna(
        pd.to_datetime(epoch, format="%Y/%j %H:%M:%S", errors="coerce"))
    for column in required[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.empty or data["Epoch"].isna().any() or not np.isfinite(data[required[1:]].to_numpy()).all():
        raise ValueError(f"{path}: 空の入力、無効な時刻または数値があります。")
    if (data.Length <= 0).any() or (data.Amp < 0).any() or (data.SNR <= 0).any():
        raise ValueError(f"{path}: Length > 0、Amp >= 0、SNR > 0 が必要です。")
    data = data.sort_values("Epoch").copy()
    center = (data.Az_Offset == 0) & (data.El_Offset == 0)
    stationary = center & np.isclose(data.Length, args.on_length, rtol=0, atol=1e-6)
    selected = stationary.copy() if args.exclude_center_crossings else center.copy()
    if args.on_amp_min is not None:
        selected &= data.Amp >= args.on_amp_min
    indices = np.flatnonzero(selected)
    times = data.Epoch.to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9
    # Preserve correlator phase convention. Holography applies antenna-34's sign once.
    field = data.Amp.to_numpy() * np.exp(1j * np.deg2rad(data.Phase.to_numpy()))
    field, _, _ = calibrate_phase(field, times, indices, args.on_phase_linear_fit)
    field, reference = calibrate_amplitude(field, times, indices)
    data["field"] = field
    # Mean-amplitude SNR convention used by Holography. ON-reference errors are not included.
    data["sigma"] = np.sqrt(2 / np.pi) * data.Amp.to_numpy() / data.SNR.to_numpy() / reference
    data = data.loc[~stationary].copy()
    if data.empty:
        raise ValueError(f"{path}: 停止ON以外の走査点がありません。")
    print(f"{path}: ON {len(indices)}点で較正、走査点 {len(data)}点")
    return data


def integrate(args):
    paths = [Path(path).resolve() for path in args.inputs]
    output = Path(args.out).resolve()
    if output in paths:
        raise ValueError("出力先に入力ファイルは指定できません。")
    if output.exists() and not args.overwrite:
        raise ValueError("出力先が存在します。上書きする場合は --overwrite を指定してください。")
    if not np.isfinite(args.grid_step).all() or np.any(np.asarray(args.grid_step) <= 0):
        raise ValueError("--grid-step は正の有限値で指定してください。")
    if not np.isfinite(args.grid_origin).all() or not np.isfinite(args.on_length) or args.on_length <= 0:
        raise ValueError("格子原点とON積分時間が不正です。")
    if args.on_amp_min is not None and (not np.isfinite(args.on_amp_min) or args.on_amp_min < 0):
        raise ValueError("--on-amp-min は0以上の有限値で指定してください。")
    frames, hashes = [], set()
    for number, path in enumerate(paths):
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest in hashes:
            raise ValueError(f"重複した入力です: {path}")
        hashes.add(digest)
        try:
            frame = read_and_calibrate(path, args)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        frame["source"] = number
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for index, axis in enumerate(("Az", "El")):
        data[axis + "_cell"] = np.rint(
            (data[axis + "_Offset"] - args.grid_origin[index]) / args.grid_step[index]).astype(np.int64)
    rows = []
    for (az_cell, el_cell), group in data.groupby(["Az_cell", "El_cell"], sort=True):
        exposure = group.Length.sum()
        weights = group.Length.to_numpy() / exposure
        field = np.sum(weights * group.field.to_numpy())
        sigma = np.sqrt(np.sum((weights * group.sigma.to_numpy()) ** 2))
        snr = np.sqrt(2 / np.pi) * abs(field) / sigma if sigma > 0 else 0.0
        rows.append(dict(Epoch=group.Epoch.min().strftime("%Y/%j %H:%M:%S.%f"),
                         Length=exposure, Amp=abs(field), Phase=np.rad2deg(np.angle(field)), SNR=snr,
                         Az_Offset=args.grid_origin[0] + az_cell * args.grid_step[0],
                         El_Offset=args.grid_origin[1] + el_cell * args.grid_step[1],
                         Beam_State="on_calibrated", Antenna_m=args.antenna,
                         Grid_Az_arcmin=args.grid_step[0], Grid_El_arcmin=args.grid_step[1],
                         N_samples=len(group), N_files=group.source.nunique()))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.15g")
    print(f"保存: {output} ({len(paths)}ファイル、{len(rows)}格子点、Length加重複素平均)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "各ファイルを自身のON点で振幅・位相補正した後、最寄りの共通格子に割り当てます。"
        "座標・位相の追加位置合わせは行いません。同じ周波数・偏波・アンテナの入力を指定してください。"))
    parser.add_argument("--in", "--input", dest="inputs", nargs="+", required=True, help="入力beam.txt（複数指定可）")
    parser.add_argument("--out", "--output", required=True, help="Holographyに渡す出力txt（CSV形式）")
    parser.add_argument("--antenna", type=int, choices=(32, 34), required=True)
    parser.add_argument("--grid-step", nargs=2, type=float, default=(1.5, 1.5), metavar=("AZ", "EL"), help="共通格子間隔 [arcmin]（既定: 1.5 1.5）")
    parser.add_argument("--grid-origin", nargs=2, type=float, default=(0, 0), metavar=("AZ", "EL"), help="共通格子原点 [arcmin]（既定: 0 0）")
    parser.add_argument("--on-length", type=float, default=10, help="停止ONのLength [s]（既定: 10）")
    parser.add_argument("--on-amp-min", type=float, help="較正に採用するONの最小Amp")
    parser.add_argument("--exclude-center-crossings", action="store_true", help="中心通過点をON較正から除外")
    parser.add_argument("--on-phase-linear-fit", action="store_true", help="ON位相を線形補間ではなく直線近似")
    parser.add_argument("--overwrite", action="store_true")
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    try:
        integrate(args)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        parser.exit(2, f"エラー: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
