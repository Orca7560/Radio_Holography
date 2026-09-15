#!/usr/bin/env python3

import sys
import os
import glob
import argparse
from bisect import bisect_right
from datetime import datetime, timedelta
import numpy as np

def parse_skd_offsets(skd_filepath):
    """SKDファイルからスキャン時刻とオフセットのマッピングを作成する"""
    skd_map = {}
    try:
        with open(skd_filepath, 'r', errors='ignore') as f:
            in_sked_block = False
            for line in f:
                if line.strip().startswith('$SKED'):
                    in_sked_block = True
                    continue
                if line.strip().startswith('$') and in_sked_block:
                    break
                if in_sked_block and line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            scan_time_str = parts[1]
                            year = scan_time_str[:2]
                            doy = scan_time_str[2:5]
                            hour = scan_time_str[5:7]
                            minute = scan_time_str[7:9]
                            second = scan_time_str[9:11]
                            
                            formatted_time = f"20{year}/{doy} {hour}:{minute}:{second}.000"

                            az_offset = float(parts[3])
                            el_offset = float(parts[4])
                            skd_map[formatted_time] = (az_offset, el_offset)
                        except (ValueError, IndexError):
                            continue
    except FileNotFoundError:
        print(f"❌ エラー: 指定されたSKDファイルが見つかりません: {skd_filepath}")
        return None
    except Exception as e:
        print(f"❌ SKDファイルの読み込み中にエラーが発生しました: {e}")
        return None
        
    print(f"✅ SKDファイルから {len(skd_map)} 点のスキャン情報を読み込みました。")
    return skd_map


def parse_prd_positions(prd_filepath):
    """Read PRD time-tagged Az/El offsets.

    The PRD records pointing knots.  Between adjacent records the commanded
    position is treated as a constant-velocity straight-line trajectory.
    """
    records = []
    try:
        with open(prd_filepath, 'r', errors='ignore') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 9 or not parts[0].isdigit() or len(parts[0]) != 13:
                    continue
                try:
                    records.append((
                        datetime.strptime(parts[0], '%Y%j%H%M%S'),
                        float(parts[7]), float(parts[8]),
                    ))
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"❌ エラー: 指定されたPRDファイルが見つかりません: {prd_filepath}")
        return None
    except Exception as e:
        print(f"❌ PRDファイルの読み込み中にエラーが発生しました: {e}")
        return None

    records.sort(key=lambda record: record[0])
    if len(records) < 2:
        print("❌ エラー: PRDから2点以上の有効な指令位置を読み取れませんでした。")
        return None
    print(f"✅ PRDファイルから {len(records)} 点の指令位置を読み込みました。")
    return records


def prd_offset_at(prd_records, prd_times, epoch, integration_length_s, scan_lag_s,
                  az_drive_speed=3.0):
    """Return midpoint pointing and scan-lag-corrected pointing from PRD.

    Epoch is the start of the fringe integration.  Since a correlation value
    represents the full interval, the pointing is evaluated at its midpoint.
    During a constant-El raster scan, the position is derived from the PRD
    segment start time and the integration length with the commanded Az speed
    (default: 3 arcmin/s), rather than from the interval's PRD coordinate
    difference. A positive lag applies theta_corr = theta - velocity * lag.
    """
    sample_time = epoch + timedelta(seconds=integration_length_s / 2.0)
    right = bisect_right(prd_times, sample_time)
    if right == 0 or right >= len(prd_records):
        return np.nan, np.nan, np.nan, np.nan

    t0, az0, el0 = prd_records[right - 1]
    t1, az1, el1 = prd_records[right]
    duration_s = (t1 - t0).total_seconds()
    if duration_s <= 0:
        return np.nan, np.nan, np.nan, np.nan

    fraction = (sample_time - t0).total_seconds() / duration_s
    az_rate = (az1 - az0) / duration_s
    el_rate = (el1 - el0) / duration_s
    # 本走査（El一定）では、PRDの区間始点からの経過時間と指定駆動速度で
    # 位置を決める。向きだけはPRDのAz変化から判定し、折返しの同時Az/El移動は
    # PRD座標をそのまま線形補間する。
    if az_drive_speed is not None and abs(el_rate) < 1e-10 and abs(az_rate) > 1e-10:
        az_rate = np.copysign(az_drive_speed, az_rate)
    az = az0 + (sample_time - t0).total_seconds() * az_rate
    el = el0 + fraction * (el1 - el0)
    return az - az_rate * scan_lag_s, el - el_rate * scan_lag_s, az_rate, el_rate

def merge_fringe_results(input_root=".", output_file=None, add_delay=False, skd_filepath=None, prd_filepath=None,
                         scan_lag_ms=0.0, az_drive_speed=None, band=None, is_maser=False):
    """
    fringe_resultsのtxtファイルを統合し、オプションに応じて情報を追加する。
    is_maserがTrueの場合、SNRの代わりにFrequencyを取得する。
    """
    dir_name = f"fringe_results_{band}" if band else "fringe_results"
    input_directory = os.path.join(input_root, dir_name)
    if output_file is None:
        out_name = f"beam_{band}.txt" if band else "beam.txt"
        output_file = os.path.join(input_root, out_name)
    
    if not os.path.isdir(input_directory):
        print(f"[ERROR] ディレクトリ '{input_directory}' が見つかりません。")
        return

    txt_files = sorted(glob.glob(os.path.join(input_directory, '*.txt')))

    if not txt_files:
        print(f"[WARN] '{input_directory}' 内に処理対象の.txtファイルが見つかりません。")
        return

    skd_data = None
    if skd_filepath:
        skd_data = parse_skd_offsets(skd_filepath)
        if skd_data is None:
            return

    prd_data = None
    if prd_filepath:
        prd_data = parse_prd_positions(prd_filepath)
        if prd_data is None:
            return
        prd_times = [record[0] for record in prd_data]
        print(f"[INFO] PRDの連続補間と走査遅れ補正を使用します: {scan_lag_ms:g} ms")
        if az_drive_speed is not None:
            print(f"[INFO] 本走査のAz駆動速度: {az_drive_speed:g} arcmin/s")

    print(f"[INFO] {len(txt_files)}個のファイルを検出しました。統合を開始します...")

    with open(output_file, 'w') as outfile:
        # --maser オプションの有無でヘッダーを変更
        # Length is retained so downstream holography can distinguish the
        # dedicated 10-s ON reference scan from a normal raster sample at
        # the same (Az, El) = (0, 0) position.
        header_parts = ["Epoch", "Length", "Amp", "Phase"]
        if is_maser:
            header_parts.append("Frequency")
        else:
            header_parts.append("SNR")
            
        if add_delay:
            header_parts.append("Res-Delay")
        if prd_data:
            header_parts.extend(["Az_Offset", "El_Offset", "Az_Rate_arcmin_s", "El_Rate_arcmin_s"])
        elif skd_data:
            header_parts.extend(["Az_Offset", "El_Offset"])
        outfile.write(", ".join(header_parts) + "\n")

        for filepath in txt_files:
            print(f"  -> 処理中: {filepath}")
            with open(filepath, 'r') as infile:
                for line in infile:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    
                    if len(parts) < 17:
                        continue
                        
                    try:
                        epoch = parts[0] + " " + parts[1]
                        
                        is_fringe_output = len(parts) >= 20
                        amp_index = 5
                        snr_index = 6
                        phase_index = 7
                        freq_index = 8
                        delay_index = 8 if is_fringe_output else 9

                        amp_val = float(parts[amp_index])
                        amp = amp_val / 100.0 if not is_fringe_output else amp_val
                        phase = float(parts[phase_index])
                        integration_length_s = float(parts[4])
                        
                        output_parts = [
                            epoch,
                            f"{integration_length_s:.6f}",
                            f"{amp:.6f}",
                            f"{phase:.3f}"
                        ]

                        # --maser オプションの有無で取得する値（FrequencyかSNR）を切り替え
                        if is_maser:
                            freq = float(parts[freq_index])
                            output_parts.append(f"{freq:.6f}")
                        else:
                            snr = float(parts[snr_index])
                            output_parts.append(f"{snr:.1f}")

                        if add_delay:
                            delay = float(parts[delay_index])
                            output_parts.append(f"{delay:.2f}")

                        if prd_data:
                            try:
                                epoch_dt = datetime.strptime(epoch, '%Y/%j %H:%M:%S.%f')
                            except ValueError:
                                epoch_dt = datetime.strptime(epoch, '%Y/%j %H:%M:%S')
                            az, el, az_rate, el_rate = prd_offset_at(
                                prd_data, prd_times, epoch_dt, integration_length_s, scan_lag_ms / 1000.0,
                                az_drive_speed=az_drive_speed
                            )
                            output_parts.extend([
                                f"{az:.6f}", f"{el:.6f}",
                                f"{az_rate:.6f}", f"{el_rate:.6f}",
                            ])
                        elif skd_data:
                            epoch_key = epoch.split('.')[0] + '.000'
                            az, el = skd_data.get(epoch_key, (np.nan, np.nan))
                            output_parts.extend([f"{az:.1f}", f"{el:.1f}"])
                        
                        new_line = ", ".join(output_parts) + "\n"
                        outfile.write(new_line)

                    except (ValueError, IndexError) as e:
                        print(f"    [WARN] 行の解析に失敗しました: {line.strip()} ({e})")

    print(f"\n[SUCCESS] 統合が完了しました。")
    print(f"出力ファイル: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fringe結果のテキストファイルを統合します。")
    parser.add_argument("--input", "--in", default=".", metavar="DIR",
                        help="観測ディレクトリ（中の fringe_results/ を統合）。未指定時はカレントディレクトリ。")
    parser.add_argument("--output", "--out", default="beam.txt", metavar="FILE",
                        help="出力するbeamファイル。未指定時は beam.txt。")
    parser.add_argument("--prd", metavar="FILE",
                        help="使用するPRDファイル。未指定時は入力ディレクトリから32m用PRDを自動検出。")
    parser.add_argument("--antenna", choices=("32", "34"), default="32",
                        help="自動検出時に使うアンテナ径（既定値: 32）。")
    parser.add_argument("--add-delay", action="store_true", help="サマリーファイルにRes-Delayの列を追加します。")
    parser.add_argument("--scan-lag-ms", type=float, default=20.0, metavar="MS",
                        help="PRD座標へ適用する走査遅れ[ms]。既定値: 20。corr_fringe側で補正済みなら0を指定。")
    parser.add_argument("--az-drive-speed", type=float, default=3.0, metavar="ARCMIN_S",
                        help="本走査中のAz駆動速度の絶対値[arcmin/s]（既定値: 3）。")
    parser.add_argument("--band-split", type=int, metavar="DIVISIONS",
                        help="8192-8704MHzを指定した分割数で分割した結果を統合します。512の約数を指定してください。")
    parser.add_argument("--maser", action="store_true", help="SNRの代わりにFrequencyを取得してファイルに書き込みます。")
    args = parser.parse_args()

    input_root = args.input
    prd_filepath = args.prd
    if prd_filepath is None:
        base = os.path.basename(os.path.normpath(input_root))
        expected = os.path.join(input_root, f"{base}{args.antenna}.prd")
        candidates = [expected] if os.path.isfile(expected) else sorted(
            glob.glob(os.path.join(input_root, f"*{args.antenna}.prd"))
        )
        if len(candidates) == 1:
            prd_filepath = candidates[0]
        else:
            parser.error("PRDを自動検出できません。--prd FILE を指定してください。")
    print(f"[INFO] PRDファイル: {prd_filepath}")

    if args.band_split is not None:
        divisions = args.band_split
        if divisions <= 0 or 512 % divisions != 0:
            parser.error("全帯域幅 512MHz の約数を --band-split に指定してください。")
        if args.output != "beam.txt":
            parser.error("--band-split 使用時は --output を指定せず、帯域ごとの beam_*.txt を出力してください。")
        step = 512 // divisions
        for i in range(divisions):
            band = f"{8192 + i*step}_{8192 + (i+1)*step}"
            print(f"\n===== バンド {band} の処理を開始します =====")
            merge_fringe_results(input_root, prd_filepath=prd_filepath,
                                 add_delay=args.add_delay, scan_lag_ms=args.scan_lag_ms,
                                 az_drive_speed=args.az_drive_speed, band=band, is_maser=args.maser)
    else:
        merge_fringe_results(input_root, output_file=args.output, prd_filepath=prd_filepath,
                             add_delay=args.add_delay, scan_lag_ms=args.scan_lag_ms,
                             az_drive_speed=args.az_drive_speed, is_maser=args.maser)
