#!/usr/bin/env python3
"""Forward-only integration-window correction.

Example:
  python corr_fringe_v5.py I25231Y --only-frinZ-freq --scan-half --scan-lag-ms -100

Negative lag moves integration boundaries earlier; positive lag moves them
later. Only the selected Az direction is corrected. Integrations shifted
completely outside the .cor data are skipped, while an integration crossing
an edge is shortened. XML/SKD and nominal output Epochs are unchanged. With
output=100, length/skip units are 10 ms, not milliseconds.
"""
import sys
import glob
import subprocess
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import re
import math
from astropy.time import Time
import argparse

output = 100  # xml ファイルの output 数
output_ON = 1 # ON の output 数
corr_step = 0.5 # offset のスキャン時間 (s)
on_length = 10 # on 点のスキャン時間 (s)
skip_step = output * corr_step # fringe実行時のskipの間隔 (floatになる可能性あり)
scan_time = 41
skip_max = scan_time * output - skip_step # fringe実行時のskipの最大値

def run_external_command(command_list, capture=False):
    """外部コマンドを実行し、必要に応じて標準出力を返す"""
    if not capture:
        print(f"[INFO] 実行コマンド: {' '.join(command_list)}")
    try:
        result = subprocess.run(
            command_list, 
            check=True, 
            capture_output=capture, 
            text=True
        )
        if capture:
            return result.stdout
    except FileNotFoundError:
        print(f"[ERROR] コマンドが見つかりません: {command_list[0]}。")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] コマンドの実行に失敗 (終了コード: {e.returncode}): {' '.join(command_list)}")
        if capture:
            return (e.stdout or "") + (e.stderr or "")
    return None

def parse_skd_schedule(skd_filename, include_positions=False):
    """SKDファイルを解析し、ON点とoffsetスキャンの時刻(datetime)リストを返す"""
    on_points = []
    offset_scans = []
    positions = []
    try:
        with open(skd_filename, 'r') as f:
            in_sked_section = False
            for line in f:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue
                
                if line.startswith('$SKED'):
                    in_sked_section = True
                    continue
                elif line.startswith('$'):
                    in_sked_section = False
                    continue
                
                if in_sked_section:
                    parts = line.split()
                    if len(parts) >= 5:
                        time_str = parts[1]
                        time_str_clean = time_str.replace('.', '')
                        
                        if len(time_str_clean) >= 11 and time_str_clean.isdigit():
                            try:
                                az_off = float(parts[3])
                                el_off = float(parts[4])
                                
                                if '.' in time_str:
                                    main_time, frac_time = time_str.split('.')
                                    dt = datetime.strptime(main_time, '%y%j%H%M%S') + timedelta(milliseconds=float("0." + frac_time) * 1000)
                                else:
                                    dt = datetime.strptime(time_str, '%y%j%H%M%S')
                                
                                positions.append((dt, az_off, el_off))
                                if az_off == 0.0 and el_off == 0.0:
                                    on_points.append(dt)
                                else:
                                    offset_scans.append(dt)
                            except ValueError:
                                continue
    except Exception as e:
        print(f"[ERROR] SKDファイルの読み込み中にエラーが発生しました: {e}")
        
    if include_positions:
        return on_points, offset_scans, sorted(positions)
    return on_points, offset_scans


def infer_scan_direction(scan_start, positions):
    """Infer +Az/-Az from same-El SKD pairs inside one .cor scan.

    Do not guess from odd/even XML process numbers: ON scans, missing files
    and multiple XML files can all break that ordering.
    """
    scan_end = scan_start + timedelta(seconds=scan_time)
    points = [p for p in positions if scan_start <= p[0] <= scan_end]
    directions = set()
    for left, right in zip(points, points[1:]):
        if right[0] <= left[0] or not math.isclose(left[2], right[2], abs_tol=1e-6):
            continue
        delta_az = right[1] - left[1]
        if abs(delta_az) > 1e-6:
            directions.add(1 if delta_az > 0 else -1)
    if len(directions) != 1:
        raise ValueError(
            f"{scan_start}: SKDから単一のAz走査方向を判定できません。"
            "同じElの座標点が複数必要です（折り返しを含む区間も補正できません）。"
        )
    return directions.pop()


def lag_to_units(lag_ms, scan_half):
    """Convert milliseconds to integer frinZ length/skip units."""
    if not math.isfinite(lag_ms):
        raise ValueError("--scan-lag-ms には有限の数値を指定してください。")
    units = lag_ms * output / 1000.0
    if not math.isclose(units, round(units), abs_tol=1e-8):
        raise ValueError(f"--scan-lag-ms は {1000.0 / output:g} ms刻みで指定してください。")
    if abs(units) >= scan_time * output:
        raise ValueError("ラグの絶対値はoffsetスキャン全体の長さより小さくしてください。")
    return int(round(units))


def apply_forward_lag(tasks, lag_ms, total_length):
    """Move boundaries, keep nominal timestamps and clip to available data.

    -100 ms at 100 samples/s:
      (skip,length) (0,25),(25,50),(75,50),... -> (0,15),(15,50),(65,50),...
    An interval moved wholly outside [0,total_length] is omitted. A partially
    overlapping interval is clipped at the edge. Thus, for 10-ms integrations
    and -100 ms, the first ten nominal integrations are omitted and the
    integration nominally starting at 100 ms reads the data at 0--10 ms.
    """
    units = lag_ms * output / 1000.0
    if not math.isfinite(units) or not math.isclose(units, round(units), abs_tol=1e-8):
        raise ValueError(f"ラグは有限かつ {1000.0 / output:g} ms刻みで指定してください。")
    units = int(round(units))
    adjusted = []
    for index, task in enumerate(tasks):
        nominal_skip = task['skip']
        start = nominal_skip + units
        end = nominal_skip + task['length'] + units
        start = max(start, 0.0)
        end = min(end, total_length)
        if end <= start:
            continue
        adjusted.append(dict(task, skip=start, length=end - start,
                             nominal_skip=nominal_skip))
    if not adjusted:
        raise ValueError("ラグ補正後に実行可能な積分区間が残りません。")
    return adjusted

# processorの引数を追加し、fringeかfrinZかでパースを分ける
def parse_fringe_output(output_str, use_freq=False, processor="fringe"):
    """fringeまたはfrinZの標準出力を解析し、辞書として返す"""
    if not output_str:
        return None

    for line in output_str.splitlines():
        if line.startswith('#'): continue
        if re.match(r'^\s*\d{4}/\d{3}', line):
            parts = line.split()
            try:
                if use_freq:
                    if processor == "frinZ" and len(parts) >= 21:
                        data = {
                            "epoch": parts[0] + " " + parts[1],
                            "label": parts[2],
                            "source": parts[3],
                            "length": float(parts[4]),
                            "amp": float(parts[5]),
                            "snr": float(parts[6]),
                            "phase": float(parts[7]),
                            "frequency": float(parts[8]),
                            "noise_level": float(parts[9]),
                            "res_rate": float(parts[10]),
                            "x_az": float(parts[11]),
                            "x_el": float(parts[12]),
                            "x_h": float(parts[13]),
                            "y_az": float(parts[14]),
                            "y_el": float(parts[15]),
                            "y_h": float(parts[16]),
                            "mjd": float(parts[17]),
                            "rfi": parts[18],
                            "bp": parts[19],
                            "acf": parts[20],
                        }
                        return data
                    elif processor == "fringe" and len(parts) >= 20:
                        amp = float(parts[5])
                        snr = float(parts[6])
                        # Noise-level が出力されないため計算
                        noise_level = (amp / snr) if snr != 0 else 0.0
                        
                        epoch_str = parts[0] + " " + parts[1]
                        epoch_for_astropy = epoch_str.replace(" ", ":", 1).replace("/", ":", 1)
                        try:
                            mjd = Time(epoch_for_astropy, format='yday', scale='utc').mjd
                        except:
                            mjd = 0.0

                        data = {
                            "epoch": epoch_str,
                            "label": parts[2],
                            "source": parts[3],
                            "length": float(parts[4]),
                            "amp": amp,
                            "snr": snr,
                            "phase": float(parts[7]),
                            "frequency": float(parts[8]),
                            "noise_level": noise_level,
                            "res_rate": float(parts[9]),
                            "x_az": float(parts[10]),
                            "x_el": float(parts[11]),
                            "x_h": float(parts[12]),
                            "y_az": float(parts[13]),
                            "y_el": float(parts[14]),
                            "y_h": float(parts[15]),
                            "mjd": mjd,
                            "rfi": "-",
                            "bp": "-",
                            "acf": "-",
                        }
                        return data
                else:
                    if len(parts) >= 17:
                        data = {
                            "epoch": parts[0] + " " + parts[1],
                            "label": parts[2],
                            "source": parts[3],
                            "length": float(parts[4]),
                            "amp": float(parts[5])/100.0,
                            "snr": float(parts[6]),
                            "phase": float(parts[7]),
                            "res_delay": float(parts[9]),
                            "res_rate": float(parts[10]),
                            "x_az": float(parts[11]),
                            "x_el": float(parts[12]),
                            "x_h": float(parts[13]),
                            "y_az": float(parts[14]),
                            "y_el": float(parts[15]),
                            "y_h": float(parts[16]),
                        }
                        if len(parts) >= 20:
                           data["amp"] = float(parts[5])
                           data["res_delay"] = float(parts[8])
                           data["res_rate"] = float(parts[9])

                        return data
            except (ValueError, IndexError):
                continue
    return None

def run_gico3_steps(step_xml_files, cpu=None):
    """gico3の処理を実行する"""
    print("--- ステップ1: gico3の処理を開始します ---")
    for xml_file in step_xml_files:
        gico3_command = ["gico3", "--schedule", xml_file, "--raw-file", "./raw", "--cor-file", "./step_cor"]
        
        if cpu is not None:
            gico3_command.extend(["--cpu", str(cpu)])
            
        run_external_command(gico3_command)
    print("--- ステップ1: gico3の処理が完了しました ---\n")

def run_fringe_steps(step_xml_files, processor="fringe", use_freq_format=False,
                     add_quick_opt=False, bandscythe_only=False, scan_half=False,
                     scan_lag_ms=0.0, on_points=None, offset_scans=None, band=None,
                     scan_positions=None, forward_direction="increasing"):
    """fringeまたはfrinZの処理を実行し、結果を整形・保存する。"""
    processor_name = "fringe" if processor == "fringe" else "frinZ.py"
    band_label = f" [{band}]" if band else ""
    print(f"--- ステップ2: {processor_name}を実行し、結果を整形・保存します{band_label} ---")
    output_dir = f"fringe_results_{band}" if band else "fringe_results"
    
    # 画像の実行履歴に合わせてディレクトリ名を修正している場合は適宜変更してください
    cor_base_dir = f"./stepcor/{band}" if band else "./stepcor"
    on_points = on_points or []
    offset_scans = offset_scans or []
    scan_positions = scan_positions or []
    forward_sign = 1 if forward_direction == "increasing" else -1

    def is_on_scan(start, label):
        if any(abs((start - t).total_seconds()) <= 1.0 for t in on_points):
            return True
        if any(abs((start - t).total_seconds()) <= 1.0 for t in offset_scans):
            return False
        return "ON" in label.upper()

    # Validate every direction before overwriting any results or invoking frinZ.
    directions = {}
    if scan_lag_ms != 0:
        if processor != "frinZ":
            raise ValueError("--scan-lag-ms はfrinZ処理でのみ使用できます。")
        lag_to_units(scan_lag_ms, scan_half)
        if not scan_positions:
            raise ValueError("往復を判定するため、Az/El座標を含むSKDファイルが必要です。")
        for xml_file in step_xml_files:
            root = ET.parse(xml_file).getroot()
            label = root.find('stream').find('label').text.strip()
            for scan in root.findall('process'):
                start = datetime.strptime(scan.find('epoch').text.strip(), '%Y/%j %H:%M:%S')
                start += timedelta(seconds=int(scan.find('skip').text))
                if not is_on_scan(start, label):
                    directions[start] = infer_scan_direction(start, scan_positions)
        if forward_sign not in directions.values():
            raise ValueError("指定した往路方向のスキャンがありません。--forward-direction を確認してください。")

    os.makedirs(output_dir, exist_ok=True)

    def write_log_line(f_obj, fringe_data, use_freq):
        if use_freq:
            line = (
                f" {fringe_data['epoch']:<21} {fringe_data['label']:>6} {fringe_data['source']:>6} "
                f"{fringe_data['length']:>9.2f} {fringe_data['amp']:>12.6f} {fringe_data['snr']:>8.1f} "
                f"{fringe_data['phase']:>12.3f} {fringe_data['frequency']:>+12.7f} {fringe_data['noise_level']:>10.6f} "
                f"{fringe_data['res_rate']:>+10.6f} {fringe_data['x_az']:>7.3f} {fringe_data['x_el']:>7.3f} {fringe_data['x_h']:>7.3f} "
                f"{fringe_data['y_az']:>8.3f} {fringe_data['y_el']:>7.3f} {fringe_data['y_h']:>7.3f}  {fringe_data['mjd']:>11.5f}   "
                f"{fringe_data['rfi']:>1}              {fringe_data['bp']} {fringe_data['acf']}\n"
            )
            f_obj.write(line)
        else:
            noise_level = (fringe_data['amp'] / fringe_data['snr']) if fringe_data['snr'] != 0 else 0
            epoch_for_astropy = fringe_data['epoch'].replace(" ", ":", 1).replace("/", ":", 1)
            mjd = Time(epoch_for_astropy, format='yday', scale='utc').mjd

            line = (
                f"{fringe_data['epoch']:<22} {fringe_data['label']:>5} {fringe_data['source']:>9} "
                f"{fringe_data['length']:>10.5f} {fringe_data['amp']*100:>10.6f} {fringe_data['snr']:>7.1f} "
                f"{fringe_data['phase']:>+8.3f} {noise_level*100:>13.6f} {fringe_data['res_delay']:>15.2f} "
                f"{fringe_data['res_rate']:>13.6f} {fringe_data['x_az']:>10.3f} {fringe_data['x_el']:>7.3f} "
                f"{fringe_data['x_h']:>8.3f} {fringe_data['y_az']:>10.3f} {fringe_data['y_el']:>7.3f} "
                f"{fringe_data['y_h']:>8.3f} {mjd:>12.5f}\n"
            )
            f_obj.write(line)

    for xml_file in step_xml_files:
        print(f"[INFO] 解析中のXMLファイル: {xml_file}")
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            scans = root.findall('process')
            stream_info = root.find('stream')
            station_info = {s.get('key'): s.find('name').text for s in root.findall('station')}
            
            base_name = os.path.splitext(os.path.basename(xml_file))[0]
            output_txt_file = os.path.join(output_dir, f"{base_name}_fringe_output.txt")
            
            with open(output_txt_file, 'w') as f:
                if use_freq_format:
                    header = (
                        f"#*******************************************************************************************************************************************************************************************************************\n"
                        f"#      Epoch         Label    Source     Length    Amp      SNR     Phase     Frequency     Noise-level      Res-Rate            {station_info.get('K', 'X')}-azel             {station_info.get('L', 'Y')}-azel        MJD        RFI        BP    ACF\n"
                        f"#                                        [s]       [%]              [deg]       [MHz]       1-sigma[%]        [Hz]        az[deg]  el[deg]  hgt[m]   az[deg]  el[deg]  hgt[m]              [MHz]      [T/F] [T/F]\n"
                        f"#*******************************************************************************************************************************************************************************************************************\n"
                    )
                else:
                    header = (
                        f"#****************************************************************************************************************************************************************************************************\n"
                        f"#      Epoch         Label     Source      Length      Amp        SNR     Phase     Noise-level      Res-Delay     Res-Rate            {station_info.get('K', 'X')}-azel               {station_info.get('L', 'Y')}-azel              MJD  \n"
                        f"#year/doy hh:mm:ss.ss                       [s]        [%]                [deg]     1-sigma[%]       [sample]        [Hz]      az[deg]  el[deg]  height[m]   az[deg]   el[deg]  height[m]          \n"
                        f"#****************************************************************************************************************************************************************************************************\n"
                    )
                f.write(header)

                for scan in scans:
                    epoch_str = scan.find('epoch').text.strip()
                    xml_skip_sec = int(scan.find('skip').text)
                    label = stream_info.find('label').text.strip()

                    base_datetime = datetime.strptime(epoch_str, '%Y/%j %H:%M:%S')
                    scan_start_datetime = base_datetime + timedelta(seconds=xml_skip_sec)
                    timestamp_for_file = scan_start_datetime.strftime('%Y%j%H%M%S')
                    
                    if bandscythe_only:
                        cor_pattern = f"{cor_base_dir}/YAMAGU32_YAMAGU34_{timestamp_for_file}_{label}*bandscythe.cor"
                    else:
                        cor_pattern = f"{cor_base_dir}/YAMAGU32_YAMAGU34_{timestamp_for_file}_{label}.cor"
                    
                    cor_files = glob.glob(cor_pattern)

                    if not cor_files:
                        print(f"  [WARN] 対応する.corファイルが見つかりません: {cor_pattern}")
                        continue
                    
                    cor_file = cor_files[0]
                    
                    is_on_point = is_on_scan(scan_start_datetime, label)
                    is_offset = not is_on_point
                    
                    tasks = []
                    if is_on_point:
                        print(f"  [INFO] ON点として処理します: {timestamp_for_file}")
                        tasks.append({'length': float(output * on_length), 'skip': 0.0})
                        
                    elif is_offset:
                        print(f"  [INFO] offsetスキャンとして処理します: {timestamp_for_file}")
                        length_val = float(corr_step * output)  # 基本となる 50.0
                        
                        if processor == "frinZ" and scan_half:
                            # Optional edge treatment: the first and last
                            # integrations are half length, with full-length
                            # integrations between them.
                            total_length = float(scan_time * output)  # 全体の長さ (4100.0)
                            half_length = length_val / 2.0            # 25.0
                            skip_val = 0.0
                            
                            # 最初のスキャン (length=25)
                            tasks.append({'length': half_length, 'skip': skip_val})
                            skip_val += half_length
                            
                            # 中間のスキャン (length=50)
                            while skip_val + length_val <= total_length + 1e-9:
                                tasks.append({'length': length_val, 'skip': skip_val})
                                skip_val += length_val
                                
                            # 最後のスキャン (余り、通常は 25)
                            rem_length = total_length - skip_val
                            if rem_length > 0.1:  # 浮動小数点の誤差を考慮して 0.1 に設定
                                tasks.append({'length': rem_length, 'skip': skip_val})
                        else:
                            # Default (also the v2 behaviour): divide the
                            # entire scan into consecutive equal-length bins.
                            skip_val = 0.0
                            while skip_val <= skip_max + 1e-9: 
                                tasks.append({'length': length_val, 'skip': skip_val})
                                skip_val += skip_step

                    if is_offset and scan_lag_ms != 0:
                        if directions[scan_start_datetime] == forward_sign:
                            original_task_count = len(tasks)
                            tasks = apply_forward_lag(tasks, scan_lag_ms, float(scan_time * output))
                            skipped_count = original_task_count - len(tasks)
                            first = tasks[0]
                            print(f"  [INFO] 往路のみ {scan_lag_ms:+g} ms補正: "
                                  f"先頭 --length {first['length']:g} --skip {first['skip']:g} "
                                  f"({first['length'] / output * 1000:g} ms積分)、"
                                  f"範囲外 {skipped_count} 区間をスキップ。Epochは補正前の時刻を保持。")
                        else:
                            print("  [INFO] 復路のためラグを適用しません。")

                    for task in tasks:
                        float_length = float(task['length'])
                        float_skip = float(task['skip'])
                        
                        int_length = int(round(float_length))
                        int_skip = int(round(float_skip))
                        
                        if processor == "fringe":
                            command = ["fringe", "--in", cor_file]
                        else:
                            command = ["frinZ", "--in", cor_file]
                            
                        command.extend(["--length", str(int_length), "--skip", str(int_skip)])
                        
                        if use_freq_format:
                            command.append("--freq")
                            
                        if add_quick_opt:
                            command.append("--quick")
                            
                        print(command)
                        
                        output_str = run_external_command(command, capture=True)
                        
                        # parserにprocessorの情報を引き継ぐ
                        fringe_data = parse_fringe_output(output_str, use_freq=use_freq_format, processor=processor)

                        if fringe_data:
                            if processor == "frinZ":
                                # Keep the original schedule/bin label. Only
                                # the frinZ integration window is shifted.
                                nominal_skip = task.get('nominal_skip', float_skip)
                                output_time = scan_start_datetime + timedelta(seconds=nominal_skip / output)
                                # YYYY/DDD HH:MM:SS.ss 形式に整形
                                exact_epoch_str = output_time.strftime('%Y/%j %H:%M:%S.%f')[:-4]
                                fringe_data['epoch'] = exact_epoch_str
                                
                                # 小数秒を加味してMJDも正確に再計算
                                epoch_for_astropy = exact_epoch_str.replace(" ", ":", 1).replace("/", ":", 1)
                                try:
                                    fringe_data['mjd'] = Time(epoch_for_astropy, format='yday', scale='utc').mjd
                                except:
                                    pass

                            write_log_line(f, fringe_data, use_freq_format)
                        else:
                            print(f"  [WARN] {cor_file} (skip={task['skip']}) の結果を解析できませんでした。")
                        
            print(f"  [SUCCESS] 結果を {output_txt_file} に保存しました。\n")
        except (ET.ParseError, AttributeError, ValueError) as e:
            print(f"  [ERROR] XML処理中にエラー: {xml_file} ({e})\n")


def main():
    parser = argparse.ArgumentParser(description="gico3とfringe/frinZの処理を実行するスクリプト。")
    parser.add_argument("obs_code", help="観測コード (例: I25231Y)")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only-corr", action="store_true", help="gico3の処理のみ実行します。")
    group.add_argument("--only-fringe", action="store_true", help="gico3の処理とfringe処理を実行します。")
    group.add_argument("--only-frinZ", action="store_true", help="gico3の処理とfrinZ.py処理を実行します。")
    group.add_argument("--only-frinZ-freq", action="store_true", help="gico3の処理とfrinZ.py(--freq)処理を実行します。")
    group.add_argument("--only-fringe-freq", action="store_true", help="gico3の処理とfringe(--quick, --freq)処理を実行します。bandscythe.corのみ対象。")
    
    parser.add_argument("--band-split", type=int, metavar="DIVISIONS",
                        help="8192-8704MHzを指定した分割数で処理します。512の約数(1, 2, 4, 8...)を指定してください。")
    
    parser.add_argument("--cpu", type=int, help="gico3実行時のCPUコア数 (例: --cpu 10)")
    parser.add_argument(
        "--scan-half", action="store_true",
        help="frinZのoffset走査で、最初と最後だけ積分長を半分にします（既定では全区間を等分割）。"
    )
    parser.add_argument(
        "--scan-lag-ms", type=float, default=0.0, metavar="MS",
        help="往路だけfrinZの積分境界をMS msずらします。負=前、正=後。length/skipを変更し、Epochは保持（例: -100）。10 ms刻み。"
    )
    parser.add_argument(
        "--forward-direction", choices=("increasing", "decreasing"), default="increasing",
        help="往路のAz方向。increasing=Az増加（既定）、decreasing=Az減少。SKD座標から判定します。"
    )
    
    args = parser.parse_args()
    obs_code = args.obs_code
    try:
        lag_to_units(args.scan_lag_ms, args.scan_half)
    except ValueError as e:
        parser.error(str(e))
    if args.scan_lag_ms != 0 and not (args.only_frinZ or args.only_frinZ_freq):
        parser.error("--scan-lag-ms は --only-frinZ または --only-frinZ-freq と併用してください。")

    try:
        os.chdir(obs_code)
        print(f"[INFO] '{obs_code}' ディレクトリに移動しました。\n")
    except FileNotFoundError:
        print(f"[ERROR] ディレクトリ '{obs_code}' が見つかりません。")
        sys.exit(1)

    # 1. SKDファイルの読み込みと判定
    skd_files = glob.glob('*.skd')
    on_points = []
    offset_scans = []
    scan_positions = []
    
    if skd_files:
        skd_file = skd_files[0]
        print(f"[INFO] SKDファイル '{skd_file}' からON点とoffsetスキャンを判定します。")
        if len(skd_files) > 1:
            print(f"  [WARN] 複数のSKDファイルが見つかりました。'{skd_file}' を使用します。")
        
        on_points, offset_scans, scan_positions = parse_skd_schedule(skd_file, include_positions=True)
        print(f"  [INFO] {len(on_points)}個のON点、{len(offset_scans)}個のoffsetスキャンを取得しました。\n")
    else:
        print("[WARN] .skdファイルが見つかりません。XMLラベル名による判定で処理を進めます。\n")

    if args.scan_lag_ms != 0 and not scan_positions:
        parser.error("往路判定用のSKD座標がありません。ラグを適用できません。")

    # 2. XMLファイルの取得
    all_step_files = glob.glob('*_KL_X_step*.xml')
    step_xml_files = sorted([f for f in all_step_files if '_fringe_search' not in f])

    if not step_xml_files:
        print("[WARN] 対象の step*.xml ファイルが見つかりませんでした。")
        return

    # 3. 実行処理
    run_corr = not (args.only_fringe or args.only_frinZ or args.only_frinZ_freq or args.only_fringe_freq)
    run_process = not args.only_corr

    if run_corr:
        run_gico3_steps(step_xml_files, cpu=args.cpu)

    if run_process:
        if args.only_frinZ or args.only_frinZ_freq:
            processor_type = "frinZ"
        else:
            processor_type = "fringe"
            
        use_freq_flag = args.only_frinZ_freq or args.only_fringe_freq
        add_quick_flag = args.only_fringe_freq
        bandscythe_only_flag = args.only_fringe_freq
        
        if args.band_split is not None:
            divisions = args.band_split
            if divisions <= 0 or 512 % divisions != 0:
                print(f"[ERROR] 全帯域幅 512MHz を {divisions} で割り切ることができません。")
                print("ヒント: 512の約数（1, 2, 4, 8, 16, 32, 64, 128, 256, 512）を指定してください。")
                sys.exit(1)

            step = 512 // divisions
            bands = [f"{8192 + i*step}_{8192 + (i+1)*step}" for i in range(divisions)]
            
            for band in bands:
                print(f"\n===== バンド {band} の処理を開始します =====")
                run_fringe_steps(
                    step_xml_files, 
                    processor=processor_type, 
                    use_freq_format=use_freq_flag,
                    add_quick_opt=add_quick_flag,
                    bandscythe_only=bandscythe_only_flag,
                    scan_half=args.scan_half,
                    scan_lag_ms=args.scan_lag_ms,
                    on_points=on_points, 
                    offset_scans=offset_scans, 
                    band=band,
                    scan_positions=scan_positions,
                    forward_direction=args.forward_direction
                )
        else:
            run_fringe_steps(
                step_xml_files, 
                processor=processor_type, 
                use_freq_format=use_freq_flag,
                add_quick_opt=add_quick_flag,
                bandscythe_only=bandscythe_only_flag,
                scan_half=args.scan_half,
                scan_lag_ms=args.scan_lag_ms,
                on_points=on_points, 
                offset_scans=offset_scans,
                scan_positions=scan_positions,
                forward_direction=args.forward_direction
            )

    print("--- 全ての処理が完了しました ---")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, ET.ParseError, AttributeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
