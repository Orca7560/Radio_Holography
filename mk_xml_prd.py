import os
import argparse
import math
import re
from bisect import bisect_right
import numpy as np
from datetime import datetime, timedelta
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

# ==========================================
# ユーザー設定
# ==========================================
# 周波数 [MHz] X-band:8192, C-band:6600
band = 8192
init_delay = 0

# 観測天体リスト。追加する場合は「名前: (J2000赤経, J2000赤緯)」を記入する。
# 赤経・赤緯はそれぞれスペース区切りの3要素とする。
TARGETS = {
    "3C273": ("12 29 06.6997", "+02 03 08.5958"),
    "3C84": ("03 19 48.1601", "+41 30 42.103"),
    "NRAO530": ("17 33 02.7057", "-13 04 49.548"),
    "J1924-2914": ("19 24 51.0559", "-29 14 30.121"),
    "3C454.3": ("22 53 57.7479", "+16 08 53.560"),
    "J0538-4405": ("05 38 50.4000", "-44 05 09.000")
}
DEFAULT_TARGET = "3C273"
target_name = DEFAULT_TARGET
target_ra_str, target_dec_str = TARGETS[target_name]

ant_meas = 32
ant_ref = 34

# ==========================================
# 1. Astropy による南中時刻精密計算
# ==========================================
def get_transit_time(date_str, ra_str, dec_str):
    yamaguchi_32m = EarthLocation.from_geocentric(
        x=-3502544.5870 * u.m,
        y=+3950966.2350 * u.m,
        z=+3566381.1920 * u.m
    )
    target = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame='icrs')
    
    # 観測日の0時(UTC)時点からの初期推定
    t = Time(f"{date_str} 00:00:00", scale='utc')
    lst_0h = t.sidereal_time('apparent', longitude=yamaguchi_32m.lon)
    ra_diff = (target.ra - lst_0h).wrap_at(24 * u.hour)
    t = t + ra_diff.hour * 0.9972695663 * u.hour
    
    # 反復計算(ニュートン法)による精密化：時角(Hour Angle)が厳密に0になる時刻を探索
    for _ in range(10):
        lst = t.sidereal_time('apparent', longitude=yamaguchi_32m.lon)
        ha_deg = (target.ra - lst).deg
        # 角度差を -180 ~ +180度 に収める
        ha_deg = (ha_deg + 180) % 360 - 180
        ha_hour = ha_deg / 15.0
        
        # 時角のズレが十分に小さくなれば（約0.3ミリ秒以下）完了
        if abs(ha_hour) < 1e-7:
            break
            
        t = t + ha_hour * 0.9972695663 * u.hour
        
    return t.to_datetime()

# ==========================================
# 2. PRD / SKD ファイル生成 (シミュレーション対応)
# ==========================================
def generate_prd_skd(obs_start_time, is_dry_run=False, output_dir=".", az_interval_sec=0.5, suffix="", scan_axis="az", half_edge=False, obs_code=None):
    if scan_axis not in ("az", "el"):
        raise ValueError("scan_axis は az または el を指定してください")
    scan_limit = 60.0
    spatial_step = 1.5

    # (走査軸, 列送り軸) の座標を PRD/SKD の (Az, El) に変換する。
    def scan_coords(scan, row):
        return (scan, row) if scan_axis == "az" else (row, scan)
    
    integ_time = float(az_interval_sec)
    total_scan_width = scan_limit * 2.0
    
    # ★スキャンにかかる時間を厳密に41.0秒に設定
    row_scan_time_sec = 41.0 
    
    # ±60′を1.5′間隔で両端を含めて走査するので81点。
    num_steps_per_row = int(total_scan_width / spatial_step) + 1
    
    cur = obs_start_time
    
    yy_str = obs_start_time.strftime("%y")
    doy_str = obs_start_time.strftime("%j")
    if obs_code is None:
        obs_code = f"I{yy_str}{doy_str}YA"
    prd_filename = f"{obs_code}{ant_meas}{suffix}.prd"

    # --- 小数点の最適化フォーマット関数群 ---
    # 値に応じて必要な分だけ小数点を残し、0.01等を.0などに勝手に丸めない
    def fmt_sec(val):
        if abs(val) < 1e-9: val = 0.0
        s = f"{val:09.6f}".rstrip("0")
        if s.endswith("."): s += "0"
        return s

    def fmt_dur(val):
        if abs(val) < 1e-9: val = 0.0
        s = f"{val:.6f}".rstrip("0")
        if s.endswith("."): s += "0"
        return f"{s:>4}"

    def fmt_angle(val):
        if abs(val) < 1e-9: val = 0.0
        s = f"{val:+.6f}".rstrip("0")
        if s.endswith("."): s += "0"
        return f"{s:>6}"
    # ----------------------------------------
    
    def yydoy_to_jd(dt):
        year, month, day = dt.year, dt.month, dt.day
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        if month <= 2: year -= 1; month += 12
        A = math.floor(year / 100)
        B = 2 - A + math.floor(A / 4)
        jd = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + B - 1524.5
        return jd + hour / 24.0

    def R_y(theta): return np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
    def R_z(theta): return np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    def calc_precession_offsets(ra_deg, dec_deg, jd_target):
        T = (jd_target - 2451545.0) / 36525.0
        zeta_A = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * (np.pi / 648000.0)
        z_A    = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * (np.pi / 648000.0)
        theta_A= (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3)* (np.pi / 648000.0)
        P = R_z(-z_A) @ R_y(theta_A) @ R_z(-zeta_A)
        alpha_0, delta_0 = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
        v_0 = np.array([np.cos(delta_0)*np.cos(alpha_0), np.cos(delta_0)*np.sin(alpha_0), np.sin(delta_0)])
        x_m, y_m, z_m = P @ v_0
        alpha_m = np.arctan2(y_m, x_m) % (2 * np.pi)
        delta_m = np.arcsin(z_m)
        delta_ra_arcmin = np.rad2deg(alpha_m - alpha_0) * 60.0 * np.cos(delta_0)
        delta_dec_arcmin = np.rad2deg(delta_m - delta_0) * 60.0
        return delta_ra_arcmin, delta_dec_arcmin

    ra_parts = target_ra_str.split()
    ra_deg = (float(ra_parts[0]) + float(ra_parts[1])/60.0 + float(ra_parts[2])/3600.0) * 15.0
    dec_parts = target_dec_str.split()
    sign = -1.0 if target_dec_str.strip().startswith("-") else 1.0
    dec_deg = sign * (abs(float(dec_parts[0])) + float(dec_parts[1])/60.0 + float(dec_parts[2])/3600.0)

    jd_start = yydoy_to_jd(obs_start_time)
    dRA, dDec = calc_precession_offsets(ra_deg, dec_deg, jd_start)

    lines_prd = [f"* EXPER {target_name}\n", "* UTC          R.A             DEC             az      el      ra      dec\n"]
    lines_skd = [f"$EXPER {obs_code}{suffix}\n", "*\n", "$SOURCES\n", f"{target_name} {target_name}    {target_ra_str} {target_dec_str} 2000.0  0  0  0  0\n", "*\n", "$SKED\n"]
    ref_lines_skd = list(lines_skd)

    def format_prd(dt, az, el):
        time_str = dt.strftime('%Y%j%H%M%S')
        return f"{time_str} {target_ra_str} {target_dec_str}      {fmt_angle(az)}   {fmt_angle(el)}    {dRA:>6.3f}  {dDec:>6.3f}\n"

    def format_skd(dt, dur, az, el):
        sec_float = dt.second + dt.microsecond / 1_000_000
        time_str = dt.strftime('%y%j%H%M') + fmt_sec(sec_float)
        return f"{target_name:<12} {time_str}     {fmt_dur(dur)}   {fmt_angle(az)} {fmt_angle(el)}  0.0  0.0\n"
    
    def format_ref_skd(dt, dur, az, el):
        time_str = dt.strftime('%y%j%H%M%S')
        return f"{target_name:<12} {time_str}     {dur:>d}   {fmt_angle(az)} {fmt_angle(el)}  0.0  0.0\n"

    def move_to_scan_start(curr, az, el):
        curr += timedelta(seconds=20)
        if not is_dry_run: lines_prd.append(format_prd(curr, az, el))
        curr += timedelta(seconds=10)
        if not is_dry_run: lines_prd.append(format_prd(curr, az, el))
        return curr

    if not is_dry_run:
        lines_prd.append(format_prd(cur, 0.0, 0.0))
        lines_skd.append(format_skd(cur, 10.0, 0.0, 0.0))
    for _ in range(5):
        cur += timedelta(seconds=2)
        if not is_dry_run: lines_prd.append(format_prd(cur, 0.0, 0.0))
    first_az, first_el = scan_coords(-scan_limit, scan_limit)
    cur = move_to_scan_start(cur, first_az, first_el)

    num_rows = int(total_scan_width / spatial_step) + 1
    prev_end_scan = -scan_limit
    center_time = None

    for r in range(num_rows):
        row_c = scan_limit - r * spatial_step
        start_scan = -scan_limit if r % 2 == 0 else scan_limit
        end_scan = scan_limit if r % 2 == 0 else -scan_limit
        start_az, start_el = scan_coords(start_scan, row_c)
        
        if r > 0:
            if r % 6 == 0:
                for _ in range(4):
                    cur += timedelta(seconds=5)
                    if not is_dry_run: lines_prd.append(format_prd(cur, 0.0, 0.0))
                
                if not is_dry_run: lines_skd.append(format_skd(cur, 10.0, 0.0, 0.0))
                
                for _ in range(5):
                    cur += timedelta(seconds=2)
                    if not is_dry_run: lines_prd.append(format_prd(cur, 0.0, 0.0))
                    
                cur = move_to_scan_start(cur, start_az, start_el)
            else:
                overshoot_scan = prev_end_scan + (1.0 if prev_end_scan > 0 else -1.0)
                cur += timedelta(seconds=1.0) 
                if not is_dry_run: lines_prd.append(format_prd(cur, *scan_coords(overshoot_scan, row_c + spatial_step / 2.0)))
                cur += timedelta(seconds=1.0) 
                if not is_dry_run: lines_prd.append(format_prd(cur, start_az, start_el))

        # 通常モードは1ステップ1.5'。両端半分モードも走査速度は3'/秒。
        scan_step_fixed = 1.5
        if r == num_rows // 2:
            # 中央の列の (Az, El)=(0, 0) 測定の中心時刻。
            center_seconds = 0.5 + scan_limit / 3.0
            center_time = cur + timedelta(seconds=center_seconds)

        for i in range(num_steps_per_row):
            dur_i = az_interval_sec / 2.0 if half_edge and i in (0, num_steps_per_row - 1) else integ_time
            offset = 0.5 + i * az_interval_sec - dur_i / 2.0
            t_i = cur + timedelta(seconds=offset)
            direction = 1 if (end_scan > start_scan) else -1
            
            scan_i = start_scan + direction * scan_step_fixed * i
            
            # 浮動小数点の丸め誤差対策で最大幅を超えないように制限
            scan_i = min(scan_i, end_scan) if direction == 1 else max(scan_i, end_scan)
            
            if not is_dry_run: lines_skd.append(format_skd(t_i, dur_i, *scan_coords(scan_i, row_c)))
        
        cur += timedelta(seconds=row_scan_time_sec)
        if not is_dry_run: lines_prd.append(format_prd(cur, *scan_coords(end_scan, row_c)))
        prev_end_scan = end_scan

    for _ in range(4):
        cur += timedelta(seconds=5)
        if not is_dry_run: lines_prd.append(format_prd(cur, 0.0, 0.0))
        
    if not is_dry_run: lines_skd.append(format_skd(cur, 10.0, 0.0, 0.0))
    for _ in range(5):
        cur += timedelta(seconds=2)
        if not is_dry_run: lines_prd.append(format_prd(cur, 0.0, 0.0))

    obs_end_time = cur
    total_seconds = int((obs_end_time - obs_start_time).total_seconds())

    if is_dry_run:
        return total_seconds, center_time - obs_start_time

    ref_lines_skd.append(format_ref_skd(obs_start_time, total_seconds, 0.0, 0.0))
    prd_path = os.path.join(output_dir, prd_filename)
    skd_meas = os.path.join(output_dir, f"{obs_code}{ant_meas}{suffix}.skd")
    skd_ref = os.path.join(output_dir, f"{obs_code}{ant_ref}{suffix}.skd")

    with open(prd_path, "w") as f: f.writelines(lines_prd)
    with open(skd_meas, "w") as f: f.writelines(lines_skd)
    with open(skd_ref, "w") as f: f.writelines(ref_lines_skd)

    # 開始・終了時刻はPRDを生成した実際の時刻を使用する（UTC）。
    memo_dir = os.path.join(os.getcwd(), "Obs_log")
    os.makedirs(memo_dir, exist_ok=True)
    memo_path = os.path.join(memo_dir, f"{obs_code}{suffix}_memo.log")
    memo_lines = [
        f"観測コード: {obs_code}{suffix}\n",
        f"観測天体: {target_name}\n",
        f"観測開始 (UTC): {obs_start_time:%y/%j %H:%M:%S} / {obs_start_time:%y/%m/%d %H:%M:%S}\n",
        f"観測終了 (UTC): {obs_end_time:%y/%j %H:%M:%S} / {obs_end_time:%y/%m/%d %H:%M:%S}\n",
        f"観測中間 (UTC): {format_utc_millis(obs_start_time + (obs_end_time - obs_start_time) / 2)}\n",
        f"観測所要時間: {total_seconds} sec\n",
        f"測定アンテナ: 山口 {ant_meas} m\n",
        f"参照アンテナ: 山口 {ant_ref} m\n",
        f"スキャン方向: {scan_axis.capitalize()}\n",
    ]
    with open(memo_path, "x", encoding="utf-8") as f:
        f.writelines(memo_lines)

    return prd_path


def generate_skd_from_prd(prd_filepath, skd_filepath, output_filepath, scan_axis):
    """PRDの走査端点を積分中心時刻へ線形補間した測定局SKDを作る。"""
    points = []
    with open(prd_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('*') or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 9:
                raise ValueError(f'PRD行を解析できません: {line.strip()}')
            points.append((datetime.strptime(fields[0], '%Y%j%H%M%S'),
                           float(fields[7]), float(fields[8])))

    axis_index = 1 if scan_axis == 'az' else 2
    fixed_index = 2 if scan_axis == 'az' else 1
    scans = []
    for p1, p2 in zip(points, points[1:]):
        dt = (p2[0] - p1[0]).total_seconds()
        if dt > 20 and abs(p2[axis_index] - p1[axis_index]) > 100 and abs(p2[fixed_index] - p1[fixed_index]) < 0.1:
            scans.append((p1, p2, dt))
    if not scans:
        raise ValueError('PRDに走査列の端点が見つかりません')
    starts = [scan[0][0] for scan in scans]

    def fmt_angle(value):
        s = f'{value:+.6f}'.rstrip('0')
        return s + '0' if s.endswith('.') else s

    updated = 0
    lines = []
    with open(skd_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.split()
            if len(fields) < 7 or not fields[1][:5].isdigit():
                lines.append(line)
                continue
            try:
                start = datetime.strptime(fields[1], '%y%j%H%M%S.%f')
                dur = float(fields[2])
            except ValueError:
                lines.append(line)
                continue
            if dur > 1.0:  # ON点と参照局の観測は変更しない
                lines.append(line)
                continue
            midpoint = start + timedelta(seconds=dur / 2.0)
            index = bisect_right(starts, midpoint) - 1
            if index < 0:
                raise ValueError(f'SKDの走査点に対応するPRD区間がありません: {fields[1]}')
            p1, p2, dt = scans[index]
            if midpoint >= p2[0]:
                raise ValueError(f'SKDの走査点がPRD区間外です: {fields[1]}')
            # Nominal raster center times are 0.5 ... 40.5 s in the 41 s leg.
            ratio = ((midpoint - p1[0]).total_seconds() - 0.5) / (dt - 1.0)
            az = p1[1] + (p2[1] - p1[1]) * ratio
            el = p1[2] + (p2[2] - p1[2]) * ratio
            match = re.match(r'^(\s*\S+\s+\S+\s+\S+\s+)(\S+)(\s+)(\S+)(.*)$', line)
            if match is None:
                raise ValueError(f'SKD行の角度を更新できません: {line.strip()}')
            lines.append(match.group(1) + fmt_angle(az).rjust(len(match.group(2))) +
                         match.group(3) + fmt_angle(el).rjust(len(match.group(4))) + match.group(5) +
                         ('\n' if line.endswith('\n') else ''))
            updated += 1

    if updated == 0:
        raise ValueError('SKDに走査点が見つかりません')
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return len(scans), updated

# ==========================================
# 3. PRD 解析 ＆ XML ファイル生成関数
# ==========================================
def parse_prd_and_generate_xml(prd_filepath, output_dir, suffix="", scan_axis="az"):
    if not os.path.exists(prd_filepath):
        print(f"エラー: {prd_filepath} が見つかりません。")
        return

    base_name = os.path.basename(prd_filepath).replace('.prd', '')
    expected_tail = f"{ant_meas}{suffix}"
    if not base_name.endswith(expected_tail):
        raise ValueError(f"PRDファイル名は {expected_tail}.prd で終わる必要があります: {base_name}")
    obscode = base_name[:-len(expected_tail)]

    lines_data = []
    with open(prd_filepath, 'r') as f:
        for line in f:
            if line.startswith('*') or not line.strip(): continue
            parts = line.split()
            if len(parts) >= 9:
                t_str = parts[0]
                dt = datetime.strptime(t_str, "%Y%j%H%M%S")
                ra_str = f"{parts[1]}h{parts[2]}m{parts[3]}"
                dec_str = f"{parts[4]}d{parts[5]}'{parts[6]}"
                az, el = float(parts[7]), float(parts[8])
                lines_data.append({'time': dt, 'ra': ra_str, 'dec': dec_str, 'az': az, 'el': el})

    if not lines_data: return

    global_epoch_dt = lines_data[0]['time']
    global_epoch_str = global_epoch_dt.strftime('%Y/%j %H:%M:%S')
    processes = []

    for i in range(len(lines_data) - 1):
        p1, p2 = lines_data[i], lines_data[i + 1]
        dt_sec = (p2['time'] - p1['time']).total_seconds()
        dAz, dEl = abs(p1['az'] - p2['az']), abs(p1['el'] - p2['el'])

        if dAz < 0.1 and dEl < 0.1 and abs(p1['az']) < 0.1 and abs(p1['el']) < 0.1 and dt_sec == 2.0:
            is_first_scan_point = True
            if i > 0:
                p0 = lines_data[i - 1]
                prev_dt = (p1['time'] - p0['time']).total_seconds()
                if prev_dt == 2.0 and abs(p0['az']) < 0.1 and abs(p0['el']) < 0.1:
                    is_first_scan_point = False
            if is_first_scan_point:
                processes.append({"type": "ON_CAL", "start": p1["time"], "length": 10, "ra": p2["ra"], "dec": p2["dec"]})
        elif ((dAz > 5.0 and dEl < 1.0) if scan_axis == "az"
              else (dEl > 5.0 and dAz < 1.0)):
            if dt_sec > 20:
                processes.append({"type": "SCAN", "start": p1["time"], "length": int(dt_sec), "ra": p1["ra"], "dec": p1["dec"]})

    # 相関用XMLだけを前後1秒延長する。PRD/SKDとON点は変更しない。
    # lengthは積分時間ではなく共通epochからの終了秒数。
    for p in processes:
        nominal_skip = int((p['start'] - global_epoch_dt).total_seconds())
        padding = 1 if p['type'] == 'SCAN' else 0
        p['adj_skip'] = nominal_skip - padding
        p['adj_length'] = nominal_skip + p['length'] + padding
        if p['adj_skip'] < 0 or global_epoch_dt + timedelta(seconds=p['adj_length']) > lines_data[-1]['time']:
            raise ValueError('スキャン前後の相関用余裕がPRDの時間範囲を超えています。')
    for previous, current in zip(processes, processes[1:]):
        if current['adj_skip'] < previous['adj_length']:
            raise ValueError('相関用XMLのprocessが重複しています。')

    def build_xml(process_list, xml_filename):
        if not process_list: return
        process_blocks = []
        for p in process_list:
            skip_val = p['adj_skip']
            length_val = p['adj_length']
            process_blocks.append(f"<process><epoch>{global_epoch_str}</epoch><skip>{skip_val:>5d}</skip><length>{length_val:>5d}</length><object>{target_name:>8}</object><stations>KL</stations></process>")
        
        process_section = "\n".join(process_blocks)
        
        # =========================== FFT points ==============================
        fft_points = 8192 if "fringe_search" in xml_filename else 1024
        # fft_points = 524288 if "fringe_search" in xml_filename else 524288
        output_points = 1 if "fringe_search" in xml_filename else 1000
        # =====================================================================

        xml_content = f"""<?xml version='1.0' encoding='UTF-8' ?>
<schedule>
<terminal name='ADS1000'><speed>1024000000</speed><channel> 1</channel><bit>2</bit><level>-1.5,+0.5,-0.5,+1.5</level></terminal>
<terminal name='ADS3000'><speed>1024000000</speed><channel>1</channel><bit>2</bit><level>-1.5,-0.5,+0.5,+1.5</level></terminal>  
<terminal name='ADS3000_OCT'><speed>1024000000</speed><channel>1</channel><bit>2</bit><level>-1.5,+0.5,-0.5,+1.5</level></terminal>
<station key='K'><name>YAMAGU32</name><pos-x>-3502544.587</pos-x><pos-y>+3950966.235</pos-y><pos-z>+3566381.192</pos-z><terminal>ADS3000</terminal></station>
<station key='L'><name>YAMAGU34</name><pos-x>-3502567.576</pos-x><pos-y>+3950885.734</pos-y><pos-z>+3566449.115</pos-z><terminal>ADS3000</terminal></station>
<shuffle key="K">24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7</shuffle>
<shuffle key="L">24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7</shuffle>
<source name='{target_name}'><ra>{process_list[0]['ra']}</ra><dec>{process_list[0]['dec']}</dec></source>
<stream>
<label>60</label><frequency>+{band}e+6</frequency><channel>1</channel><fft>{fft_points}</fft><output>{output_points}</output>
<special key='K'><sideband>LSB</sideband></special>
<special key='L'><sideband>LSB</sideband></special>
</stream>
<clock key='K'><epoch>{global_epoch_str}</epoch><delay>+{init_delay}</delay><rate>+0</rate></clock>
{process_section}
</schedule>"""
        out_path = os.path.join(output_dir, xml_filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

    build_xml(processes, f"{obscode}{suffix}_KL_X_all.xml")

    blocks, current_block = [], []
    for p in processes:
        if p['type'] == 'ON_CAL':
            if current_block:
                blocks.append(current_block)
                current_block = []
            current_block.append(p)
        else:
            if current_block: current_block.append(p)
    if current_block: blocks.append(current_block)

    for idx, block in enumerate(blocks):
        build_xml(block, f"{obscode}{suffix}_KL_X_step{idx+1:02d}.xml")
        if block and block[0]['type'] == 'ON_CAL':
            build_xml([block[0]], f"{obscode}{suffix}_KL_X_step{idx+1:02d}_fringe_search.xml")

def reserve_observation_directory(obs_start_time, output_root):
    """同じ年・DOYの観測をYA, YB, ... YZの順に新規フォルダーへ保存する。"""
    base_code = f"I{obs_start_time.strftime('%y%j')}Y"
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        obs_code = f"{base_code}{letter}"
        output_dir = os.path.join(output_root, obs_code)
        # 集約先にログが残っているコードも使用済みとして扱う。
        if os.path.lexists(os.path.join(os.getcwd(), "Obs_log", f"{obs_code}_memo.log")):
            continue
        try:
            # 存在確認と作成を同時に行い、既存フォルダーは決して再利用しない。
            os.mkdir(output_dir)
        except FileExistsError:
            continue
        return obs_code, output_dir
    raise FileExistsError(
        f"{base_code}A〜{base_code}Z はすべて使用済みです。保存先を変更してください。")


def generate_schedule(date_str, selected_target, antenna, scan_axis, output_root='.', half_edge=False, start_time=None):
    """0.5秒間隔のPRD/SKD/XMLを生成し、出力先を返す。"""
    global target_name, target_ra_str, target_dec_str, ant_meas, ant_ref
    datetime.strptime(date_str, '%Y-%m-%d')
    if selected_target not in TARGETS:
        raise ValueError(f'未登録の天体です: {selected_target}')
    if antenna not in (32, 34) or scan_axis not in ('az', 'el'):
        raise ValueError('アンテナまたは走査方向が不正です')
    target_name = selected_target
    target_ra_str, target_dec_str = TARGETS[target_name]
    ant_meas, ant_ref = antenna, 34 if antenna == 32 else 32

    transit_time_exact = get_transit_time(date_str, target_ra_str, target_dec_str)
    dummy_epoch = datetime(2026, 1, 1)
    total_seconds, center_offset = generate_prd_skd(dummy_epoch, is_dry_run=True, az_interval_sec=0.5, scan_axis=scan_axis, half_edge=half_edge)
    # ON点と移動時間を含む観測全体を南中の前後に均等に配置し、仰角変動を抑える。
    if start_time is None:
        midpoint_offset = timedelta(seconds=total_seconds / 2)
        obs_start_time = (transit_time_exact - midpoint_offset + timedelta(microseconds=500000)).replace(microsecond=0)
    else:
        # 既存観測の再生成時には、指定された開始時刻を使用する。
        obs_start_time = start_time
    center_scan_time = obs_start_time + center_offset
    obs_code, output_dir = reserve_observation_directory(obs_start_time, output_root)

    prd_filepath = generate_prd_skd(obs_start_time, output_dir=output_dir, az_interval_sec=0.5, scan_axis=scan_axis, half_edge=half_edge, obs_code=obs_code)
    measured_skd = os.path.join(output_dir, f'{obs_code}{ant_meas}.skd')
    prd_based_skd = os.path.join(output_dir, f'{obs_code}{ant_meas}_from_prd.skd')
    generate_skd_from_prd(prd_filepath, measured_skd, prd_based_skd, scan_axis)
    parse_prd_and_generate_xml(prd_filepath, output_dir=output_dir, scan_axis=scan_axis)

    return output_dir, transit_time_exact, obs_start_time, center_scan_time


def format_utc_millis(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S') + f'.{dt.microsecond // 1000:03d}'


def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import threading

    root = tk.Tk()
    root.title('山口 ホログラフィ観測スケジュール')
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=18)
    frame.grid(sticky='nsew')

    date_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
    target_var = tk.StringVar(value=DEFAULT_TARGET)
    antenna_var = tk.IntVar(value=32)
    axis_var = tk.StringVar(value='az')
    half_edge_var = tk.BooleanVar(value=False)
    output_var = tk.StringVar(value=os.path.dirname(os.path.abspath(__file__)))
    coords_var = tk.StringVar()
    status_var = tk.StringVar(value='観測日をUTCの日付で入力してください。観測全体の中間を南中に合わせます。')

    ttk.Label(frame, text='観測日 (UTC)').grid(row=0, column=0, sticky='w', pady=5)
    ttk.Entry(frame, textvariable=date_var, width=16).grid(row=0, column=1, sticky='w', pady=5)
    ttk.Label(frame, text='YYYY-MM-DD').grid(row=0, column=2, sticky='w', padx=8)
    ttk.Label(frame, text='天体').grid(row=1, column=0, sticky='w', pady=5)
    target_box = ttk.Combobox(frame, textvariable=target_var, values=list(TARGETS), state='readonly', width=18)
    target_box.grid(row=1, column=1, sticky='w', pady=5)
    ttk.Label(frame, textvariable=coords_var).grid(row=2, column=1, columnspan=2, sticky='w')

    def update_coords(*_):
        ra, dec = TARGETS[target_var.get()]
        coords_var.set(f'J2000  RA {ra}   Dec {dec}')

    target_var.trace_add('write', update_coords)
    update_coords()
    ttk.Label(frame, text='測定アンテナ').grid(row=3, column=0, sticky='w', pady=8)
    antenna_frame = ttk.Frame(frame)
    antenna_frame.grid(row=3, column=1, sticky='w')
    ttk.Radiobutton(antenna_frame, text='32 m', variable=antenna_var, value=32).pack(side='left')
    ttk.Radiobutton(antenna_frame, text='34 m', variable=antenna_var, value=34).pack(side='left', padx=12)
    ttk.Label(frame, text='走査方向').grid(row=4, column=0, sticky='w', pady=8)
    axis_frame = ttk.Frame(frame)
    axis_frame.grid(row=4, column=1, sticky='w')
    ttk.Radiobutton(axis_frame, text='Az', variable=axis_var, value='az').pack(side='left')
    ttk.Radiobutton(axis_frame, text='El', variable=axis_var, value='el').pack(side='left', padx=18)
    ttk.Label(frame, text='スキャン両端').grid(row=5, column=0, sticky='w', pady=5)
    ttk.Checkbutton(frame, text='最初と最後の積分時間を半分にする (0.25 s)',
                    variable=half_edge_var).grid(row=5, column=1, columnspan=3, sticky='w')
    ttk.Label(frame, text='保存先の親フォルダー').grid(row=6, column=0, sticky='w', pady=5, padx=(0, 12))
    ttk.Entry(frame, textvariable=output_var, width=48).grid(row=6, column=1, columnspan=2, sticky='ew', pady=5)

    def browse():
        chosen = filedialog.askdirectory(parent=root, initialdir=output_var.get())
        if chosen:
            output_var.set(chosen)

    ttk.Button(frame, text='参照...', command=browse).grid(row=6, column=3, padx=(8, 0))

    def finish(result=None, error=None):
        generate_button.config(state='normal')
        if error is not None:
            status_var.set('生成に失敗しました。')
            messagebox.showerror('生成エラー', str(error), parent=root)
        else:
            output_dir, transit, start, center = result
            status_var.set(f'生成完了: {output_dir}')
            messagebox.showinfo('生成完了',
                f'出力先: {output_dir}\n南中（観測全体の中間の目標）: {format_utc_millis(transit)} UTC\n'
                f'中央走査 (Az=0, El=0): {format_utc_millis(center)} UTC\n'
                f'開始: {start:%Y-%m-%d %H:%M:%S} UTC',
                parent=root)

    def generate():
        date_str = date_var.get().strip()
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            messagebox.showerror('入力エラー', '観測日は YYYY-MM-DD 形式で入力してください。', parent=root)
            return
        output_root = output_var.get().strip()
        if not output_root or not os.path.isdir(output_root):
            messagebox.showerror('入力エラー', '存在する保存先フォルダーを選んでください。', parent=root)
            return
        name, antenna, axis, half_edge = target_var.get(), antenna_var.get(), axis_var.get(), half_edge_var.get()
        generate_button.config(state='disabled')
        status_var.set('生成中...')

        def worker():
            try:
                result = generate_schedule(date_str, name, antenna, axis, output_root, half_edge)
            except Exception as exc:
                root.after(0, lambda exc=exc: finish(error=exc))
            else:
                root.after(0, lambda result=result: finish(result=result))

        threading.Thread(target=worker, daemon=True).start()

    generate_button = ttk.Button(frame, text='スケジュールを生成', command=generate)
    generate_button.grid(row=7, column=1, sticky='w', pady=(14, 8))
    ttk.Label(frame, textvariable=status_var, wraplength=550).grid(row=8, column=0, columnspan=4, sticky='w')
    root.mainloop()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1 or sys.argv[1:] == ['--gui']:
        launch_gui()
    else:
        parser = argparse.ArgumentParser(description='観測スケジュール生成（観測全体の中間を南中に合わせる。引数なしでGUIを起動）')
        parser.add_argument('--date', help='観測日 UTC (例: 2026-09-29)')
        parser.add_argument('--target-name', default=DEFAULT_TARGET, help='登録済み天体名')
        parser.add_argument('--list-targets', action='store_true', help='登録済み天体を表示')
        parser.add_argument('--antenna', type=int, choices=(32, 34), default=32)
        parser.add_argument('--scan-axis', choices=('az', 'el'), default='az')
        parser.add_argument('--half-edge', action='store_true', help='各走査列の最初と最後を0.25秒積分にする')
        parser.add_argument('--output-dir', default='.', help='保存先の親フォルダー')
        parser.add_argument('--start-time', type=lambda value: datetime.strptime(value, '%Y-%m-%d %H:%M:%S'), help='観測開始UTCを明示指定（例: 2026-07-03 08:21:00）')
        parser.add_argument('--from-prd', help='既存PRDの時刻・座標を維持してXMLのみを再生成する')
        args = parser.parse_args()
        if args.list_targets:
            for name, (ra, dec) in TARGETS.items():
                print(f'{name:<12} RA {ra}  Dec {dec}')
            parser.exit()
        if args.from_prd:
            filename = os.path.basename(args.from_prd)
            match = re.fullmatch(r'.+(32|34)\.prd', filename)
            if not match:
                parser.error('--from-prd は観測コード32.prd または観測コード34.prd を指定してください')
            ant_meas = int(match.group(1))
            ant_ref = 34 if ant_meas == 32 else 32
            with open(args.from_prd, encoding='utf-8-sig') as fp:
                source = next((line.split()[2] for line in fp if line.startswith('* EXPER ') and len(line.split()) >= 3), None)
            if not source:
                parser.error('PRDのEXPER行に天体名がありません')
            target_name = source
            if not os.path.isdir(args.output_dir):
                parser.error('保存先フォルダーが存在しません')
            parse_prd_and_generate_xml(args.from_prd, args.output_dir, scan_axis=args.scan_axis)
            print(f'PRDの時刻を維持してXMLを生成しました: {args.output_dir}')
            parser.exit()
        if not args.date:
            parser.error('--date を指定してください')
        selected = next((name for name in TARGETS if name.casefold() == args.target_name.casefold()), None)
        if selected is None:
            parser.error(f'未登録の天体です: {args.target_name}。--list-targets で一覧を確認してください')
        if not os.path.isdir(args.output_dir):
            parser.error(f'保存先フォルダーが存在しません: {args.output_dir}')
        output_dir, transit, start, center = generate_schedule(
            args.date, selected, args.antenna, args.scan_axis, args.output_dir, args.half_edge, args.start_time)
        print(f'天体: {selected}  測定アンテナ: {args.antenna} m  走査方向: {args.scan_axis.upper()}')
        print(f'南中: {format_utc_millis(transit)} UTC')
        print(f'中央走査 (Az=0, El=0): {format_utc_millis(center)} UTC')
        print(f'開始: {start:%Y-%m-%d %H:%M:%S} UTC')
        print(f'出力先: {output_dir}')

