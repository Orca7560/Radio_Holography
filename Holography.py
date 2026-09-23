import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 画面表示なし
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ==========================================
# 🔧 ユーザー設定パラメータ
# ==========================================
# 処理に使用するAz/Elのオフセット範囲（±何分角以内か）。制限しない場合は None にする
# OFFSET_LIMIT_ARCMIN = 9.0 
OFFSET_LIMIT_ARCMIN = None

# スケジュールの時刻とビームデータ時刻のズレ（タイムラグ）を自動補正するかどうか
# かなり細かく見ている場合のみ
AUTO_TIME_ALIGN = True
AUTO_TIME_ALIGN = False
MANUAL_TIME_OFFSET_SEC = 0.0

# スライスグラフ（Az vs Amp/Phase）を生成するかどうか
GENERATE_SLICES = True
GENERATE_SLICES = False

# 開口面位相の全スライスプロットを生成するかどうか
GENERATE_APERTURE_SLICES = True
GENERATE_APERTURE_SLICES = False

# dBスケールビームパターンの表示下限（例: -30.0）
DB_MIN = -45.0

# ★ ビームパターンのズーム画像を出力する際の表示範囲（±何分角か）。Noneにすると出力しない
ZOOM_ARCMIN = 25.0

# SNR閾値：beam_10.txt の SNR 列を使い、この値以下のデータ点はSNフィルタ済みビームマップに表示しない
# ★ 定量評価を実行するために、Noneではなく数値を指定してください。
SNR_THRESHOLD = 7.0
# SNR_THRESHOLD = None  # <-- コメントアウトしました

# 鏡面誤差として許容する最大値（これを超える部分はブロッキング等とみなしRMSから除外）
SURFACE_ERR_THRESHOLD_MM = 3.0 

# 副鏡ブロッキングとして除外する中心の正方形サイズ（メートル）
CENTER_BLOCK_SIZE_M = 3.0

# 観測コード（ディレクトリ名）
OBS_CODE = "I26149Y" # sys.argv[1] の代わりに固定値にするか、元に戻してください
# ==========================================

# =========================
# コマンドライン引数
# =========================
parser = argparse.ArgumentParser(
    description="電波ホログラフィ解析"
)
parser.add_argument(
    "obs_code", nargs="?", default=OBS_CODE,
    help="観測コード（既定: 設定値を使用）"
)
parser.add_argument(
    "--beam", action="store_true",
    help="El=0 のAzスキャンを、Az ±20′・最大0 dBでプロットする"
)
args = parser.parse_args()
OBS_CODE = args.obs_code
PLOT_BEAM_CUT = args.beam

# =========================
# ディレクトリ設定
# =========================

BASE_DIR = OBS_CODE
BEAM_FILE = os.path.join(BASE_DIR, "beam_half.txt")
SKD_FILE  = os.path.join(BASE_DIR, "schedule_half.skd")

OUT_DIR       = os.path.join(BASE_DIR, f"output_limited_half")
SLICE_AMP_DIR = os.path.join(BASE_DIR, "slice", "Amp")
SLICE_PH_DIR  = os.path.join(BASE_DIR, "slice", "Phase")

os.makedirs(OUT_DIR,       exist_ok=True)
if GENERATE_SLICES:
    os.makedirs(SLICE_AMP_DIR, exist_ok=True)
    os.makedirs(SLICE_PH_DIR,  exist_ok=True)

print(f"OBS_CODE : {OBS_CODE}")
print(f"Input    : {BASE_DIR}/")

# =========================
# アンテナ・観測パラメータ
# =========================
c = 3e8
f = 8.448e9
wavelength = c / f
D = 32.0
arcmin_to_rad = np.pi / (180 * 60)

# =========================
# ファイル読み込み
# =========================
try:
    beam = pd.read_csv(BEAM_FILE, encoding="utf-8-sig")
except FileNotFoundError:
    raise FileNotFoundError(f"エラー: {BEAM_FILE} が見つかりません。")

beam.columns = beam.columns.str.strip()
beam["Epoch"] = pd.to_datetime(beam["Epoch"], format="%Y/%j %H:%M:%S.%f")
beam["E"] = beam["Amp"] * np.exp(1j * np.deg2rad(beam["Phase"]))
if "SNR" not in beam.columns:
    raise ValueError("エラー: beamファイルに 'SNR' 列が見つかりません。")

def parse_skd(filename):
    rows = []
    try:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('$') or line.startswith('*'):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    float(parts[1])
                except ValueError:
                    continue
                timestr, az, el = parts[1], float(parts[3]), float(parts[4])
                year = int("20" + timestr[:2])
                doy  = int(timestr[2:5])
                hh   = int(timestr[5:7])
                mm   = int(timestr[7:9])
                ss   = float(timestr[9:])
                dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
                rows.append({"Epoch": dt, "Az": az, "El": el})
    except FileNotFoundError:
        raise FileNotFoundError(f"エラー: {filename} が見つかりません。")
    return pd.DataFrame(rows)

sked = parse_skd(SKD_FILE)

# --- スケジュールから Az と El のスキャン間隔を独立して自動推定 ---
def estimate_scan_steps(df):
    d_az = np.abs(np.diff(df["Az"]))
    d_el = np.abs(np.diff(df["El"]))
    
    # 連続スキャン (0.03'等) に対応するため、下限を 0.01 に設定
    valid_az = d_az[(d_az > 0.01) & (d_az < 20.0)]
    valid_el = d_el[(d_el > 0.01) & (d_el < 20.0)]
    
    step_az = 3.0 # デフォルト値
    if len(valid_az) > 0:
        vals_az, counts_az = np.unique(np.round(valid_az, 3), return_counts=True)
        step_az = vals_az[np.argmax(counts_az)]
        
    step_el = 3.0 # デフォルト値
    if len(valid_el) > 0:
        vals_el, counts_el = np.unique(np.round(valid_el, 3), return_counts=True)
        step_el = vals_el[np.argmax(counts_el)]
        
    return step_az, step_el

SCAN_STEP_AZ, SCAN_STEP_EL = estimate_scan_steps(sked)
print(f"自動推定されたスキャン間隔 -> Az: {SCAN_STEP_AZ} arcmin, El: {SCAN_STEP_EL} arcmin")

beam = beam.sort_values("Epoch")
sked = sked.sort_values("Epoch")

# =========================
# スケジュールとビームデータの結合 (線形補間とタイムラグ補正)
# =========================
beam_time = beam["Epoch"].astype(np.int64).values / 1e9
sked_time = sked["Epoch"].astype(np.int64).values / 1e9

time_offset = MANUAL_TIME_OFFSET_SEC
if AUTO_TIME_ALIGN:
    print("\nタイムラグ（アンテナ動作とデータ記録の時刻ズレ）を自動推定しています...")
    
    # キャリブレーション用の長時間ON点（連続して(0,0)にいる期間）を除外してモデルを作る
    sked_r2 = sked["Az"].values**2 + sked["El"].values**2
    sked_is_center = sked_r2 < 0.1
    sked_center_mask = sked_is_center & (np.append(np.diff(sked_time), 0) > 1.0) 
    
    sked_scan_mask = ~sked_center_mask
    sked_time_for_align = sked_time[sked_scan_mask]
    sked_r2_for_align = sked_r2[sked_scan_mask]
    
    sked_pseudo_amp = np.exp(-sked_r2_for_align / (2 * 10.0**2)) 
    
    t_min = max(beam_time.min(), sked_time_for_align.min())
    t_max = min(beam_time.max(), sked_time_for_align.max())
    t_common = np.arange(t_min, t_max, 0.1)
    
    if len(t_common) > 10:
        beam_amp_interp = np.interp(t_common, beam_time, beam["Amp"].values)
        sked_model_interp = np.interp(t_common, sked_time_for_align, sked_pseudo_amp)
        
        correlation = np.correlate(beam_amp_interp - beam_amp_interp.mean(), 
                                   sked_model_interp - sked_model_interp.mean(), mode='full')
        lags = np.arange(-len(t_common) + 1, len(t_common)) * 0.1
        best_lag_idx = np.argmax(correlation)
        time_offset = lags[best_lag_idx]
        print(f"-> 自動推定されたタイムラグ: {time_offset:+.2f} 秒 (スケジュール時刻を補正します)")
    else:
        print("-> 有効なオーバーラップ範囲が狭いため、自動推定をスキップします。")

# タイムラグの適用
sked_time_corrected = sked_time + time_offset

# --- 線形補間（Interpolation）による座標割り当て ---
beam["Az"] = np.interp(beam_time, sked_time_corrected, sked["Az"].values)
beam["El"] = np.interp(beam_time, sked_time_corrected, sked["El"].values)

# スケジュールの時間範囲外のデータを除外
valid_mask = (beam_time >= sked_time_corrected.min()) & (beam_time <= sked_time_corrected.max())
df = beam[valid_mask].copy()

initial_len = len(beam)
dropped_len = initial_len - len(df)

if len(df) == 0:
    raise ValueError("エラー: スケジュールとビームデータの時刻が全く一致しません。")
elif dropped_len > 0:
    print(f"情報: スケジュールに対応しない時間帯のビームデータを {dropped_len} 件除外して処理を続行します。\n")

# =========================
# 変数抽出 ＆ 位相・振幅(ON点)補正
# =========================
E  = df["E"].values
Az = df["Az"].values
El = df["El"].values
SNR = df["SNR"].values
times_sec = df["Epoch"].astype(np.int64).values / 1e9  

on_idx = np.where((Az == 0) & (El == 0) & (np.abs(E) > 1.0))[0]
E_corr = E.copy()

if len(on_idx) >= 2:
    on_phases = np.unwrap(np.angle(E[on_idx]))
    on_amps = np.abs(E[on_idx])
    
    # インデックスではなく、正確な「時間(秒)」を基準にして変動を補間する
    phi_ref_interp = np.interp(times_sec, times_sec[on_idx], on_phases)
    amp_ref_interp = np.interp(times_sec, times_sec[on_idx], on_amps)
    
    # 振幅変動を平均値(mean_on_amp)に揃えるように補正
    mean_on_amp = np.mean(on_amps)
    
    # 位相の引き算と、振幅の割り算(補正)を同時に適用
    E_corr = E_corr * np.exp(-1j * phi_ref_interp) * (mean_on_amp / amp_ref_interp)
    
    print(f"ON点補正: Amp>1 を満たす {len(on_idx)} 箇所を用いて位相および振幅の補正を行いました。")
else:
    print("警告: 有効なON点が少ないため、位相・振幅補正をスキップしました。")

# =========================
# データの範囲（オフセット）制限
# =========================
if OFFSET_LIMIT_ARCMIN is not None:
    print(f"オフセット範囲を ±{OFFSET_LIMIT_ARCMIN} arcmin に制限します。")
    limit_mask = (np.abs(Az) <= OFFSET_LIMIT_ARCMIN) & (np.abs(El) <= OFFSET_LIMIT_ARCMIN)
    Az = Az[limit_mask]
    El = El[limit_mask]
    E_corr = E_corr[limit_mask]
    SNR = SNR[limit_mask]
    times_sec = times_sec[limit_mask]

# =========================
# キャリブレーション用ON点の除外 (スキャン通過点のみ採用)
# =========================
valid_scan_mask = np.ones(len(Az), dtype=bool)
is_center = (Az == 0) & (El == 0)

if np.sum(is_center) > 1:
    valid_scan_mask[is_center] = False # 一旦すべての (0, 0) を除外
    
    # El=0 の横スキャンに属する (0, 0) を見つけて復活させる
    is_el0_scan = (El == 0) & (Az != 0)
    if np.sum(is_el0_scan) > 0:
        mean_time_el0 = np.mean(times_sec[is_el0_scan])
        center_indices = np.where(is_center)[0]
        closest_idx = center_indices[np.argmin(np.abs(times_sec[center_indices] - mean_time_el0))]
        valid_scan_mask[closest_idx] = True
        
    # Az=0の縦スキャンが存在する場合の保険
    is_az0_scan = (Az == 0) & (El != 0)
    if np.sum(is_az0_scan) > 0:
        mean_time_az0 = np.mean(times_sec[is_az0_scan])
        center_indices = np.where(is_center)[0]
        closest_idx = center_indices[np.argmin(np.abs(times_sec[center_indices] - mean_time_az0))]
        valid_scan_mask[closest_idx] = True

# 全データから不要なON点を完全に消し去る
Az        = Az[valid_scan_mask]
El        = El[valid_scan_mask]
E_corr    = E_corr[valid_scan_mask]
SNR       = SNR[valid_scan_mask]
times_sec = times_sec[valid_scan_mask]

# =========================
# El=0 ビームカット（--beam）
# =========================
if PLOT_BEAM_CUT:
    beam_cut_mask = (np.isclose(El, 0.0, atol=1e-6)) & (np.abs(Az) <= 20.0)
    if not np.any(beam_cut_mask):
        print("[WARN] --beam: El=0 かつ Az ±20′ 内のデータがないため、ビームカットを出力しません。")
    else:
        az_cut = Az[beam_cut_mask]
        amp_cut = np.abs(E_corr[beam_cut_mask])
        order = np.argsort(az_cut)
        az_cut = az_cut[order]
        amp_cut = amp_cut[order]

        peak_cut = np.max(amp_cut)
        if peak_cut <= 0:
            print("[WARN] --beam: El=0 ビームカットの最大振幅が0のため、dBプロットを出力しません。")
        else:
            with np.errstate(divide="ignore"):
                amp_cut_db = 20.0 * np.log10(amp_cut / peak_cut)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(az_cut, amp_cut_db, marker="o", markersize=3, linewidth=1.0)
            ax.set_xlim(-20.0, 20.0)
            ax.set_ylim(DB_MIN, 0.0)
            ax.set_xlabel("Az offset [arcmin]")
            ax.set_ylabel("Normalized amplitude [dB]")
            ax.set_title("Beam Cut at El = 0")
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            beam_cut_path = os.path.join(OUT_DIR, "beam_cut_el0_dB.png")
            fig.savefig(beam_cut_path, dpi=150)
            plt.close(fig)
            print(f"[INFO] --beam: El=0 のAzビームカットを保存しました: {beam_cut_path}")

# =========================
# スライスプロット生成
# =========================
if GENERATE_SLICES:
    print("スライスプロットを生成中...")

    # 1. El=0 スキャン専用プロット
    mask_el0 = (El == 0)
    if np.any(mask_el0):
        Az_el0         = Az[mask_el0]
        Amp_el0        = np.abs(E_corr[mask_el0])
        Ph_el0         = np.rad2deg(np.angle(E_corr[mask_el0]))
        
        sort_idx       = np.argsort(Az_el0)
        Az_el0_sorted  = Az_el0[sort_idx]
        Amp_el0_sorted = Amp_el0[sort_idx]
        Ph_el0_sorted  = Ph_el0[sort_idx]

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        axes[0].plot(Az_el0_sorted, Amp_el0_sorted, marker='o', markersize=3, linewidth=1.0)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("El=0 Scan: Az offset vs Amplitude & Phase")
        axes[0].grid(True, linestyle='--', alpha=0.5)

        axes[1].plot(Az_el0_sorted, Ph_el0_sorted, marker='o', markersize=3, linewidth=1.0)
        axes[1].set_xlabel("Az offset [arcmin]")
        axes[1].set_ylabel("Phase [deg]")
        axes[1].set_ylim(-180, 180)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "el0_az_scan_amp_phase.png"), dpi=150)
        plt.close(fig)

    # 2. 全El offsetスライスプロット
    el_unique = np.unique(El)
    for el_val in el_unique:
        mask_el   = (El == el_val)
        Az_slice  = Az[mask_el]
        sort_idx  = np.argsort(Az_slice)
        Az_sorted = Az_slice[sort_idx]
        E_slice   = E_corr[mask_el][sort_idx]
        Amp_slice = np.abs(E_slice)
        Ph_slice  = np.rad2deg(np.angle(E_slice))

        el_label = f"El{el_val:+.1f}arcmin".replace("+", "p").replace("-", "m").replace(".", "_")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(Az_sorted, Amp_slice, marker='o', markersize=3, linewidth=1.0)
        ax.set_xlabel("Az offset [arcmin]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Az Scan Amplitude (El = {el_val:+.1f}')")
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(SLICE_AMP_DIR, f"{el_label}.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(Az_sorted, Ph_slice, marker='o', markersize=3, linewidth=1.0)
        ax.set_xlabel("Az offset [arcmin]")
        ax.set_ylabel("Phase [deg]")
        ax.set_title(f"Az Scan Phase (El = {el_val:+.1f}')")
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(SLICE_PH_DIR, f"{el_label}.png"), dpi=150)
        plt.close(fig)

# =========================
# 格子化 (アベレージングによる強固なグリッド作成)
# =========================
az_min, az_max = np.min(Az), np.max(Az)
el_min, el_max = np.min(El), np.max(El)

# 独立して求めた SCAN_STEP_AZ と SCAN_STEP_EL を使って方眼紙の軸を作成
tx_arcmin = np.arange(az_min, az_max + SCAN_STEP_AZ/2, SCAN_STEP_AZ)
ty_arcmin = np.arange(el_min, el_max + SCAN_STEP_EL/2, SCAN_STEP_EL)

tx = tx_arcmin * arcmin_to_rad
ty = ty_arcmin * arcmin_to_rad
nx, ny = len(tx), len(ty)

beam_grid_sum = np.zeros((ny, nx), dtype=complex)
snr_grid_sum  = np.zeros((ny, nx), dtype=float)
count_grid    = np.zeros((ny, nx), dtype=int)

for i in range(len(Az)):
    ix = np.argmin(np.abs(tx_arcmin - Az[i]))
    iy = np.argmin(np.abs(ty_arcmin - El[i]))
    
    beam_grid_sum[iy, ix] += E_corr[i]
    snr_grid_sum[iy, ix]  += SNR[i]
    count_grid[iy, ix]    += 1

# データが存在するセルだけ平均化、それ以外は 0j / NaN とする
valid_cells = count_grid > 0
beam_grid = np.zeros((ny, nx), dtype=complex)
snr_grid  = np.full((ny, nx), np.nan)

beam_grid[valid_cells] = beam_grid_sum[valid_cells] / count_grid[valid_cells]
snr_grid[valid_cells]  = snr_grid_sum[valid_cells] / count_grid[valid_cells]

beam_grid_amp = np.abs(beam_grid)

az_min_arc, az_max_arc = tx_arcmin.min(), tx_arcmin.max()
el_min_arc, el_max_arc = ty_arcmin.min(), ty_arcmin.max()
extent_vals = [az_max_arc, az_min_arc, el_min_arc, el_max_arc] # X軸反転

# =========================
# 複素ビームパターン (Linear Amp & Phase)
# =========================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im = axes[0].imshow(beam_grid_amp, extent=extent_vals, cmap="viridis", origin='lower', aspect='auto')
fig.colorbar(im, ax=axes[0], label="Amplitude")
axes[0].set_title("Complex Beam Pattern Amplitude")
axes[0].set_xlabel("Az offset [arcmin]")
axes[0].set_ylabel("El offset [arcmin]")

im = axes[1].imshow(np.rad2deg(np.angle(beam_grid)), extent=extent_vals, cmap="twilight", origin='lower', aspect='auto')
fig.colorbar(im, ax=axes[1], label="Phase [deg]")
axes[1].set_title("Complex Beam Pattern Phase")
axes[1].set_xlabel("Az offset [arcmin]")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "beam_pattern.png"), dpi=150)

if ZOOM_ARCMIN is not None:
    axes[0].set_xlim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
    axes[0].set_ylim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
    axes[1].set_xlim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
    axes[1].set_ylim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
    fig.savefig(os.path.join(OUT_DIR, "beam_pattern_zoom.png"), dpi=150)

plt.close(fig)

# =========================
# dBスケール ビームパターン
# =========================
peak_amp = np.nanmax(beam_grid_amp)
with np.errstate(divide='ignore', invalid='ignore'):
    beam_grid_db = 20 * np.log10(beam_grid_amp / peak_amp)
    beam_grid_db = np.clip(beam_grid_db, a_min=DB_MIN, a_max=0)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    beam_grid_db, 
    extent=extent_vals, 
    cmap="inferno",
    vmin=DB_MIN, 
    vmax=0, 
    origin='lower',
    aspect='auto'
)
fig.colorbar(im, ax=ax, label="Normalized Amplitude [dB]")
ax.set_title("Beam Pattern (dB Scale)")
ax.set_xlabel("Az offset [arcmin]")
ax.set_ylabel("El offset [arcmin]")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "beam_pattern_dB.png"), dpi=150)

if ZOOM_ARCMIN is not None:
    ax.set_xlim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
    ax.set_ylim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
    fig.savefig(os.path.join(OUT_DIR, "beam_pattern_dB_zoom.png"), dpi=150)

plt.close(fig)

# =========================
# SNRフィルタ済みビームマップ
# =========================
if SNR_THRESHOLD is not None:
    snr_mask = snr_grid >= SNR_THRESHOLD

    beam_grid_amp_sn = np.where(snr_mask, beam_grid_amp, np.nan)
    beam_grid_phase_sn = np.where(snr_mask, np.rad2deg(np.angle(beam_grid)), np.nan)

    n_total  = np.sum(~np.isnan(snr_grid))
    n_masked = np.sum(~np.isnan(snr_grid) & ~snr_mask)
    print(f"SNRフィルタ: 閾値 = {SNR_THRESHOLD}")
    print(f"  除外データ点: {n_masked} / {n_total} ({100*n_masked/max(n_total,1):.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im = axes[0].imshow(
        beam_grid_amp_sn,
        extent=extent_vals,
        cmap="viridis",
        origin='lower',
        aspect='auto'
    )
    fig.colorbar(im, ax=axes[0], label="Amplitude")
    axes[0].set_title(f"SNR-Filtered Beam Amplitude\n(SNR >= {SNR_THRESHOLD})")
    axes[0].set_xlabel("Az offset [arcmin]")
    axes[0].set_ylabel("El offset [arcmin]")

    im = axes[1].imshow(
        beam_grid_phase_sn,
        extent=extent_vals,
        cmap="twilight",
        vmin=-180,
        vmax=180,
        origin='lower',
        aspect='auto'
    )
    fig.colorbar(im, ax=axes[1], label="Phase [deg]")
    axes[1].set_title(f"SNR-Filtered Beam Phase\n(SNR >= {SNR_THRESHOLD})")
    axes[1].set_xlabel("Az offset [arcmin]")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "beam_pattern_SNR_filtered.png"), dpi=150)

    if ZOOM_ARCMIN is not None:
        axes[0].set_xlim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
        axes[0].set_ylim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
        axes[1].set_xlim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
        axes[1].set_ylim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
        fig.savefig(os.path.join(OUT_DIR, "beam_pattern_SNR_filtered_zoom.png"), dpi=150)

    plt.close(fig)

    with np.errstate(divide='ignore', invalid='ignore'):
        beam_grid_db_sn = 20 * np.log10(beam_grid_amp_sn / peak_amp)
        beam_grid_db_sn = np.clip(beam_grid_db_sn, a_min=DB_MIN, a_max=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        beam_grid_db_sn,
        extent=extent_vals,
        cmap="inferno",
        vmin=DB_MIN,
        vmax=0,
        origin='lower',
        aspect='auto'
    )
    fig.colorbar(im, ax=ax, label="Normalized Amplitude [dB]")
    ax.set_title(f"SNR-Filtered Beam Pattern (dB)\n(SNR >= {SNR_THRESHOLD})")
    ax.set_xlabel("Az offset [arcmin]")
    ax.set_ylabel("El offset [arcmin]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "beam_pattern_dB_SNR_filtered.png"), dpi=150)

    if ZOOM_ARCMIN is not None:
        ax.set_xlim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
        ax.set_ylim(-ZOOM_ARCMIN, ZOOM_ARCMIN)
        fig.savefig(os.path.join(OUT_DIR, "beam_pattern_dB_SNR_filtered_zoom.png"), dpi=150)

    plt.close(fig)

    print(f"SNRフィルタ済みビームマップを保存しました: beam_pattern_SNR_filtered.png, beam_pattern_dB_SNR_filtered.png")

# =========================
# FFT -> 開口面電場分布
# =========================
aperture = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(beam_grid)))

# dtheta_x, dtheta_y は欠損に影響されない完璧な刻み
dtheta_x = np.mean(np.diff(tx))
dtheta_y = np.mean(np.diff(ty))
dx, dy = wavelength / (nx * dtheta_x), wavelength / (ny * dtheta_y)
x, y = (np.arange(nx) - nx // 2) * dx, (np.arange(ny) - ny // 2) * dy
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
mask = R < (D / 2)

# =========================
# 開口面電場分布（振幅・位相）のプロット
# =========================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
extent_aperture = [x.min(), x.max(), y.min(), y.max()]

im = axes[0].imshow(
    np.abs(aperture),
    extent=extent_aperture,
    cmap="viridis",
    origin='lower'
)
fig.colorbar(im, ax=axes[0], label="Amplitude")
axes[0].set_title("Aperture Field Amplitude")
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("y [m]")

im = axes[1].imshow(
    np.rad2deg(np.angle(aperture)),
    extent=extent_aperture,
    cmap="twilight",
    vmin=-180, vmax=180,
    origin='lower'
)
fig.colorbar(im, ax=axes[1], label="Phase [deg]")
axes[1].set_title("Aperture Field Phase (Raw)")
axes[1].set_xlabel("x [m]")

# 表示範囲を ±20m に固定
axes[0].set_xlim(-20, 20)
axes[0].set_ylim(-20, 20)
axes[1].set_xlim(-20, 20)
axes[1].set_ylim(-20, 20)

fig.tight_layout()
fpath_aperture = os.path.join(OUT_DIR, "aperture_field.png")
fig.savefig(fpath_aperture, dpi=150)
plt.close(fig)

# =========================
# 位相処理 (180度補正機能)
# =========================
phase_raw = np.angle(aperture)
phase_corrected = phase_raw.copy()
threshold = np.deg2rad(135)
shift_val = np.deg2rad(180)
phase_corrected = np.where(phase_raw > threshold, phase_raw - shift_val, phase_corrected)
phase_corrected = np.where(phase_raw < -threshold, phase_raw + shift_val, phase_corrected)
phase_unwrapped = phase_corrected

# =========================
# 1. チルト(平面)のみのフィットと除去
# =========================
# 副鏡の影響を排除しないよう、内側の制限を 0.0 に設定
FIT_RADIUS_MIN = 0.0
FIT_RADIUS_MAX = 20.0
fit_mask = mask & (R >= FIT_RADIUS_MIN) & (R <= FIT_RADIUS_MAX)

if np.sum(fit_mask) < 10:
    print("警告: フィット範囲のデータが少なすぎます。全体マスクで処理します。")
    fit_mask = mask

Xf_tilt, Yf_tilt, Zf_tilt = X[fit_mask].flatten(), Y[fit_mask].flatten(), phase_unwrapped[fit_mask].flatten()
A_tilt = np.c_[Xf_tilt, Yf_tilt, np.ones_like(Xf_tilt)]
C_tilt, _, _, _ = np.linalg.lstsq(A_tilt, Zf_tilt, rcond=None)
plane = C_tilt[0]*X + C_tilt[1]*Y + C_tilt[2]
phase_tilt_only = phase_unwrapped - plane

# =========================
# 2. ゼルニケ多項式フィットと大域的歪みの除去
# =========================
def zernike_basis(rho, phi, num_terms=9):
    """単位円上の極座標(rho, phi)に対してゼルニケ基底を生成"""
    Z = np.zeros((num_terms, rho.size))
    if num_terms > 0: Z[0] = 1.0                                      # Z0: Piston
    if num_terms > 1: Z[1] = rho * np.cos(phi)                        # Z1: X-Tilt
    if num_terms > 2: Z[2] = rho * np.sin(phi)                        # Z2: Y-Tilt
    if num_terms > 3: Z[3] = 2 * rho**2 - 1.0                         # Z3: Defocus
    if num_terms > 4: Z[4] = rho**2 * np.cos(2*phi)                   # Z4: Astigmatism X
    if num_terms > 5: Z[5] = rho**2 * np.sin(2*phi)                   # Z5: Astigmatism Y
    if num_terms > 6: Z[6] = (3 * rho**3 - 2 * rho) * np.cos(phi)     # Z6: Coma X
    if num_terms > 7: Z[7] = (3 * rho**3 - 2 * rho) * np.sin(phi)     # Z7: Coma Y
    if num_terms > 8: Z[8] = 6 * rho**4 - 6 * rho**2 + 1.0            # Z8: Spherical
    return Z

# 座標を単位円に正規化 (rho <= 1)
rho_all = R / (D / 2.0)
phi_all = np.arctan2(Y, X)

NUM_ZERNIKE = 9
Z_all = zernike_basis(rho_all.flatten(), phi_all.flatten(), num_terms=NUM_ZERNIKE)

# 信頼できる領域のデータだけで最小二乗法を実行
Z_fit = Z_all[:, fit_mask.flatten()].T
phase_fit = phase_unwrapped[fit_mask].flatten()
C_zernike, _, _, _ = np.linalg.lstsq(Z_fit, phase_fit, rcond=None)

zernike_surface = np.dot(C_zernike, Z_all).reshape(X.shape)
phase_zernike = phase_unwrapped - zernike_surface

print(f"\n--- ゼルニケ多項式フィット成分 (項数: {NUM_ZERNIKE}) ---")
coeff_names = ["Piston", "X-Tilt", "Y-Tilt", "Defocus", "Astig X", "Astig Y", "Coma X", "Coma Y", "Spherical"]
for i in range(min(NUM_ZERNIKE, len(coeff_names))):
    print(f"  Z{i} ({coeff_names[i]:>9}): {np.rad2deg(C_zernike[i]):>7.2f} deg")

# =========================
# 平面除去前・Tilt除去・Zernike除去の 3連比較プロット
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

phase_before_display = np.where(mask, np.rad2deg(phase_unwrapped), np.nan)
im0 = axes[0].imshow(phase_before_display, extent=extent_aperture, cmap="twilight", vmin=-180, vmax=180, origin='lower')
fig.colorbar(im0, ax=axes[0], label="Phase [deg]")
axes[0].set_title("Aperture Phase (Raw)")
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("y [m]")
axes[0].add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))

phase_tilt_display = np.where(mask, np.rad2deg(phase_tilt_only), np.nan)
im1 = axes[1].imshow(phase_tilt_display, extent=extent_aperture, cmap="twilight", vmin=-180, vmax=180, origin='lower')
fig.colorbar(im1, ax=axes[1], label="Phase [deg]")
axes[1].set_title("Aperture Phase (Tilt Removed Only)")
axes[1].set_xlabel("x [m]")
axes[1].add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))

phase_zernike_display = np.where(mask, np.rad2deg(phase_zernike), np.nan)
im2 = axes[2].imshow(phase_zernike_display, extent=extent_aperture, cmap="twilight", vmin=-180, vmax=180, origin='lower')
fig.colorbar(im2, ax=axes[2], label="Phase [deg]")
axes[2].set_title("Aperture Phase (Zernike Removed)")
axes[2].set_xlabel("x [m]")
axes[2].add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))

for ax in axes:
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

fig.tight_layout()
fpath_phase_comp = os.path.join(OUT_DIR, "aperture_phase_comparison.png")
fig.savefig(fpath_phase_comp, dpi=150)
plt.close(fig)

# ==========================================
# チルト（傾き）除去の残差確認用 中央スライスプロット (Tilt除去のみのデータを使用)
# ==========================================
print("\nチルト除去確認用の中央スライスプロットを生成中...")

phase_deg_masked_tilt = np.where(mask, np.rad2deg(phase_tilt_only), np.nan)

ix_center = nx // 2
iy_center = ny // 2

phase_x_slice = phase_deg_masked_tilt[iy_center, :]
phase_y_slice = phase_deg_masked_tilt[:, ix_center]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左：X軸スライス ---
axes[0].plot(x, phase_x_slice, marker='o', color='C0', markersize=4, label='Phase along X (Y=0)')
axes[0].axvspan(-CENTER_BLOCK_SIZE_M/2, CENTER_BLOCK_SIZE_M/2, color='red', alpha=0.1, label='Sub-reflector')
axes[0].axhline(0, color='black', linestyle='--')

valid_x = ~np.isnan(phase_x_slice) & ~((x >= -CENTER_BLOCK_SIZE_M/2) & (x <= CENTER_BLOCK_SIZE_M/2))
if np.sum(valid_x) > 2:
    popt_x = np.polyfit(x[valid_x], phase_x_slice[valid_x], 1)
    fit_x = np.polyval(popt_x, x)
    axes[0].plot(x, fit_x, color='red', linestyle='-', linewidth=2, label=f'Linear Fit (Slope: {popt_x[0]:.2f} deg/m)')

axes[0].set_title("Central Slice along X-axis (Tilt Check)")
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("Phase [deg]")
axes[0].set_xlim(-D/2, D/2)
axes[0].set_ylim(-180, 180)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(loc='upper right')

# --- 右：Y軸スライス ---
axes[1].plot(y, phase_y_slice, marker='o', color='C1', markersize=4, label='Phase along Y (X=0)')
axes[1].axvspan(-CENTER_BLOCK_SIZE_M/2, CENTER_BLOCK_SIZE_M/2, color='red', alpha=0.1, label='Sub-reflector')
axes[1].axhline(0, color='black', linestyle='--')

valid_y = ~np.isnan(phase_y_slice) & ~((y >= -CENTER_BLOCK_SIZE_M/2) & (y <= CENTER_BLOCK_SIZE_M/2))
if np.sum(valid_y) > 2:
    popt_y = np.polyfit(y[valid_y], phase_y_slice[valid_y], 1)
    fit_y = np.polyval(popt_y, y)
    axes[1].plot(y, fit_y, color='red', linestyle='-', linewidth=2, label=f'Linear Fit (Slope: {popt_y[0]:.2f} deg/m)')

axes[1].set_title("Central Slice along Y-axis (Tilt Check)")
axes[1].set_xlabel("y [m]")
axes[1].set_xlim(-D/2, D/2)
axes[1].set_ylim(-180, 180)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='upper right')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "aperture_phase_tilt_check.png"), dpi=150)
plt.close(fig)

# 以降の定量評価用には Zernike除去ベースの位相を代表として扱う
phase_after = phase_zernike
phase_deg_2d = np.rad2deg(phase_after)
phase_deg_masked = np.where(mask, phase_deg_2d, np.nan)

# ==========================================
# 開口面電場分布（位相）の全スライスプロット出力 (Zernikeベースで出力)
# ==========================================
if GENERATE_APERTURE_SLICES:
    print("\n開口面電場分布（位相）の全スライスプロットを生成中...")

    AP_SLICE_X_DIR = os.path.join(OUT_DIR, "aperture_slice", "along_X_fixed_Y")
    AP_SLICE_Y_DIR = os.path.join(OUT_DIR, "aperture_slice", "along_Y_fixed_X")
    os.makedirs(AP_SLICE_X_DIR, exist_ok=True)
    os.makedirs(AP_SLICE_Y_DIR, exist_ok=True)

    # ① [Yを固定] して X方向に走る水平スライス全パターン
    for iy in range(ny):
        y_val = y[iy]
        if np.any(mask[iy, :]):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, phase_deg_masked[iy, :], marker='o', markersize=3, color='C0', linewidth=1.2)
            ax.axvspan(-CENTER_BLOCK_SIZE_M/2, CENTER_BLOCK_SIZE_M/2, color='red', alpha=0.1, label='Sub-reflector')
            
            ax.set_title(f"Aperture Phase along X (Fixed Y = {y_val:+.2f} m)")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("Phase [deg]")
            ax.set_xlim(-D/2 - 1, D/2 + 1)
            ax.set_ylim(-180, 180)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            
            fname = f"slice_Y{iy:02d}_y{y_val:+.2f}m.png".replace("+", "p").replace("-", "m").replace(".", "_")
            fig.savefig(os.path.join(AP_SLICE_X_DIR, fname), dpi=150)
            plt.close(fig)

    # ② [Xを固定] して Y方向に走る垂直スライス全パターン
    for ix in range(nx):
        x_val = x[ix]
        if np.any(mask[:, ix]):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(y, phase_deg_masked[:, ix], marker='o', markersize=3, color='C1', linewidth=1.2)
            ax.axvspan(-CENTER_BLOCK_SIZE_M/2, CENTER_BLOCK_SIZE_M/2, color='red', alpha=0.1, label='Sub-reflector')
            
            ax.set_title(f"Aperture Phase along Y (Fixed X = {x_val:+.2f} m)")
            ax.set_xlabel("y [m]")
            ax.set_ylabel("Phase [deg]")
            ax.set_xlim(-D/2 - 1, D/2 + 1)
            ax.set_ylim(-180, 180)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            
            fname = f"slice_X{ix:02d}_x{x_val:+.2f}m.png".replace("+", "p").replace("-", "m").replace(".", "_")
            fig.savefig(os.path.join(AP_SLICE_Y_DIR, fname), dpi=150)
            plt.close(fig)

    # ③ 全スライスを1枚に重ねた「サマリープロット」
    fig_sum, axes_sum = plt.subplots(1, 2, figsize=(14, 5))

    for iy in range(ny):
        if np.any(mask[iy, :]):
            axes_sum[0].plot(x, phase_deg_masked[iy, :], alpha=0.25, color='C0', linewidth=1.0)
    axes_sum[0].axvspan(-CENTER_BLOCK_SIZE_M/2, CENTER_BLOCK_SIZE_M/2, color='red', alpha=0.1)
    axes_sum[0].set_title("All X-Slices Overlaid (Fixed Y)")
    axes_sum[0].set_xlabel("x [m]")
    axes_sum[0].set_ylabel("Phase [deg]")
    axes_sum[0].set_xlim(-D/2, D/2)
    axes_sum[0].set_ylim(-180, 180)
    axes_sum[0].grid(True, linestyle='--', alpha=0.5)

    for ix in range(nx):
        if np.any(mask[:, ix]):
            axes_sum[1].plot(y, phase_deg_masked[:, ix], alpha=0.25, color='C1', linewidth=1.0)
    axes_sum[1].axvspan(-CENTER_BLOCK_SIZE_M/2, CENTER_BLOCK_SIZE_M/2, color='red', alpha=0.1)
    axes_sum[1].set_title("All Y-Slices Overlaid (Fixed X)")
    axes_sum[1].set_xlabel("y [m]")
    axes_sum[1].set_xlim(-D/2, D/2)
    axes_sum[1].set_ylim(-180, 180)
    axes_sum[1].grid(True, linestyle='--', alpha=0.5)

    fig_sum.tight_layout()
    fig_sum.savefig(os.path.join(OUT_DIR, "aperture_phase_slices_summary.png"), dpi=150)
    plt.close(fig_sum)

    print(f"開口面位相の全スライスを保存しました:\n  -> {AP_SLICE_X_DIR}/\n  -> {AP_SLICE_Y_DIR}/")

# =========================
# 鏡面誤差計算とマスキング (RMS 2種評価)
# =========================
surface_tilt_mm = (wavelength / (4 * np.pi) * phase_tilt_only) * 1e3
surface_zernike_mm = (wavelength / (4 * np.pi) * phase_zernike) * 1e3

# ブロッキング除外用マスクの準備
half_size = CENTER_BLOCK_SIZE_M / 2.0
center_block_mask = ~((X >= -half_size) & (X <= half_size) & (Y >= -half_size) & (Y <= half_size))
valid_mask_center = mask & center_block_mask

# --- 1. Tilt除去のみ のRMS評価 ---
valid_mask_threshold_tilt = mask & (np.abs(surface_tilt_mm) <= SURFACE_ERR_THRESHOLD_MM)
rms_before_tilt = np.sqrt(np.mean(surface_tilt_mm[mask]**2))
rms_after_thresh_tilt  = np.sqrt(np.mean(surface_tilt_mm[valid_mask_threshold_tilt]**2))
rms_after_center_tilt = np.sqrt(np.mean(surface_tilt_mm[valid_mask_center]**2))

print(f"\n--- 鏡面精度評価 (Tilt除去のみ) ---")
print(f"Surface RMS (除外前): {rms_before_tilt:.3f} mm")
print(f"Surface RMS (閾値除外後): {rms_after_thresh_tilt:.3f} mm")
print(f"Surface RMS (中心除外後): {rms_after_center_tilt:.3f} mm")

# --- 2. Zernike除去後 のRMS評価 ---
valid_mask_threshold_zernike = mask & (np.abs(surface_zernike_mm) <= SURFACE_ERR_THRESHOLD_MM)
rms_before_zernike = np.sqrt(np.mean(surface_zernike_mm[mask]**2))
rms_after_thresh_zernike  = np.sqrt(np.mean(surface_zernike_mm[valid_mask_threshold_zernike]**2))
rms_after_center_zernike = np.sqrt(np.mean(surface_zernike_mm[valid_mask_center]**2))

print(f"\n--- 鏡面精度評価 (Zernike除去後) ---")
print(f"Surface RMS (除外前): {rms_before_zernike:.3f} mm")
print(f"Surface RMS (閾値除外後): {rms_after_thresh_zernike:.3f} mm")
print(f"Surface RMS (中心除外後): {rms_after_center_zernike:.3f} mm")

# =========================
# 鏡面誤差のプロット出力 (共通描画関数)
# =========================
def plot_surface_error(surface_data, valid_mask_threshold, rms_vals, title_prefix, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin_val, vmax_val = -SURFACE_ERR_THRESHOLD_MM, SURFACE_ERR_THRESHOLD_MM
    extent_ap = [x.min() - dx/2, x.max() + dx/2, y.min() - dy/2, y.max() + dy/2]

    surface_before_display = np.where(mask, surface_data, np.nan)
    im0 = axes[0].imshow(surface_before_display, extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(im0, ax=axes[0], label="Surface Error [mm]")
    axes[0].set_title(f"{title_prefix}: Before Masking\n(RMS: {rms_vals[0]:.3f} mm)")
    axes[0].set_xlabel("x [m]"), axes[0].set_ylabel("y [m]")
    axes[0].add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))

    surface_thresh_display = np.where(valid_mask_threshold, surface_data, np.nan)
    im1 = axes[1].imshow(surface_thresh_display, extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(im1, ax=axes[1], label="Surface Error [mm]")
    axes[1].set_title(f"{title_prefix}: Threshold Masking\n(RMS: {rms_vals[1]:.3f} mm)")
    axes[1].set_xlabel("x [m]")
    axes[1].add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))

    surface_center_display = np.where(valid_mask_center, surface_data, np.nan)
    im2 = axes[2].imshow(surface_center_display, extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(im2, ax=axes[2], label="Surface Error [mm]")
    axes[2].set_title(f"{title_prefix}: Center Masking ({CENTER_BLOCK_SIZE_M}m)\n(RMS: {rms_vals[2]:.3f} mm)")
    axes[2].set_xlabel("x [m]")
    axes[2].add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))
    rect = plt.Rectangle((-half_size, -half_size), CENTER_BLOCK_SIZE_M, CENTER_BLOCK_SIZE_M, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
    axes[2].add_patch(rect)

    for ax in axes:
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close(fig)

plot_surface_error(surface_tilt_mm, valid_mask_threshold_tilt, 
                   [rms_before_tilt, rms_after_thresh_tilt, rms_after_center_tilt], 
                   "Tilt Only", "surface_error_tilt_only.png")

plot_surface_error(surface_zernike_mm, valid_mask_threshold_zernike, 
                   [rms_before_zernike, rms_after_thresh_zernike, rms_after_center_zernike], 
                   "Zernike Removed", "surface_error_zernike.png")

# 以降の定量評価（ノイズ影響やクロップ評価）には、より高精度なZernike除去ベースの誤差データを使用
surface_mm = surface_zernike_mm 
rms_after_center = rms_after_center_zernike

# =========================
# 低S/Nデータが鏡面に与える影響の定量評価（差分評価）
# =========================
if SNR_THRESHOLD is not None:
    print("\n==============================================")
    print(f"S/Nフィルタ(閾値: {SNR_THRESHOLD})による影響の定量評価を開始します...")
    
    beam_grid_filtered = np.where(snr_mask, beam_grid, 0j)
    
    energy_raw = np.nansum(np.abs(beam_grid)**2)
    energy_filtered = np.nansum(np.abs(beam_grid_filtered)**2)
    energy_loss_percent = (1.0 - (energy_filtered / energy_raw)) * 100
    print(f"失われたシグナルエネルギー: 全体の {energy_loss_percent:.4f} %")

    aperture_filtered = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(beam_grid_filtered)))

    phase_filt = np.angle(aperture_filtered)
    threshold_rad = np.deg2rad(135)
    shift_rad = np.deg2rad(180)
    phase_filt = np.where(phase_filt > threshold_rad, phase_filt - shift_rad, phase_filt)
    phase_filt = np.where(phase_filt < -threshold_rad, phase_filt + shift_rad, phase_filt)
    
    # チルト除去(S/Nフィルタ版はTiltのみ除去で簡略化)
    Xf_filt, Yf_filt, Zf_filt = X[mask].flatten(), Y[mask].flatten(), phase_filt[mask].flatten()
    A_filt = np.c_[Xf_filt, Yf_filt, np.ones_like(Xf_filt)]
    C_filt, _, _, _ = np.linalg.lstsq(A_filt, Zf_filt, rcond=None)
    plane_filt = C_filt[0]*X + C_filt[1]*Y + C_filt[2]
    phase_after_filt = phase_filt - plane_filt
    
    surface_filtered_mm = (wavelength / (4 * np.pi) * phase_after_filt) * 1e3

    # 差分評価
    surface_diff = surface_tilt_mm - surface_filtered_mm 
    diff_rms = np.sqrt(np.mean(surface_diff[mask]**2))
    print(f"低S/Nデータが鏡面に与えていた影響(ノイズによるウソの凹凸): RMS {diff_rms:.3f} mm")
    print("==============================================\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    vmin_val, vmax_val = -SURFACE_ERR_THRESHOLD_MM, SURFACE_ERR_THRESHOLD_MM
    extent_ap = [x.min() - dx/2, x.max() + dx/2, y.min() - dy/2, y.max() + dy/2]

    im0 = axes[0].imshow(np.where(mask, surface_tilt_mm, np.nan), extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
    axes[0].set_title("Raw Surface (Tilt Only)")
    
    im1 = axes[1].imshow(np.where(mask, surface_filtered_mm, np.nan), extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
    axes[1].set_title(f"Filtered Surface (S/N >= {SNR_THRESHOLD})")
    
    im2 = axes[2].imshow(np.where(mask, surface_diff, np.nan), extent=extent_ap, origin='lower', cmap='PRGn', vmin=-1.0, vmax=1.0)
    fig.colorbar(im2, ax=axes[2], label="Difference [mm]")
    axes[2].set_title(f"Impact of Low S/N Data\n(Difference RMS: {diff_rms:.3f} mm)")

    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))
    
    axes[0].set_ylabel("y [m]")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "surface_quantitative_diff.png"), dpi=150)
    plt.close(fig)

# =========================
# 空間クロップ（ZOOM_ARCMIN範囲のみを配列から切り出し）による粗解像度での鏡面評価
# =========================
if ZOOM_ARCMIN is not None:
    print("\n==============================================")
    print(f"空間クロップ(±{ZOOM_ARCMIN} arcmin以内のデータのみを使用)による本来の解像度での評価を開始します...")
    
    idx_x = np.where(np.abs(tx_arcmin) <= ZOOM_ARCMIN)[0]
    idx_y = np.where(np.abs(ty_arcmin) <= ZOOM_ARCMIN)[0]
    
    if len(idx_x) > 0 and len(idx_y) > 0:
        tx_crop = tx_arcmin[idx_x]
        ty_crop = ty_arcmin[idx_y]
        
        beam_grid_crop = beam_grid[idx_y[0]:idx_y[-1]+1, idx_x[0]:idx_x[-1]+1]
        
        nx_crop, ny_crop = len(tx_crop), len(ty_crop)
        
        aperture_crop = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(beam_grid_crop)))
        
        dx_crop = wavelength / (nx_crop * dtheta_x)
        dy_crop = wavelength / (ny_crop * dtheta_y)
        
        print(f"切り出し後の配列サイズ: {nx_crop} x {ny_crop} ピクセル")
        print(f"実質的なピクセル解像度: {dx_crop:.2f} m x {dy_crop:.2f} m")
        
        x_crop = (np.arange(nx_crop) - nx_crop // 2) * dx_crop
        y_crop = (np.arange(ny_crop) - ny_crop // 2) * dy_crop
        X_crop, Y_crop = np.meshgrid(x_crop, y_crop)
        R_crop = np.sqrt(X_crop**2 + Y_crop**2)
        mask_crop = R_crop < (D / 2)
        
        center_block_mask_crop = ~((X_crop >= -half_size) & (X_crop <= half_size) & (Y_crop >= -half_size) & (Y_crop <= half_size))
        valid_mask_center_crop = mask_crop & center_block_mask_crop
        
        phase_crop = np.angle(aperture_crop)
        threshold_rad = np.deg2rad(135)
        shift_rad = np.deg2rad(180)
        phase_crop = np.where(phase_crop > threshold_rad, phase_crop - shift_rad, phase_crop)
        phase_crop = np.where(phase_crop < -threshold_rad, phase_crop + shift_rad, phase_crop)
        
        if np.sum(mask_crop) > 3:
            Xf_c, Yf_c, Zf_c = X_crop[mask_crop].flatten(), Y_crop[mask_crop].flatten(), phase_crop[mask_crop].flatten()
            A_c = np.c_[Xf_c, Yf_c, np.ones_like(Xf_c)]
            C_c, _, _, _ = np.linalg.lstsq(A_c, Zf_c, rcond=None)
            plane_crop = C_c[0]*X_crop + C_c[1]*Y_crop + C_c[2]
            phase_after_crop = phase_crop - plane_crop
            
            surface_crop_mm = (wavelength / (4 * np.pi) * phase_after_crop) * 1e3
            
            rms_crop_before = np.sqrt(np.mean(surface_crop_mm[mask_crop]**2))
            rms_crop_after_center = np.sqrt(np.mean(surface_crop_mm[valid_mask_center_crop]**2))
            
            print(f"Surface RMS (切り出しデータのみ / 除外前): {rms_crop_before:.3f} mm")
            print(f"Surface RMS (切り出しデータのみ / 中心{CENTER_BLOCK_SIZE_M}m除外後): {rms_crop_after_center:.3f} mm")
            print("==============================================\n")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            vmin_val, vmax_val = -SURFACE_ERR_THRESHOLD_MM, SURFACE_ERR_THRESHOLD_MM
            
            im0 = axes[0].imshow(np.where(valid_mask_center, surface_zernike_mm, np.nan), extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
            axes[0].set_title(f"Full Data (Zernike, Pixel: {dx:.2f}m x {dy:.2f}m)\nRMS: {rms_after_center_zernike:.3f} mm")
            axes[0].set_xlabel("x [m]")
            axes[0].set_ylabel("y [m]")
            axes[0].add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))
            rect0 = plt.Rectangle((-half_size, -half_size), CENTER_BLOCK_SIZE_M, CENTER_BLOCK_SIZE_M, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
            axes[0].add_patch(rect0)
            
            extent_crop = [x_crop.min() - dx_crop/2, x_crop.max() + dx_crop/2, y_crop.min() - dy_crop/2, y_crop.max() + dy_crop/2]
            im1 = axes[1].imshow(np.where(valid_mask_center_crop, surface_crop_mm, np.nan), extent=extent_crop, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
            fig.colorbar(im1, ax=axes[1], label="Surface Error [mm]")
            axes[1].set_title(f"Cropped (±{ZOOM_ARCMIN}', Tilt Only, Pixel: {dx_crop:.2f}m x {dy_crop:.2f}m)\nRMS: {rms_crop_after_center:.3f} mm")
            axes[1].set_xlabel("x [m]")
            axes[1].add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))
            rect1 = plt.Rectangle((-half_size, -half_size), CENTER_BLOCK_SIZE_M, CENTER_BLOCK_SIZE_M, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
            axes[1].add_patch(rect1)
            
            for ax in axes:
                ax.set_xlim(-20, 20)
                ax.set_ylim(-20, 20)
                
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "surface_cropped_comparison.png"), dpi=150)
            plt.close(fig)
        else:
            print("エラー: 有効なデータ点が少なすぎて平面フィットができません。")
    else:
        print("エラー: ZOOM_ARCMIN の範囲内にデータが存在しません。")

print("完了しました。")
