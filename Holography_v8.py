import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 画面表示なし（ファイル出力専用）
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import griddata

# ==========================================
# 🔧 変更履歴
# 2026-08-27: 
# - ルッツの式（Ruze's equation）を用いて、算出したRMS(rms_after_center_tilt)から
#   6.5GHz～12.5GHzにおける表面効率（η）を計算・プロットする機能を追加。
# ==========================================

# ==========================================
# 🔧 ユーザー設定パラメータ
# ==========================================
# 処理に使用するAz/Elのオフセット範囲（±何分角以内か）。制限しない場合は None にする
OFFSET_LIMIT_ARCMIN = None

# PRD由来の補正済み座標をbeam.txtから直接読むため、SKDとの時刻合わせは行わない。

# --- 空間ズレ（分角）補正 ---
AUTO_SPACE_ALIGN = True
# AUTO_SPACE_ALIGN = False
MANUAL_AZ_OFFSET_ARCMIN = 0.0
MANUAL_EL_OFFSET_ARCMIN = 0.0

# 偶数スキャン専用のAzオフセット量 [arcmin]
EVEN_SCAN_AZ_OFFSET_ARCMIN = 0.0

# スライスグラフ（Az vs Amp/Phase）を生成するかどうか
GENERATE_SLICES = True
GENERATE_SLICES = False

# 開口面位相の全スライスプロットを生成するかどうか
GENERATE_APERTURE_SLICES = True
GENERATE_APERTURE_SLICES = False

# dBスケールビームパターンの表示下限（例: -30.0）
DB_MIN = -45.0

# ビームパターンのズーム画像の一辺の幅 [分角]。Noneにすると出力しない
ZOOM_SIZE_ARCMIN = 50.0  # 50' x 50'

# SNR閾値：beam_10.txt の SNR 列を使い、この値以下のデータ点はSNフィルタ済みビームマップに表示しない
SNR_THRESHOLD = 7.0

# 鏡面誤差として許容する最大値（これを超える部分はブロッキング等とみなしRMSから除外）
SURFACE_ERR_THRESHOLD_MM = 3.0 

# 副鏡ブロッキングとして除外する中心の正方形サイズ（メートル）
CENTER_BLOCK_SIZE_M = 3.0

# 観測コード（ディレクトリ名）
OBS_CODE = "I26149Y"

# ピークから何arcmin以上離れたデータを「背景（ベースライン）」とみなすか
BASELINE_EXCLUDE_ARCMIN = 40.0

# ガウシアンフィッティングに使用するメインビームの下限(dB)
GAUSS_FIT_CUTOFF_DB = -5.0

# =========================
# ディレクトリ・基本設定
# =========================
if len(sys.argv) >= 2:
    OBS_CODE = sys.argv[1]

BASE_DIR = OBS_CODE
BEAM_FILE = os.path.join(BASE_DIR, "beam.txt")

OUT_DIR       = os.path.join(BASE_DIR, "output_limited_test_100_hosei")

SLICE_DIR = os.path.join(BASE_DIR, "slice")
SLICE_AMP_DIR = os.path.join(SLICE_DIR, "Amp")
SLICE_PH_DIR  = os.path.join(SLICE_DIR, "Phase")

os.makedirs(OUT_DIR, exist_ok=True)
if GENERATE_SLICES:
    os.makedirs(SLICE_AMP_DIR, exist_ok=True)
    os.makedirs(SLICE_PH_DIR,  exist_ok=True)

print(f"OBS_CODE : {OBS_CODE}")
print(f"Input    : {BASE_DIR}/")

c = 3e8
f = 8.448e9
wavelength = c / f
D = 32.0
arcmin_to_rad = np.pi / (180 * 60)

# ==========================================
# 共通関数定義
# ==========================================
def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def analyze_beam_width(x_data, y_data):
    peak_amp = np.max(y_data)
    with np.errstate(divide='ignore', invalid='ignore'):
        y_db = 20 * np.log10(y_data / peak_amp)

    valid_idx = y_db >= GAUSS_FIT_CUTOFF_DB
    
    # 1. ガウシアンフィット
    popt_gauss = None
    gauss_fit_x0 = 0.0
    beamwidth_3db_arcmin = 0.0
    if np.sum(valid_idx) > 3: 
        x_fit_g, y_fit_g = x_data[valid_idx], y_data[valid_idx]
        try:
            p0 = [max(y_fit_g) - min(y_fit_g), x_fit_g[np.argmax(y_fit_g)], 1.0, min(y_fit_g)]
            popt_gauss, _ = curve_fit(gaussian, x_fit_g, y_fit_g, p0=p0)
            gauss_fit_x0 = popt_gauss[1]
            beamwidth_3db_arcmin = 2.355 * abs(popt_gauss[2])
        except Exception:
            popt_gauss = None

    # 2. 4次関数フィット
    popt_poly = None
    poly_fit_x0 = 0.0
    fwhm_poly_arcmin = 0.0
    if np.sum(valid_idx) > 4: 
        x_fit_p, y_fit_p = x_data[valid_idx], y_data[valid_idx]
        try:
            popt_poly = np.polyfit(x_fit_p, y_fit_p, 4)
            p_q = np.poly1d(popt_poly)
            
            dp = p_q.deriv()
            real_roots = dp.roots[np.isreal(dp.roots)].real
            if len(real_roots) > 0:
                valid_roots = real_roots[(real_roots >= min(x_fit_p)) & (real_roots <= max(x_fit_p))]
                poly_fit_x0 = valid_roots[np.argmax(p_q(valid_roots))] if len(valid_roots) > 0 else real_roots[np.argmax(p_q(real_roots))]
                peak_y = p_q(poly_fit_x0)
                
                baseline = np.min(y_data)
                half_max = baseline + (peak_y - baseline) / 2.0
                
                roots_half = (p_q - half_max).roots
                real_roots_half = np.sort(roots_half[np.isreal(roots_half)].real)
                
                left_roots = real_roots_half[real_roots_half < poly_fit_x0]
                right_roots = real_roots_half[real_roots_half > poly_fit_x0]
                
                if len(left_roots) > 0 and len(right_roots) > 0:
                    fwhm_poly_arcmin = right_roots[0] - left_roots[-1]
                else:
                    popt_poly = None
        except Exception:
            popt_poly = None

    # 3. 直接補間 (RAW Data Interpolation)
    beamwidth_3db_direct = np.nan
    direct_center = np.nan
    try:
        peak_idx = np.argmax(y_db)
        idx_left = np.where(y_db[:peak_idx] <= -3.0)[0]
        idx_right = np.where(y_db[peak_idx:] <= -3.0)[0] + peak_idx
        
        if len(idx_left) > 0 and len(idx_right) > 0:
            x1, y1 = x_data[idx_left[-1]], y_db[idx_left[-1]]
            x2, y2 = x_data[idx_left[-1]+1], y_db[idx_left[-1]+1]
            x_left = x1 + (-3.0 - y1) * (x2 - x1) / (y2 - y1)
            
            x3, y3 = x_data[idx_right[0]-1], y_db[idx_right[0]-1]
            x4, y4 = x_data[idx_right[0]], y_db[idx_right[0]]
            x_right = x3 + (-3.0 - y3) * (x4 - x3) / (y4 - y3)
            
            beamwidth_3db_direct = x_right - x_left
            direct_center = (x_left + x_right) / 2.0
    except Exception:
        pass

    print(f"\n--- メインビーム解析結果 ---")
    if popt_poly is not None:
        print(f"📍 4次関数 : Center = {poly_fit_x0:+.2f} arcmin, FWHM = {fwhm_poly_arcmin:.3f} arcmin")
    if popt_gauss is not None:
        print(f"📍 ガウシアン: Center = {gauss_fit_x0:+.2f} arcmin, BW(-3dB) = {beamwidth_3db_arcmin:.3f} arcmin")
    if not np.isnan(beamwidth_3db_direct):
        print(f"📍 直接補間  : Center = {direct_center:+.2f} arcmin, BW(-3dB) = {beamwidth_3db_direct:.3f} arcmin (RAW Data Interpolation)")
    print("----------------------------\n")

    return {
        'gauss': {'popt': popt_gauss, 'fwhm': beamwidth_3db_arcmin} if popt_gauss is not None else None,
        'poly': {'poly1d': np.poly1d(popt_poly) if popt_poly is not None else None, 'fwhm': fwhm_poly_arcmin} if popt_poly is not None else None
    }

# =========================
# ファイル読み込み
# =========================
try:
    beam = pd.read_csv(BEAM_FILE, encoding="utf-8-sig")
except FileNotFoundError:
    raise FileNotFoundError(f"エラー: {BEAM_FILE} が見つかりません。")

beam.columns = beam.columns.str.strip()
beam["Epoch"] = pd.to_datetime(beam["Epoch"], format="%Y/%j %H:%M:%S.%f", errors="coerce").fillna(
    pd.to_datetime(beam["Epoch"], format="%Y/%j %H:%M:%S", errors="coerce")
)
required_columns = {"Epoch", "Length", "Amp", "Phase", "SNR", "Az_Offset", "El_Offset"}
missing_columns = required_columns - set(beam.columns)
if missing_columns:
    raise ValueError(
        "エラー: beam.txtに必要な列がありません: "
        f"{', '.join(sorted(missing_columns))}\n"
        "group_up_txt_prd.pyで作成したbeam.txtを入力してください。"
    )
for column in ("Length", "Amp", "Phase", "SNR", "Az_Offset", "El_Offset"):
    beam[column] = pd.to_numeric(beam[column], errors="coerce")
for column in ("Az_Rate_arcmin_s", "El_Rate_arcmin_s"):
    if column in beam.columns:
        beam[column] = pd.to_numeric(beam[column], errors="coerce")
beam = beam.dropna(subset=["Epoch", "Length", "Amp", "Phase", "SNR", "Az_Offset", "El_Offset"])
beam["E"] = beam["Amp"] * np.exp(1j * np.deg2rad(beam["Phase"]))

def stationary_el_levels(df):
    """Return the intended raster-row El levels, excluding turnaround motion."""
    if "El_Rate_arcmin_s" in df.columns:
        stationary = df.loc[
            np.isclose(df["El_Rate_arcmin_s"].to_numpy(), 0.0, atol=1e-8), "El"
        ].to_numpy()
    else:
        stationary = df["El"].to_numpy()
    return np.unique(np.round(stationary, 6))


def estimate_scan_steps_from_beam(df, el_levels):
    """Infer sampling from in-row Az samples and the intended El row levels."""
    d_az = np.abs(np.diff(df["Az"].to_numpy()))
    if "El_Rate_arcmin_s" in df.columns:
        in_row = np.isclose(df["El_Rate_arcmin_s"].to_numpy(), 0.0, atol=1e-8)
        d_az = d_az[in_row[:-1] & in_row[1:]]
    d_el = np.abs(np.diff(el_levels))
    valid_az = d_az[(d_az > 0.01) & (d_az < 20.0)]
    valid_el = d_el[(d_el > 0.01) & (d_el < 20.0)]
    step_az = np.unique(np.round(valid_az, 3))[np.argmax(np.unique(np.round(valid_az, 3), return_counts=True)[1])] if len(valid_az) > 0 else 3.0
    step_el = np.unique(np.round(valid_el, 3))[np.argmax(np.unique(np.round(valid_el, 3), return_counts=True)[1])] if len(valid_el) > 0 else 3.0
    return step_az, step_el

beam = beam.sort_values("Epoch")
beam["Az"] = beam["Az_Offset"]
beam["El"] = beam["El_Offset"]
el_levels = stationary_el_levels(beam)
if len(el_levels) < 2:
    raise ValueError("エラー: PRD座標から2本以上の一定El走査列を検出できませんでした。")
# PRDには折返しのための短いEl移動が入る。これはビーム格子の新しい
# El列ではないため、最寄りの本走査El行へ割り当てる。
beam["El_PRD_continuous"] = beam["El"]
beam["El"] = el_levels[np.abs(beam["El"].to_numpy()[:, None] - el_levels).argmin(axis=1)]
df = beam.copy()
SCAN_STEP_AZ, SCAN_STEP_EL = estimate_scan_steps_from_beam(df, el_levels)
print(f"beam.txtのPRD補正済み座標から推定したスキャン間隔 -> Az: {SCAN_STEP_AZ} arcmin, El: {SCAN_STEP_EL} arcmin")
print("[INFO] PRDの折返し中のEl座標は、最寄りの本走査El行へスナップして格子化します。")

# =========================
# 変数抽出 ＆ 空間ズレ補正 ＆ 位相補正
# =========================
E, Az, El, SNR = df["E"].values, df["Az"].values, df["El"].values, df["SNR"].values
times_sec = df["Epoch"].astype(np.int64).values / 1e9  

az_offset = MANUAL_AZ_OFFSET_ARCMIN
el_offset = MANUAL_EL_OFFSET_ARCMIN

if AUTO_SPACE_ALIGN:
    max_idx = np.argmax(np.abs(E))
    az_offset = Az[max_idx]
    el_offset = El[max_idx]

Az_shifted = Az - az_offset
El_shifted = El - el_offset

unique_els, el_inverse = np.unique(np.round(El_shifted, 2), return_inverse=True)
is_even_scan = (el_inverse % 2 == 1)

if EVEN_SCAN_AZ_OFFSET_ARCMIN != 0.0:
    Az_shifted[is_even_scan] += EVEN_SCAN_AZ_OFFSET_ARCMIN

on_idx = np.where((Az == 0) & (El == 0) & (np.abs(E) > 1.0))[0]
E_corr = E.copy()

if len(on_idx) >= 2:
    on_phases = np.unwrap(np.angle(E[on_idx]))
    phi_ref_interp = np.interp(times_sec, times_sec[on_idx], on_phases)
    E_corr = E_corr * np.exp(-1j * phi_ref_interp)

if OFFSET_LIMIT_ARCMIN is not None:
    limit_mask = (np.abs(Az_shifted) <= OFFSET_LIMIT_ARCMIN) & (np.abs(El_shifted) <= OFFSET_LIMIT_ARCMIN)
    Az, El, Az_shifted, El_shifted, E_corr, SNR, times_sec = Az[limit_mask], El[limit_mask], Az_shifted[limit_mask], El_shifted[limit_mask], E_corr[limit_mask], SNR[limit_mask], times_sec[limit_mask]

valid_scan_mask = np.ones(len(Az), dtype=bool)
is_center = (Az == 0) & (El == 0)

if np.sum(is_center) > 1:
    valid_scan_mask[is_center] = False 
    for is_scan in [(El == 0) & (Az != 0), (Az == 0) & (El != 0)]:
        if np.sum(is_scan) > 0:
            valid_scan_mask[np.where(is_center)[0][np.argmin(np.abs(times_sec[np.where(is_center)[0]] - np.mean(times_sec[is_scan])))]] = True

Az_orig = Az[valid_scan_mask]
El_orig = El[valid_scan_mask]

Az = Az_shifted[valid_scan_mask]
El = El_shifted[valid_scan_mask]
E_corr = E_corr[valid_scan_mask]
SNR = SNR[valid_scan_mask]
times_sec = times_sec[valid_scan_mask]

# =========================
# スライスプロット＆ビーム幅(FWHM)解析の生成
# =========================
if GENERATE_SLICES:
    print("Generating slice plots and beam-width (FWHM) analysis...")
    mask_el0 = (El_orig == 0)
    if np.any(mask_el0):
        Az_el0 = Az[mask_el0]
        Amp_el0 = np.abs(E_corr[mask_el0])
        Ph_el0 = np.rad2deg(np.angle(E_corr[mask_el0]))
        sort_idx = np.argsort(Az_el0)
        Az_el0, Amp_el0, Ph_el0 = Az_el0[sort_idx], Amp_el0[sort_idx], Ph_el0[sort_idx]

        peak_amp = np.max(Amp_el0)
        with np.errstate(divide='ignore', invalid='ignore'):
            Amp_el0_db = 20 * np.log10(Amp_el0 / peak_amp)
        fit_res = analyze_beam_width(Az_el0, Amp_el0)
        valid_idx = Amp_el0_db >= GAUSS_FIT_CUTOFF_DB

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(Az_el0, Amp_el0_db, 'k.', alpha=0.3, label="All Data")
        if np.any(valid_idx):
            Az_fit = Az_el0[valid_idx]
            axes[0].plot(Az_fit, Amp_el0_db[valid_idx], 'ro', markersize=4, label="Fit Range")
            x_fit = np.linspace(Az_fit.min(), Az_fit.max(), 300)
            if fit_res.get('gauss'):
                g_y = gaussian(x_fit, *fit_res['gauss']['popt'])
                with np.errstate(divide='ignore', invalid='ignore'):
                    g_y_db = 20 * np.log10(g_y / peak_amp)
                axes[0].plot(x_fit, g_y_db, 'r--', lw=2,
                             label=f"Gauss FWHM: {fit_res['gauss']['fwhm']:.2f}'")
            if fit_res.get('poly'):
                q_y = fit_res['poly']['poly1d'](x_fit)
                with np.errstate(divide='ignore', invalid='ignore'):
                    q_y_db = 20 * np.log10(np.maximum(q_y, 1e-10) / peak_amp)
                axes[0].plot(x_fit, q_y_db, 'b-', label="4th order fit")
            axes[0].set_xlim(Az_fit.min() - 5.0, Az_fit.max() + 5.0)
        axes[0].axhline(-3.0, color='gray', linestyle=':', alpha=0.7, label="-3 dB Line")
        axes[0].set_ylim(GAUSS_FIT_CUTOFF_DB - 5, 2.0)
        axes[0].set_ylabel("Normalized Amplitude [dB]")
        axes[0].set_title("El=0 Scan: Az offset vs Amplitude [dB] & Phase")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[1].plot(Az_el0, Ph_el0, '-', linewidth=1.0, color='purple')
        axes[1].set_xlabel("Az offset [arcmin]")
        axes[1].set_ylabel("Phase [deg]")
        axes[1].set_ylim(-180, 180)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "el0_az_scan_dB_phase.png"), dpi=150)
        plt.close(fig)

    for el_val_orig in np.unique(El_orig):
        mask_el = (El_orig == el_val_orig)
        sort_idx = np.argsort(Az[mask_el])
        Az_sorted = Az[mask_el][sort_idx]
        E_sorted = E_corr[mask_el][sort_idx]
        el_val_actual = np.mean(El[mask_el])
        el_label = f"El{el_val_actual:+.1f}arcmin".replace("+", "p").replace("-", "m").replace(".", "_")
        for values, ylabel, title, output_dir in (
            (np.abs(E_sorted), "Amplitude", "Amplitude", SLICE_AMP_DIR),
            (np.rad2deg(np.angle(E_sorted)), "Phase [deg]", "Phase", SLICE_PH_DIR),
        ):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(Az_sorted, values, marker='o', markersize=3, linewidth=1.0)
            ax.set_xlabel("Az offset [arcmin]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Az Scan {title} (El = {el_val_actual:+.1f}')")
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{el_label}.png"), dpi=150)
            plt.close(fig)

# =========================
# 格子化 (アベレージングによる強固なグリッド作成)
# =========================
az_min, az_max = np.min(Az), np.max(Az)
el_min, el_max = np.min(El), np.max(El)

tx_arcmin = np.arange(az_min, az_max + SCAN_STEP_AZ/2, SCAN_STEP_AZ)
ty_arcmin = np.arange(el_min, el_max + SCAN_STEP_EL/2, SCAN_STEP_EL)

tx = tx_arcmin * arcmin_to_rad
ty = ty_arcmin * arcmin_to_rad
nx, ny = len(tx), len(ty)

beam_grid_sum, snr_grid_sum, count_grid = np.zeros((ny, nx), dtype=complex), np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=int)

for i in range(len(Az)):
    ix, iy = np.argmin(np.abs(tx_arcmin - Az[i])), np.argmin(np.abs(ty_arcmin - El[i]))
    beam_grid_sum[iy, ix] += E_corr[i]
    snr_grid_sum[iy, ix]  += SNR[i]
    count_grid[iy, ix]    += 1

valid_cells = count_grid > 0
beam_grid, snr_grid = np.zeros((ny, nx), dtype=complex), np.full((ny, nx), np.nan)
beam_grid[valid_cells] = beam_grid_sum[valid_cells] / count_grid[valid_cells]
snr_grid[valid_cells]  = snr_grid_sum[valid_cells] / count_grid[valid_cells]

if np.any(~valid_cells):
    Y_valid, X_valid = np.where(valid_cells)
    Y_all, X_all = np.mgrid[0:ny, 0:nx]
    beam_grid = griddata((Y_valid, X_valid), beam_grid[valid_cells], (Y_all, X_all), method='nearest')
    snr_grid = griddata((Y_valid, X_valid), snr_grid[valid_cells], (Y_all, X_all), method='nearest')

beam_grid_amp = np.abs(beam_grid)
extent_vals = [tx_arcmin.max(), tx_arcmin.min(), ty_arcmin.min(), ty_arcmin.max()] 

# =========================
# 複素ビームパターン (Linear & dB)
# =========================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im = axes[0].imshow(beam_grid_amp, extent=extent_vals, cmap="viridis", origin='lower', aspect='auto')
fig.colorbar(im, ax=axes[0], label="Amplitude")
axes[0].set_title("Complex Beam Pattern Amplitude"), axes[0].set_xlabel("Az offset [arcmin]"), axes[0].set_ylabel("El offset [arcmin]")

im = axes[1].imshow(np.rad2deg(np.angle(beam_grid)), extent=extent_vals, cmap="twilight", origin='lower', aspect='auto')
fig.colorbar(im, ax=axes[1], label="Phase [deg]")
axes[1].set_title("Complex Beam Pattern Phase"), axes[1].set_xlabel("Az offset [arcmin]")
fig.tight_layout(), fig.savefig(os.path.join(OUT_DIR, "beam_pattern.png"), dpi=150), plt.close(fig)

peak_amp = np.nanmax(beam_grid_amp)
with np.errstate(divide='ignore', invalid='ignore'):
    beam_grid_db = 20 * np.log10(beam_grid_amp / peak_amp)
    beam_grid_db = np.clip(beam_grid_db, a_min=DB_MIN, a_max=0)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(beam_grid_db, extent=extent_vals, cmap="inferno", vmin=DB_MIN, vmax=0, origin='lower', aspect='auto')
fig.colorbar(im, ax=ax, label="Normalized Amplitude [dB]")
ax.set_title("Beam Pattern (dB Scale)"), ax.set_xlabel("Az offset [arcmin]"), ax.set_ylabel("El offset [arcmin]")
fig.tight_layout(), fig.savefig(os.path.join(OUT_DIR, "beam_pattern_dB.png"), dpi=150)
if ZOOM_SIZE_ARCMIN is not None:
    zoom_half_width = ZOOM_SIZE_ARCMIN / 2.0
    zoom_size_label = f"{ZOOM_SIZE_ARCMIN:g}".replace(".", "p")

    # Zoom plots deliberately use beam_grid_amp / beam_grid_db directly.
    # The SNR_THRESHOLD mask is only applied later in the surface-error comparison.
    fig_zoom_linear, ax_zoom_linear = plt.subplots(figsize=(6, 5))
    im_zoom_linear = ax_zoom_linear.imshow(
        beam_grid_amp, extent=extent_vals, cmap="viridis", origin='lower', aspect='auto'
    )
    fig_zoom_linear.colorbar(im_zoom_linear, ax=ax_zoom_linear, label="Amplitude")
    ax_zoom_linear.set_xlim(zoom_half_width, -zoom_half_width)
    ax_zoom_linear.set_ylim(-zoom_half_width, zoom_half_width)
    ax_zoom_linear.set_title(
        f"Beam Pattern (Linear, {ZOOM_SIZE_ARCMIN:g}' x {ZOOM_SIZE_ARCMIN:g}', All SNR)"
    )
    ax_zoom_linear.set_xlabel("Az offset [arcmin]")
    ax_zoom_linear.set_ylabel("El offset [arcmin]")
    fig_zoom_linear.tight_layout()
    fig_zoom_linear.savefig(
        os.path.join(OUT_DIR, f"beam_pattern_linear_zoom_{zoom_size_label}x{zoom_size_label}arcmin.png"),
        dpi=150,
    )
    plt.close(fig_zoom_linear)

    fig_zoom_db, ax_zoom_db = plt.subplots(figsize=(6, 5))
    im_zoom_db = ax_zoom_db.imshow(
        beam_grid_db, extent=extent_vals, cmap="inferno", vmin=DB_MIN, vmax=0,
        origin='lower', aspect='auto'
    )
    fig_zoom_db.colorbar(im_zoom_db, ax=ax_zoom_db, label="Normalized Amplitude [dB]")
    ax_zoom_db.set_xlim(zoom_half_width, -zoom_half_width)
    ax_zoom_db.set_ylim(-zoom_half_width, zoom_half_width)
    ax_zoom_db.set_title(
        f"Beam Pattern (dB, {ZOOM_SIZE_ARCMIN:g}' x {ZOOM_SIZE_ARCMIN:g}', All SNR)"
    )
    ax_zoom_db.set_xlabel("Az offset [arcmin]")
    ax_zoom_db.set_ylabel("El offset [arcmin]")
    fig_zoom_db.tight_layout()
    fig_zoom_db.savefig(
        os.path.join(OUT_DIR, f"beam_pattern_dB_zoom_{zoom_size_label}x{zoom_size_label}arcmin.png"),
        dpi=150,
    )
    plt.close(fig_zoom_db)

    # Linear amplitude, dB amplitude, and phase in one zoomed figure.
    # As with the individual zoom plots, no SNR threshold is applied here.
    beam_grid_phase_deg = np.rad2deg(np.angle(beam_grid))
    fig_zoom_all, axes_zoom = plt.subplots(1, 3, figsize=(16, 5))

    im_zoom_amp = axes_zoom[0].imshow(
        beam_grid_amp, extent=extent_vals, cmap="viridis", origin='lower', aspect='auto'
    )
    fig_zoom_all.colorbar(im_zoom_amp, ax=axes_zoom[0], label="Amplitude")
    axes_zoom[0].set_title("Linear Amplitude")

    im_zoom_amp_db = axes_zoom[1].imshow(
        beam_grid_db, extent=extent_vals, cmap="inferno", vmin=DB_MIN, vmax=0,
        origin='lower', aspect='auto'
    )
    fig_zoom_all.colorbar(im_zoom_amp_db, ax=axes_zoom[1], label="Normalized Amplitude [dB]")
    axes_zoom[1].set_title("Amplitude [dB]")

    im_zoom_phase = axes_zoom[2].imshow(
        beam_grid_phase_deg, extent=extent_vals, cmap="twilight", vmin=-180, vmax=180,
        origin='lower', aspect='auto'
    )
    fig_zoom_all.colorbar(im_zoom_phase, ax=axes_zoom[2], label="Phase [deg]")
    axes_zoom[2].set_title("Phase")

    for ax_zoom in axes_zoom:
        ax_zoom.set_xlim(zoom_half_width, -zoom_half_width)
        ax_zoom.set_ylim(-zoom_half_width, zoom_half_width)
        ax_zoom.set_xlabel("Az offset [arcmin]")
        ax_zoom.set_ylabel("El offset [arcmin]")

    fig_zoom_all.suptitle(
        f"Beam Pattern ({ZOOM_SIZE_ARCMIN:g}' x {ZOOM_SIZE_ARCMIN:g}', All SNR)"
    )
    fig_zoom_all.tight_layout()
    fig_zoom_all.savefig(
        os.path.join(OUT_DIR, f"beam_pattern_zoom_{zoom_size_label}x{zoom_size_label}arcmin.png"),
        dpi=150,
    )
    plt.close(fig_zoom_all)
plt.close(fig)

# =========================
# FFT -> 開口面電場分布
# =========================
aperture = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(beam_grid)))

dtheta_x, dtheta_y = np.mean(np.diff(tx)), np.mean(np.diff(ty))
dx, dy = wavelength / (nx * dtheta_x), wavelength / (ny * dtheta_y)
x, y = (np.arange(nx) - nx // 2) * dx, (np.arange(ny) - ny // 2) * dy
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
mask = R < (D / 2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
extent_aperture = [x.min(), x.max(), y.min(), y.max()]

# 振幅 (Linear)
aperture_amp = np.abs(aperture)
im = axes[0].imshow(aperture_amp, extent=extent_aperture, cmap="viridis", origin='lower')
fig.colorbar(im, ax=axes[0], label="Amplitude")
axes[0].set_title("Aperture Field Amplitude"), axes[0].set_xlabel("x [m]"), axes[0].set_ylabel("y [m]")

# 振幅 (dBスケール追加)
with np.errstate(divide='ignore', invalid='ignore'):
    aperture_db = 20 * np.log10(aperture_amp / np.nanmax(aperture_amp))
    aperture_db = np.clip(aperture_db, a_min=DB_MIN, a_max=0)
im_db = axes[1].imshow(aperture_db, extent=extent_aperture, cmap="inferno", origin='lower', vmin=DB_MIN, vmax=0)
fig.colorbar(im_db, ax=axes[1], label="Amplitude [dB]")
axes[1].set_title("Aperture Field Amplitude (dB)"), axes[1].set_xlabel("x [m]")

# 位相
im_ph = axes[2].imshow(np.rad2deg(np.angle(aperture)), extent=extent_aperture, cmap="twilight", vmin=-180, vmax=180, origin='lower')
fig.colorbar(im_ph, ax=axes[2], label="Phase [deg]")
axes[2].set_title("Aperture Field Phase (Raw)"), axes[2].set_xlabel("x [m]")

for ax in axes:
    ax.set_xlim(-20, 20), ax.set_ylim(-20, 20)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "aperture_field_no_circle.png"), dpi=150) # ★点線なし保存

for ax in axes:
    ax.add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))
fig.savefig(os.path.join(OUT_DIR, "aperture_field.png"), dpi=150) # ★点線あり保存
plt.close(fig)

# =========================
# 位相処理 (180度補正機能)
# =========================
phase_raw = np.angle(aperture)
threshold, shift_val = np.deg2rad(135), np.deg2rad(180)
phase_corrected = np.where(phase_raw > threshold, phase_raw - shift_val, phase_raw)
phase_corrected = np.where(phase_corrected < -threshold, phase_corrected + shift_val, phase_corrected)
phase_unwrapped = phase_corrected

# =========================
# チルト(平面)のみのフィットと除去
# =========================
FIT_RADIUS_MIN = 2.0
FIT_RADIUS_MAX = 14.0
fit_mask = mask & (R >= FIT_RADIUS_MIN) & (R <= FIT_RADIUS_MAX)

if np.sum(fit_mask) < 10:
    fit_mask = mask

Xf_tilt, Yf_tilt, Zf_tilt = X[fit_mask].flatten(), Y[fit_mask].flatten(), phase_unwrapped[fit_mask].flatten()
A_tilt = np.c_[Xf_tilt, Yf_tilt, np.ones_like(Xf_tilt)]
C_tilt, _, _, _ = np.linalg.lstsq(A_tilt, Zf_tilt, rcond=None)
plane = C_tilt[0]*X + C_tilt[1]*Y + C_tilt[2]
phase_tilt_only = phase_unwrapped - plane

# 平面除去前・Tilt除去の 比較プロット
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# マスクを外し全域表示
phase_before_display = np.rad2deg(phase_unwrapped)
im0 = axes[0].imshow(phase_before_display, extent=extent_aperture, cmap="twilight", vmin=-180, vmax=180, origin='lower')
fig.colorbar(im0, ax=axes[0], label="Phase [deg]")
axes[0].set_title("Aperture Phase (Raw)")
axes[0].set_xlabel("x [m]"), axes[0].set_ylabel("y [m]")

# マスクを外し全域表示
phase_tilt_display = np.rad2deg(phase_tilt_only)
im1 = axes[1].imshow(phase_tilt_display, extent=extent_aperture, cmap="twilight", vmin=-180, vmax=180, origin='lower')
fig.colorbar(im1, ax=axes[1], label="Phase [deg]")
axes[1].set_title("Aperture Phase (Tilt Removed Only)")
axes[1].set_xlabel("x [m]")

for ax in axes:
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "aperture_phase_comparison_no_circle.png"), dpi=150) # ★点線なし保存

for ax in axes:
    ax.add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))
fig.savefig(os.path.join(OUT_DIR, "aperture_phase_comparison.png"), dpi=150) # ★点線あり保存
plt.close(fig)

# ==========================================
# チルト（傾き）除去の残差確認用 中央スライスプロット
# ==========================================
phase_deg_masked_tilt = np.where(mask, np.rad2deg(phase_tilt_only), np.nan)
ix_center, iy_center = nx // 2, ny // 2
phase_x_slice = phase_deg_masked_tilt[iy_center, :]
phase_y_slice = phase_deg_masked_tilt[:, ix_center]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, coord, values, axis_name, color in (
    (axes[0], x, phase_x_slice, 'X', 'C0'),
    (axes[1], y, phase_y_slice, 'Y', 'C1'),
):
    ax.plot(coord, values, marker='o', color=color, markersize=4,
            label=f'Phase along {axis_name}')
    ax.axvspan(-CENTER_BLOCK_SIZE_M/2, CENTER_BLOCK_SIZE_M/2,
               color='red', alpha=0.1, label='Sub-reflector')
    ax.axhline(0, color='black', linestyle='--')
    valid = ~np.isnan(values) & ~((coord >= -CENTER_BLOCK_SIZE_M/2) &
                                  (coord <= CENTER_BLOCK_SIZE_M/2))
    if np.sum(valid) > 2:
        fit = np.polyfit(coord[valid], values[valid], 1)
        ax.plot(coord, np.polyval(fit, coord), color='red', linewidth=2,
                label=f'Linear Fit (Slope: {fit[0]:.2f} deg/m)')
    ax.set_title(f"Central Slice along {axis_name}-axis (Tilt Check)")
    ax.set_xlabel(f"{axis_name.lower()} [m]")
    ax.set_ylabel("Phase [deg]")
    ax.set_xlim(-D/2, D/2)
    ax.set_ylim(-180, 180)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "aperture_phase_tilt_check.png"), dpi=150)
plt.close(fig)

phase_after = phase_tilt_only
phase_deg_2d = np.rad2deg(phase_after)
# 外側も表示するため np.where によるマスクを外す
phase_deg_masked = phase_deg_2d

# ==========================================
# 開口面電場分布（位相）の全スライスプロット出力
# ==========================================
if GENERATE_APERTURE_SLICES:
    AP_SLICE_X_DIR = os.path.join(OUT_DIR, "aperture_slice", "along_X_fixed_Y")
    AP_SLICE_Y_DIR = os.path.join(OUT_DIR, "aperture_slice", "along_Y_fixed_X")
    os.makedirs(AP_SLICE_X_DIR, exist_ok=True)
    os.makedirs(AP_SLICE_Y_DIR, exist_ok=True)

    for index, coord, values, fixed_coord, output_dir, prefix in (
        *[(iy, x, phase_deg_masked[iy, :], y[iy], AP_SLICE_X_DIR, 'Y') for iy in range(ny) if np.any(mask[iy, :])],
        *[(ix, y, phase_deg_masked[:, ix], x[ix], AP_SLICE_Y_DIR, 'X') for ix in range(nx) if np.any(mask[:, ix])],
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(coord, values, marker='o', markersize=3, linewidth=1.2)
        ax.axvspan(-CENTER_BLOCK_SIZE_M/2, CENTER_BLOCK_SIZE_M/2, color='red', alpha=0.1)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f"Aperture Phase (Fixed {prefix} = {fixed_coord:+.2f} m)")
        ax.set_xlabel(f"{'x' if prefix == 'Y' else 'y'} [m]")
        ax.set_ylabel("Phase [deg]")
        ax.set_xlim(-D/2 - 1, D/2 + 1)
        ax.set_ylim(-180, 180)
        ax.grid(True, linestyle='--', alpha=0.5)
        filename = f"slice_{prefix}{index:02d}_{fixed_coord:+.2f}m.png".replace("+", "p").replace("-", "m").replace(".", "_")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for iy in range(ny):
        if np.any(mask[iy, :]):
            axes[0].plot(x, phase_deg_masked[iy, :], alpha=0.25, color='C0')
    for ix in range(nx):
        if np.any(mask[:, ix]):
            axes[1].plot(y, phase_deg_masked[:, ix], alpha=0.25, color='C1')
    for ax, axis_name in zip(axes, ('X', 'Y')):
        ax.axvspan(-CENTER_BLOCK_SIZE_M/2, CENTER_BLOCK_SIZE_M/2, color='red', alpha=0.1)
        ax.set_title(f"All {axis_name}-Slices Overlaid")
        ax.set_xlabel(f"{axis_name.lower()} [m]")
        ax.set_ylabel("Phase [deg]")
        ax.set_xlim(-D/2, D/2)
        ax.set_ylim(-180, 180)
        ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "aperture_phase_slices_summary.png"), dpi=150)
    plt.close(fig)

# =========================
# 鏡面誤差計算とマスキング (Tilt除去のみのRMS評価)
# =========================
surface_tilt_mm = (wavelength / (4 * np.pi) * phase_tilt_only) * 1e3

half_size = CENTER_BLOCK_SIZE_M / 2.0
center_block_mask = ~((X >= -half_size) & (X <= half_size) & (Y >= -half_size) & (Y <= half_size))
valid_mask_center = mask & center_block_mask

valid_mask_threshold_tilt = mask & (np.abs(surface_tilt_mm) <= SURFACE_ERR_THRESHOLD_MM)
rms_before_tilt = np.sqrt(np.mean(surface_tilt_mm[mask]**2))
rms_after_thresh_tilt  = np.sqrt(np.mean(surface_tilt_mm[valid_mask_threshold_tilt]**2))
rms_after_center_tilt = np.sqrt(np.mean(surface_tilt_mm[valid_mask_center]**2))

print(f"\n--- 鏡面精度評価 (Tilt除去のみ) ---")
print(f"Surface RMS (除外前): {rms_before_tilt:.3f} mm")
print(f"Surface RMS (閾値除外後): {rms_after_thresh_tilt:.3f} mm")
print(f"Surface RMS (中心除外後): {rms_after_center_tilt:.3f} mm")

# =========================
# 鏡面誤差のプロット出力
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmin_val, vmax_val = -SURFACE_ERR_THRESHOLD_MM, SURFACE_ERR_THRESHOLD_MM
extent_ap = [x.min() - dx/2, x.max() + dx/2, y.min() - dy/2, y.max() + dy/2]

# マスクを外して全域表示
im0 = axes[0].imshow(surface_tilt_mm, extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
fig.colorbar(im0, ax=axes[0], label="Surface Error [mm]")
axes[0].set_title(f"Tilt Only: Before Masking\n(RMS: {rms_before_tilt:.3f} mm)")
axes[0].set_xlabel("x [m]"), axes[0].set_ylabel("y [m]")

# 表示用の閾値マスク（16m外側も値が閾値以下なら描画させる）
display_threshold_mask = np.abs(surface_tilt_mm) <= SURFACE_ERR_THRESHOLD_MM
im1 = axes[1].imshow(np.where(display_threshold_mask, surface_tilt_mm, np.nan), extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
fig.colorbar(im1, ax=axes[1], label="Surface Error [mm]")
axes[1].set_title(f"Tilt Only: Threshold Masking\n(RMS: {rms_after_thresh_tilt:.3f} mm)")
axes[1].set_xlabel("x [m]")

# 中心除外マスクのみ適用（外側は描画）
im2 = axes[2].imshow(np.where(center_block_mask, surface_tilt_mm, np.nan), extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
fig.colorbar(im2, ax=axes[2], label="Surface Error [mm]")
axes[2].set_title(f"Tilt Only: Center Masking ({CENTER_BLOCK_SIZE_M}m)\n(RMS: {rms_after_center_tilt:.3f} mm)")
axes[2].set_xlabel("x [m]")
axes[2].add_patch(plt.Rectangle((-half_size, -half_size), CENTER_BLOCK_SIZE_M, CENTER_BLOCK_SIZE_M, linewidth=1, edgecolor='black', facecolor='none', linestyle='--'))

for ax in axes:
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "surface_error_tilt_only_no_circle.png"), dpi=150) # ★点線なし保存

for ax in axes:
    ax.add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))
fig.savefig(os.path.join(OUT_DIR, "surface_error_tilt_only.png"), dpi=150) # ★点線あり保存
plt.close(fig)

# =========================
# ★ 低S/Nデータが鏡面に与える影響の定量評価（差分評価）
# =========================
if SNR_THRESHOLD is not None:
    snr_mask = snr_grid >= SNR_THRESHOLD
    beam_grid_filtered = np.where(snr_mask, beam_grid, 0j)
    
    aperture_filtered = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(beam_grid_filtered)))

    phase_filt = np.angle(aperture_filtered)
    threshold_rad = np.deg2rad(135)
    shift_rad = np.deg2rad(180)
    phase_filt = np.where(phase_filt > threshold_rad, phase_filt - shift_rad, phase_filt)
    phase_filt = np.where(phase_filt < -threshold_rad, phase_filt + shift_rad, phase_filt)
    
    if np.sum(fit_mask) > 3:
        Xf_filt, Yf_filt, Zf_filt = X[fit_mask].flatten(), Y[fit_mask].flatten(), phase_filt[fit_mask].flatten()
        A_filt = np.c_[Xf_filt, Yf_filt, np.ones_like(Xf_filt)]
        C_filt, _, _, _ = np.linalg.lstsq(A_filt, Zf_filt, rcond=None)
        plane_filt = C_filt[0]*X + C_filt[1]*Y + C_filt[2]
        phase_after_filt = phase_filt - plane_filt
    else:
        phase_after_filt = phase_filt
    
    surface_filtered_mm = (wavelength / (4 * np.pi) * phase_after_filt) * 1e3

    surface_diff = surface_tilt_mm - surface_filtered_mm
    diff_rms = np.sqrt(np.mean(surface_diff[mask]**2))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmin_val, vmax_val = -SURFACE_ERR_THRESHOLD_MM, SURFACE_ERR_THRESHOLD_MM

    im0 = axes[0].imshow(surface_tilt_mm, extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
    axes[0].set_title("Raw Surface (All Data, Tilt Only)")
    
    im1 = axes[1].imshow(surface_filtered_mm, extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
    axes[1].set_title(f"Filtered Surface (S/N >= {SNR_THRESHOLD}, Tilt Only)")
    
    im2 = axes[2].imshow(surface_diff, extent=extent_ap, origin='lower', cmap='PRGn', vmin=-1.0, vmax=1.0)
    fig.colorbar(im2, ax=axes[2], label="Difference [mm]")
    axes[2].set_title(f"Impact of Low S/N Data\n(Difference RMS: {diff_rms:.3f} mm)")

    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
    axes[0].set_ylabel("y [m]")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "surface_quantitative_diff_no_circle.png"), dpi=150) # ★点線なし保存

    for ax in axes:
        ax.add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))
    fig.savefig(os.path.join(OUT_DIR, "surface_quantitative_diff.png"), dpi=150) # ★点線あり保存
    plt.close(fig)

# =========================
# ★ 空間クロップ（粗解像度での鏡面評価）Tilt 補正のみ
# =========================
if ZOOM_SIZE_ARCMIN is not None:
    zoom_half_width = ZOOM_SIZE_ARCMIN / 2.0
    idx_x = np.where(np.abs(tx_arcmin) <= zoom_half_width)[0]
    idx_y = np.where(np.abs(ty_arcmin) <= zoom_half_width)[0]
    
    if len(idx_x) > 0 and len(idx_y) > 0:
        tx_crop = tx_arcmin[idx_x]
        ty_crop = ty_arcmin[idx_y]
        
        beam_grid_crop = beam_grid[idx_y[0]:idx_y[-1]+1, idx_x[0]:idx_x[-1]+1]
        nx_crop, ny_crop = len(tx_crop), len(ty_crop)
        
        aperture_crop = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(beam_grid_crop)))
        
        dx_crop = wavelength / (nx_crop * dtheta_x)
        dy_crop = wavelength / (ny_crop * dtheta_y)
        
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
        
        fit_mask_crop = mask_crop & (R_crop >= FIT_RADIUS_MIN) & (R_crop <= FIT_RADIUS_MAX)
        
        if np.sum(fit_mask_crop) > 3:
            Xf_c = X_crop[fit_mask_crop].flatten()
            Yf_c = Y_crop[fit_mask_crop].flatten()
            Zf_c = phase_crop[fit_mask_crop].flatten()
            
            A_c = np.c_[Xf_c, Yf_c, np.ones_like(Xf_c)]
            C_c, _, _, _ = np.linalg.lstsq(A_c, Zf_c, rcond=None)
            
            plane_crop = C_c[0]*X_crop + C_c[1]*Y_crop + C_c[2]
            phase_after_crop = phase_crop - plane_crop
            surface_crop_mm = (wavelength / (4 * np.pi) * phase_after_crop) * 1e3
            
            rms_crop_after_center = np.sqrt(np.mean(surface_crop_mm[valid_mask_center_crop]**2))
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            vmin_val, vmax_val = -SURFACE_ERR_THRESHOLD_MM, SURFACE_ERR_THRESHOLD_MM
            
            im0 = axes[0].imshow(np.where(center_block_mask, surface_tilt_mm, np.nan), extent=extent_ap, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
            axes[0].set_title(f"Full Data (Tilt Only, Pixel: {dx:.2f}m x {dy:.2f}m)\nRMS: {rms_after_center_tilt:.3f} mm")
            axes[0].set_xlabel("x [m]"), axes[0].set_ylabel("y [m]")
            axes[0].add_patch(plt.Rectangle((-half_size, -half_size), CENTER_BLOCK_SIZE_M, CENTER_BLOCK_SIZE_M, linewidth=1, edgecolor='black', facecolor='none', linestyle='--'))
            
            extent_crop = [x_crop.min() - dx_crop/2, x_crop.max() + dx_crop/2, y_crop.min() - dy_crop/2, y_crop.max() + dy_crop/2]
            im1 = axes[1].imshow(np.where(center_block_mask_crop, surface_crop_mm, np.nan), extent=extent_crop, origin='lower', cmap='coolwarm', vmin=vmin_val, vmax=vmax_val)
            fig.colorbar(im1, ax=axes[1], label="Surface Error [mm]")
            axes[1].set_title(f"Cropped ({ZOOM_SIZE_ARCMIN:g}' x {ZOOM_SIZE_ARCMIN:g}', Tilt Only, Pixel: {dx_crop:.2f}m x {dy_crop:.2f}m)\nRMS: {rms_crop_after_center:.3f} mm")
            axes[1].set_xlabel("x [m]")
            axes[1].add_patch(plt.Rectangle((-half_size, -half_size), CENTER_BLOCK_SIZE_M, CENTER_BLOCK_SIZE_M, linewidth=1, edgecolor='black', facecolor='none', linestyle='--'))
            
            for ax in axes:
                ax.set_xlim(-20, 20)
                ax.set_ylim(-20, 20)
                
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "surface_cropped_comparison_no_circle.png"), dpi=150) # ★点線なし保存

            for ax in axes:
                ax.add_patch(plt.Circle((0, 0), D/2, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.7))
            fig.savefig(os.path.join(OUT_DIR, "surface_cropped_comparison.png"), dpi=150) # ★点線あり保存
            plt.close(fig)

# =========================
# ★ルッツの式 (Ruze Equation) による表面効率 (eta) のグラフ生成
# 2026-08-27 追加
# =========================
print("\n--- ルッツの式 (Ruze Equation) による表面効率評価 ---")
freq_min = 6.5e9
freq_max = 12.5e9
# 周波数配列 (Hz)
freq_array = np.linspace(freq_min, freq_max, 200)
wavelength_array = c / freq_array

# 中心ブロック除外後のRMS (mm -> m)
rms_val_m = rms_after_center_tilt * 1e-3

# ルッツの式: eta = exp( - (4 * pi * epsilon / lambda)^2 )
eta_array = np.exp(- (4 * np.pi * rms_val_m / wavelength_array)**2)

fig, ax = plt.subplots(figsize=(8, 6))
# 求めたRMSに対するグラフ
ax.plot(freq_array / 1e9, eta_array, label=f'Calculated RMS: {rms_after_center_tilt:.3f} mm', color='blue', linewidth=2.5)

# 比較用リファレンス (0.3mm, 0.7mm, 1.0mm)
for ref_rms_mm in [0.3, 0.7, 1.0]:
    if abs(ref_rms_mm - rms_after_center_tilt) > 0.05:
        ref_eta = np.exp(- (4 * np.pi * (ref_rms_mm * 1e-3) / wavelength_array)**2)
        ax.plot(freq_array / 1e9, ref_eta, linestyle='--', label=f'Ref RMS: {ref_rms_mm:.1f} mm')

ax.set_title("Surface Efficiency vs Frequency (Ruze\'s Equation)", fontsize=14)
ax.set_xlabel("Frequency [GHz]", fontsize=12)
ax.set_ylabel(r"Surface Efficiency ($\eta$)", fontsize=12)
ax.set_xlim(6.5, 12.5)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(fontsize=11)
fig.tight_layout()

ruze_out_path = os.path.join(OUT_DIR, "ruze_efficiency.png")
fig.savefig(ruze_out_path, dpi=150)
print(f"ルッツの式に基づく効率グラフを生成しました: {ruze_out_path}")
plt.close(fig)

print("完了しました。")
