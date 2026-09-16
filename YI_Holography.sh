#!/usr/bin/env bash
#
# YI_Holography.sh — ホログラフィ解析の一連の処理を通しで実行する。
#
# 通常モード（既定, 全8ステップ）:
#   1. fringe_search.py                     delay収束
#   2. corr_fringe.py --only-corr        gico3実行
#   3. corr_fringe.py --only-frinZ       frinZ実行（lag未補正）
#   4. group_up_txt_prd.py                  beam_search.txt を作成
#   5. scanning_effect.py                   走査lagを推定
#   6. corr_fringe.py --only-frinZ       frinZ再実行（lag補正あり）
#   7. group_up_txt_prd.py                  beam.txt（最終）を作成
#   8. Holography.py                        ホログラフィ解析
#
# --after-corr モード（1〜4は既に実行済みという前提で、5→6→7→8のみ実行）:
#   既存の beam_search.txt / SKD / (gico3済みの)raw・stepcor を使い、
#   lag推定 → lag補正frinZ再実行 → beam統合 → Holography のみをやり直す。
#   ※ Holographyはgico3/frinZの生出力ではなく統合済みbeam.txtを必要とする
#     ため、群化ステップ(旧7)もこのモードに含めている。
#
# 使い方:
#   ./YI_Holography.sh OBS_CODE [オプション]
#   ./YI_Holography.sh I26184Y
#   ./YI_Holography.sh I26184Y --antenna 34 --cpu 10 --polar
#   ./YI_Holography.sh I26184Y --dry-run
#   ./YI_Holography.sh I26184Y --after-corr --invert-lag-sign
#
# パス規則:
#   SKDファイル: ${OBS_CODE}/${OBS_CODE}${ANTENNA}.skd
#   PRDファイル: ${OBS_CODE}/${OBS_CODE}${ANTENNA}.prd
#   出力先:     ${OBS_CODE}/results[_接尾辞]         (--output-suffix)
#   beamファイル: ${OBS_CODE}/beam[_接尾辞].txt        （最終, --output-suffixと共通）
#               ${OBS_CODE}/beam[_接尾辞]_search.txt （lag推定用の暫定beam）
#
set -euo pipefail

# ── スクリプト自身の場所（同じディレクトリのpythonスクリプトを呼ぶ） ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_FRINGE_SEARCH="${SCRIPT_DIR}/fringe_search.py"
PY_CORR_FRINGE="${SCRIPT_DIR}/corr_fringe.py"
PY_GROUP_UP="${SCRIPT_DIR}/group_up_txt_prd.py"
PY_SCANNING="${SCRIPT_DIR}/scanning_effect.py"
PY_HOLOGRAPHY="${SCRIPT_DIR}/Holography.py"

# ── 既定値 ──
ANTENNA="32"
OUTPUT_SUFFIX=""
CPU=""
BAND_SPLIT=""
SCAN_HALF="false"
FORWARD_DIRECTION="increasing"
MASER="false"
ADD_DELAY="false"

# scanning_effect.py 用。lagサーチは既定1ms刻みで探索し、その値をそのまま
# corr_fringe_v6.py --scan-lag-ms へ渡す（丸めない）。
# ※ corr_fringe_v6.py 側は integration window を 10ms（=1/output, output=100
#   固定）単位でしか動かせないため、1msの倍数かつ10msの倍数でない値を渡すと
#   corr_fringe_v6.py 自身がエラーで停止する。その場合は --lag-step-ms 10 を
#   指定するか、--invert-lag-sign 等で得られた値を確認のうえ調整すること。
MATCH_TOLERANCE=""
ROW_GAP=""
MIN_SNR=""
MAX_LAG_MS=""
LAG_STEP_MS="1"
GRID_SIZE=""
SCAN_BAND=""          # --band-split 使用時に走査lag推定へ使う帯域を明示指定

# Holography.py 用（スライス出力は既定で両方ON。--no-sliceで両方OFF）
POLAR="false"
NO_SLICE="false"
DB_MIN=""
ZOOM_SIZE=""
CENTER_BLOCK_SIZE=""
HOLOGRAPHY_BAND=""    # --band-split 使用時にHolographyへ渡す帯域を明示指定

INVERT_LAG_SIGN="false"
DRY_RUN="false"
AFTER_CORR="false"

usage() {
    cat <<'EOF'
使い方: YI_Holography.sh OBS_CODE [オプション]

位置引数:
  OBS_CODE                    観測コード（ディレクトリ名。例: I26184Y）

パス関連:
  --antenna {32,34}           SKD/PRDファイル名の{ANTENNA}部分（既定: 32）
                               -> OBS_CODE/OBS_CODE{ANTENNA}.skd / .prd
  --output-suffix NAME        図の出力先を OBS_CODE/results_NAME に、
                               beamファイル名を beam_NAME.txt / beam_NAME_search.txt
                               にする（未指定時は results / beam.txt / beam_search.txt）

corr_fringe_v6.py 関連:
  --cpu N                     gico3実行時のCPUコア数
  --band-split N               8192-8704MHzをN分割して処理（512の約数）
  --scan-half                 frinZのoffset走査で最初と最後の積分長を半分にする
  --forward-direction {increasing,decreasing}
                               往路のAz方向（既定: increasing）

group_up_txt_prd.py 関連:
  --maser                     SNRの代わりにFrequencyを取得する
  --add-delay                 出力にRes-Delay列を追加する

scanning_effect.py 関連:
  --match-tolerance SEC       時刻照合の許容誤差[s]
  --row-gap SEC                走査行を区切る時間ギャップ[s]
  --min-snr SNR                lag推定に使う最小SNR
  --max-lag-ms MS              lag探索範囲 ±MS[ms]
  --lag-step-ms MS              lag探索の刻み幅[ms]（既定: 1、丸めずcorr_fringeへ渡す）
  --grid-size N                 比較用グリッドの分割数
  --scan-band BAND              --band-split使用時、lag推定に使う帯域
                               （例: 8192_8256）。band-split時は必須。
  --invert-lag-sign            推定lagの符号を反転してcorr_fringeへ渡す
                               （符号の定義がscanning_effect.pyとcorr_fringe_v6.py
                               で逆だった場合に使用）

Holography.py 関連:
  --polar                      開口面の極座標表示を追加する
  --no-slice                    ビーム/開口面のスライス出力を両方OFFにする
                               （既定は --slice-beam --slice-aperture 相当で両方ON）
  --db-min DB                    dBスケールの下限
  --zoom-size ARCMIN              ビームパターンのズーム幅[arcmin]
  --center-block-size M           副鏡ブロッキングとして除外する正方形の一辺[m]
  --holography-band BAND        --band-split使用時、Holographyに渡す帯域
                               （例: 8192_8256）。band-split時は必須。

実行モード:
  --after-corr                  1〜4（delay収束/gico3/初回frinZ/暫定beam作成）を
                               スキップし、既存の beam_search.txt を使って
                               5.scanning_effect → 6.corr_fringe(lag補正)
                               → 7.beam統合 → 8.Holography だけを実行する。
                               事前に通常モードを一度実行し、raw/stepcorと
                               beam_search.txtが揃っている必要がある。
  --dry-run                    実行はせず、各ステップのコマンドを表示するだけ
  -h, --help                    このヘルプを表示する
EOF
}

# ── 引数パース ──
if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

OBS_CODE="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --antenna) ANTENNA="$2"; shift 2 ;;
        --output-suffix) OUTPUT_SUFFIX="$2"; shift 2 ;;
        --cpu) CPU="$2"; shift 2 ;;
        --band-split) BAND_SPLIT="$2"; shift 2 ;;
        --scan-half) SCAN_HALF="true"; shift ;;
        --forward-direction) FORWARD_DIRECTION="$2"; shift 2 ;;
        --maser) MASER="true"; shift ;;
        --add-delay) ADD_DELAY="true"; shift ;;
        --match-tolerance) MATCH_TOLERANCE="$2"; shift 2 ;;
        --row-gap) ROW_GAP="$2"; shift 2 ;;
        --min-snr) MIN_SNR="$2"; shift 2 ;;
        --max-lag-ms) MAX_LAG_MS="$2"; shift 2 ;;
        --lag-step-ms) LAG_STEP_MS="$2"; shift 2 ;;
        --grid-size) GRID_SIZE="$2"; shift 2 ;;
        --scan-band) SCAN_BAND="$2"; shift 2 ;;
        --invert-lag-sign) INVERT_LAG_SIGN="true"; shift ;;
        --polar) POLAR="true"; shift ;;
        --no-slice) NO_SLICE="true"; shift ;;
        --db-min) DB_MIN="$2"; shift 2 ;;
        --zoom-size) ZOOM_SIZE="$2"; shift 2 ;;
        --center-block-size) CENTER_BLOCK_SIZE="$2"; shift 2 ;;
        --holography-band) HOLOGRAPHY_BAND="$2"; shift 2 ;;
        --after-corr) AFTER_CORR="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] 不明なオプション: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "$ANTENNA" != "32" && "$ANTENNA" != "34" ]]; then
    echo "[ERROR] --antenna は 32 か 34 を指定してください: $ANTENNA" >&2
    exit 1
fi

if [[ -n "$BAND_SPLIT" ]]; then
    if [[ -z "$SCAN_BAND" ]]; then
        echo "[ERROR] --band-split 使用時は --scan-band で走査lag推定に使う帯域を指定してください（例: 8192_8256）。" >&2
        exit 1
    fi
    if [[ -z "$HOLOGRAPHY_BAND" ]]; then
        echo "[ERROR] --band-split 使用時は --holography-band でHolographyに渡す帯域を指定してください（例: 8192_8256）。" >&2
        exit 1
    fi
fi

# ── パス構築 ──
SKD_FILE="${OBS_CODE}/${OBS_CODE}${ANTENNA}.skd"
PRD_FILE="${OBS_CODE}/${OBS_CODE}${ANTENNA}.prd"
OUTPUT_DIR="${OBS_CODE}/results${OUTPUT_SUFFIX:+_${OUTPUT_SUFFIX}}"

# beamファイル名: 既定 beam.txt / beam_search.txt。--output-suffixを付けると
# beam_<suffix>.txt / beam_<suffix>_search.txt になる（複数バージョンを
# 上書きせず残しておくためのマージン）。
# --band-split使用時はgroup_up_txt_prd.py側の制約で --output を指定できず
# beam_<band>.txt 固定になるため、suffixはband-split時には適用されない。
BEAM_BASENAME="beam${OUTPUT_SUFFIX:+_${OUTPUT_SUFFIX}}"
if [[ -n "$BAND_SPLIT" ]]; then
    BEAM_SEARCH_FILE="${OBS_CODE}/beam_${SCAN_BAND}_search.txt"
    BEAM_FINAL_FILE="${OBS_CODE}/beam_${HOLOGRAPHY_BAND}.txt"
else
    BEAM_SEARCH_FILE="${OBS_CODE}/${BEAM_BASENAME}_search.txt"
    BEAM_FINAL_FILE="${OBS_CODE}/${BEAM_BASENAME}.txt"
fi

for f in "$SKD_FILE" "$PRD_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] 見つかりません: $f" >&2
        exit 1
    fi
done

if [[ "$AFTER_CORR" == "true" && ! -f "$BEAM_SEARCH_FILE" && "$DRY_RUN" != "true" ]]; then
    echo "[ERROR] --after-corr には $BEAM_SEARCH_FILE が必要です。先に通常モードで一度実行してください。" >&2
    exit 1
fi

# ── 実行ヘルパー ──
# dry-run時は表示のみ。それ以外は実行し、失敗したら即座に停止する（set -e）。
run() {
    echo "+ $*"
    if [[ "$DRY_RUN" != "true" ]]; then
        "$@"
    fi
}

step() {
    echo
    echo "=== [$1] $2 ==="
}

# corr_fringe_v6.py --only-corr は gico3実行のみを行うため、band-split /
# scan-half / forward-direction は意味を持たない（--cpu のみ有効）。
corr_fringe_corr_opts=()
[[ -n "$CPU" ]] && corr_fringe_corr_opts+=(--cpu "$CPU")

# corr_fringe_v6.py --only-frinZ（frinZ実行）に共通で渡すオプション
corr_fringe_common_opts=()
[[ -n "$CPU" ]] && corr_fringe_common_opts+=(--cpu "$CPU")
[[ -n "$BAND_SPLIT" ]] && corr_fringe_common_opts+=(--band-split "$BAND_SPLIT")
[[ "$SCAN_HALF" == "true" ]] && corr_fringe_common_opts+=(--scan-half)
corr_fringe_common_opts+=(--forward-direction "$FORWARD_DIRECTION")

group_up_common_opts=(--antenna "$ANTENNA")
[[ "$MASER" == "true" ]] && group_up_common_opts+=(--maser)
[[ "$ADD_DELAY" == "true" ]] && group_up_common_opts+=(--add-delay)
[[ -n "$BAND_SPLIT" ]] && group_up_common_opts+=(--band-split "$BAND_SPLIT")

scanning_opts=()
[[ -n "$MATCH_TOLERANCE" ]] && scanning_opts+=(--match-tolerance "$MATCH_TOLERANCE")
[[ -n "$ROW_GAP" ]] && scanning_opts+=(--row-gap "$ROW_GAP")
[[ -n "$MIN_SNR" ]] && scanning_opts+=(--min-snr "$MIN_SNR")
[[ -n "$MAX_LAG_MS" ]] && scanning_opts+=(--max-lag-ms "$MAX_LAG_MS")
scanning_opts+=(--lag-step-ms "$LAG_STEP_MS")
[[ -n "$GRID_SIZE" ]] && scanning_opts+=(--grid-size "$GRID_SIZE")

# スライス出力は既定でON。--no-sliceが指定されたときだけ両方OFFにする。
holography_opts=()
[[ "$POLAR" == "true" ]] && holography_opts+=(--polar)
if [[ "$NO_SLICE" != "true" ]]; then
    holography_opts+=(--slice-beam --slice-aperture)
fi
[[ -n "$DB_MIN" ]] && holography_opts+=(--db-min "$DB_MIN")
[[ -n "$ZOOM_SIZE" ]] && holography_opts+=(--zoom-size "$ZOOM_SIZE")
[[ -n "$CENTER_BLOCK_SIZE" ]] && holography_opts+=(--center-block-size "$CENTER_BLOCK_SIZE")

echo "OBS_CODE     : $OBS_CODE"
echo "ANTENNA      : $ANTENNA"
echo "SKD_FILE     : $SKD_FILE"
echo "PRD_FILE     : $PRD_FILE"
echo "OUTPUT_DIR   : $OUTPUT_DIR"
echo "BEAM_SEARCH  : $BEAM_SEARCH_FILE"
echo "BEAM_FINAL   : $BEAM_FINAL_FILE"
[[ -n "$BAND_SPLIT" ]] && echo "BAND_SPLIT   : $BAND_SPLIT (scan-band=$SCAN_BAND, holography-band=$HOLOGRAPHY_BAND)"
[[ "$AFTER_CORR" == "true" ]] && echo "MODE         : --after-corr（5→6→7→8のみ実行）"

# lag_search.csv から mismatch 最小の lag_ms を取得する共通処理。
# 丸めは行わず、scanning_effect.py の --lag-step-ms 刻みの値をそのまま返す。
estimate_lag_ms() {
    local csv_path="$1"
    python3 - "$csv_path" "$INVERT_LAG_SIGN" <<'PYEOF'
import sys
import pandas as pd

csv_path, invert = sys.argv[1], sys.argv[2] == "true"
df = pd.read_csv(csv_path).dropna(subset=["mismatch"])
if df.empty:
    print("[ERROR] lag_search.csv に有効なmismatch値がありません。", file=sys.stderr)
    sys.exit(1)

best_lag_ms = float(df.loc[df["mismatch"].idxmin(), "lag_ms"])
if invert:
    best_lag_ms = -best_lag_ms
print(f"{best_lag_ms:.3f}")
PYEOF
}

if [[ "$AFTER_CORR" != "true" ]]; then
    # ── 1. fringe_search.py : delay収束 ──
    step 1/8 "fringe_search.py（delay収束）"
    run python3 "$PY_FRINGE_SEARCH" --workdir "$OBS_CODE"

    # ── 2. corr_fringe_v6.py --only-corr : gico3実行 ──
    step 2/8 "corr_fringe.py --only-corr（gico3実行）"
    run python3 "$PY_CORR_FRINGE" "$OBS_CODE" --only-corr "${corr_fringe_corr_opts[@]}"

    # ── 3. corr_fringe_v6.py --only-frinZ : frinZ実行（lag未補正） ──
    step 3/8 "corr_fringe.py --only-frinZ（lag未補正）"
    run python3 "$PY_CORR_FRINGE" "$OBS_CODE" --only-frinZ "${corr_fringe_common_opts[@]}"

    # ── 4. group_up_txt_prd.py : beam_search.txt 作成 ──
    step 4/8 "group_up_txt_prd.py（${BEAM_SEARCH_FILE} 作成）"
    if [[ -n "$BAND_SPLIT" ]]; then
        run python3 "$PY_GROUP_UP" --input "$OBS_CODE" --prd "$PRD_FILE" "${group_up_common_opts[@]}"
        # band-split時は --output を指定できないため beam_<band>.txt が既定名で
        # 作られる。lag補正後の同名ファイルに上書きされる前に "_search" 付きで
        # 退避しておく。
        if [[ "$DRY_RUN" != "true" ]]; then
            for f in "${OBS_CODE}"/beam_*.txt; do
                [[ -e "$f" ]] || continue
                base="$(basename "$f" .txt)"
                cp "$f" "${OBS_CODE}/${base}_search.txt"
            done
        fi
    else
        run python3 "$PY_GROUP_UP" --input "$OBS_CODE" --prd "$PRD_FILE" \
            --output "$BEAM_SEARCH_FILE" "${group_up_common_opts[@]}"
    fi

    if [[ ! -f "$BEAM_SEARCH_FILE" && "$DRY_RUN" != "true" ]]; then
        echo "[ERROR] $BEAM_SEARCH_FILE が作成されませんでした。--scan-band の指定を確認してください。" >&2
        exit 1
    fi

    STEP5_LABEL="5/8"; STEP6_LABEL="6/8"; STEP7_LABEL="7/8"; STEP8_LABEL="8/8"
else
    STEP5_LABEL="1/4"; STEP6_LABEL="2/4"; STEP7_LABEL="3/4"; STEP8_LABEL="4/4"
fi

# ── scanning_effect.py : 走査lag推定 ──
step "$STEP5_LABEL" "scanning_effect.py（走査lag推定, 探索刻み ${LAG_STEP_MS} ms）"
run python3 "$PY_SCANNING" "$BEAM_SEARCH_FILE" "$SKD_FILE" \
    --outdir "${OBS_CODE}/scanning_result" "${scanning_opts[@]}"

LAG_SEARCH_CSV="${OBS_CODE}/scanning_result/lag_search.csv"

if [[ "$DRY_RUN" == "true" ]]; then
    BEST_LAG_MS="0"
else
    # scanning_effect.pyの探索刻み(既定1ms)そのままの値を丸めずに使う。
    # corr_fringe.pyは10ms刻みしか受け付けないため、10の倍数でない値を
    # 渡すとcorr_fringe.py自身がエラーで停止する。
    BEST_LAG_MS=$(estimate_lag_ms "$LAG_SEARCH_CSV")
fi

echo "[INFO] 推定lag（探索刻み ${LAG_STEP_MS} ms, 丸めなし, 符号反転=${INVERT_LAG_SIGN}）: ${BEST_LAG_MS} ms"

# ── corr_fringe.py --only-frinZ : frinZ再実行（lag補正あり） ──
step "$STEP6_LABEL" "corr_fringe.py --only-frinZ（lag補正 ${BEST_LAG_MS} ms）"
run python3 "$PY_CORR_FRINGE" "$OBS_CODE" --only-frinZ \
    --scan-lag-ms "$BEST_LAG_MS" "${corr_fringe_common_opts[@]}"

# ── group_up_txt_prd.py : beam.txt（最終）作成 ──
step "$STEP7_LABEL" "group_up_txt_prd.py（${BEAM_FINAL_FILE} 作成）"
if [[ -n "$BAND_SPLIT" ]]; then
    run python3 "$PY_GROUP_UP" --input "$OBS_CODE" --prd "$PRD_FILE" "${group_up_common_opts[@]}"
else
    run python3 "$PY_GROUP_UP" --input "$OBS_CODE" --prd "$PRD_FILE" \
        --output "$BEAM_FINAL_FILE" "${group_up_common_opts[@]}"
fi

if [[ ! -f "$BEAM_FINAL_FILE" && "$DRY_RUN" != "true" ]]; then
    echo "[ERROR] $BEAM_FINAL_FILE が作成されませんでした。--holography-band の指定を確認してください。" >&2
    exit 1
fi

# ── Holography.py : ホログラフィ解析 ──
step "$STEP8_LABEL" "Holography.py（ホログラフィ解析）"
run python3 "$PY_HOLOGRAPHY" --input "$BEAM_FINAL_FILE" --output "$OUTPUT_DIR" "${holography_opts[@]}"

echo
echo "--- 全ステップが完了しました ---"
echo "最終beam    : $BEAM_FINAL_FILE"
echo "出力ディレクトリ: $OUTPUT_DIR"
