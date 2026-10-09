#!/usr/bin/env bash
# IP61 の .dat を解析サーバーへ転送し、観測日付の .raw リンクを作って
# YI_Holography.sh を実行する。
set -euo pipefail

usage() {
    cat <<'EOF'
使い方: autoHolo.sh OBS_CODE [オプション]

位置引数:
  OBS_CODE                    観測コード（例: I26184Y）。YYDDD を観測日とする

autoHolo.sh 固有:
  --dat-day-offset DAYS       .datの日付 - 観測日の日数（既定: 2）
                               2=観測日より2日後、0=同日、-2=2日前

出力・入力（以下は YI_Holography.sh に渡す）:
  --antenna {32,34}           測定アンテナ径 [m]（既定: 32）。SKD/PRD選択と解析に使用
                               34ではHolography.pyが入力Phaseを反転して解析
  --output-suffix NAME        results_NAME、beam_NAME.txt 等に保存

相関処理・主走査:
  --cpu N                     gico3 のCPUコア数
  --band-split N              8192–8704 MHz をN分割（512の約数）
  --include-center-crossings   位相補正に中心通過点も含める（中心座標かつAmp > 1）
  --on-length SEC             ON判定・ON積分・Holography較正用ONのLength [s]（既定: 10）
  --offset-scan-time SEC      OFF点の積分時間と時間間隔 [s]（既定: 0.5、ON積分時間は--on-length）
  --scan-half                 offset走査の最初と最後の積分長を半分にする
  --scan-axis {az,el}         主走査軸（既定: az、縦走査は el）
  --forward-direction {increasing,decreasing}
                               片方向ラグ指定時の往路方向。正逆別推定には影響しない

beam.txt の作成:
  --el-drive-speed ARCMIN_S   El走査の駆動速度[arcmin/s]。省略時はPRDから算出
  --maser                     SNRの代わりにFrequencyを保存
  --add-delay                 Res-Delay列を追加

走査ラグの推定:
  --match-tolerance SEC       測定値とSKDの時刻照合の許容差[s]
  --row-gap SEC               走査列を分ける時間間隔[s]
  --min-snr SNR               ラグ推定に使うSNRの下限
  --max-lag-ms MS             ラグの探索範囲 ±MS[ms]
  --lag-step-ms MS            正逆別ラグの最終探索刻み[ms]（既定: 1）
  --grid-size N               振幅マップ比較用グリッドの分割数
  --model-el-deg DEG          Airy主ビーム比較時のAz射影の仰角（既定: 57.3）
  --airy-radius-arcmin ARCMIN Airy主ビームの比較半径（既定: 12）
  --bidirectional-lags        正逆両方向のラグを組として同時探索（既定: 一方固定）
  --scan-band BAND            分割時のラグ推定用帯域（例: 8192_8256）
  --invert-lag-sign           推定ラグの符号を反転してfrinZへ渡す

ホログラフィ解析:
  --polar                     開口面の極座標表示を追加
  --no-slice                  ビーム・開口面のスライス出力を抑制
  --db-min DB                 dB表示の下限
  --zoom-size ARCMIN          ビーム図のズーム幅[arcmin]
  --center-block-size M       中心の副鏡遮蔽領域の一辺[m]
  --holography-band BAND      分割時の解析用帯域（例: 8192_8256）

実行モード:
  --after-corr                delay探索・初回相関処理を省き既存結果から解析を再開
  --dry-run                   YI_Holography.sh の解析コマンドだけ表示
                               注意: autoHolo.sh の転送・リンク作成は実行される
  -h, --help                  このヘルプを表示して終了

--band-split 使用時は --scan-band と --holography-band の両方が必要。
上記の解析オプションは解析サーバーの YI_Holography.sh にそのまま渡す。

処理の流れ:
  1. 解析サーバー（IP20）に観測コードのディレクトリを作成
  2. IP61 の /mnt/raid から同じ時刻の S1/S2 .dat を照合して転送
     ゲートに pv があれば転送速度と残り時間も表示
  3. 観測日 YYYYDOY の .raw リンクを作成し、解析サーバーの
     /mnt/yi_raid5/Holography/YI_Holography.sh を実行

実行例:
  ./autoHolo.sh I26184Y --scan-axis az
  ./autoHolo.sh I26184Y --scan-axis el --antenna 32 --cpu 10
  ./autoHolo.sh I26184Y --scan-axis el --el-drive-speed 3 --scan-half --polar
  ./autoHolo.sh I26184Y --scan-axis el --band-split 8 \
    --scan-band 8192_8256 --holography-band 8192_8256
  ./autoHolo.sh I26184Y --dat-day-offset 0 --output-suffix test
EOF
}

if [[ $# -eq 0 ]]; then usage; exit 1; fi
if [[ $1 == -h || $1 == --help ]]; then usage; exit 0; fi

OBS_CODE=$1
shift
if [[ ! $OBS_CODE =~ ^[A-Za-z][0-9]{5}[A-Za-z]$ ]]; then
    echo "[ERROR] 観測コードは I26184Y のような形式で指定してください。" >&2
    exit 1
fi

DAT_DAY_OFFSET=2
ANTENNA=32
YI_OPTS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --antenna)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --antenna には 32 か 34 を指定してください。" >&2
                exit 1
            fi
            if [[ $2 != 32 && $2 != 34 ]]; then
                echo "[ERROR] --antenna は 32 か 34 を指定してください: $2" >&2
                exit 1
            fi
            ANTENNA=$2
            shift 2 ;;
        --dat-day-offset)
            if [[ $# -lt 2 || ! $2 =~ ^[+-]?[0-9]+$ ]]; then
                echo "[ERROR] --dat-day-offset には整数の日数が必要です。" >&2
                exit 1
            fi
            DAT_DAY_OFFSET=$2
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) YI_OPTS+=("$1"); shift ;;
    esac
done
YI_OPTS+=(--antenna "$ANTENNA")

for program in ssh sshpass date; do
    command -v "$program" >/dev/null || { echo "[ERROR] $program が必要です。" >&2; exit 1; }
done

# 観測日のDOYを検証し、UTCで日数を足す（年越し・閏年に対応）。
OBS_YEAR=$((2000 + 10#${OBS_CODE:1:2}))
OBS_DOY=$((10#${OBS_CODE:3:3}))
DAYS_IN_YEAR=365
if (( OBS_YEAR % 4 == 0 && (OBS_YEAR % 100 != 0 || OBS_YEAR % 400 == 0) )); then
    DAYS_IN_YEAR=366
fi
if (( OBS_DOY < 1 || OBS_DOY > DAYS_IN_YEAR )); then
    echo "[ERROR] 観測コードに存在しない DOY: $OBS_CODE" >&2
    exit 1
fi
if [[ $DAT_DAY_OFFSET == -* ]]; then
    DAT_OFFSET_DAYS=$((-10#${DAT_DAY_OFFSET#-}))
else
    DAT_OFFSET_DAYS=$((10#${DAT_DAY_OFFSET#+}))
fi
if ! YEAR_START_EPOCH=$(date -u -d "${OBS_YEAR}-01-01" +%s); then
    echo "[ERROR] 実行元に GNU date が必要です。観測年: $OBS_YEAR" >&2
    exit 1
fi
OBS_DAY=$(date -u -d "@$((YEAR_START_EPOCH + (OBS_DOY - 1) * 86400))" +%Y%j)
DAT_DAY=$(date -u -d "@$((YEAR_START_EPOCH + (OBS_DOY - 1 + DAT_OFFSET_DAYS) * 86400))" +%Y%j)
HOLO_DIR=/mnt/yi_raid5/Holography
TARGET_DIR=$HOLO_DIR/$OBS_CODE

echo "観測コード: $OBS_CODE / raw リンクの日付: $OBS_DAY / dat の日付: $DAT_DAY"
read -rsp 'Enter SSH Password (for IP61 and IP20): ' SSH_PASS
echo
export SSHPASS=$SSH_PASS
unset SSH_PASS
trap 'unset SSHPASS' EXIT

# 引数ごとに引用してリモートの bash に渡す。
remote_bash() {
    local host=$1 arg quoted command='bash -s --'
    shift
    for arg in "$@"; do
        printf -v quoted '%q' "$arg"
        command+=" $quoted"
    done
    sshpass -e ssh -T "$host" "$command"
}

echo '=== 1. 解析サーバーに転送先を作成 ==='
remote_bash oper@192.168.0.20 "$TARGET_DIR" <<'REMOTE'
set -euo pipefail
mkdir -p -- "$1/raw"
REMOTE

echo '=== 2. IP61 の S1/S2 ペアを検証して転送 ==='
# ゲートから両サーバーに認証し、IP61 → ゲート → 解析サーバーとストリーム転送する。
# IP61 上で解析サーバーへ SSH する必要はない。
# IP61 が先頭に送る合計ファイルサイズを読み、pv があれば転送速度と残り時間を表示する。
receive_with_progress() {
    local total_size
    if ! IFS= read -r total_size || [[ ! $total_size =~ ^[0-9]+$ ]]; then
        echo '[ERROR] IP61 から転送サイズを取得できませんでした。' >&2
        return 1
    fi
    if command -v pv >/dev/null 2>&1; then
        pv -f -pterb -s "$total_size" | \
            sshpass -e ssh -T oper@192.168.0.20 "tar -C '$TARGET_DIR' -xf -"
    else
        echo '[INFO] ゲートに pv がないため残り時間は表示できません（転送は続行します）。' >&2
        sshpass -e ssh -T oper@192.168.0.20 "tar -C '$TARGET_DIR' -xf -"
    fi
}
remote_bash ymgusr@192.168.6.61 "$OBS_CODE" "$DAT_DAY" <<'REMOTE' | receive_with_progress
set -euo pipefail
obs_code=$1
dat_day=$2
shopt -s nullglob
s1_files=(/mnt/raid/"${obs_code}"_ymg1_S1_tid01-"${dat_day}"-*.dat)
s2_files=(/mnt/raid/"${obs_code}"_ymg1_S2_tid02-"${dat_day}"-*.dat)
if (( ${#s1_files[@]} == 0 || ${#s2_files[@]} == 0 )); then
    echo "[ERROR] $obs_code / $dat_day の S1/S2 の両方が必要です。" >&2
    exit 1
fi

declare -A s1_by_time=() s2_by_time=()
for file in "${s1_files[@]}"; do
    name=${file##*/}
    if [[ $name =~ ^${obs_code}_ymg1_S1_tid01-${dat_day}-([0-9]{6})\.dat$ ]]; then
        s1_by_time[${BASH_REMATCH[1]}]=$file
    else
        echo "[ERROR] 未対応の S1 ファイル名: $name" >&2
        exit 1
    fi
done
for file in "${s2_files[@]}"; do
    name=${file##*/}
    if [[ $name =~ ^${obs_code}_ymg1_S2_tid02-${dat_day}-([0-9]{6})\.dat$ ]]; then
        s2_by_time[${BASH_REMATCH[1]}]=$file
    else
        echo "[ERROR] 未対応の S2 ファイル名: $name" >&2
        exit 1
    fi
done
for time in "${!s1_by_time[@]}"; do
    if [[ -z ${s2_by_time[$time]+x} ]]; then
        echo "[ERROR] S1 ${time} に対応する S2 がありません。" >&2
        exit 1
    fi
done
for time in "${!s2_by_time[@]}"; do
    if [[ -z ${s1_by_time[$time]+x} ]]; then
        echo "[ERROR] S2 ${time} に対応する S1 がありません。" >&2
        exit 1
    fi
done

echo "${#s1_files[@]} 組の .dat をゲート経由で転送します。" >&2
names=()
total_size=0
for file in "${s1_files[@]}" "${s2_files[@]}"; do
    names+=("${file##*/}")
    size=$(stat -c %s -- "$file")
    total_size=$((total_size + size))
done
printf '%s\n' "$total_size"
tar -C /mnt/raid -cf - -- "${names[@]}"
REMOTE
echo 'S1/S2 の転送が完了しました。'

echo '=== 3. 観測日の .raw リンクを作り YI_Holography.sh を実行 ==='
remote_bash oper@192.168.0.20 "$OBS_CODE" "$OBS_DAY" "$DAT_DAY" "$HOLO_DIR" "${YI_OPTS[@]}" <<'REMOTE'
set -euo pipefail
obs_code=$1
obs_day=$2
dat_day=$3
holo_dir=$4
shift 4
target_dir=$holo_dir/$obs_code
shopt -s nullglob

s1_files=("$target_dir"/"${obs_code}"_ymg1_S1_tid01-"${dat_day}"-*.dat)
s2_files=("$target_dir"/"${obs_code}"_ymg1_S2_tid02-"${dat_day}"-*.dat)
if (( ${#s1_files[@]} == 0 || ${#s1_files[@]} != ${#s2_files[@]} )); then
    echo '[ERROR] 転送先に揃った S1/S2 がありません。' >&2
    exit 1
fi

declare -A s2_by_time=()
for file in "${s2_files[@]}"; do
    name=${file##*/}
    if [[ $name =~ ^${obs_code}_ymg1_S2_tid02-${dat_day}-([0-9]{6})\.dat$ ]]; then
        s2_by_time[${BASH_REMATCH[1]}]=$name
    else
        echo "[ERROR] 未対応の S2 ファイル名: $name" >&2
        exit 1
    fi
done
for file in "${s1_files[@]}"; do
    name=${file##*/}
    if [[ ! $name =~ ^${obs_code}_ymg1_S1_tid01-${dat_day}-([0-9]{6})\.dat$ ]]; then
        echo "[ERROR] 未対応の S1 ファイル名: $name" >&2
        exit 1
    fi
    time=${BASH_REMATCH[1]}
    if [[ -z ${s2_by_time[$time]+x} ]]; then
        echo "[ERROR] S1 ${time} に対応する S2 がありません。" >&2
        exit 1
    fi
done

for file in "${s1_files[@]}"; do
    name=${file##*/}
    [[ $name =~ -([0-9]{6})\.dat$ ]]
    time=${BASH_REMATCH[1]}
    for antenna in 34 32; do
        link="$target_dir/raw/YAMAGU${antenna}_${obs_day}${time}.raw"
        if [[ -e $link && ! -L $link ]]; then
            echo "[ERROR] 既存の通常ファイルを上書きしません: $link" >&2
            exit 1
        fi
    done
    ln -sfnT -- "../$name" "$target_dir/raw/YAMAGU34_${obs_day}${time}.raw"
    ln -sfnT -- "../${s2_by_time[$time]}" "$target_dir/raw/YAMAGU32_${obs_day}${time}.raw"
    echo "リンク作成: YAMAGU34/YAMAGU32_${obs_day}${time}.raw"
done

cd -- "$holo_dir"
if [[ ! -f YI_Holography.sh ]]; then
    echo "[ERROR] 解析サーバーに $holo_dir/YI_Holography.sh がありません。" >&2
    exit 1
fi
bash YI_Holography.sh "$obs_code" "$@"
REMOTE

echo '転送と解析が完了しました。'
