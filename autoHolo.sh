#!/usr/bin/env bash
# IP61 の .dat を解析サーバーへ転送し、観測日付の .raw リンクを作って
# YI_Holography.sh を実行する。
#
# 例: ./autoHolo.sh I26184Y --antenna 32 --scan-half --cpu 10
# 既定では .dat のファイル名の日付を観測日より 2 日後とみなす。
# 異なる場合: --dat-day-offset -2 (または 0 など)
set -euo pipefail

usage() {
    cat <<'EOF'
使い方: autoHolo.sh OBS_CODE [--dat-day-offset DAYS] [YI_Holography.sh のオプション...]

  OBS_CODE                  観測コード（例: I26184Y）
  --dat-day-offset DAYS     .dat の日付 - 観測日の日数（既定: 2）
  --help, -h                このヘルプを表示

残りのオプションは YI_Holography.sh にそのまま渡します。
IP61 の /mnt/raid にある OBS_CODE の S1/S2 ペアを転送します。
リンク名の YYYYDOY は OBS_CODE に記録された観測日を使用します。
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
YI_OPTS=()
while [[ $# -gt 0 ]]; do
    case $1 in
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

for program in ssh sshpass python3; do
    command -v "$program" >/dev/null || { echo "[ERROR] $program が必要です。" >&2; exit 1; }
done

# DOY の年越しと閏年を日付として計算する。strftime はリンク名の YYYYDOY と同じ書式。
mapfile -t DAYS < <(python3 - "$OBS_CODE" "$DAT_DAY_OFFSET" <<'PY'
from datetime import datetime, timedelta
import sys

code, offset = sys.argv[1:]
year = 2000 + int(code[1:3])
doy = int(code[3:6])
observed = datetime.strptime(f"{year}{doy:03d}", "%Y%j")
if observed.year != year:  # strptime は平年の 366 日目を翌年へ繰り上げ得る
    raise SystemExit(f"[ERROR] 観測コードに存在しない DOY: {code}")
print(observed.strftime("%Y%j"))
print((observed + timedelta(days=int(offset))).strftime("%Y%j"))
PY
)
if [[ ${#DAYS[@]} -ne 2 ]]; then
    echo "[ERROR] 観測コードから日付を読み取れません: $OBS_CODE" >&2
    exit 1
fi
OBS_DAY=${DAYS[0]}
DAT_DAY=${DAYS[1]}
HOLO_DIR=/mnt/yi_raid5/Holography
TARGET_DIR=$HOLO_DIR/$OBS_CODE

echo "観測コード: $OBS_CODE / raw リンクの日付: $OBS_DAY / dat の日付: $DAT_DAY"
read -rsp 'Enter SSH Password (for IP61 and IP20): ' SSH_PASS
echo
export SSHPASS=$SSH_PASS
unset SSH_PASS
trap 'unset SSHPASS' EXIT

# ssh は渡した引数をリモート側のシェルで解釈するため、引数ごとに引用する。
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
remote_bash ymgusr@192.168.6.61 "$OBS_CODE" "$DAT_DAY" "$TARGET_DIR" <<'REMOTE'
set -euo pipefail
obs_code=$1
dat_day=$2
target_dir=$3
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
        echo "[ERROR] 未対応の S1 ファイル名: $name" >&2; exit 1
    fi
done
for file in "${s2_files[@]}"; do
    name=${file##*/}
    if [[ $name =~ ^${obs_code}_ymg1_S2_tid02-${dat_day}-([0-9]{6})\.dat$ ]]; then
        s2_by_time[${BASH_REMATCH[1]}]=$file
    else
        echo "[ERROR] 未対応の S2 ファイル名: $name" >&2; exit 1
    fi
done
for time in "${!s1_by_time[@]}"; do
    if [[ ! -v s2_by_time[$time] ]]; then
        echo "[ERROR] S1 ${time} に対応する S2 がありません。" >&2; exit 1
    fi
done
for time in "${!s2_by_time[@]}"; do
    if [[ ! -v s1_by_time[$time] ]]; then
        echo "[ERROR] S2 ${time} に対応する S1 がありません。" >&2; exit 1
    fi
done
echo "${#s1_files[@]} 組の .dat を corr5_10G に転送します。"
scp -- "${s1_files[@]}" "${s2_files[@]}" "corr5_10G:${target_dir}/"
REMOTE

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
    echo '[ERROR] 転送先に揃った S1/S2 がありません。' >&2; exit 1
fi
declare -A s2_by_time=()
for file in "${s2_files[@]}"; do
    name=${file##*/}
    if [[ $name =~ ^${obs_code}_ymg1_S2_tid02-${dat_day}-([0-9]{6})\.dat$ ]]; then
        s2_by_time[${BASH_REMATCH[1]}]=$name
    else
        echo "[ERROR] 未対応の S2 ファイル名: $name" >&2; exit 1
    fi
done
for file in "${s1_files[@]}"; do
    name=${file##*/}
    if [[ ! $name =~ ^${obs_code}_ymg1_S1_tid01-${dat_day}-([0-9]{6})\.dat$ ]]; then
        echo "[ERROR] 未対応の S1 ファイル名: $name" >&2; exit 1
    fi
    time=${BASH_REMATCH[1]}
    if [[ ! -v s2_by_time[$time] ]]; then
        echo "[ERROR] S1 ${time} に対応する S2 がありません。" >&2; exit 1
    fi
done
for file in "${s1_files[@]}"; do
    name=${file##*/}
    [[ $name =~ -([0-9]{6})\.dat$ ]]
    time=${BASH_REMATCH[1]}
    for antenna in 34 32; do
        link="$target_dir/raw/YAMAGU${antenna}_${obs_day}${time}.raw"
        if [[ -e $link && ! -L $link ]]; then
            echo "[ERROR] 既存の通常ファイルを上書きしません: $link" >&2; exit 1
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
