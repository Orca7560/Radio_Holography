#!/usr/bin/env bash
# Reprocess an existing beam file using scan-lag correction.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  Holography_step2.sh --in BEAM.txt --antenna NUM --suffix NAME [OPTIONS]

Run:
  scanning_effect -> corr_fringe --only-frinZ (lag corrected)
  -> group_up_txt -> Holography

Required:
  -i, --in, --input FILE    Existing beam file used for lag estimation.
  -a, --antenna NUM         Antenna suffix used for CODE<NUM>.prd/.skd.
      --suffix NAME         Output tag; the input beam is never overwritten.

OBS_CODE defaults to the parent directory name of --in:
  I26184Y/beam.txt with --antenna 32
  -> I26184Y/I26184Y32.prd, I26184Y/I26184Y32.skd
  -> I26184Y/beam_NAME.txt, I26184Y/results_NAME/

Options:
  -O, --obs-code CODE       Override the OBS_CODE inferred from --in.
      --cpu N               CPU count for corr_fringe.
      --max-lag-ms MS       Lag-search range (default: 1000).
      --lag-step-ms MS      Lag-search step (default: 5).
      --min-snr N           Minimum SNR for lag fitting (default: 3).
      --lag-ms MS           Do not search; use this lag value.
      --polar               Add polar aperture plots.
      --no-slice            Do not generate beam or aperture slices.
      --db-min DB           Holography dB lower limit.
      --zoom-size ARCMIN    Holography beam zoom width.
      --center-block-size M Holography center blocking size.
      --dry-run             Print commands without running them.
  -h, --help                Show this help.
EOF
}

input="" antenna="" obs_code="" suffix="" cpu=""
max_lag_ms="1000" lag_step_ms="5" min_snr="3" manual_lag=""
polar=0 slices=1 dry_run=0 db_min="" zoom_size="" center_block_size=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--in|--input) input="$2"; shift 2 ;;
    -a|--antenna) antenna="$2"; shift 2 ;;
    -O|--obs-code) obs_code="$2"; shift 2 ;;
    --suffix) suffix="$2"; shift 2 ;;
    --cpu) cpu="$2"; shift 2 ;;
    --max-lag-ms) max_lag_ms="$2"; shift 2 ;;
    --lag-step-ms) lag_step_ms="$2"; shift 2 ;;
    --min-snr) min_snr="$2"; shift 2 ;;
    --lag-ms) manual_lag="$2"; shift 2 ;;
    --polar) polar=1; shift ;;
    --no-slice) slices=0; shift ;;
    --db-min) db_min="$2"; shift 2 ;;
    --zoom-size) zoom_size="$2"; shift 2 ;;
    --center-block-size) center_block_size="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$input" && -n "$antenna" && -n "$suffix" ]] || { usage >&2; exit 2; }
[[ "$suffix" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid --suffix" >&2; exit 2; }
[[ -f "$input" ]] || { echo "Input beam not found: $input" >&2; exit 2; }

obs_dir="$(cd "$(dirname "$input")" && pwd)"
input="$obs_dir/$(basename "$input")"
[[ -n "$obs_code" ]] || obs_code="$(basename "$obs_dir")"
prd="$obs_dir/${obs_code}${antenna}.prd"
skd="$obs_dir/${obs_code}${antenna}.skd"
beam="$obs_dir/beam_${suffix}.txt"
scan_dir="$obs_dir/scanning_result_${suffix}"
results_dir="$obs_dir/results_${suffix}"

[[ "$input" != "$beam" ]] || { echo "Refusing to overwrite the input beam." >&2; exit 2; }
for command_name in corr_fringe group_up_txt scanning_effect Holography; do
  command -v "$command_name" >/dev/null 2>&1 || { echo "Command not found in PATH: $command_name" >&2; exit 127; }
done
[[ -f "$prd" && -f "$skd" ]] || { echo "PRD or SKD is missing." >&2; exit 2; }

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  [[ "$dry_run" -eq 1 ]] || "$@"
}

corr=(corr_fringe --workdir "$obs_dir")
[[ -n "$cpu" ]] && corr+=(--cpu "$cpu")
scan=(scanning_effect --in "$input" --skd "$skd" --out "$scan_dir" --max-lag-ms "$max_lag_ms" --lag-step-ms "$lag_step_ms" --min-snr "$min_snr")
[[ -n "$manual_lag" ]] && scan+=(--lag-ms "$manual_lag")
holo=(Holography --in "$beam" --out "$results_dir")
[[ "$slices" -eq 1 ]] && holo+=(--slice-beam --slice-aperture)
[[ "$polar" -eq 1 ]] && holo+=(--polar)
[[ -n "$db_min" ]] && holo+=(--db-min "$db_min")
[[ -n "$zoom_size" ]] && holo+=(--zoom-size "$zoom_size")
[[ -n "$center_block_size" ]] && holo+=(--center-block-size "$center_block_size")

run "${scan[@]}"
# Do not silently run an uncorrected second frinZ pass.
if ! corr_fringe --help 2>&1 | grep -q -- '--scan-result'; then
  echo "corr_fringe does not support --scan-result yet; refusing uncorrected reprocessing." >&2
  exit 2
fi
run "${corr[@]}" --only-frinZ --scan-result "$scan_dir"
run group_up_txt --in "$obs_dir/frinz_results" --prd "$prd" --out "$beam"
run "${holo[@]}"

echo "Completed: $results_dir"
