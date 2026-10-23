# Central beam re-integration

Examples:

    python3 beam_analysis.py --in I26287YA --antenna 34 --scan-axis az --scan-lag-ms 779
    python3 beam_analysis.py --input I26287YA --antenna 32 --scan-axis el --dry-run

No arguments displays help. --in/--input selects the observation directory;
--workdir is retained as an alias.

Required files: WORKDIR/step_cor/*.cor, observation XML files with
stream/label, stream/output and process/epoch,length,skip, plus
WORKDIR/OBS_CODE32.prd or OBS_CODE34.prd. The data must already be correlated.
Use --prd FILE and repeatable --xml FILE for explicit metadata files.

The script selects raster legs intersecting Az/El ±12 arcmin and uses PRD
positions at each nominal integration midpoint to keep only moving samples
inside the square. This includes the central leg and neighboring fixed-axis
positions within ±12 arcmin. Calibration ON files are excluded using XML
length-skip=10 s (--on-length overrides this value). Stationary and turnaround
motion are excluded.

## Integration commands and units

frinZ is called ONCE PER INTEGRATION, with --length and --skip:

    frinZ --in FILE.cor --length 10 --skip 100
    frinZ --in FILE.cor --length 10 --skip 110
    frinZ --in FILE.cor --length 10 --skip 120

There is no frinZ --step option. --step-samples belongs to beam_analysis.py
and determines the increment in --skip. Default length/step counts are 10/10.
Only intervals whose nominal midpoint lies inside the selected square are
executed, rather than processing every window in each full file.

The sample rate is read separately from each XML stream/output. Override it
with --samples-per-second HZ only when needed. Missing rates cause an error
instead of assuming 1000 Hz.

- output=100 Hz: length 10 means 100 ms; spacing 10 means 100 ms.
- output=1000 Hz: length 10 means 10 ms; spacing 10 means 10 ms.
- For 10 ms integrations at 100 Hz, use --length-samples 1 --step-samples 1.

The returned integration duration is checked against this conversion.

## Lag

--scan-lag-ms follows corr_fringe's one-direction convention: positive shifts
the physical integration later, negative earlier. Increasing scans are selected
by default; --forward-direction decreasing selects the other direction.
The unselected direction has zero lag. Keep nominal Epoch/pointing labels
while shifting --skip. Out-of-file windows are omitted.

Lag is rounded to the nearest actual correlation sample (ties away from zero).
A warning shows requested/applied lag if rounding is necessary. For example,
779 ms at 100 Hz becomes 780 ms. Applied_Lag_ms records the effective value.
The output's time resolution cannot be improved by fractional sample skips.

## Outputs

- WORKDIR/ANA_beam.txt: CSV with Epoch, Length, Amp, Phase, SNR, PRD offsets,
  rates and applied lag. --output FILE overrides this path.
- WORKDIR/beam_analysis/*.stdout.txt and *.stderr.txt: raw results and commands.
- WORKDIR/beam_analysis/ANA_beam_map.png
- WORKDIR/beam_analysis/pdf/ANA_beam_map.pdf

--outdir overrides the figure/raw-log directory. The map shows amplitude dB
and raw phase on square ±12 arcmin axes. Complex linear interpolation is for
the figure only, without extrapolation. The CSV preserves individual results.
ON phase calibration and 34 m phase inversion are not performed; --antenna
selects the PRD. --dry-run lists selected commands without running frinZ.
