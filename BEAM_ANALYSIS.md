# Central beam re-integration

Examples:

    python3 beam_analysis.py --workdir I26287YA --antenna 32 --scan-axis az
    python3 beam_analysis.py --workdir I26287YA --antenna 34 --scan-axis el --scan-lag-ms 100

Required files: WORKDIR/step_cor/*.cor (already correlated), WORKDIR/*.xml,
and WORKDIR/OBS_CODE32.prd or OBS_CODE34.prd. XML needs stream/label and
process/epoch,length,skip. This script does not run gico3.

Use --prd FILE and repeatable --xml FILE for explicit files. Their paths,
like --output and --outdir, are relative to the calling directory.

Select moving raster legs intersecting Az/El ±12 arcmin: the leg through
(0,0) and neighboring legs within ±12 arcmin in the fixed coordinate.
Run frinZ --length 10 --step 10 once per selected .cor file, adding --skip
when needed for lag. Each command processes the full remaining file; the
summary retains only nominal integration midpoints inside the square and
excludes stationary/turnaround motion. XML length-skip=10 s identifies ON
files, which are excluded; --on-length overrides it.

Length/step are correlation sample counts. At the default
--samples-per-second 1000, 10 samples means 10 ms, not 10 s.
Set the actual output rate if different. --length-samples and --step-samples
override 10. An incompatible result duration causes an error.

--scan-lag-ms follows corr_fringe's one-direction convention: positive shifts
integration later, negative earlier. Increasing scans are selected by default;
--forward-direction decreasing selects the other direction. The unselected
direction has zero lag. Windows outside nominal or actual file bounds are
omitted. Epoch is the nominal integration start and coordinates its midpoint;
the physical data window is shifted by the lag. Applied_Lag_ms records it.

Outputs:

- WORKDIR/ANA_beam.txt: CSV with Epoch, Length, Amp, Phase, SNR, PRD offsets,
  PRD rates and applied lag; --output FILE overrides the path.
- WORKDIR/beam_analysis/*.stdout.txt and *.stderr.txt: raw frinZ logs.
- WORKDIR/beam_analysis/ANA_beam_map.png
- WORKDIR/beam_analysis/pdf/ANA_beam_map.pdf

The map shows peak-normalized amplitude in dB and raw phase on square
±12 arcmin axes. Complex linear interpolation is used for the figure only,
with no extrapolation outside sampled coverage. CSV values retain the frinZ
samples. ON phase calibration and 34 m phase sign inversion are not applied;
--antenna selects PRD only.

Use --dry-run to inspect files/commands without running frinZ. The real
frinZ must support --step and output one result row for each successive
integration window; raw logs are saved for inspection.
