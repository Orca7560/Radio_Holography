    usable = usable[usable.overlap_pixels >= min_overlap * usable.overlap_pixels.max()]
    best = usable.sort_values(["xcorr", "overlap_pixels"], ascending=[False, False]).iloc[0]
    lag = float(best.lag_ms)
    return result, lag, amplitude_map(moving, lag, gx, gy)


def save_diagnostic(path: Path, results: dict, plus: pd.DataFrame,
                    minus: pd.DataFrame, width: float, grid_size: int) -> None:
    """Plot a fine central grid; leave the search grid and scores untouched."""
    axis = np.linspace(-width, width, grid_size)
    gx, gy = np.meshgrid(axis, axis)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    extent = [-width, width, -width, width]
    for row_index, (name, fixed_data, moving_data) in enumerate((
            ("+Az fixed / -Az shifted", plus, minus),
            ("-Az fixed / +Az shifted", minus, plus))):
        table, lag, _ = results[name]
        fixed = amplitude_map(fixed_data, 0, gx, gy)
        moving_raw = amplitude_map(moving_data, 0, gx, gy)
        moved = amplitude_map(moving_data, lag, gx, gy)
        ax = axes[row_index]
        ax[0].plot(table.lag_ms, table.xcorr, color="navy")
        ax[0].axvline(lag, color="crimson", label=f"{lag:g} ms")
        ax[0].set(xlabel="Moving direction lag [ms]", ylabel="Normalized correlation",
                  title=name)
        ax[0].legend()
        vmax = max(np.nanmax(fixed), np.nanmax(moving_raw), np.nanmax(moved))
        for pane, image, title in ((ax[1], fixed, "Fixed direction"),
                                   (ax[2], moving_raw, "Moving: before"),
                                   (ax[3], moved, "Moving: selected lag")):
            im = pane.imshow(image, origin="lower", extent=extent, cmap="viridis",
                             vmin=0, vmax=vmax, aspect="equal")
            if pane is not ax[1]:
                pane.contour(gx, gy, fixed, levels=[0.5 * np.nanmax(fixed)],
                             colors="white", linewidths=1.2)
            pane.set(title=title, xlabel="Az [arcmin]",
                     ylabel="El [arcmin]")
            fig.colorbar(im, ax=pane, shrink=0.75)
    fig.suptitle("Central beam; white contour = fixed direction at 50% peak", fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path, help="beam.txt: Epoch, Amp, Phase, SNR")
    parser.add_argument("schedule", type=Path, help="SKD file with $SKED offsets")
    parser.add_argument("--outdir", type=Path, default=Path("scanning_xcorr_result"),
                        help="separate result directory (default: scanning_xcorr_result)")
    parser.add_argument("--max-lag-ms", type=float, default=1000,
                        help="search each moving direction from -MAX to +MAX ms (default: 1000)")
    parser.add_argument("--lag-step-ms", type=float, default=5,
                        help="lag spacing in ms (default: 5)")
    parser.add_argument("--match-tolerance", type=float, default=0.006,
                        help="SKD/beam timestamp matching tolerance in seconds (default: 0.006)")
    parser.add_argument("--row-gap", type=float, default=2.0,
                        help="time gap separating scan rows in seconds (default: 2)")
    parser.add_argument("--min-snr", type=float, default=3,
                        help="minimum SNR for lag search (default: 3)")
    parser.add_argument("--grid-size", type=int, default=121,
                        help="number of comparison pixels per map axis (default: 121)")
    parser.add_argument("--main-beam-fraction", type=float, default=0.05,
                        help="only pixels above this fraction of the peak (default: 0.05)")
    parser.add_argument("--min-pixels", type=int, default=20,
                        help="minimum common main-beam pixels (default: 20)")
    parser.add_argument("--min-overlap", type=float, default=0.8,
                        help="minimum useful overlap relative to the maximum (default: 0.8)")
    parser.add_argument("--plot-width-arcmin", type=float, default=12,
                        help="central diagnostic map half-width in arcmin (default: 12)")
    parser.add_argument("--plot-grid-size", type=int, default=241,
                        help="pixels per axis for diagnostic maps only (default: 241)")
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    args = parser.parse_args()
    if (args.max_lag_ms <= 0 or args.lag_step_ms <= 0 or args.grid_size < 5
            or args.min_pixels < 3 or not 0 <= args.main_beam_fraction < 1
            or not 0 < args.min_overlap <= 1 or args.plot_width_arcmin <= 0
            or args.plot_grid_size < 5):
        parser.error("invalid lag/grid/overlap/beam threshold")
    data = classify(measurements(args.measurements), schedule(args.schedule),
                    args.match_tolerance, args.row_gap)
    usable = data[data.SNR >= args.min_snr]
    plus, minus = usable[usable.direction == 1], usable[usable.direction == -1]
    if min(len(plus), len(minus)) < 3:
        raise ValueError("both +Az and -Az scans with enough SNR are required")
    qx = np.linspace(data.az_arcmin.quantile(0.01), data.az_arcmin.quantile(0.99),
                     args.grid_size)
    qy = np.linspace(data.el_arcmin.quantile(0.01), data.el_arcmin.quantile(0.99),
                     args.grid_size)
    gx, gy = np.meshgrid(qx, qy)
    plus_map, minus_map = (amplitude_map(plus, 0, gx, gy),
                           amplitude_map(minus, 0, gx, gy))
    lags = np.arange(-args.max_lag_ms, args.max_lag_ms + args.lag_step_ms / 2,
                     args.lag_step_ms)
    results = {}
    for name, fixed, moving in (("+Az fixed / -Az shifted", plus_map, minus),
                                ("-Az fixed / +Az shifted", minus_map, plus)):
        print(f"Searching {name}: {len(lags)} trials", file=sys.stderr, flush=True)
        results[name] = search(fixed, moving, gx, gy, lags,
                               args.main_beam_fraction, args.min_pixels,
                               args.min_overlap)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (table, lag, _) in results.items():
        table.to_csv(args.outdir / ("lag_search_minus_moving.csv" if name.startswith("+Az")
                                        else "lag_search_plus_moving.csv"), index=False)
        best = table.loc[np.isclose(table.lag_ms, lag)].iloc[0]
        rows.append({"fixed_direction": "+Az" if name.startswith("+Az") else "-Az",
                     "moving_direction": "-Az" if name.startswith("+Az") else "+Az",
                     "best_moving_lag_ms": lag,
                     "normalized_xcorr": float(best.xcorr),
                     "overlap_pixels": int(best.overlap_pixels)})
    pd.DataFrame(rows).to_csv(args.outdir / "best_lags_xcorr.csv", index=False)
    save_diagnostic(args.outdir / "scanning_xcorr_diagnostics.png", results,
                    plus, minus, args.plot_width_arcmin, args.plot_grid_size)
    for result in rows:
        print(f"{result['fixed_direction']} fixed, {result['moving_direction']} shifted: "
              f"lag={result['best_moving_lag_ms']:g} ms, "
              f"r={result['normalized_xcorr']:.5f}, "
              f"pixels={result['overlap_pixels']}")
    print("These are alternative relative alignments, not two independent absolute lags.")
    print(f"Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
