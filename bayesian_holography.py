#!/usr/bin/env python3
"""Standalone complex-aperture Bayesian reconstruction; see BAYESIAN_HOLOGRAPHY.md."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats

ARCMIN = np.pi / 10800


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--in', '--input', dest='input', type=Path, help='beam.txt (Epoch, Length, Amp, Phase, SNR, Az_Offset, El_Offset)')
    p.add_argument('--out', '--output', dest='output', type=Path, default=Path('bayesian_result'))
    p.add_argument('--replot', type=Path, metavar='RESULT_DIR', help='replot saved results without MCMC; specify a different --out')
    p.add_argument('--db-min', type=float, default=-45., help='dB display floor, matching Holography.py')
    p.add_argument('--surface-limit-mm', type=float, default=3., help='symmetric surface-error display limits +/- this value [mm]')
    p.add_argument('--zoom-size', type=float, default=24., help='full width of measured/model comparison zoom [arcmin]')
    p.add_argument('--slice-half-width', type=float, default=.75, help='cross-axis half-width of measured/model slices [arcmin]')
    p.add_argument('--demo', action='store_true', help='generate a known asymmetric aperture and noisy beam, then reconstruct')
    p.add_argument('--antenna', type=int, choices=(32, 34), default=32)
    p.add_argument('--frequency-ghz', type=float, default=8.448)
    p.add_argument('--grid-size', type=int, default=32, help='aperture pixels per side; dense cubic-cost linear algebra')
    p.add_argument('--center-block-size', type=float, default=3.0, help='known square blockage [m]; 0 disables')
    p.add_argument('--calibration', choices=('on', 'none'), default='on', help='ON correction or already-calibrated data')
    p.add_argument('--on-length', type=float, default=10.)
    p.add_argument('--on-amp-min', type=float, help='optional ON amplitude threshold; default no threshold')
    p.add_argument('--exclude-center-crossings', action='store_true')
    p.add_argument('--on-phase-linear-fit', action='store_true')
    p.add_argument('--exclude-on-time', action='append', default=[], help='integration-start Epoch; repeatable')
    p.add_argument('--noise-sigma', type=float, help='override raw-data quadrature noise SD; otherwise Amp/SNR, with zero SNR filled by mean sigma at matching Length')
    p.add_argument('--scan-width', type=float, help='use offsets within +/- this value in both axes [arcmin]')
    p.add_argument('--motion-average', action='store_true', help='average a constant-rate scan over Length; coordinates must be midpoints')
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--iterations', type=int, default=2000, help='iterations per chain, including warmup')
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--thin', type=int, default=1, help='retained-draw spacing; normally leave 1')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--length-prior-m', type=float, default=3., help='median smoothness length prior [m]')
    p.add_argument('--length-log-sd', type=float, default=.8)
    p.add_argument('--scale-prior', type=float, default=1., help='median complex-field spectral-scale prior')
    p.add_argument('--scale-log-sd', type=float, default=2.)
    p.add_argument('--proposal-sd', type=float, default=.15, help='initial log-hyperparameter random-walk SD')
    p.add_argument('--beam-grid', type=int, default=121, help='continuous forward-model beam display grid')
    p.add_argument('--predict-draws', type=int, default=100, help='posterior draws for beam uncertainty images')
    p.add_argument('--progress-every', type=int, default=50)
    return p


def geometry(diameter, n, blockage):
    dx = diameter / n
    axis = (np.arange(n) + .5) * dx - diameter / 2
    x, y = np.meshgrid(axis, axis)
    # Pixel support is the union of selected square cells, not an exact circle.
    support = (x*x + y*y < (diameter / 2)**2)
    support &= ~((np.abs(x) < blockage/2) & (np.abs(y) < blockage/2))
    if np.count_nonzero(support) < 4:
        raise ValueError('Too few supported aperture pixels')
    return axis, x, y, support, dx


def fourier_operator(az, el, x, y, wavelength, dx, *, length=None, rates=None):
    """B(theta)=area-normalized integral A(r) exp(-2pi i theta.r/lambda) dr.

    Square pixels are integrated exactly (sinc factors). Optional temporal
    averaging assumes linear motion about each supplied angular midpoint.
    """
    az, el = np.asarray(az), np.asarray(el)
    if rates is not None:
        length, rates = np.asarray(length), np.asarray(rates)
        sweep = np.max(np.abs(length[:, None]*ARCMIN/wavelength *
                              (rates[:, 0, None]*x + rates[:, 1, None]*y)))
        nodes = max(8, int(np.ceil(2*np.pi*sweep))+8)
        if nodes > 256:
            raise ValueError('Excessive motion integration; check Length and rates units')
        abscissa, weights = np.polynomial.legendre.leggauss(nodes)
        averaged = np.zeros((len(az), len(x)), complex)
        for t, weight in zip(abscissa, weights):
            averaged += weight/2*fourier_operator(az + t*length/2*rates[:, 0],
                el + t*length/2*rates[:, 1], x, y, wavelength, dx)
        return averaged
    u, v = az*ARCMIN/wavelength, el*ARCMIN/wavelength
    f = np.exp(-2j*np.pi*(u[:, None]*x + v[:, None]*y)) / len(x)
    f *= (np.sinc(u*dx)*np.sinc(v*dx))[:, None]
    return f


def smooth_basis(support, dx):
    """Neumann graph Laplacian: no smoothness edges across known blockage."""
    ids = np.full(support.shape, -1, dtype=int)
    ids[support] = np.arange(support.sum())
    lap = np.zeros((support.sum(), support.sum()))
    for shift in (1, support.shape[1]):
        a, b = ids.ravel()[:-shift], ids.ravel()[shift:]
        ok = (a >= 0) & (b >= 0)
        if shift == 1:
            ok &= np.arange(len(a)) % support.shape[1] != support.shape[1]-1
        a, b = a[ok], b[ok]
        np.add.at(lap, (a, a), 1/dx**2)
        np.add.at(lap, (b, b), 1/dx**2)
        lap[a, b] -= 1/dx**2
        lap[b, a] -= 1/dx**2
    eig, basis = linalg.eigh(lap, check_finite=False)
    return np.maximum(eig, 0), basis


def measurement_sigmas(table, override=None):
    """Fill zero SNR from arithmetic mean raw sigma at matching Length."""
    amp = table.Amp.to_numpy(dtype=float)
    snr = table.SNR.to_numpy(dtype=float)
    length = table.Length.to_numpy(dtype=float)
    if np.any(snr < 0):
        raise ValueError('SNR must be nonnegative; negative SNR cannot be imputed')
    imputed = np.zeros(len(table), dtype=bool)
    if override is not None:
        return np.full(len(table), override), imputed
    valid = snr > 0
    if np.any(amp[valid] <= 0):
        raise ValueError('Amp must be positive when SNR is positive')
    sigma = np.full(len(table), np.nan)
    sigma[valid] = amp[valid]/snr[valid]
    if not np.all(np.isfinite(sigma[valid]) & (sigma[valid] > 0)):
        raise ValueError('Amp/SNR must give a positive finite sigma')
    # Donors are original positive-SNR points only: never reuse filled values.
    for i in np.flatnonzero(~valid):
        peers = valid & np.isclose(length, length[i], rtol=0., atol=1e-6)
        if not peers.any():
            raise ValueError(f'Cannot fill SNR=0 at {table.Epoch.iloc[i]}: '
                             f'no positive-SNR points with Length={length[i]:g} s')
        sigma[i] = np.mean(sigma[peers])
        imputed[i] = True
        print(f'[INFO] SNR=0: Epoch={table.Epoch.iloc[i]}, Length={length[i]:g} s; '
              f'raw sigma={sigma[i]:.10g} (arithmetic mean of {peers.sum()} matching-Length points)',
              flush=True)
    return sigma, imputed


def read_data(args):
    table = pd.read_csv(args.input, skipinitialspace=True)
    table.columns = table.columns.str.strip()
    required = ['Epoch', 'Length', 'Amp', 'Phase', 'SNR', 'Az_Offset', 'El_Offset']
    missing = set(required)-set(table.columns)
    if missing:
        raise ValueError(f'Missing beam columns: {sorted(missing)}')
    for key in required[1:]:
        table[key] = pd.to_numeric(table[key], errors='raise')
    epoch = pd.to_datetime(table.Epoch, format='%Y/%j %H:%M:%S.%f', errors='coerce')
    epoch = epoch.fillna(pd.to_datetime(table.Epoch, format='%Y/%j %H:%M:%S', errors='coerce'))
    if epoch.isna().any() or not np.isfinite(table[required[1:]].to_numpy()).all():
        raise ValueError('Invalid Epoch or nonfinite measurement; fix input rather than silently dropping rows')
    table['time'] = epoch.astype('int64') / 1e9
    table = table.sort_values('time').reset_index(drop=True)
    if (table.Length <= 0).any() or (table.Amp < 0).any():
        raise ValueError('Length must be positive and Amp nonnegative')
    field = table.Amp.to_numpy()*np.exp(1j*np.deg2rad(table.Phase.to_numpy()) * (-1 if args.antenna == 34 else 1))
    sigma, sigma_imputed = measurement_sigmas(table, args.noise_sigma)
    table['raw_quadrature_sigma'] = sigma.copy()
    table['quadrature_sigma_imputed'] = sigma_imputed
    center = np.isclose(table.Az_Offset, 0, atol=1e-8, rtol=0) & np.isclose(table.El_Offset, 0, atol=1e-8, rtol=0)
    stationary = center & np.isclose(table.Length, args.on_length, atol=1e-6, rtol=0)
    on = stationary.copy() if args.exclude_center_crossings else center.copy()
    if args.on_amp_min is not None:
        on &= table.Amp >= args.on_amp_min
    for text in args.exclude_on_time:
        excluded = pd.to_datetime(text, format='%Y/%j %H:%M:%S.%f', errors='coerce')
        if pd.isna(excluded):
            excluded = pd.to_datetime(text, format='%Y/%j %H:%M:%S', errors='raise')
        matches = np.isclose(table.time, excluded.value/1e9, rtol=0, atol=1e-6)
        if not np.any(matches & on):
            print(f'[WARN] no ON candidate matched --exclude-on-time {text}', flush=True)
        on &= ~matches
    ref_amp = np.ones(len(table))
    ref_phase = np.zeros(len(table))
    if args.calibration == 'on':
        ti = table.time.to_numpy()[on]
        if len(ti) < 2 or len(np.unique(ti)) != len(ti):
            raise ValueError('Need at least 2 ON points at distinct epochs; --calibration none only for pre-calibrated input')
        ai, pi = np.abs(field[on]), np.unwrap(np.angle(field[on]))
        if np.any(ai <= 0):
            raise ValueError('Zero ON amplitude cannot calibrate gain')
        t = table.time.to_numpy()
        ref_amp = np.interp(t, ti, ai)
        if args.on_phase_linear_fit:
            coeff = np.polyfit(ti-ti[0], pi, 1)
            ref_phase = np.polyval(coeff, t-ti[0])
        else:
            ref_phase = np.interp(t, ti, pi)
        field = field/ref_amp*np.exp(-1j*ref_phase)
        sigma = sigma/ref_amp
        print(f'[INFO] ON calibration: {on.sum()} points; gain held fixed during inference', flush=True)
    use = ~stationary
    if args.scan_width is not None:
        use &= (np.abs(table.Az_Offset) <= args.scan_width) & (np.abs(table.El_Offset) <= args.scan_width)
    table['used'] = use
    table['calibration_on'] = on
    table['reference_amplitude'] = ref_amp
    table['reference_phase_rad'] = ref_phase
    table['calibrated_real'] = field.real
    table['calibrated_imag'] = field.imag
    table['quadrature_sigma'] = sigma
    used = table.loc[use].copy()
    if len(used) < 4:
        raise ValueError('Fewer than 4 raster measurements')
    if np.ptp(used.Az_Offset) <= 0 or np.ptp(used.El_Offset) <= 0:
        raise ValueError('Need nonzero offset coverage in both axes for 2D reconstruction')
    starts, durations = used.time.to_numpy(), used.Length.to_numpy()
    overlap = int(np.sum(starts[1:] < np.maximum.accumulate(starts+durations)[:-1]-1e-6))
    if overlap:
        raise ValueError(f'{overlap} overlapping integration intervals: this likelihood assumes independent samples; use a nonoverlapping beam.txt')
    rates = None
    if args.motion_average:
        keys = ['Az_Rate_arcmin_s', 'El_Rate_arcmin_s']
        if any(key not in used for key in keys):
            raise ValueError('--motion-average requires both rate columns')
        rates = used[keys].to_numpy(dtype=float)
        if not np.isfinite(rates).all():
            raise ValueError('Nonfinite scan rates')
    return table, used, field[np.asarray(use)], sigma[np.asarray(use)], rates


class GaussianAperture:
    """Collapsed log posterior of two log hyperparameters, complex proper Gaussian."""
    def __init__(self, design, data, sigma, eig, args):
        weighted = design/sigma[:, None]
        self.gram = weighted.conj().T @ weighted
        self.rhs = weighted.conj().T @ (data/sigma)
        self.eig = eig
        self.prior_mean = np.log([args.length_prior_m, args.scale_prior])
        self.prior_sd = np.array([args.length_log_sd, args.scale_log_sd])

    def conditional(self, log_parameters):
        ell, tau = np.exp(log_parameters)
        q = (1 + ell**2*self.eig)**2/tau**2
        precision = self.gram.copy()
        precision.flat[::len(q)+1] += q
        chol = linalg.cholesky(precision, lower=True, check_finite=False)
        mean = linalg.cho_solve((chol, True), self.rhs, check_finite=False)
        # Complex density exp(-a* Q a / 2): determinant terms have no 1/2.
        logp = (np.log(q).sum() - 2*np.log(chol.diagonal().real).sum()
                + .5*np.vdot(self.rhs, mean).real
                - .5*np.sum(((log_parameters-self.prior_mean)/self.prior_sd)**2))
        return float(logp), mean, chol

    @staticmethod
    def draw(mean, chol, rng):
        z = rng.normal(size=len(mean)) + 1j*rng.normal(size=len(mean))
        return mean + linalg.solve_triangular(chol.conj().T, z, lower=False, check_finite=False)


def mcmc(model, args):
    fields, parameters, acceptances = [], [], []
    for chain, seed in enumerate(np.random.SeedSequence(args.seed).spawn(args.chains)):
        rng = np.random.default_rng(seed)
        point = model.prior_mean + rng.normal(0, .4, 2)
        current = model.conditional(point)
        log_step = np.log(args.proposal_sd)
        retained_fields, retained_parameters = [], []
        accepted = sampling_accepted = 0
        warmup_points = []
        proposal_shape = np.eye(2)
        for iteration in range(args.iterations):
            proposed = point + proposal_shape@rng.normal(size=2)*np.exp(log_step)
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    candidate = model.conditional(proposed)
                success = np.log(rng.random()) < candidate[0]-current[0]
            except (linalg.LinAlgError, FloatingPointError):
                success = False
            if success:
                point, current = proposed, candidate
                accepted += 1
                if iteration >= args.warmup:
                    sampling_accepted += 1
            if iteration < args.warmup:
                warmup_points.append(point.copy())
                if iteration >= 50 and (iteration+1) % 25 == 0:
                    covariance = np.cov(np.asarray(warmup_points).T) + np.eye(2)*1e-4
                    proposal_shape = linalg.cholesky(covariance, lower=True)
                # Diminishing adaptation ONLY in discarded warmup; freeze afterward.
                log_step += (float(success)-.3)/(iteration+10)**.6
                log_step = np.clip(log_step, np.log(1e-5), np.log(2.))
            elif (iteration-args.warmup) % args.thin == 0:
                retained_parameters.append(point.copy())
                retained_fields.append(model.draw(current[1], current[2], rng))
            if (iteration+1) % args.progress_every == 0 or iteration+1 == args.iterations:
                print(f'[MCMC] chain {chain+1}/{args.chains} {iteration+1}/{args.iterations}; acceptance={accepted/(iteration+1):.3f}', flush=True)
        fields.append(np.asarray(retained_fields))
        parameters.append(np.asarray(retained_parameters))
        acceptances.append(sampling_accepted/(args.iterations-args.warmup))
        # Preserve each completed chain even if a later chain is interrupted.
        np.savez_compressed(args.output/f'chain_{chain+1:02d}.npz',
                            coefficients=fields[-1], log_parameters=parameters[-1],
                            seed_entropy=args.seed, seed_spawn_key=seed.spawn_key)
    return np.asarray(fields), np.asarray(parameters), acceptances


def diagnostics(chains):
    """Rank/folded split R-hat and initial-positive-sequence ESS on ranks."""
    m, n = chains.shape
    half = n//2
    if half < 4 or m < 2:
        return {'rank_split_rhat': None, 'bulk_ess': None}
    split = np.concatenate((chains[:, :half], chains[:, -half:]))
    rank = stats.norm.ppf((stats.rankdata(split.ravel()).reshape(split.shape)-.375)/(split.size+.25))
    def variance(z):
        w = np.var(z, axis=1, ddof=1).mean()
        b = half*np.var(z.mean(axis=1), ddof=1)
        return w, (half-1)/half*w + b/half
    def rhat(z):
        w, v = variance(z)
        return float(np.sqrt(v/w)) if w > 0 else None
    folded = np.abs(split-np.median(split))
    folded = stats.norm.ppf((stats.rankdata(folded.ravel()).reshape(split.shape)-.375)/(split.size+.25))
    rhats = [r for r in (rhat(rank), rhat(folded)) if r is not None]
    w, v = variance(rank)
    ess = None
    if v > 0:
        z = rank-rank.mean(axis=1, keepdims=True)
        f = np.fft.rfft(z, n=2*half, axis=1)
        acov = np.fft.irfft(f*f.conj(), n=2*half, axis=1)[:, :half]/half
        rho = 1-(w-acov.mean(axis=0))/v
        rho[0] = 1
        pairs = []
        for t in range(0, half-1, 2):
            pair = rho[t]+rho[t+1]
            if pair < 0:
                break
            pairs.append(min(pair, pairs[-1]) if pairs else pair)
        tau = max(1., -1+2*sum(pairs))
        ess = float(min(split.size, split.size/tau))
    return {'rank_split_rhat': max(rhats) if rhats else None, 'bulk_ess': ess}


def surface_draws(fields, x, y, support, dx, wavelength, diameter):
    from skimage.restoration import unwrap_phase
    import inspect
    unwrap_seed = {'rng': 0} if 'rng' in inspect.signature(unwrap_phase).parameters else {'seed': 0}
    phase, surfaces = [], []
    r = np.hypot(x[support], y[support])
    tilt = (r >= 2) & (r <= (14 if diameter == 32 else 15))
    full = np.hypot(np.abs(x[support])+dx/2, np.abs(y[support])+dx/2) < diameter/2
    annulus = (r >= 2) & (r <= 12)
    # Exclude EVERY pixel touching the central 3x3 m square, also for alternate support blockage.
    outside_center = (np.abs(x[support])-dx/2 >= 1.5) | (np.abs(y[support])-dx/2 >= 1.5)
    full &= outside_center
    annulus &= outside_center
    if tilt.sum() < 3 or full.sum() < 2 or annulus.sum() < 2:
        raise ValueError('Grid/support too coarse for tilt and RMS masks')
    plane = np.column_stack((x[support], y[support], np.ones(support.sum())))
    for draw in fields:
        image = np.zeros(support.shape)
        image[support] = np.angle(draw)
        unwrapped = np.asarray(unwrap_phase(np.ma.array(image, mask=~support), **unwrap_seed))[support]
        coeff = linalg.lstsq(plane[tilt], unwrapped[tilt], check_finite=False)[0]
        residual = unwrapped-plane@coeff
        phase.append(residual)
        surfaces.append(residual*wavelength*1000/(4*np.pi))
    surfaces = np.asarray(surfaces)
    return np.asarray(phase), surfaces, full, annulus


def demo(args):
    rng = np.random.default_rng(args.seed+1)
    wave = 299792458/(args.frequency_ghz*1e9)
    axis, x, y, support, dx = geometry(args.antenna, args.grid_size, args.center_block_size)
    xx, yy = x[support], y[support]
    r = np.hypot(xx, yy)/(args.antenna/2)
    amp = (1-.25*r*r)*(1-.3*np.exp(-((xx-7)**2+(yy+5)**2)/5))
    height = .7*np.exp(-((xx+5)**2+(yy-4)**2)/15)-.5*np.exp(-((xx-6)**2+(yy+5)**2)/10)
    truth = amp*np.exp(4j*np.pi*height/(wave*1000))
    # Independent sampling; demo angles stay below pixel Nyquist even on a small test grid.
    span = min(60., .85*wave/(2*dx)/ARCMIN)
    coord = np.linspace(-span, span, 31)
    az, el = np.meshgrid(coord, coord)
    az, el = az.ravel(), el.ravel()
    clean = fourier_operator(az, el, xx, yy, wave, dx)@truth
    sigma = np.full(len(clean), args.noise_sigma if args.noise_sigma is not None else .002)
    noisy = clean+sigma*(rng.normal(size=len(clean))+1j*rng.normal(size=len(clean)))
    times = pd.date_range('2026-07-03T08:00:00', periods=len(clean), freq='1s')
    table = pd.DataFrame(dict(Epoch=times.strftime('%Y/%j %H:%M:%S.%f'), Length=.5,
                              Amp=np.abs(noisy), Phase=np.rad2deg(np.angle(noisy))*(-1 if args.antenna == 34 else 1),
                              SNR=np.abs(noisy)/sigma, Az_Offset=az, El_Offset=el))
    args.input = args.output/'demo_beam.txt'
    table.to_csv(args.input, index=False)
    args.calibration = 'none'
    np.savez_compressed(args.output/'demo_truth.npz', field=truth, height_mm=height,
                        x_axis=axis, support=support, noiseless_beam=clean)
    return truth


def display_style(kind, db_min=-45., surface_limit=3.):
    """Match Holography.py for physical maps; SD remains nonnegative."""
    if kind == 'db':
        return dict(cmap='inferno', vmin=db_min, vmax=0), np.arange(np.ceil(db_min/10)*10, 1, 10)
    if kind == 'phase':
        return dict(cmap='twilight', vmin=-180, vmax=180), np.arange(-180, 181, 60)
    if kind == 'surface':
        return dict(cmap='coolwarm', vmin=-surface_limit, vmax=surface_limit), None
    if kind == 'residual':
        return dict(cmap='coolwarm', vmin=-5, vmax=5), np.arange(-4, 5, 2)
    return dict(cmap='viridis', vmin=0), None


def fit_statistics(data, predicted, sigma):
    """Same measured coordinates and SAME linear units; no fitted rescaling."""
    residual = (data-predicted)/sigma
    high_snr = np.abs(data)/sigma >= 5
    phase_error = np.rad2deg(np.angle(data*np.conj(predicted)))
    return dict(count=len(data),
                measured_peak_amplitude=float(np.abs(data).max()),
                predicted_peak_amplitude=float(np.abs(predicted).max()),
                predicted_over_measured_peak=float(np.abs(predicted).max()/np.abs(data).max()) if np.abs(data).max() > 0 else None,
                mean_squared_standardized_quadrature_residual=float(np.mean(np.abs(residual)**2)/2),
                real_residual_mean_sigma=float(residual.real.mean()),
                imag_residual_mean_sigma=float(residual.imag.mean()),
                complex_rmse=float(np.sqrt(np.mean(np.abs(data-predicted)**2))),
                amplitude_rmse=float(np.sqrt(np.mean((np.abs(data)-np.abs(predicted))**2))),
                high_snr_count=int(high_snr.sum()),
                high_snr_phase_rmse_deg=float(np.sqrt(np.mean(phase_error[high_snr]**2))) if high_snr.any() else None)


def plot_saved_results(source, output, args):
    """Replot compressed summary products; never rerun MCMC or load field draws."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source, output = Path(source), Path(output)
    required = ['summary.json', 'posterior_aperture.npz', 'posterior_beam.npz',
                'beam_residuals.csv', 'posterior_trace.csv']
    for name in required:
        if not (source/name).exists():
            raise ValueError(f'Replot needs {source/name}')
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    residual_table = pd.read_csv(source/'beam_residuals.csv')
    trace = pd.read_csv(source/'posterior_trace.csv')
    data = residual_table.data_real.to_numpy()+1j*residual_table.data_imag.to_numpy()
    predicted = residual_table.predicted_real.to_numpy()+1j*residual_table.predicted_imag.to_numpy()
    sigma = residual_table.sigma.to_numpy()
    peak = float(np.max(np.abs(data)))
    if peak <= 0:
        raise ValueError('Measured beam has no positive peak')
    metrics = fit_statistics(data, predicted, sigma)
    rms_angle = np.hypot(residual_table.Az_Offset, residual_table.El_Offset).to_numpy()
    region_metrics = {'all': metrics}
    metrics['measured_peak_offset_arcmin'] = residual_table.loc[int(np.argmax(np.abs(data))), ['Az_Offset', 'El_Offset']].to_list()
    metrics['predicted_peak_offset_arcmin'] = residual_table.loc[int(np.argmax(np.abs(predicted))), ['Az_Offset', 'El_Offset']].to_list()
    for name, select in [('within_12arcmin', rms_angle <= 12), ('outside_12arcmin', rms_angle > 12)]:
        if select.any():
            region_metrics[name] = fit_statistics(data[select], predicted[select], sigma[select])
    output.mkdir(parents=True, exist_ok=True)
    (output/'pdf').mkdir(exist_ok=True)
    (output/'beam_fit_quality.json').write_text(json.dumps(region_metrics, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'[FIT] same-coordinate mean squared standardized quadrature residual='
          f'{metrics["mean_squared_standardized_quadrature_residual"]:.4g}; '
          f'amplitude RMSE={metrics["amplitude_rmse"]:.6g}', flush=True)
    if metrics['mean_squared_standardized_quadrature_residual'] > 2:
        print('[WARN] Data-model residuals exceed the assumed noise scale substantially. '
              'Inspect residual maps/calibration/trajectory/prior; more MCMC iterations alone may not fix model mismatch.', flush=True)
    print('[PLOT] Comparison panels: left measured, right model at identical coordinates; '
          'same measured peak for BOTH dB references. Slice dots=data, solid line=model.', flush=True)
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    def save(fig, name):
        fig.savefig(output/f'{name}.png', dpi=160, bbox_inches='tight')
        fig.savefig(output/'pdf'/f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)
    def maps(values, labels, kinds, name, extent, aperture=False):
        fig, axes = plt.subplots(1, len(values), figsize=(7*len(values), 6), squeeze=False)
        for ax, value, label, kind in zip(axes.ravel(), values, labels, kinds):
            style, ticks = display_style(kind, args.db_min, args.surface_limit_mm)
            im = ax.imshow(value, origin='lower', extent=extent, aspect='equal', interpolation='nearest', **style)
            ax.set_xlabel('x [m]' if aperture else 'Az offset [arcmin]')
            ax.set_ylabel('y [m]' if aperture else 'El offset [arcmin]')
            ax.set_box_aspect(1)
            fig.colorbar(im, ax=ax, shrink=.8, label=label, ticks=ticks)
        fig.tight_layout()
        save(fig, name)
    def db(values, reference):
        return 20*np.log10(np.maximum(np.abs(values)/reference, 1e-15))
    with np.load(source/'posterior_aperture.npz') as ap:
        support, axis = ap['support'], ap['x_axis']
        dx = float(axis[1]-axis[0])
        diameter = dx*len(axis)
        extent = [-diameter/2, diameter/2]*2
        def image(value):
            grid = np.full(support.shape, np.nan)
            grid[support] = value
            return grid
        amp, mean = ap['amplitude_mean'], ap['field_mean']
        maps([image(db(amp, amp.max())), image(np.rad2deg(ap['phase_mean_rad']))],
             ['Amplitude [dB]', 'Phase [deg]'], ['db', 'phase'], 'aperture_mean', extent, True)
        maps([image(ap['amplitude_sd']), image(np.rad2deg(ap['phase_circular_sd_rad']))],
             ['Amplitude SD', 'Circular phase SD [deg]'], ['sd', 'sd'], 'aperture_uncertainty', extent, True)
        maps([image(ap['surface_mean_mm']), image(ap['surface_sd_mm'])],
             ['Surface error mean [mm]', 'Surface error SD [mm]'], ['surface', 'sd'], 'surface_posterior', extent, True)
        if (source/'demo_truth.npz').exists():
            with np.load(source/'demo_truth.npz') as truth:
                maps([image(np.abs(truth['field'])), image(np.abs(mean-truth['field']))],
                     ['True amplitude', 'Complex-field absolute error'], ['sd', 'sd'], 'demo_recovery', extent, True)
        field_count = int(support.sum())
    with np.load(source/'posterior_beam.npz') as beam:
        azaxis, elaxis = beam['az_arcmin'], beam['el_arcmin']
        extent = [azaxis[0], azaxis[-1], elaxis[0], elaxis[-1]]
        maps([db(beam['complex_mean'], peak), np.rad2deg(np.angle(beam['complex_mean']))],
             ['Amplitude [dB]', 'Phase [deg]'], ['db', 'phase'], 'beam_mean', extent)
        maps([beam['amplitude_sd']], ['Amplitude SD'], ['sd'], 'beam_uncertainty', extent)
    # Raw scatter avoids introducing a second interpolation model in fit checks.
    az, el = residual_table.Az_Offset.to_numpy(), residual_table.El_Offset.to_numpy()
    bound = max(np.max(np.abs(az)), np.max(np.abs(el)))
    def point_maps(values, label, kind, name, zoom=None):
        fig, axes = plt.subplots(1, len(values), figsize=(7*len(values), 6), squeeze=False)
        for ax, value in zip(axes.ravel(), values):
            style, ticks = display_style(kind, args.db_min, args.surface_limit_mm)
            im = ax.scatter(az, el, c=value, marker='s', s=16 if zoom is None else 40,
                            linewidths=0, **style)
            half = bound if zoom is None else zoom/2
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_aspect('equal')
            ax.set_box_aspect(1)
            ax.set_xlabel('Az offset [arcmin]')
            ax.set_ylabel('El offset [arcmin]')
            fig.colorbar(im, ax=ax, shrink=.8, label=label, ticks=ticks)
        fig.tight_layout()
        save(fig, name)
    point_maps([db(data, peak), db(predicted, peak)], 'Amplitude [dB]', 'db', 'beam_data_model_db')
    point_maps([db(data, peak), db(predicted, peak)], 'Amplitude [dB]', 'db',
               'beam_data_model_db_zoom', args.zoom_size)
    point_maps([np.rad2deg(np.angle(data)), np.rad2deg(np.angle(predicted))],
               'Phase [deg]', 'phase', 'beam_data_model_phase')
    standardized = (data-predicted)/sigma
    point_maps([standardized.real, standardized.imag], 'Residual / sigma', 'residual', 'beam_standardized_residuals')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    selected_counts = {}
    for col, (coord, cross, axis_label) in enumerate([(az, el, 'Az'), (el, az, 'El')]):
        # Do not snap PRD coordinates; state the exact cross-axis strip width.
        select = np.abs(cross) <= args.slice_half_width
        select &= np.abs(coord) <= args.zoom_size/2
        order = np.argsort(coord[select])
        selected_counts[axis_label] = int(select.sum())
        location, measured, model = coord[select][order], data[select][order], predicted[select][order]
        axes[0, col].plot(location, db(measured, peak), 'o', color='black', markersize=3)
        axes[0, col].plot(location, db(model, peak), '-', color='black', linewidth=1)
        measured_phase = np.rad2deg(np.angle(measured))
        model_phase = np.rad2deg(np.angle(model))
        if len(model_phase):
            model_phase[np.r_[False, np.abs(np.diff(model_phase)) > 180]] = np.nan
        axes[1, col].plot(location, measured_phase, 'o', color='black', markersize=3)
        axes[1, col].plot(location, model_phase, '-', color='black', linewidth=1)
        axes[0, col].set_ylim(args.db_min, 1)
        axes[1, col].set_ylim(-180, 180)
        for row in (0, 1):
            axes[row, col].set_xlim(-args.zoom_size/2, args.zoom_size/2)
            axes[row, col].set_xlabel(f'{axis_label} offset [arcmin]')
        axes[0, col].set_ylabel('Amplitude [dB]')
        axes[1, col].set_ylabel('Phase [deg]')
        if not select.any():
            print(f'[WARN] No {axis_label} slice points in cross-axis strip +/-{args.slice_half_width:g} arcmin.', flush=True)
    fig.tight_layout()
    save(fig, 'beam_fit_slices')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    u = np.linspace(-5, 5, 300)
    for ax, value in zip(axes, [standardized.real, standardized.imag]):
        ax.hist(value, bins=60, density=True, histtype='step', color='black')
        ax.plot(u, stats.norm.pdf(u), color='black', linewidth=1)
        ax.set_xlabel('Residual / sigma')
        ax.set_ylabel('Density')
    fig.tight_layout()
    save(fig, 'beam_residual_histogram')
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    keys = ['length_m', 'prior_scale', 'rms_full_mm', 'rms_annulus_mm']
    labels = ['Smoothness ell [m]', 'Field prior scale tau', 'Derived full RMS [mm]', 'Derived annulus RMS [mm]']
    parameter_rows = []
    for ax, key, label in zip(axes, keys, labels):
        for chain, frame in trace.groupby('chain'):
            ax.plot(frame.draw, frame[key], color='black', alpha=.45, linewidth=.7)
        ax.set_ylabel(label)
        quantiles = np.percentile(trace[key], [16, 50, 84])
        parameter_rows.append(dict(parameter=key, role='MH hyperparameter' if key in keys[:2] else 'derived statistic',
                                   p16=quantiles[0], median=quantiles[1], p84=quantiles[2]))
    axes[-1].set_xlabel('Retained draw per chain')
    fig.tight_layout()
    save(fig, 'mcmc_trace')
    pd.DataFrame(parameter_rows).to_csv(output/'parameter_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key in zip(axes, keys[2:]):
        ax.hist(trace[key], bins=40, density=True, histtype='step', color='black')
        ax.set_xlabel('Surface RMS [mm]')
        ax.set_ylabel('Posterior density')
    fig.tight_layout()
    save(fig, 'surface_rms_posterior')
    model_description = dict(complex_aperture_pixels=field_count, real_field_unknowns=2*field_count,
        mh_parameters={'log_ell': 'log smoothness length [m]', 'log_tau': 'log field coefficient prior scale'},
        conditional_parameters=f'{field_count} complex aperture coefficients drawn jointly per retained step',
        derived_not_sampled=['surface RMS', 'surface height after phase unwrap and tilt removal', 'predicted beam pattern'],
        fixed=['measured beam values and positions', 'sigma=Amp/SNR with documented zero-SNR fill',
               'ON calibration', 'aperture support', 'frequency'],
        hyperpriors={name: summary['parameters'].get(name) for name in ['length_prior_m', 'length_log_sd', 'scale_prior', 'scale_log_sd']},
        noise_sigma_override=summary['parameters'].get('noise_sigma'),
        note='The sampler does not jitter observed amplitudes/phases as in noise-propagation Monte Carlo.')
    (output/'inference_parameters.json').write_text(json.dumps(model_description, indent=2)+'\n', encoding='utf-8')
    (output/'plot_settings.json').write_text(json.dumps(dict(source=str(source.resolve()),
        db_min=args.db_min, surface_limit_mm=args.surface_limit_mm,
        shared_beam_reference_measured_peak=peak, zoom_size_arcmin=args.zoom_size,
        slice_cross_half_width_arcmin=args.slice_half_width, slice_point_counts=selected_counts,
        comparison_panels='left measured, right model at same measured coordinates',
        phase_comparison='all points; low-amplitude phases may be uninformative',
        continuous_beam='instantaneous prediction; motion-average training predictions are in the point comparisons'), indent=2)+'\n', encoding='utf-8')
    print(f'[PARAMETERS] MH samples log(ell), log(tau); jointly draws {field_count} complex field coefficients '
          f'({2*field_count} real unknowns). RMS is derived. See inference_parameters.json and parameter_summary.csv.', flush=True)
    return region_metrics


def save_results(args, axis, x, y, support, dx, basis, coefficients, parameters, used, data, sigma, design, wave, acceptances, truth):
    shape = coefficients.shape[:2]
    # Coefficients are in the graph eigenbasis; output fields are native aperture pixels.
    fields = coefficients.reshape(-1, basis.shape[0])@basis.T
    phase, surface, full, annulus = surface_draws(fields, x, y, support, dx, wave, args.antenna)
    rms_full = np.sqrt(np.mean(surface[:, full]**2, axis=1)).reshape(shape)
    rms_annulus = np.sqrt(np.mean(surface[:, annulus]**2, axis=1)).reshape(shape)
    mean = fields.mean(axis=0)
    circular_mean = np.angle(np.exp(1j*np.angle(fields)).mean(axis=0))
    circular_sd = np.sqrt(-2*np.log(np.clip(np.abs(np.exp(1j*np.angle(fields)).mean(axis=0)), 1e-15, 1)))
    pred = design@coefficients.reshape(-1, basis.shape[0]).mean(axis=0)
    np.savez_compressed(args.output/'posterior_aperture.npz', x_axis=axis, y_axis=axis, support=support,
                        field_draws=fields, field_mean=mean, amplitude_mean=np.abs(fields).mean(axis=0),
                        amplitude_sd=np.abs(fields).std(axis=0, ddof=1), phase_mean_rad=circular_mean,
                        phase_circular_sd_rad=circular_sd, surface_draws_mm=surface,
                        surface_mean_mm=surface.mean(axis=0), surface_sd_mm=surface.std(axis=0, ddof=1),
                        rms_full_mask=full, rms_annulus_mask=annulus)
    trace = pd.DataFrame({'chain': np.repeat(np.arange(shape[0])+1, shape[1]),
                          'draw': np.tile(np.arange(shape[1])+1, shape[0]),
                          'length_m': np.exp(parameters[:, :, 0]).ravel(),
                          'prior_scale': np.exp(parameters[:, :, 1]).ravel(),
                          'rms_full_mm': rms_full.ravel(), 'rms_annulus_mm': rms_annulus.ravel()})
    trace.to_csv(args.output/'posterior_trace.csv', index=False)
    pd.DataFrame(dict(Az_Offset=used.Az_Offset, El_Offset=used.El_Offset,
                      data_real=data.real, data_imag=data.imag, sigma=sigma,
                      predicted_real=pred.real, predicted_imag=pred.imag,
                      residual_real_sigma=(data-pred).real/sigma,
                      residual_imag_sigma=(data-pred).imag/sigma)).to_csv(args.output/'beam_residuals.csv', index=False)
    diagnostic = {}
    for name, values in [('length_m', parameters[:, :, 0]), ('prior_scale', parameters[:, :, 1]),
                         ('rms_full_mm', rms_full), ('rms_annulus_mm', rms_annulus)]:
        diagnostic[name] = diagnostics(values)
    summaries = {}
    for name, values in [('full_aperture', rms_full), ('annulus', rms_annulus)]:
        q = np.percentile(values, [2.5, 16, 50, 84, 97.5])
        summaries[name] = dict(zip(['p2.5_mm', 'p16_mm', 'median_mm', 'p84_mm', 'p97.5_mm'], map(float, q)))
        print(f'[POSTERIOR] {name}: median={q[2]:.4f} mm; 16-84%=[{q[1]:.4f}, {q[3]:.4f}]', flush=True)
    import scipy, skimage, platform, hashlib
    summary = {'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'pandas': pd.__version__,
                            'skimage': skimage.__version__},
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'model': 'proper complex Gaussian aperture with graph-smoothness prior; collapsed MH hyperparameters plus exact conditional field draws',
               'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'acceptance_sampling': acceptances, 'diagnostics': diagnostic, 'rms': summaries,
               'quadrature_residual_mean_square': float(np.mean(np.abs((data-pred)/sigma)**2)/2),
               'pixel_size_m': dx, 'nominal_fourier_resolution_m': [float(wave/(np.ptp(used.Az_Offset)*ARCMIN)), float(wave/(np.ptp(used.El_Offset)*ARCMIN))],
               'conditions': ['ON gains fixed; ON calibration uncertainty/correlations not included',
                              'independent Gaussian noise; raw sigma=Amp/SNR, zero SNR filled by arithmetic mean at matching Length unless overridden',
                              'support and zero blockage imposed; Gaussian prior on complex field, not on surface height',
                              'surface sign is lambda*phase/(4pi); physical front/back sign and sky handedness require external validation',
                              'tilt removed after masked 2D phase unwrapping, unlike legacy 135-degree branch rule',
                              'fine grid is not evidence of improved resolution; prior sensitivity and injection tests required']}
    if truth is not None:
        summary['demo_field_rmse'] = float(np.sqrt(np.mean(np.abs(mean-truth)**2)))
        _, true_surface, _, _ = surface_draws(truth[None, :], x, y, support, dx, wave, args.antenna)
        summary['demo_true_rms_full_mm'] = float(np.sqrt(np.mean(true_surface[:, full]**2)))
        summary['demo_true_rms_annulus_mm'] = float(np.sqrt(np.mean(true_surface[:, annulus]**2)))
    if any(d['rank_split_rhat'] is None or d['rank_split_rhat'] > 1.01 or d['bulk_ess'] is None or d['bulk_ess'] < 400 for d in diagnostic.values()):
        print('[WARN] MCMC convergence target not met (rank split R-hat<=1.01, ESS>=400). Inspect traces and extend chains.', flush=True)
    if np.any(np.percentile(circular_sd[full], 95) > .5):
        print('[WARN] Large aperture phase uncertainty: surface unwrapping and Gaussian summary may be unreliable.', flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    # Continuous beam is the forward transform of posterior aperture samples.
    n = args.beam_grid
    azaxis = np.linspace(used.Az_Offset.min(), used.Az_Offset.max(), n)
    elaxis = np.linspace(used.El_Offset.min(), used.El_Offset.max(), n)
    az, el = np.meshgrid(azaxis, elaxis)
    chosen = np.linspace(0, len(fields)-1, min(args.predict_draws, len(fields)), dtype=int)
    beam_mean, beam_amp_sd = np.empty(n*n, complex), np.empty(n*n)
    for start in range(0, n*n, 512):
        f = fourier_operator(az.ravel()[start:start+512], el.ravel()[start:start+512], x[support], y[support], wave, dx)
        samples = f@fields[chosen].T
        beam_mean[start:start+512] = f@mean
        beam_amp_sd[start:start+512] = np.std(np.abs(samples), axis=1, ddof=1)
    beam_mean = beam_mean.reshape(n, n)
    beam_amp_sd = beam_amp_sd.reshape(n, n)
    np.savez_compressed(args.output/'posterior_beam.npz', az_arcmin=azaxis, el_arcmin=elaxis,
                        complex_mean=beam_mean, amplitude_sd=beam_amp_sd, uncertainty_draw_count=len(chosen))
    fit = plot_saved_results(args.output, args.output, args)
    summary['fit_quality'] = fit
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return summary


def main(argv=None):
    p = parser()
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        p.print_help()
        return 0
    args = p.parse_args(argv)
    if sum(map(bool, [args.input, args.demo, args.replot])) != 1:
        p.error('specify exactly one of --in, --demo, or --replot')
    if not np.isfinite(args.db_min) or args.db_min >= 0:
        p.error('--db-min must be finite and negative')
    for key in ['surface_limit_mm', 'zoom_size', 'slice_half_width']:
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f'--{key.replace("_", "-")} must be finite and positive')
    if args.replot and args.replot.resolve() == args.output.resolve():
        p.error('--replot requires a different --out to preserve source results')
    for key in ['frequency_ghz', 'on_length', 'length_prior_m', 'length_log_sd', 'scale_prior', 'scale_log_sd', 'proposal_sd']:
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f'--{key.replace("_", "-")} must be finite and positive')
    for key in ['noise_sigma', 'scan_width']:
        value = getattr(args, key)
        if value is not None and (not np.isfinite(value) or value <= 0):
            p.error(f'--{key.replace("_", "-")} must be finite and positive')
    if args.on_amp_min is not None and (not np.isfinite(args.on_amp_min) or args.on_amp_min < 0):
        p.error('--on-amp-min must be finite and nonnegative')
    if not np.isfinite(args.center_block_size) or not 0 <= args.center_block_size < args.antenna:
        p.error('invalid --center-block-size')
    if args.grid_size < 8 or args.chains < 2 or args.iterations-args.warmup < 8 or args.warmup < 0 or args.thin < 1 or (args.iterations-args.warmup-1)//args.thin+1 < 8 or args.predict_draws < 2 or args.beam_grid < 3 or args.progress_every < 1 or args.seed < 0:
        p.error('need grid>=8, chains>=2, >=8 retained draws per chain, valid warmup/thin, predict-draws>=2, beam-grid>=3, nonnegative seed')
    args.output.mkdir(parents=True, exist_ok=True)
    # Avoid destroying previous expensive chains/results by accident.
    if (args.output/'summary.json').exists() or list(args.output.glob('chain_*.npz')):
        p.error('output already contains posterior results; choose a new --out directory')
    class Tee:
        def __init__(self, stream, log):
            self.stream, self.log = stream, log
        def write(self, text):
            self.stream.write(text)
            self.log.write(text)
            self.flush()
        def flush(self):
            self.stream.flush()
            self.log.flush()
    oldout, olderr = sys.stdout, sys.stderr
    with (args.output/'bayesian_holography.log').open('w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(oldout, log), Tee(olderr, log)
        try:
            start = time.monotonic()
            if args.replot:
                plot_saved_results(args.replot, args.output, args)
                print(f'[INFO] Replotted saved results without MCMC: {args.output}', flush=True)
                return 0
            truth = demo(args) if args.demo else None
            all_rows, used, data, sigma, rates = read_data(args)
            all_rows.to_csv(args.output/'input_calibration.csv', index=False)
            axis, x, y, support, dx = geometry(args.antenna, args.grid_size, args.center_block_size)
            wave = 299792458/(args.frequency_ghz*1e9)
            if np.max(np.abs(used[['Az_Offset', 'El_Offset']].to_numpy()))*ARCMIN > wave/(2*dx):
                raise ValueError('Offsets exceed aperture-grid Nyquist range; increase --grid-size')
            print(f'[INFO] {len(data)} samples; {support.sum()} complex aperture pixels; dx={dx:.4g} m', flush=True)
            print('[INFO] Fixed ON calibration; independent quadrature Gaussian noise; dense Cholesky per MCMC proposal.', flush=True)
            print(f'[PARAMETERS] MH: log(ell) and log(tau); conditional draws: {support.sum()} complex aperture coefficients. Observed beam and ON calibration stay fixed; RMS is derived.', flush=True)
            eig, basis = smooth_basis(support, dx)
            f = fourier_operator(used.Az_Offset, used.El_Offset, x[support], y[support], wave, dx,
                                 length=used.Length.to_numpy(), rates=rates)
            design = f@basis
            model = GaussianAperture(design, data, sigma, eig, args)
            coefficients, parameters, acceptances = mcmc(model, args)
            save_results(args, axis, x, y, support, dx, basis, coefficients, parameters, used, data, sigma, design, wave, acceptances, truth)
            print(f'[INFO] Finished in {time.monotonic()-start:.1f} s; results: {args.output}', flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            return 1
        finally:
            sys.stdout, sys.stderr = oldout, olderr
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
