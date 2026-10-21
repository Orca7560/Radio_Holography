"""Shared numerical steps and thermal-noise Monte Carlo for Holography.py."""
import numpy as np
from scipy.interpolate import griddata


def calibrate_phase(field, times, on_indices, linear_fit=False):
    """Re-estimate the ON reference; return corrected field, reference, slope."""
    corrected = np.asarray(field, dtype=complex).copy()
    if len(on_indices) < 2:
        if linear_fit:
            raise ValueError('Linear ON fitting requires at least two ON points.')
        return corrected, None, None
    phases = np.unwrap(np.angle(corrected[on_indices]))
    slope = None
    if linear_fit:
        relative = times[on_indices] - times[on_indices[0]]
        if np.unique(relative).size < 2:
            raise ValueError('Linear ON fitting requires distinct ON epochs.')
        slope, intercept = np.polyfit(relative, phases, 1)
        reference = slope * (times - times[on_indices[0]]) + intercept
    else:
        reference = np.interp(times, times[on_indices], phases)
    corrected *= np.exp(-1j * reference)
    return corrected, reference, slope


def calibrate_amplitude(field, times, on_indices):
    """Normalize by the linearly interpolated ON amplitude (endpoint hold)."""
    corrected = np.asarray(field, dtype=complex).copy()
    if len(on_indices) < 2:
        raise ValueError('ON amplitude normalization requires at least two ON points.')
    on_times = np.asarray(times)[on_indices]
    amplitudes = np.abs(corrected[on_indices])
    if not np.all(np.diff(on_times) > 0):
        raise ValueError('ON amplitude normalization requires increasing distinct ON epochs.')
    if not np.all(np.isfinite(amplitudes) & (amplitudes > 0)):
        raise ValueError('ON amplitudes must be finite and positive.')
    reference = np.interp(times, on_times, amplitudes)
    return corrected / reference, reference


def grid_layout(az, el, x_axis, y_axis):
    """Cache exactly the nearest-cell assignment and nearest gap filling."""
    ix = np.array([np.argmin(np.abs(x_axis - value)) for value in az])
    iy = np.array([np.argmin(np.abs(y_axis - value)) for value in el])
    shape = (len(y_axis), len(x_axis))
    cell = iy * shape[1] + ix
    count = np.bincount(cell, minlength=np.prod(shape)).reshape(shape)
    occupied = count > 0
    observed = np.flatnonzero(occupied)
    yy, xx = np.where(occupied)
    all_y, all_x = np.mgrid[:shape[0], :shape[1]]
    nearest = griddata((yy, xx), observed, (all_y, all_x), method='nearest').astype(int)
    return cell, count, nearest


def average_grid(values, layout):
    cell, count, nearest = layout
    sums = np.zeros(count.size, dtype=np.asarray(values).dtype)
    np.add.at(sums, cell, values)
    raw = np.zeros(count.size, dtype=sums.dtype)
    occupied = count.ravel() > 0
    raw[occupied] = sums[occupied] / count.ravel()[occupied]
    return raw[nearest], raw.reshape(count.shape)


def aperture_from_beam(beam):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(beam)))


def correct_aperture_phase(field):
    """Preserve the existing +/-135 degree, 180 degree correction exactly."""
    phase = np.angle(field)
    phase = np.where(phase > np.deg2rad(135), phase - np.pi, phase)
    return np.where(phase < -np.deg2rad(135), phase + np.pi, phase)


def remove_phase_tilt(phase, x, y, fit_mask):
    matrix = np.c_[x[fit_mask].ravel(), y[fit_mask].ravel(), np.ones(np.count_nonzero(fit_mask))]
    coefficients, _, _, _ = np.linalg.lstsq(matrix, phase[fit_mask].ravel(), rcond=None)
    plane = coefficients[0] * x + coefficients[1] * y + coefficients[2]
    return phase - plane, plane, coefficients


def noise_sigmas(amplitude, snr, length):
    """A/SNR per quadrature; impute undefined values only within same Length."""
    amplitude, snr, length = map(lambda v: np.asarray(v, dtype=float), (amplitude, snr, length))
    if not np.all(np.isfinite(amplitude) & (amplitude >= 0) & np.isfinite(length) & (length > 0)):
        raise ValueError('Monte Carlo requires finite nonnegative Amp and positive Length.')
    valid = np.isfinite(snr) & (snr > 0) & (amplitude > 0)
    sigma = np.full(amplitude.shape, np.nan)
    sigma[valid] = amplitude[valid] / snr[valid]
    valid &= np.isfinite(sigma) & (sigma > 0)
    imputed = ~valid
    for index in np.flatnonzero(imputed):
        peers = valid & np.isclose(length, length[index], rtol=0, atol=1e-6)
        if not np.any(peers):
            raise ValueError(f'No valid A/SNR at Length={length[index]:g} s to estimate noise for point {index}.')
        sigma[index] = np.median(sigma[peers])
    return sigma, imputed


def run_monte_carlo(field, sigma, times, on_indices, measurement_indices, layout,
                    x, y, fit_mask, rms_masks, wavelength, trials, seed=None,
                    linear_fit=False, progress=None, normalize_amplitude=False):
    """Propagate independent complex noise on original points including ONs.

    Geometry, peak shift, gridding assignment, ON membership and RMS masks are
    fixed at their nominal values. Every trial refits the ON phase and tilt.
    """
    rng = np.random.default_rng(seed)
    shape = layout[1].shape
    amplitudes = np.empty((trials,) + shape)
    raw_phases = np.empty_like(amplitudes)
    surfaces = np.empty_like(amplitudes)
    metrics = np.empty((trials, len(rms_masks)))
    factor = wavelength * 1000 / (4 * np.pi)
    nominal, _, _ = calibrate_phase(field, times, on_indices, linear_fit)
    if normalize_amplitude:
        nominal, _ = calibrate_amplitude(nominal, times, on_indices)
    nominal_grid, _ = average_grid(nominal[measurement_indices], layout)
    nominal_aperture = aperture_from_beam(nominal_grid)
    nominal_phase = np.angle(nominal_aperture)
    nominal_branch = np.abs(nominal_phase) > np.deg2rad(135)
    branch_count = np.zeros(shape, dtype=int)
    for trial in range(trials):
        noisy = field + sigma * (rng.standard_normal(len(field)) + 1j * rng.standard_normal(len(field)))
        corrected, _, _ = calibrate_phase(noisy, times, on_indices, linear_fit)
        if normalize_amplitude:
            corrected, _ = calibrate_amplitude(corrected, times, on_indices)
        beam, _ = average_grid(corrected[measurement_indices], layout)
        aperture = aperture_from_beam(beam)
        amplitudes[trial] = np.abs(aperture)
        raw_phases[trial] = np.angle(aperture)
        branch_count += ((np.abs(raw_phases[trial]) > np.deg2rad(135)) != nominal_branch)
        phase, _, _ = remove_phase_tilt(correct_aperture_phase(aperture), x, y, fit_mask)
        surfaces[trial] = phase * factor
        for column, mask in enumerate(rms_masks):
            metrics[trial, column] = np.sqrt(np.mean(surfaces[trial][mask] ** 2))
        if progress is not None:
            progress(trial + 1, trials)
    # Wrapped differences avoid artificial 360 degree jumps in raw phase.
    residual_phase = np.angle(np.exp(1j * (raw_phases - nominal_phase)))
    circular_mean = np.angle(np.mean(np.exp(1j * raw_phases), axis=0))
    resultant = np.abs(np.mean(np.exp(1j * raw_phases), axis=0))
    circular_std = np.sqrt(-2 * np.log(np.clip(resultant, np.finfo(float).tiny, 1)))
    return dict(amplitude_mean=amplitudes.mean(axis=0),
                amplitude_std=amplitudes.std(axis=0, ddof=1),
                raw_phase_mean_rad=circular_mean, raw_phase_circular_std_rad=circular_std,
                raw_phase_delta_p16_rad=np.percentile(residual_phase, 16, axis=0),
                raw_phase_delta_p84_rad=np.percentile(residual_phase, 84, axis=0),
                surface_mean_mm=surfaces.mean(axis=0), surface_std_mm=surfaces.std(axis=0, ddof=1),
                surface_p16_mm=np.percentile(surfaces, 16, axis=0),
                surface_p84_mm=np.percentile(surfaces, 84, axis=0),
                surface_trials_mm=surfaces, rms_trials_mm=metrics,
                phase_branch_change_fraction=branch_count / trials,
                nominal_aperture=nominal_aperture)

