import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt, hilbert, lfilter

from .config import Filters, SimulationConfig, SystemConfig

HILBERT_DEMOD_LOWPASS_HZ = 5.0
HILBERT_DEMOD_LOWPASS_ORDER = 2
HILBERT_SMOOTH_LEN = 100


def _causal_analytic(
    v_filt: np.ndarray, system: SystemConfig, sim: SimulationConfig
) -> np.ndarray:
    """Causal narrow-band analytic-signal generator.

    The imaginary component is obtained from a quarter-cycle delayed copy of
    the real component. This is appropriate here because the signal is already
    tightly band-pass filtered around f0. At 50 Hz and 10 kHz, the quadrature
    delay is 50 samples = 5 ms.
    """
    quarter_cycle_samples = max(1, int(round(sim.fs / (4.0 * system.f0))))
    delay_kernel = np.zeros(quarter_cycle_samples + 1)
    delay_kernel[quarter_cycle_samples] = 1.0
    v_quadrature = lfilter(delay_kernel, [1.0], v_filt)
    return v_filt + 1j * v_quadrature


def _causal_smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing moving average of length n. Strictly causal."""
    if n <= 1:
        return x
    return lfilter(np.ones(n) / n, [1.0], x)


def _phase_frequency_causal(phase: np.ndarray, sim: SimulationConfig) -> np.ndarray:
    """Backward phase derivative in Hz."""
    f_inst = np.empty_like(phase)
    f_inst[0] = 0.0
    f_inst[1:] = np.diff(phase) / (2 * np.pi * sim.dt)
    return f_inst


def estimate_hilbert(
    v: np.ndarray, system: SystemConfig, sim: SimulationConfig, filters: Filters
) -> np.ndarray:
    """Strictly causal Hilbert-based instantaneous-frequency estimator.

    Chain: causal I/Q demodulation around f0 -> causal low-pass filtering
    of the baseband components -> backward phase derivative -> causal
    trailing-window smoothing.
    """
    t = sim.t
    lp_b, lp_a = butter(
        HILBERT_DEMOD_LOWPASS_ORDER,
        HILBERT_DEMOD_LOWPASS_HZ / (sim.fs / 2.0),
        btype="low",
    )
    i_base = 2.0 * lfilter(lp_b, lp_a, v * np.cos(2 * np.pi * system.f0 * t))
    q_base = -2.0 * lfilter(lp_b, lp_a, v * np.sin(2 * np.pi * system.f0 * t))
    z = i_base + 1j * q_base
    phase = np.unwrap(np.angle(z))
    return _causal_smooth(
        system.f0 + _phase_frequency_causal(phase, sim),
        HILBERT_SMOOTH_LEN,
    )


def estimate_hilbert_acausal(
    v: np.ndarray, system: SystemConfig, sim: SimulationConfig, filters: Filters
) -> np.ndarray:
    """Acausal Hilbert reference. Uses filtfilt + scipy.signal.hilbert and a
    centered moving-average smoother. Not realizable in deployment; used
    only as the analytical-definition reference trace in Fig. 7.
    """
    v_filt = filtfilt(filters.bp_b, filters.bp_a, v)
    z = hilbert(v_filt)
    phase = np.unwrap(np.angle(z))
    f_inst = np.empty_like(phase)
    f_inst[1:-1] = (phase[2:] - phase[:-2]) / (4 * np.pi * sim.dt)
    f_inst[0] = f_inst[1]
    f_inst[-1] = f_inst[-2]
    return np.convolve(f_inst, np.ones(HILBERT_SMOOTH_LEN) / HILBERT_SMOOTH_LEN,
                       mode="same")


def estimate_stft_vectorized(
    v: np.ndarray,
    system: SystemConfig,
    sim: SimulationConfig,
    win_cycles: int = 8,
    hop_ratio: int = 4,
    bw_search_hz: float = 5.0,
    zero_pad: int = 8,
) -> np.ndarray:
    win_len = int(round(win_cycles / system.f0 * sim.fs))
    hop = max(1, win_len // hop_ratio)
    n_frames = (len(v) - win_len) // hop + 1
    starts = np.arange(n_frames) * hop
    frames = sliding_window_view(v, win_len)[starts]
    frames = frames * np.hanning(win_len)

    nfft = zero_pad * win_len
    freqs = np.fft.rfftfreq(nfft, d=sim.dt)
    band_idx = np.where(
        (freqs >= system.f0 - bw_search_hz) & (freqs <= system.f0 + bw_search_hz)
    )[0]
    spec = np.abs(np.fft.rfft(frames, n=nfft, axis=1))[:, band_idx]
    pk = np.argmax(spec, axis=1)

    left = np.maximum(pk - 1, 0)
    right = np.minimum(pk + 1, spec.shape[1] - 1)
    rows = np.arange(n_frames)
    y0 = spec[rows, left]
    y1 = spec[rows, pk]
    y2 = spec[rows, right]
    denom = y0 - 2 * y1 + y2
    delta = np.divide(
        0.5 * (y0 - y2), denom, out=np.zeros_like(denom), where=denom != 0
    )
    delta[(pk == 0) | (pk == spec.shape[1] - 1)] = 0.0

    df_bin = freqs[1] - freqs[0]
    f_frame = freqs[band_idx[pk]] + delta * df_bin
    # Strictly causal labeling: each frame's estimate is timestamped at the
    # END of its analysis window. This forbids any look-ahead into future
    # samples, at the cost of a one-window warm-up before the first valid
    # estimate. Centroid labeling (start + win_len/2) would be standard in
    # post-processing but leaks T_win/2 of future data — unacceptable for
    # the deployment-realistic latency claim of Sec. V.
    t_frame = (starts + win_len) / sim.fs
    return np.interp(sim.t, t_frame, f_frame, left=system.f0, right=f_frame[-1])


def estimate_stft_compatible(
    v: np.ndarray,
    system: SystemConfig,
    sim: SimulationConfig,
    win_cycles: int = 8,
    hop_ratio: int = 4,
    bw_search_hz: float = 5.0,
    zero_pad: int = 8,
) -> np.ndarray:
    """Strictly causal STFT estimator (per-frame loop variant of the
    vectorized version). Each frame's frequency estimate is timestamped at
    the END of its analysis window so no future samples are used. The
    one-window warm-up returns the nominal carrier f0.
    """
    win_len = int(round(win_cycles / system.f0 * sim.fs))
    hop = max(1, win_len // hop_ratio)
    window = np.hanning(win_len)
    nfft = zero_pad * win_len
    freqs = np.fft.rfftfreq(nfft, d=sim.dt)
    band_idx = np.where(
        (freqs >= system.f0 - bw_search_hz) & (freqs <= system.f0 + bw_search_hz)
    )[0]
    n_frames = (len(v) - win_len) // hop + 1
    f_frame = np.zeros(n_frames)
    t_frame = np.zeros(n_frames)
    df_bin = freqs[1] - freqs[0]
    for k in range(n_frames):
        start = k * hop
        seg = v[start:start + win_len] * window
        spec = np.abs(np.fft.rfft(seg, n=nfft))
        local = spec[band_idx]
        pk = np.argmax(local)
        if 0 < pk < len(local) - 1:
            y0, y1, y2 = local[pk - 1], local[pk], local[pk + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        else:
            delta = 0.0
        f_frame[k] = freqs[band_idx[pk]] + delta * df_bin
        t_frame[k] = (start + win_len) / sim.fs
    return np.interp(sim.t, t_frame, f_frame, left=system.f0, right=f_frame[-1])


def estimate_zero_crossing(
    v: np.ndarray, system: SystemConfig, sim: SimulationConfig, median_len: int = 5
) -> np.ndarray:
    """Strictly causal zero-crossing frequency estimator.

    Each period estimate (1 / inter-crossing-interval) is timestamped at
    the END of its second crossing — never at the midpoint, which would
    leak half a cycle of future data. Smoothing uses a trailing rolling
    median (causal), not the centered scipy.signal.medfilt.
    """
    signs = np.sign(v)
    signs[signs == 0] = 1
    cross_idx = np.where(np.diff(signs) > 0)[0]
    if len(cross_idx) < 3:
        return np.full(len(v), system.f0)
    v0 = v[cross_idx]
    v1 = v[cross_idx + 1]
    frac = np.divide(-v0, v1 - v0, out=np.full_like(v0, 0.5), where=v1 != v0)
    t_cross = (cross_idx + frac) / sim.fs
    f_event = 1.0 / np.diff(t_cross)
    if median_len > 1 and len(f_event) >= median_len:
        windows = sliding_window_view(f_event, median_len)
        rolling = np.median(windows, axis=1)
        f_smoothed = np.empty_like(f_event)
        f_smoothed[:median_len - 1] = f_event[:median_len - 1]
        f_smoothed[median_len - 1:] = rolling
        f_event = f_smoothed
    # Causal labeling: each f_event[k] is known only after t_cross[k+1].
    t_event = t_cross[1:]
    return np.interp(sim.t, t_event, f_event, left=system.f0, right=f_event[-1])


def estimate_pll(
    v: np.ndarray,
    system: SystemConfig,
    sim: SimulationConfig,
    filters: Filters,
    bw_hz: float = 15.0,
    damping: float = 0.707,
) -> np.ndarray:
    """Strictly causal type-II PLL.

    Prefilter is causal lfilter; the analytic-signal input is generated via
    the 1/4-cycle delay trick (causal, T/4 = 5 ms group delay at f0=50 Hz)
    instead of scipy.signal.hilbert (FFT-based, acausal).
    """
    v_filt = lfilter(filters.bp_b, filters.bp_a, v)
    z = _causal_analytic(v_filt, system, sim)
    omega_n = 2 * np.pi * bw_hz
    kp = 2 * damping * omega_n
    ki = omega_n ** 2
    omega0 = 2 * np.pi * system.f0
    f_hat = np.empty(len(v))
    theta = 0.0
    integ = 0.0
    for n, sample in enumerate(z):
        err = np.angle(sample * np.exp(-1j * theta))
        integ += err * sim.dt
        omega_inst = omega0 + kp * err + ki * integ
        f_hat[n] = omega_inst / (2 * np.pi)
        theta += omega_inst * sim.dt
    return f_hat
