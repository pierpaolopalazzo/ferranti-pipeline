"""Extension 1.F — end-to-end latency characterization.

This module implements the deployment-realistic (causal) variant of the
Hilbert chain and decomposes the total latency into the three structural
stages declared in the paper Sec. V:

  (a) anti-aliasing band-pass prefilter (causal lfilter, Butterworth)
  (b) sliding window of the disturbance-power metric P_Df (length T)
  (c) post-derivative smoothing on the instantaneous-frequency trace

It also provides the small step-injection helper used to measure the
empirical t_90 latency on a Df = 0.1 Hz step.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import group_delay

from .config import Filters, LineState, SimulationConfig, SystemConfig
from .estimators import (
    HILBERT_DEMOD_LOWPASS_HZ,
    HILBERT_DEMOD_LOWPASS_ORDER,
    HILBERT_SMOOTH_LEN,
)


def bandpass_group_delay_at_f0_ms(
    filters: Filters, system: SystemConfig, sim: SimulationConfig
) -> float:
    """Group delay of the band-pass prefilter evaluated at f0, in ms.

    For a 2nd-order Butterworth band-pass with +-2 Hz bandwidth around 50 Hz
    this is in the order of a few tens of ms; the exact value depends on the
    sampling rate and filter order (sim.bp_order, sim.bp_bw_hz).
    """
    w, gd = group_delay((filters.bp_b, filters.bp_a), w=4096, fs=sim.fs)
    idx = int(np.argmin(np.abs(w - system.f0)))
    return float(gd[idx] * sim.dt * 1e3)


def latency_stages(
    filters: Filters, system: SystemConfig, sim: SimulationConfig
) -> dict[str, float]:
    """Per-stage latency contributions of the causal Hilbert chain (ms).

    Stages, in order along the data path:
      - causal I/Q low-pass group delay near DC
      - 1/4-cycle quadrature reference interval at f0
      - sliding P_Df window (T/2 effective lag of a trailing window centroid)
      - causal trailing smoothing on the f_inst trace
    """
    # Low-frequency group-delay approximation for the causal I/Q demodulator.
    # For a second-order Butterworth low-pass this is close to the empirical
    # step delay, and avoids the numerical group_delay warning of the narrow
    # 50 Hz band-pass IIR.
    prefilter_ms = (
        HILBERT_DEMOD_LOWPASS_ORDER
        / (2.0 * np.pi * HILBERT_DEMOD_LOWPASS_HZ)
        * 1e3
    )
    analytic_ms = 1e3 / (4.0 * system.f0)
    window_ms = 0.5 * sim.window_s * 1e3
    smoothing_ms = 0.5 * HILBERT_SMOOTH_LEN * sim.dt * 1e3
    total_ms = prefilter_ms + analytic_ms + window_ms + smoothing_ms
    return {
        "prefilter_ms": prefilter_ms,
        "analytic_ms": analytic_ms,
        "window_ms": window_ms,
        "smoothing_ms": smoothing_ms,
        "total_ms": total_ms,
    }


def step_injection_voltage(
    step_hz: float,
    system: SystemConfig,
    sim: SimulationConfig,
    line: LineState,
    seed: int = 999,
    snr_db: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a clean step in instantaneous frequency at sim.disturbance_time_s.

    Returns (df_step, v_step) so callers can also use df_step as ground truth.
    """
    t = sim.t
    df_step = np.where(t >= sim.disturbance_time_s, step_hz, 0.0)
    rng = np.random.default_rng(seed + 100000)
    phase_mod = 2 * np.pi * np.cumsum(df_step) * sim.dt
    v_clean = line.v_peak * np.cos(2 * np.pi * system.f0 * t + phase_mod)
    snr = sim.snr_db if snr_db is None else snr_db
    noise_power = np.mean(v_clean ** 2) / (10 ** (snr / 10))
    v = v_clean + np.sqrt(noise_power) * rng.standard_normal(sim.n_samples)
    return df_step, v


def time_to_target(
    f_hat: np.ndarray,
    f0: float,
    sim: SimulationConfig,
    step_hz: float,
    fraction: float = 0.9,
) -> float:
    """Time from disturbance onset to first crossing of `fraction * step_hz` (ms).

    Returns nan if the threshold is never crossed in the post-event horizon.
    """
    target = fraction * step_hz
    post = np.where(sim.t >= sim.disturbance_time_s)[0]
    hit = np.where((f_hat[post] - f0) >= target)[0]
    if len(hit) == 0:
        return float("nan")
    return float((sim.t[post[hit[0]]] - sim.disturbance_time_s) * 1e3)
