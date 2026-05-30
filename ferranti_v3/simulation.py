import numpy as np
from scipy.signal import butter, filtfilt

from .config import Filters, LineState, SimulationConfig, SystemConfig


def build_line_state(system: SystemConfig) -> LineState:
    omega0 = 2 * np.pi * system.f0
    z_per_m = system.r_per_m + 1j * omega0 * system.l_per_m
    y_per_m = system.g_per_m + 1j * omega0 * system.c_per_m
    gamma = np.sqrt(z_per_m * y_per_m)
    z0 = np.sqrt(z_per_m / y_per_m)
    ferranti_ratio = 1.0 / np.cosh(gamma * system.line_length_m)
    v_r = system.vs_rms * ferranti_ratio
    v_peak = float(np.sqrt(2) * v_r.real)
    return LineState(gamma=gamma, z0=z0, ferranti_ratio=ferranti_ratio,
                     v_r=v_r, v_peak=v_peak)


def build_filters(system: SystemConfig, sim: SimulationConfig) -> Filters:
    nyq = sim.fs / 2
    subhz_b, subhz_a = butter(
        2, [sim.subhz_low / nyq, sim.subhz_high / nyq], btype="band"
    )
    bp_b, bp_a = butter(
        sim.bp_order,
        [(system.f0 - sim.bp_bw_hz) / nyq, (system.f0 + sim.bp_bw_hz) / nyq],
        btype="band",
    )
    return Filters(subhz_b=subhz_b, subhz_a=subhz_a, bp_b=bp_b, bp_a=bp_a)


def ferranti_profile(system: SystemConfig, line: LineState, n_points: int = 500):
    x = np.linspace(0, system.line_length_m, n_points)
    v_x = line.v_r * np.cosh(line.gamma * (system.line_length_m - x))
    return x, np.abs(v_x) / system.vs_rms


def raised_cosine_activation(t: np.ndarray, start_s: float, ramp_s: float = 0.3):
    activation = np.zeros_like(t)
    end = start_s + ramp_s
    ramp = (t > start_s) & (t < end)
    activation[t >= end] = 1.0
    activation[ramp] = 0.5 * (1 - np.cos(np.pi * (t[ramp] - start_s) / ramp_s))
    return activation


def generate_delta_f_stochastic(
    seed: int, sim: SimulationConfig, filters: Filters
) -> np.ndarray:
    t = sim.t
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(sim.n_samples)
    bandlimited = filtfilt(filters.subhz_b, filters.subhz_a, white)
    bandlimited *= sim.df_std_target / np.std(bandlimited)
    return bandlimited * raised_cosine_activation(t, sim.disturbance_time_s)


def generate_delta_f_swing(
    seed: int, h_eq: float, system: SystemConfig, sim: SimulationConfig, filters: Filters
) -> np.ndarray:
    t = sim.t
    rng = np.random.default_rng(seed + 200000)
    white = rng.standard_normal(sim.n_samples)
    d_p = filtfilt(filters.subhz_b, filters.subhz_a, white)
    d_p *= sim.load_mod_pct / np.std(d_p)
    d_p *= raised_cosine_activation(t, sim.disturbance_time_s)
    df_pu = -np.cumsum(d_p) * sim.dt / (2 * h_eq)
    df_pu -= np.mean(df_pu[t > sim.disturbance_time_s + 1.0])
    return df_pu * system.f0


def synthesize_voltage(
    delta_f: np.ndarray,
    seed: int,
    system: SystemConfig,
    sim: SimulationConfig,
    line: LineState,
    snr_db: float | None = None,
) -> np.ndarray:
    t = sim.t
    rng = np.random.default_rng(seed + 100000)
    phase_mod = 2 * np.pi * np.cumsum(delta_f) * sim.dt
    v_clean = line.v_peak * np.cos(2 * np.pi * system.f0 * t + phase_mod)
    snr = sim.snr_db if snr_db is None else snr_db
    signal_power = np.mean(v_clean ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    return v_clean + np.sqrt(noise_power) * rng.standard_normal(len(delta_f))
