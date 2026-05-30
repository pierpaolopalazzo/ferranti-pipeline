import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import SimulationConfig


def sliding_mean_square(signal: np.ndarray, n_win: int) -> np.ndarray:
    sq = np.asarray(signal) ** 2
    cumsum = np.cumsum(sq)
    out = np.empty_like(sq, dtype=float)
    warm = min(n_win, len(signal))
    out[:warm] = cumsum[:warm] / np.arange(1, warm + 1)
    if len(signal) > n_win:
        out[n_win:] = (cumsum[n_win:] - cumsum[:-n_win]) / n_win
    return out


def rms_sliding_window(signal: np.ndarray, n_win: int) -> np.ndarray:
    return np.sqrt(sliding_mean_square(signal, n_win))


def compute_rocof(f_inst: np.ndarray, sim: SimulationConfig) -> np.ndarray:
    rocof = np.empty_like(f_inst)
    rocof[1:-1] = (f_inst[2:] - f_inst[:-2]) / (2 * sim.dt)
    rocof[0] = (f_inst[1] - f_inst[0]) / sim.dt
    rocof[-1] = (f_inst[-1] - f_inst[-2]) / sim.dt
    return rocof


def rmse_vs_truth(df_est: np.ndarray, df_true: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((df_est[:, mask] - df_true[:, mask]) ** 2, axis=1))))


def noise_floor(df_est: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.std(df_est[:, mask], axis=1)))


def event_alarm_flags(score: np.ndarray, thresholds: np.ndarray,
                      m_samples: int) -> np.ndarray:
    """Per ogni soglia in `thresholds`, ritorna un array bool (n_real, n_th)
    che dice se la realizzazione triggererebbe l'allarme event-level
    (regola: `score > theta` per `m_samples` campioni consecutivi).

    `score` ha shape (n_real, n_samples). Lavora vettorialmente sulla griglia θ.
    """
    score = np.asarray(score)
    thresholds = np.asarray(thresholds)
    if score.size == 0 or score.shape[1] < m_samples:
        return np.zeros((score.shape[0], len(thresholds)), dtype=bool)
    if m_samples <= 1:
        # Per ogni θ: alarm se max(score) > θ.
        max_per_row = score.max(axis=1)
        return max_per_row[:, None] > thresholds[None, :]
    # Massimo della media scorrevole di lunghezza m_samples per ogni riga.
    # Una riga triggera al θ corrente se questo massimo supera θ.
    windows = sliding_window_view(score, m_samples, axis=1)
    # Una corsa di m_samples sopra θ ⇔ il minimo della finestra > θ.
    min_in_window = windows.min(axis=-1)         # (n_real, n_samples - m + 1)
    max_min = min_in_window.max(axis=1)          # (n_real,)
    return max_min[:, None] > thresholds[None, :]


def roc_event_level(score_pre: np.ndarray, score_post: np.ndarray,
                    thresholds: np.ndarray, m_samples: int):
    """Curva ROC event-level. Ritorna (FAR_percent, Detection_percent) ordinati
    per soglia crescente; entrambi vettori len(thresholds).
    """
    pre_flags = event_alarm_flags(score_pre, thresholds, m_samples)
    post_flags = event_alarm_flags(score_post, thresholds, m_samples)
    far = 100.0 * pre_flags.mean(axis=0)
    det = 100.0 * post_flags.mean(axis=0)
    return far, det


def detection_at_far_target(far: np.ndarray, det: np.ndarray,
                            far_target_pct: float = 1.0) -> float:
    """Massima detection rate fra i punti ROC con FAR ≤ far_target_pct.
    Ritorna NaN se nessuna soglia rispetta il vincolo.
    """
    mask = far <= far_target_pct
    if not np.any(mask):
        return float("nan")
    return float(det[mask].max())


def time_to_90(f_hat: np.ndarray, f0: float, sim: SimulationConfig,
               step_hz: float = 0.3) -> float:
    target = 0.9 * step_hz
    post = np.where(sim.t >= sim.disturbance_time_s)[0]
    hit = np.where((f_hat[post] - f0) >= target)[0]
    if len(hit) == 0:
        return float("nan")
    return float((sim.t[post[hit[0]]] - sim.disturbance_time_s) * 1e3)

