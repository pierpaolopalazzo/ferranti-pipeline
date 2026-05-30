import numpy as np

from .config import Filters, LineState, SimulationConfig, SystemConfig
from .estimators import estimate_hilbert
from .metrics import sliding_mean_square
from .simulation import generate_delta_f_stochastic, synthesize_voltage


def ensemble_hilbert(
    seeds: list[int],
    system: SystemConfig,
    sim: SimulationConfig,
    line: LineState,
    filters: Filters,
):
    f_hat = np.empty((len(seeds), sim.n_samples))
    df_true = np.empty_like(f_hat)
    p_df = np.empty_like(f_hat)
    for row, seed in enumerate(seeds):
        df = generate_delta_f_stochastic(seed, sim, filters)
        v = synthesize_voltage(df, seed, system, sim, line)
        f = estimate_hilbert(v, system, sim, filters)
        f_hat[row] = f
        df_true[row] = df
        p_df[row] = sliding_mean_square(f - system.f0, sim.n_win)
    return df_true, f_hat, p_df
