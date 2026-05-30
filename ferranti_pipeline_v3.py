import argparse
import sys
import io
import numpy as np

from ferranti_v3.config import (
    PlotSettings,
    SimulationConfig,
    SystemConfig,
    apply_matplotlib_style,
)
from ferranti_v3.estimators import (
    estimate_hilbert,
    estimate_hilbert_acausal,
    estimate_pll,
    estimate_stft_compatible,
    estimate_zero_crossing,
)
from ferranti_v3.latency import (
    latency_stages,
    step_injection_voltage,
    time_to_target,
)
from ferranti_v3.metrics import (
    compute_rocof,
    detection_at_far_target,
    noise_floor,
    rmse_vs_truth,
    rms_sliding_window,
    roc_event_level,
    sliding_mean_square,
    time_to_90,
)
from ferranti_v3.simulation import (
    build_filters,
    build_line_state,
    generate_delta_f_stochastic,
    synthesize_voltage,
)
from ferranti_v3.plots import (
    plot_coherent_vs_noise,
    plot_frequency_and_pdf,
    plot_latency_breakdown,
    plot_method_benchmark,
    plot_mc_convergence,
    plot_noise_resilience,
    plot_roc_detection,
    plot_rocof_comparison,
    plot_sweeps,
    plot_threshold_calibration,
)
from ferranti_v3.workflow import ensemble_hilbert


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def run_pipeline(args):
    if args.config is None:
        # No TOML preset: matplotlib defaults and direct output in figures/v3
        # unless --output-dir overrides.
        settings = PlotSettings(data={"output": {"nickname": "",
                                                 "dir": "figures/v3",
                                                 "dpi": 100,
                                                 "formats": ["png"]}})
    else:
        settings = PlotSettings.load(args.config)
        apply_matplotlib_style(settings)
    system = SystemConfig()
    sim = SimulationConfig()
    line = build_line_state(system)
    filters = build_filters(system, sim)
    output_dir = settings.output_dir(args.output_dir)

    n_mc = args.n_mc
    seeds = [42 + k for k in range(n_mc)]

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    def wanted(*figs: str) -> bool:
        return only is None or any(f in only for f in figs)

    print(f"[v3] Ferranti |V_R/V_S| = {abs(line.ferranti_ratio):.4f} "
          f"({100 * (abs(line.ferranti_ratio) - 1):+.2f}%)")
    if only is not None:
        print(f"[v3] --only filter active: {sorted(only)}")

    df_true = f_hilbert = p_hilbert = None
    needs_ensemble = wanted("001", "004", "005", "007", "008", "009", "010")
    if needs_ensemble:
        print(f"[v3] Hilbert ensemble: N_MC={n_mc}")
        df_true, f_hilbert, p_hilbert = ensemble_hilbert(seeds, system, sim, line, filters)

    if wanted("001"):
        plot_frequency_and_pdf(output_dir, system, sim, f_hilbert, p_hilbert, settings)
    if wanted("002"):
        plot_sweeps(output_dir, system, sim, line, filters, settings)
    if wanted("003"):
        plot_coherent_vs_noise(output_dir, system, sim, line, filters, settings)
    if wanted("004"):
        plot_rocof_comparison(output_dir, system, sim, f_hilbert,
                              f_hilbert - system.f0, p_hilbert, settings)

    methods = ["Hilbert", "STFT", "Zero-Cross", "PLL"]
    colors = {
        "Hilbert": settings.color("hilbert", "m"),
        "STFT": settings.color("stft", "b"),
        "Zero-Cross": settings.color("zero_cross", "g"),
        "PLL": settings.color("pll", "orange"),
    }
    t = sim.t
    mask_post = (t > sim.disturbance_time_s + 0.5) & (t < 12.0)
    mask_pre_eval = (t > 6.0) & (t < sim.disturbance_time_s - 0.1)
    mask_pre_pdf = (t > 6.5) & (t < sim.disturbance_time_s - 0.1)

    if wanted("005"):
        df_est = {"Hilbert": f_hilbert - system.f0}

        print("[v3] Method benchmark")
        f_stft = np.empty_like(f_hilbert)
        f_zc = np.empty_like(f_hilbert)
        f_pll = np.empty_like(f_hilbert)
        for row, seed in enumerate(seeds):
            v = synthesize_voltage(df_true[row], seed, system, sim, line)
            f_stft[row] = estimate_stft_compatible(v, system, sim)
            f_zc[row] = estimate_zero_crossing(v, system, sim)
            f_pll[row] = estimate_pll(v, system, sim, filters)
        df_est["STFT"] = f_stft - system.f0
        df_est["Zero-Cross"] = f_zc - system.f0
        df_est["PLL"] = f_pll - system.f0

        pdf = {method: np.vstack([sliding_mean_square(row, sim.n_win)
                                  for row in values])
               for method, values in df_est.items()}

        metrics = {}
        for method in methods:
            pre = float(np.mean(np.mean(pdf[method], axis=0)[mask_pre_pdf]))
            post = float(np.mean(np.mean(pdf[method], axis=0)[mask_post]))
            metrics[method] = {
                "rmse_mhz": 1000 * rmse_vs_truth(df_est[method], df_true, mask_post),
                "noise_mhz": 1000 * noise_floor(df_est[method], mask_pre_eval),
                "pdf_ratio": post / max(pre, 1e-15),
            }

        df_step = np.where(t >= sim.disturbance_time_s, 0.3, 0.0)
        v_step = synthesize_voltage(df_step, 999, system, sim, line)
        f_step = {
            "Hilbert": estimate_hilbert(v_step, system, sim, filters),
            "STFT": estimate_stft_compatible(v_step, system, sim),
            "Zero-Cross": estimate_zero_crossing(v_step, system, sim),
            "PLL": estimate_pll(v_step, system, sim, filters),
        }
        for method in methods:
            metrics[method]["latency_ms"] = time_to_90(f_step[method], system.f0, sim)

        plot_method_benchmark(output_dir, sim, system, methods, colors, df_est, pdf, f_step,
                              metrics, settings)

        print("\n[v3] Metriche principali")
        print(f"{'Method':<12} {'RMSE [mHz]':>12} {'Noise [mHz]':>13} "
              f"{'t90 [ms]':>10} {'P_df post/pre':>15}")
        for method in methods:
            m = metrics[method]
            print(f"{method:<12} {m['rmse_mhz']:>12.2f} {m['noise_mhz']:>13.2f} "
                  f"{m['latency_ms']:>10.1f} {m['pdf_ratio']:>14.1f}x")

    # ----- Extension 1.F: latency breakdown (Fig. 006) -----
    if wanted("006"):
        print("[v3] Latency breakdown (Ext. 1.F)")
        step_hz = 0.1
        df_step_lat, v_step_lat = step_injection_voltage(step_hz, system, sim, line)
        f_traces = {
            "Hilbert (acausal, reference)": estimate_hilbert_acausal(v_step_lat, system, sim, filters),
            "Hilbert (causal)": estimate_hilbert(v_step_lat, system, sim, filters),
            "STFT (causal)": estimate_stft_compatible(v_step_lat, system, sim),
        }
        t90_ms = {
            label: time_to_target(f_hat, system.f0, sim, step_hz)
            for label, f_hat in f_traces.items()
        }
        stages_ms = latency_stages(filters, system, sim)
        plot_latency_breakdown(output_dir, system, sim, step_hz, f_traces, t90_ms, stages_ms,
                               settings)
        print(f"           I/Q LPF     = {stages_ms['prefilter_ms']:6.1f} ms")
        print(f"           quadrature  = {stages_ms['analytic_ms']:6.1f} ms")
        print(f"           window T/2  = {stages_ms['window_ms']:6.1f} ms")
        print(f"           smoothing   = {stages_ms['smoothing_ms']:6.1f} ms")
        print(f"           total pred. = {stages_ms['total_ms']:6.1f} ms")
        for label, value in t90_ms.items():
            print(f"           t90 {label:<32s} = {value:6.1f} ms")

    # ----- Extension 1.G + 1.H: noise resilience + ROC (Fig. 007 + 010) -----
    if not wanted("007", "010"):
        if wanted("008"):
            print("[v3] Monte Carlo convergence (Ext. 1.H)")
            plot_mc_convergence(output_dir, sim, p_hilbert, settings)
        if wanted("009"):
            print("[v3] Threshold calibration (Ext. 1.M)")
            plot_threshold_calibration(output_dir, sim, p_hilbert, settings)
        print(f"\n[v3] Figure salvate in {output_dir}")
        return

    print("\n[v3] Noise resilience sweep (Ext. 1.G)")
    snr_sweep = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    noise_n_mc = args.noise_n_mc
    # Per noise_n_mc > n_mc replico ciclicamente i df_true: rumore di carico
    # idem, ma seed AWGN distinto → l'ensemble del rumore di misura cresce.
    noise_seeds = [10_000 + k for k in range(noise_n_mc)]
    df_indices = [k % n_mc for k in range(noise_n_mc)]
    print(f"           N_noise={noise_n_mc} per SNR (N_load_realizations={n_mc})")
    freq_noise_mhz = {
        method: np.empty((len(snr_sweep), noise_n_mc), dtype=float)
        for method in methods
    }
    pdf_floor_hz2 = {
        method: np.empty((len(snr_sweep), noise_n_mc), dtype=float)
        for method in methods
    }

    # ----- Estensione 1.H: accumulazione score per ROC event-level -----
    # Detector: P_Df-Hilbert, P_Df-STFT, ROCOF_RMS-Hilbert.
    # Score time-series ristretti a mask_pre_pdf (regime normale)
    # e mask_post (regime disturbato).
    n_pre = int(mask_pre_pdf.sum())
    n_post = int(mask_post.sum())
    detectors = ["P_df_Hilbert", "P_df_STFT", "ROCOF_RMS_Hilbert"]
    score_pre = {
        d: np.empty((len(snr_sweep), noise_n_mc, n_pre), dtype=float)
        for d in detectors
    }
    score_post = {
        d: np.empty((len(snr_sweep), noise_n_mc, n_post), dtype=float)
        for d in detectors
    }

    for snr_idx, snr_db in enumerate(snr_sweep):
        for row, seed in enumerate(noise_seeds):
            v = synthesize_voltage(df_true[df_indices[row]], seed, system, sim, line,
                                   snr_db=float(snr_db))
            estimates = {
                "Hilbert": estimate_hilbert(v, system, sim, filters) - system.f0,
                "STFT": estimate_stft_compatible(v, system, sim) - system.f0,
                "Zero-Cross": estimate_zero_crossing(v, system, sim) - system.f0,
                "PLL": estimate_pll(v, system, sim, filters) - system.f0,
            }
            for method in methods:
                freq_noise_mhz[method][snr_idx, row] = (
                    1000 * np.std(estimates[method][mask_pre_eval])
                )
                pdf_row = sliding_mean_square(estimates[method], sim.n_win)
                pdf_floor_hz2[method][snr_idx, row] = np.mean(pdf_row[mask_pre_pdf])

            # Score per ROC: P_Df Hilbert/STFT e ROCOF_RMS Hilbert.
            p_hilb = sliding_mean_square(estimates["Hilbert"], sim.n_win)
            p_stft_score = sliding_mean_square(estimates["STFT"], sim.n_win)
            rocof_hilb = compute_rocof(estimates["Hilbert"] + system.f0, sim)
            rocof_rms = rms_sliding_window(rocof_hilb, sim.n_win)
            score_pre["P_df_Hilbert"][snr_idx, row] = p_hilb[mask_pre_pdf]
            score_post["P_df_Hilbert"][snr_idx, row] = p_hilb[mask_post]
            score_pre["P_df_STFT"][snr_idx, row] = p_stft_score[mask_pre_pdf]
            score_post["P_df_STFT"][snr_idx, row] = p_stft_score[mask_post]
            score_pre["ROCOF_RMS_Hilbert"][snr_idx, row] = rocof_rms[mask_pre_pdf]
            score_post["ROCOF_RMS_Hilbert"][snr_idx, row] = rocof_rms[mask_post]
        print(f"           SNR {snr_db:4.0f} dB done")
    if wanted("007"):
        plot_noise_resilience(output_dir, sim, methods, colors, snr_sweep,
                              freq_noise_mhz, pdf_floor_hz2, settings)

    # ----- Estensione 1.H: ROC + detection vs SNR (Fig. 010) -----
    print("[v3] ROC event-level + detection-vs-SNR (Ext. 1.H)")
    m_event_ms = 50.0
    m_event_samples = max(1, int(round(m_event_ms * sim.fs / 1000.0)))
    far_target_pct = 1.0
    n_thresholds = 80
    threshold_grids = {}
    for d in detectors:
        all_scores = np.concatenate([
            score_pre[d].ravel(), score_post[d].ravel()
        ])
        all_scores = all_scores[np.isfinite(all_scores) & (all_scores > 0)]
        lo = np.quantile(all_scores, 0.001)
        hi = np.quantile(all_scores, 0.999)
        threshold_grids[d] = np.logspace(np.log10(lo), np.log10(hi), n_thresholds)
    roc_far = {d: np.zeros((len(snr_sweep), n_thresholds)) for d in detectors}
    roc_det = {d: np.zeros((len(snr_sweep), n_thresholds)) for d in detectors}
    detection_at_far = {d: np.zeros(len(snr_sweep)) for d in detectors}
    for snr_idx, snr_db in enumerate(snr_sweep):
        for d in detectors:
            far, det = roc_event_level(
                score_pre[d][snr_idx], score_post[d][snr_idx],
                threshold_grids[d], m_event_samples,
            )
            roc_far[d][snr_idx] = far
            roc_det[d][snr_idx] = det
            detection_at_far[d][snr_idx] = detection_at_far_target(
                far, det, far_target_pct=far_target_pct)
        print(f"           SNR {snr_db:4.0f} dB → "
              + ", ".join(f"{d}={detection_at_far[d][snr_idx]:5.1f}%"
                          for d in detectors))
    if wanted("010"):
        plot_roc_detection(output_dir, sim, snr_sweep, detectors,
                           threshold_grids, roc_far, roc_det,
                           detection_at_far, m_event_ms, far_target_pct, settings)

    if wanted("008"):
        print("[v3] Monte Carlo convergence (Ext. 1.H)")
        plot_mc_convergence(output_dir, sim, p_hilbert, settings)

    if wanted("009"):
        print("[v3] Threshold calibration (Ext. 1.M)")
        plot_threshold_calibration(output_dir, sim, p_hilbert, settings)
    print(f"\n[v3] Figure salvate in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ferranti P_Delta_f pipeline v3")
    parser.add_argument("--n-mc", type=int, default=30,
                        help="Monte Carlo realizations for the ensemble")
    parser.add_argument("--noise-n-mc", type=int, default=100,
                        help="Monte Carlo realizations per SNR for Fig. 007/010")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for v3 figures")
    parser.add_argument("--config", default=None,
                        help="TOML settings file for output and plotting style")
    parser.add_argument("--only", default=None,
                        help="Comma-separated figure numbers to (re)generate "
                             "(e.g. '004' or '001,004'). Upstream compute and "
                             "downstream stages are skipped when possible.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
