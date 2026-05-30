import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import filtfilt

from .config import Filters, LineState, PlotSettings, SimulationConfig, SystemConfig
from .estimators import estimate_hilbert
from .metrics import compute_rocof, rms_sliding_window, sliding_mean_square
from .simulation import generate_delta_f_swing, raised_cosine_activation, synthesize_voltage


def plot_latency_breakdown(
    output_dir: Path,
    system: SystemConfig,
    sim: SimulationConfig,
    step_hz: float,
    f_traces: dict,
    t90_ms: dict,
    stages_ms: dict,
    settings: PlotSettings,
):
    """Figure 006 — end-to-end latency breakdown (Extension 1.F).

    Two-panel composition:
      (a) Step response: superimposed f_hat(t) of three estimators on a
          step Df = step_hz, with markers at t_90.
      (b) Bar chart of per-stage latency for the causal I/Q Hilbert chain
          (baseband low-pass delay, window T/2, smoothing) plus the
          empirical end-to-end value, side-by-side with STFT and the
          acausal filtfilt-Hilbert reference.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t = sim.t
    zoom = (t >= sim.disturbance_time_s - 0.05) & (t <= sim.disturbance_time_s + 0.40)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=settings.figsize("006", (13, 4.8)))
    grid_alpha = settings.style("grid_alpha", 0.3)

    color_map = {
        "Hilbert (acausal, reference)": ("k", "--"),
        "Hilbert (causal)": (settings.color("hilbert", "m"), "-"),
        "STFT (causal)": (settings.color("stft", "b"), "-"),
    }
    for label, f_hat in f_traces.items():
        color, ls = color_map.get(label, ("gray", "-"))
        ax_a.plot(
            t[zoom] - sim.disturbance_time_s,
            (f_hat - system.f0)[zoom],
            color=color,
            linestyle=ls,
            linewidth=settings.line_width("006", thick=True),
            label=label,
        )
        if label in t90_ms and not np.isnan(t90_ms[label]):
            ax_a.axvline(
                t90_ms[label] * 1e-3,
                color=color,
                linestyle=":",
                alpha=0.5,
                linewidth=0.9,
            )
    ax_a.axhline(step_hz, color="k", linestyle="--", alpha=0.45,
                 label=f"$\\Delta f$ step ({step_hz:.2f} Hz)")
    ax_a.axhline(0.9 * step_hz, color="gray", linestyle=":", alpha=0.55,
                 label="90% target")
    ax_a.axvline(0, color=settings.color("event", "r"), linestyle=":",
                 alpha=settings.style("event_alpha", 0.7))
    ax_a.set_xlabel("$t - t_d$ [s]")
    ax_a.set_ylabel("$\\hat{\\Delta f}$ [Hz]")
    ax_a.set_xlim(-0.05, 0.40)
    ax_a.set_ylim(-0.02, step_hz * 1.35)
    set_panel_title(ax_a, f"Step response on $\\Delta f$ = {step_hz:.2f} Hz "
                    "(SNR 50 dB)", "(a)", settings)
    ax_a.legend(loc="lower right",
                fontsize=settings.style("legend_font_size", 8),
                frameon=False)
    ax_a.grid(True, alpha=grid_alpha)

    stage_labels = [
        "IQ LPF",
        "Quad.",
        "$P_{\\Delta f}$",
        "Avg.",
        "Pred.",
        "t$_{90}$",
    ]
    causal_hilbert_label = "Hilbert (causal)"
    measured = t90_ms.get(causal_hilbert_label, float("nan"))
    bar_values = [
        stages_ms["prefilter_ms"],
        stages_ms["analytic_ms"],
        stages_ms["window_ms"],
        stages_ms["smoothing_ms"],
        stages_ms["total_ms"],
        measured,
    ]
    bar_colors = ["steelblue", "purple", "mediumseagreen", "goldenrod", "gray", "salmon"]
    x_pos = np.arange(len(stage_labels))
    bars = ax_b.bar(x_pos, bar_values, color=bar_colors,
                    width=0.65, edgecolor="k", linewidth=0.5)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(stage_labels, fontsize=settings.style("tick_label_size", 8),
                         rotation=20, ha="right")
    ax_b.set_ylabel("Latency [ms]")
    set_panel_title(ax_b, "Causal I/Q Hilbert chain — latency stage breakdown",
                    "(b)", settings)
    ax_b.grid(True, alpha=grid_alpha, axis="y")
    bar_label_size = settings.style("bar_label_size", 9)
    for bar, value in zip(bars, bar_values):
        if not np.isnan(value):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center", va="bottom",
                fontsize=bar_label_size,
            )
    # Reference horizontal lines: STFT and acausal Hilbert measured t90
    ref_y = 0.95
    for label, color, ls in [
        ("STFT (causal)", settings.color("stft", "b"), "--"),
        ("Hilbert (acausal, reference)", "k", ":"),
    ]:
        v = t90_ms.get(label)
        if v is not None and not np.isnan(v):
            ax_b.axhline(v, color=color, linestyle=ls, linewidth=1.0,
                         alpha=0.7)
            ax_b.text(
                0.04, ref_y, f"{label}: {v:.1f} ms",
                transform=ax_b.transAxes,
                color=color,
                fontsize=settings.style("legend_font_size", 8),
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
            )
            ref_y -= 0.08
    ax_b.set_ylim(0, max(bar_values) * 1.35)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "006-latency_breakdown",
        settings,
        {
            "figure_number": "006",
            "latex_label": "fig:latency",
            "short_title": "End-to-end latency breakdown",
            "caption": (
                "End-to-end latency characterization (Extension 1.F). "
                "(a) Step response on a $\\Delta f$ = "
                f"{step_hz:.2f}"
                " Hz injection at $t_d$ for the causal I/Q Hilbert chain, "
                "the STFT chain, and the acausal filtfilt-Hilbert reference. "
                "(b) Per-stage latency decomposition of the causal Hilbert "
                "chain: baseband low-pass delay, quadrature reference interval, "
                "sliding-window $T/2$, "
                "and post-derivative smoothing. The predicted total matches the "
                "measured $t_{90}$ on the step injection."
            ),
            "notes": (
                "The acausal filtfilt baseline is reported only as analytical "
                "reference; it is not realizable in deployment."
            ),
            "metrics": {
                "step_hz": step_hz,
                "stages_ms": stages_ms,
                "t90_ms": t90_ms,
            },
        },
    )
    plt.close(fig)

def save_figure(
    fig,
    output_dir: Path,
    stem: str,
    settings: PlotSettings,
    metadata: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = settings.section("output").get("formats", ["png", "svg"])
    for fmt in formats:
        if fmt == "json":
            payload = {
                "file_stem": stem,
                "outputs": [f"{stem}.{f}" for f in formats if f != "json"],
                **(metadata or {}),
            }
            (output_dir / f"{stem}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            continue
        # pad_inches=0.02 trims the residual ~0.1" whitespace that
        # bbox_inches='tight' leaves around the content; the actual
        # figure breathing room is then controlled by the LaTeX
        # \abovecaptionskip / \textfloatsep, not by per-figure padding.
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if fmt == "png":
            kwargs["dpi"] = settings.dpi()
        fig.savefig(output_dir / f"{stem}.{fmt}", **kwargs)


def legend_outside(ax, settings: PlotSettings, **overrides):
    """Place legend in the right margin, frameless. Avoids overlapping panel labels."""
    kwargs = dict(loc="center left", bbox_to_anchor=(1.005, 0.5),
                  fontsize=settings.style("legend_font_size", 8),
                  frameon=False, borderaxespad=0.0)
    kwargs.update(overrides)
    return ax.legend(**kwargs)


def rotate_legend_text(legend, angle: float = 90.0) -> None:
    """Rotate legend text labels by `angle` degrees (counter-clockwise)."""
    for txt in legend.get_texts():
        txt.set_rotation(angle)
        txt.set_verticalalignment("center")


def vertical_strip_legend(ax, entries, settings,
                          x: float = 1.03,
                          handle_height: float = 0.10,
                          text_pad: float = 0.015,
                          entry_gap: float = 0.12) -> None:
    """Draw entries stacked vertically in a single narrow column at the right margin.

    Each entry occupies a vertical slot: a short vertical handle (line sample)
    sits at the bottom of the slot, with the rotated text label above it.
    Reading bottom-to-top within each entry: handle, then label.

    `entries`: iterable of (color, linestyle, label) tuples. The first entry
    appears at the bottom, the last at the top.
    Coordinates are in axes-fraction space; artists are drawn with
    clip_on=False so they appear outside the axes box.
    """
    lw = settings.style("line_width", 1.2)
    fs = settings.style("legend_font_size", 8)
    n = len(entries)
    slot = 1.0 / n
    for i, (color, ls, label) in enumerate(entries):
        y0 = i * slot + 0.02 + i * entry_gap
        y_handle_end = y0 + handle_height
        y_text = y_handle_end + text_pad
        ax.plot([x, x], [y0, y_handle_end], color=color, linestyle=ls,
                linewidth=lw, transform=ax.transAxes, clip_on=False)
        ax.text(x, y_text, label, transform=ax.transAxes,
                rotation=90, va="bottom", ha="center", fontsize=fs)


def set_panel_title(ax, title: str, panel: str, settings: PlotSettings,
                    fig: str | None = None) -> None:
    if settings.style("show_titles", True, fig=fig):
        ax.set_title(title)
    if settings.style("show_panel_labels", False, fig=fig):
        ax.text(
            settings.style("panel_label_x", 0.02, fig=fig),
            settings.style("panel_label_y", 0.92, fig=fig),
            panel,
            transform=ax.transAxes,
            fontsize=settings.style("panel_label_size", 9, fig=fig),
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )


def rolling_envelope(y: np.ndarray, window_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if window_samples <= 1:
        return y, y
    pad_left = window_samples // 2
    pad_right = window_samples - 1 - pad_left
    y_pad = np.pad(y, (pad_left, pad_right), mode="edge")
    windows = sliding_window_view(y_pad, window_samples)
    return np.min(windows, axis=1), np.max(windows, axis=1)


def format_sci3_math(value: float) -> str:
    if value == 0 or not np.isfinite(value):
        return f"{value:.3g}"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.3g}$\\cdot 10^{{{exponent}}}$"


def plot_voltage_profile(
    output_dir: Path,
    system: SystemConfig,
    line: LineState,
    x: np.ndarray,
    voltage_profile: np.ndarray,
    settings: PlotSettings,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize("001", (7.16, 2.4)))
    grid_alpha = settings.style("grid_alpha", 0.3)
    overvoltage_pct = 100 * (voltage_profile - 1.0)
    ax.plot(x / 1e3, overvoltage_pct, "b-",
            linewidth=settings.line_width("001", thick=True))
    ax.axhline(0.0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("Distance from sending end [km]")
    ax.set_ylabel("Overvoltage [%]")
    set_panel_title(ax, f"Fig. 1 — Ferranti overvoltage profile along 300 km line "
                    f"($|V_R/V_S|$ = {abs(line.ferranti_ratio):.3f})", "(a)", settings)
    ax.grid(True, alpha=grid_alpha)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "001-ferranti_voltage_profile",
        settings,
        {
            "figure_number": "001",
            "latex_label": "fig:ferranti-voltage-profile",
            "short_title": "Ferranti voltage profile",
            "caption": (
                "Overvoltage profile along the 300 km line illustrating the "
                "steady-state Ferranti rise relative to the sending-end voltage."
            ),
            "notes": "Spatial-only figure; separated from the time-domain diagnostic to keep axes consistent.",
        },
    )
    plt.close(fig)


def plot_frequency_and_pdf(
    output_dir: Path,
    system: SystemConfig,
    sim: SimulationConfig,
    f_hat: np.ndarray,
    p_df: np.ndarray,
    settings: PlotSettings,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    t = sim.t
    view = (t >= 6.0) & (t <= 12.0)
    p_mean = np.mean(p_df, axis=0)
    p10 = np.percentile(p_df, 10, axis=0)
    p90 = np.percentile(p_df, 90, axis=0)

    ramp_s = 0.3  # matches default in simulation.raised_cosine_activation
    t_ramp_end = sim.disturbance_time_s + ramp_s

    fig, axes = plt.subplots(2, 1, figsize=settings.figsize("001", (7.16, 4.2)))
    grid_alpha = settings.style("grid_alpha", 0.3)

    axes[0].plot(t[view], f_hat[0, view], color=settings.color("hilbert", "b"),
                 linewidth=settings.line_width("001"),
                 label="$f_i(t)$ estimated")
    axes[0].axhline(system.f0, color="k", linestyle="--", alpha=0.6,
                    label="$f_0$ = 50 Hz")
    axes[0].axvline(sim.disturbance_time_s, color=settings.color("event", "r"),
                    linestyle=":", alpha=settings.style("event_alpha", 0.7),
                    label=f"$t_d$ = {sim.disturbance_time_s:.1f} s")
    axes[0].axvline(t_ramp_end, color=settings.color("event", "r"),
                    linestyle="--", alpha=0.4,
                    label=f"ramp end ($t_d$ + {int(ramp_s*1000)} ms)")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Inst. frequency [Hz]")
    axes[0].set_ylim(49.5, 50.5)
    set_panel_title(axes[0], "Extracted instantaneous frequency $f_i(t)$ "
                    "at the receiving end", "(a)", settings)
    legend_outside(axes[0], settings)
    axes[0].grid(True, alpha=grid_alpha)

    pdf_color = settings.color("pdf", "m")
    axes[1].fill_between(t[view], p10[view], p90[view], color=pdf_color,
                         alpha=settings.style("band_alpha", 0.15),
                         label=f"10-90% band (N={len(p_df)})")
    axes[1].plot(t[view], p_df[0, view], color=pdf_color, alpha=0.35,
                 linewidth=settings.line_width("001") * 0.67,
                 label="single realization")
    axes[1].plot(t[view], p_mean[view], color=pdf_color,
                 linewidth=settings.line_width("001", thick=True),
                 label="ensemble mean")
    axes[1].axhline(0.04, color=settings.color("threshold", "r"), linestyle="--",
                    alpha=settings.style("event_alpha", 0.7),
                    label="$\\theta_{\\rm op}$ = 0.04 Hz$^2$")
    axes[1].axvline(sim.disturbance_time_s, color=settings.color("event", "r"),
                    linestyle=":", alpha=settings.style("event_alpha", 0.7),
                    label="Disturbance onset")
    axes[1].axvline(t_ramp_end, color=settings.color("event", "r"),
                    linestyle="--", alpha=0.4,
                    label=f"ramp end ($t_d$ + {int(ramp_s*1000)} ms)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("$P_{\\Delta f}$ [Hz$^2$]")
    axes[1].set_ylim(0, 0.1)
    set_panel_title(axes[1], "Temporal evolution of disturbance power "
                    f"$P_{{\\Delta f}}$ (ensemble of {len(p_df)} stochastic realizations)",
                    "(b)", settings)
    legend_outside(axes[1], settings)
    axes[1].grid(True, alpha=grid_alpha)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "001-frequency_and_pdf",
        settings,
        {
            "figure_number": "001",
            "latex_label": "fig:frequency_pdf",
            "short_title": "Instantaneous frequency and disturbance power",
            "caption": (
                "Time-domain diagnostic after stochastic low-inertia modulation. "
                "(a) Hilbert-estimated instantaneous frequency at the receiving end. "
                "(b) Sliding-window disturbance power $P_{\\Delta f}$ with ensemble "
                "spread and operational severity threshold $\\theta_{\\rm op}$. "
                "The disturbance is activated at $t_d$ through a causal "
                "raised-cosine ramp of 300\\,ms (dashed marker at $t_d$ + 300\\,ms); "
                "the visible delay between $t_d$ and the $P_{\\Delta f}$ rise is the "
                "sum of the ramp, the causal Hilbert/IQ chain group delay, and the "
                "sliding-window aggregation, decomposed quantitatively in Fig.~6."
            ),
            "notes": "First active figure in the paper-ready sequence.",
        },
    )
    plt.close(fig)


def plot_sweeps(output_dir: Path, system: SystemConfig, sim: SimulationConfig,
                line: LineState, filters: Filters, settings: PlotSettings):
    h_sweep = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0])
    p_vs_h = []
    for h_val in h_sweep:
        vals = []
        for k in range(15):
            seed = 42 + k
            df = generate_delta_f_swing(seed, h_val, system, sim, filters)
            v = synthesize_voltage(df, seed + 200000, system, sim, line)
            f = estimate_hilbert(v, system, sim, filters)
            p = sliding_mean_square(f - system.f0, sim.n_win)
            vals.append(np.mean(p[(sim.t > sim.disturbance_time_s + 0.5)
                                  & (sim.t < sim.duration_s - 0.2)]))
        p_vs_h.append(np.mean(vals))
    p_vs_h = np.array(p_vs_h)
    alpha = -np.polyfit(np.log(h_sweep), np.log(p_vs_h), 1)[0]

    fig, ax = plt.subplots(figsize=settings.figsize("002", (6.5, 4.5)))
    grid_alpha = settings.style("grid_alpha", 0.3)
    ax.plot(h_sweep, p_vs_h, "mo-", linewidth=settings.line_width("002", thick=True),
            markersize=settings.style("marker_size", 6),
            label="P$_{\\Delta f}$ plateau")
    anchor_idx = int(np.argmin(np.abs(h_sweep - 5.0)))
    ax.plot(h_sweep, p_vs_h[anchor_idx] * (h_sweep[anchor_idx] / h_sweep), "k--",
            linewidth=1, alpha=0.6, label="$\\propto 1/H$ reference")
    ax.set_xlabel("Inertia constant H [s]")
    ax.set_ylabel("P$_{\\Delta f}$ plateau [Hz$^2$]")
    set_panel_title(ax, "P$_{\\Delta f}$ vs system inertia\n"
                    "(swing equation, fit: P$_{\\Delta f} \\propto 1/H^2$)",
                    "(a)", settings)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(h_sweep)
    ax.set_xticklabels([f"{h:g}" for h in h_sweep])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.03, 0.05, 0.1, 0.5, 1.0])
    ax.set_yticklabels(["0.03", "0.05", "0.1", "0.5", "1"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=grid_alpha)
    ax.legend(loc="lower left", fontsize=settings.style("legend_font_size", 8),
              framealpha=0.9)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "002-inertia_sensitivity",
        settings,
        {
            "figure_number": "002",
            "latex_label": "fig:sweeps",
            "short_title": "Inertia sweep",
            "caption": (
                "Disturbance-power plateau versus equivalent inertia (N$_{MC}$=15 "
                "stochastic realizations per H value), showing that lower inertia "
                "increases the diagnostic significance of $P_{\\Delta f}$ in the "
                "considered Ferranti scenario."
            ),
            "notes": (
                "The previous line-length panel was removed from the paper "
                "because the Ferranti voltage-rise profile is treated as known."
            ),
            "metrics": {
                "h_sweep_s": h_sweep.tolist(),
                "p_delta_f_plateau_hz2": p_vs_h.tolist(),
                "fit_alpha": float(alpha),
            },
        },
    )
    plt.close(fig)


def plot_coherent_vs_noise(output_dir: Path, system: SystemConfig, sim: SimulationConfig,
                           line: LineState, filters: Filters, settings: PlotSettings):
    t = sim.t
    activation = raised_cosine_activation(t, sim.disturbance_time_s)
    a_target = 0.4 / np.sqrt(2)
    df_a = a_target * np.sin(2 * np.pi * 0.3 * (t - sim.disturbance_time_s)) * activation

    rng = np.random.default_rng(seed=99)
    df_b = filtfilt(filters.subhz_b, filters.subhz_a, rng.standard_normal(sim.n_samples))
    mask_cal = (t > sim.disturbance_time_s + 0.5) & (t < sim.duration_s - 0.2)
    df_b *= np.sqrt((a_target ** 2 / 2) / np.var(df_b[mask_cal]))
    df_b *= activation

    def run_case(df, seed_noise):
        rng = np.random.default_rng(seed_noise + 400000)
        phase_mod = 2 * np.pi * np.cumsum(df) * sim.dt
        v_clean = line.v_peak * np.cos(2 * np.pi * system.f0 * t + phase_mod)
        noise_power = np.mean(v_clean ** 2) / (10 ** (sim.snr_db / 10))
        v = v_clean + np.sqrt(noise_power) * rng.standard_normal(sim.n_samples)
        f = estimate_hilbert(v, system, sim, filters)
        return sliding_mean_square(f - system.f0, sim.n_win)

    p_a = run_case(df_a, 1)
    p_b = run_case(df_b, 2)
    mask_plat = (t > sim.disturbance_time_s + 0.5) & (t < sim.duration_s - 0.2)
    p_a_mean, p_a_std = np.mean(p_a[mask_plat]), np.std(p_a[mask_plat])
    p_b_mean, p_b_std = np.mean(p_b[mask_plat]), np.std(p_b[mask_plat])

    view = (t >= 6.0) & (t <= 12.0)
    fig, axes = plt.subplots(2, 2, figsize=settings.figsize("003", (13, 7)))
    grid_alpha = settings.style("grid_alpha", 0.3)
    axes[0, 0].plot(t[view], df_a[view], "b-", linewidth=settings.line_width("003"))
    axes[0, 0].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("$\\Delta f$ [Hz]")
    set_panel_title(axes[0, 0], "Case A — Coherent $\\Delta f$ (sinusoidal, 0.3 Hz)",
                    "(a)", settings)
    axes[0, 0].grid(True, alpha=grid_alpha)
    axes[0, 0].set_ylim(-0.65, 0.4)

    axes[0, 1].plot(t[view], df_b[view], "g-", linewidth=settings.line_width("003"))
    axes[0, 1].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7)
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("$\\Delta f$ [Hz]")
    set_panel_title(axes[0, 1], "Case B — Stochastic $\\Delta f$ (band-limited [0.05, 1] Hz)",
                    "(b)", settings)
    axes[0, 1].grid(True, alpha=grid_alpha)
    axes[0, 1].set_ylim(-0.65, 0.4)

    ymax = max(p_a_mean + 4 * p_a_std, p_b_mean + 4 * p_b_std)
    axes[1, 0].plot(t[view], p_a[view], "b-", linewidth=settings.line_width("003", thick=True),
                    label="$P_{\\Delta f}(t)$")
    axes[1, 0].axhline(p_a_mean, color="b", linestyle="--", alpha=0.6,
                       label=f"mean = {p_a_mean:.3f} Hz$^2$")
    axes[1, 0].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("$P_{\\Delta f}$ [Hz²]")
    set_panel_title(axes[1, 0],
                    f"Case A — $P_{{\\Delta f}}$: oscillates at $2 f_{{osc}}$ = 0.6 Hz\n"
                    f"(coeff. var. = {p_a_std / p_a_mean:.2f})", "(c)", settings)
    axes[1, 0].grid(True, alpha=grid_alpha)
    axes[1, 0].set_ylim(-0.1, 0.4)
    vertical_strip_legend(axes[1, 0], [
        ("b", "--", f"mean = {p_a_mean:.3f} Hz$^2$"),
        ("b", "-", "$P_{\\Delta f}(t)$"),
    ], settings)

    axes[1, 1].plot(t[view], p_b[view], "g-", linewidth=settings.line_width("003", thick=True),
                    label="$P_{\\Delta f}(t)$")
    axes[1, 1].axhline(p_b_mean, color="g", linestyle="--", alpha=0.6,
                       label=f"mean = {p_b_mean:.3f} Hz$^2$")
    axes[1, 1].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("$P_{\\Delta f}$ [Hz²]")
    set_panel_title(axes[1, 1],
                    f"Case B — $P_{{\\Delta f}}$: plateau (random fluctuations)\n"
                    f"(coeff. var. = {p_b_std / p_b_mean:.2f})", "(d)", settings)
    axes[1, 1].grid(True, alpha=grid_alpha)
    axes[1, 1].set_ylim(-0.1, 0.4)
    vertical_strip_legend(axes[1, 1], [
        ("g", "--", f"mean = {p_b_mean:.3f} Hz$^2$"),
        ("g", "-", "$P_{\\Delta f}(t)$"),
    ], settings)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "003-coherent_vs_stochastic",
        settings,
        {
            "figure_number": "003",
            "latex_label": "fig:coherent_stochastic",
            "short_title": "Coherent versus stochastic frequency modulation",
            "caption": (
                "Comparison between coherent sinusoidal and stochastic band-limited "
                "frequency deviations with matched variance. The mean disturbance "
                "power is comparable, while the temporal structure changes the "
                "appearance of the $P_{\\Delta f}$ trajectory."
            ),
            "notes": "Useful as support material; likely not essential for the 6-page final paper.",
        },
    )
    plt.close(fig)


def plot_rocof_comparison(output_dir: Path, system: SystemConfig, sim: SimulationConfig,
                          f_hilbert: np.ndarray, df_hilbert: np.ndarray,
                          p_hilbert: np.ndarray, settings: PlotSettings):
    t = sim.t
    view = (t >= 6.0) & (t <= 12.0)
    rocof_rms = np.vstack([rms_sliding_window(compute_rocof(row, sim), sim.n_win)
                           for row in f_hilbert])
    fdev_rms = np.vstack([rms_sliding_window(row, sim.n_win) for row in df_hilbert])
    p_mean = np.mean(p_hilbert, axis=0)
    p10 = np.percentile(p_hilbert, 10, axis=0)
    p90 = np.percentile(p_hilbert, 90, axis=0)
    rocof_mean = np.mean(rocof_rms, axis=0)
    fdev_mean = np.mean(fdev_rms, axis=0)

    fig, axes = plt.subplots(4, 1, figsize=settings.figsize("004", (10, 11)), sharex=True)
    grid_alpha = settings.style("grid_alpha", 0.3)
    axes[0].plot(t[view], p_mean[view], color=settings.color("pdf", "m"),
                 linewidth=settings.line_width("004", thick=True),
                 label="$P_{\\Delta f}$ (ensemble mean)")
    axes[0].fill_between(t[view], p10[view], p90[view], color=settings.color("pdf", "m"),
                         alpha=settings.style("band_alpha", 0.15),
                         label="10-90% band")
    axes[0].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7)
    axes[0].set_ylabel("\n$P_{\\Delta f}$ [Hz²]")
    set_panel_title(axes[0], "Fig. 4a — Disturbance power $P_{\\Delta f}(t)$",
                    "(a)", settings, fig="004")
    legend_outside(axes[0], settings)
    axes[0].grid(True, alpha=grid_alpha)

    axes[1].plot(t[view], fdev_mean[view], color=settings.color("freq_rms", "b"),
                 linewidth=settings.line_width("004", thick=True),
                 label="Freq. deviation RMS = $\\sqrt{P_{\\Delta f}}$")
    axes[1].fill_between(t[view], np.percentile(fdev_rms, 10, axis=0)[view],
                         np.percentile(fdev_rms, 90, axis=0)[view], color="b",
                         alpha=0.15, label="10-90% band")
    axes[1].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7)
    axes[1].set_ylabel("\nRMS $\\Delta f$ [Hz]")
    set_panel_title(axes[1],
                    "Fig. 4b — Frequency deviation RMS (algebraically = $\\sqrt{P_{\\Delta f}}$)",
                    "(b)", settings, fig="004")
    legend_outside(axes[1], settings)
    axes[1].grid(True, alpha=grid_alpha)

    axes[2].plot(t[view], rocof_mean[view], color=settings.color("rocof", "g"),
                 linewidth=settings.line_width("004", thick=True),
                 label="ROCOF$_{\\mathrm{RMS}}$ (ensemble mean)")
    axes[2].fill_between(t[view], np.percentile(rocof_rms, 10, axis=0)[view],
                         np.percentile(rocof_rms, 90, axis=0)[view], color="g",
                         alpha=0.15, label="10-90% band")
    axes[2].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7)
    axes[2].set_ylabel("\nROCOF$_{\\mathrm{RMS}}$ [Hz/s]")
    set_panel_title(axes[2], "Fig. 4c — Rate of Change of Frequency (ROCOF) RMS",
                    "(c)", settings, fig="004")
    legend_outside(axes[2], settings)
    axes[2].grid(True, alpha=grid_alpha)

    base = (t >= 6.0) & (t < sim.disturbance_time_s - 0.5)
    axes[3].plot(t[view], p_mean[view] / np.mean(p_mean[base]), "m-", linewidth=2,
                 label="$P_{\\Delta f}$ / baseline")
    axes[3].plot(t[view], fdev_mean[view] / np.mean(fdev_mean[base]), "b-",
                 linewidth=2, label="Freq. dev. RMS / baseline")
    axes[3].plot(t[view], rocof_mean[view] / np.mean(rocof_mean[base]), "g-",
                 linewidth=2, label="ROCOF$_{\\mathrm{RMS}}$ / baseline")
    axes[3].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7)
    axes[3].axhline(1.0, color="k", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Normalized\n(×baseline)")
    set_panel_title(axes[3], "Fig. 4d — Normalized comparison: sensitivity to sub-Hz disturbance",
                    "(d)", settings, fig="004")
    axes[3].set_yscale("log")
    legend_outside(axes[3], settings)
    axes[3].grid(True, alpha=grid_alpha)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "004-comparison_pdf_rocof",
        settings,
        {
            "figure_number": "004",
            "latex_label": "fig:rocof",
            "short_title": "P_Delta_f, frequency RMS, and ROCOF comparison",
            "caption": (
                "Comparison of disturbance power, frequency-deviation RMS, and "
                "ROCOF RMS under the same stochastic Ferranti scenario "
                f"(ensemble of {len(f_hilbert)} realizations, "
                f"sliding window = {sim.window_s * 1e3:.0f} ms). "
                "(a) Disturbance power $P_{\\Delta f}(t)$; "
                "(b) frequency-deviation RMS, algebraically $\\sqrt{P_{\\Delta f}}$; "
                "(c) rate-of-change-of-frequency (ROCOF) RMS; "
                "(d) normalized comparison highlighting sensitivity to sub-Hz "
                "disturbance. "
                "The metrics are complementary: $P_{\\Delta f}$ captures sustained "
                "low-frequency disturbance energy, whereas ROCOF emphasizes faster "
                "frequency variations."
            ),
            "notes": "Consider reducing to fewer panels if space is limited.",
        },
    )
    plt.close(fig)


def plot_method_benchmark(output_dir: Path, sim: SimulationConfig, system: SystemConfig,
                          methods: list[str], colors: dict[str, str],
                          df_est: dict[str, np.ndarray],
                          pdf: dict[str, np.ndarray],
                          f_step: dict[str, np.ndarray],
                          metrics: dict[str, dict[str, float]],
                          settings: PlotSettings):
    t = sim.t
    view = (t >= 6.0) & (t <= 12.0)
    zoom = (t >= sim.disturbance_time_s - 0.05) & (t <= sim.disturbance_time_s + 0.25)
    fig, axes = plt.subplots(4, 1, figsize=settings.figsize("005", (10, 14)))
    grid_alpha = settings.style("grid_alpha", 0.3)
    method_keys = {"Hilbert": "hilbert", "STFT": "stft",
                   "Zero-Cross": "zero_cross", "PLL": "pll"}
    linestyles = {m: settings.linestyle(method_keys.get(m, ""), "-") for m in methods}
    legend_kwargs = dict(loc="center left", bbox_to_anchor=(1.005, 0.5),
                         fontsize=settings.style("legend_font_size", 8),
                         frameon=False, borderaxespad=0.0)

    for method in methods:
        f_mean = np.mean(df_est[method] + system.f0, axis=0)
        if method == "PLL":
            env_lo, env_hi = rolling_envelope(f_mean, int(round(0.20 * sim.fs)))
            axes[0].plot(t[view], f_mean[view],
                         color=colors[method], linestyle=linestyles[method],
                         linewidth=settings.line_width("005") * 0.7,
                         label=method, alpha=0.22)
            axes[0].plot(t[view], env_lo[view],
                         color=colors[method], linestyle=linestyles[method],
                         linewidth=settings.line_width("005"),
                         alpha=0.95)
            axes[0].plot(t[view], env_hi[view],
                         color=colors[method], linestyle=linestyles[method],
                         linewidth=settings.line_width("005"),
                         alpha=0.95)
            continue
        axes[0].plot(t[view], f_mean[view],
                     color=colors[method], linestyle=linestyles[method],
                     linewidth=settings.line_width("005"),
                     label=method, alpha=0.9)
    axes[0].axhline(system.f0, color="k", linestyle="--", alpha=0.5)
    axes[0].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7,
                    label=f"$t_d$ = {sim.disturbance_time_s:.1f}s")
    axes[0].set_xticks([6, 7, 8, 9, 10, 11, 12])
    axes[0].set_xticklabels(["6", "7", "8", "Time [s]", "10", "11", "12"])
    axes[0].set_ylabel("$\\hat f(t)$ [Hz]")
    axes[0].set_ylim(49.92, 50.14)
    set_panel_title(axes[0], f"Fig. 5a — Ensemble-mean instantaneous frequency "
                    f"(N={len(next(iter(df_est.values())))}, SNR=50 dB, stochastic $\\Delta f$)",
                    "(a)", settings, fig="005")
    axes[0].legend(**legend_kwargs)
    axes[0].grid(True, alpha=grid_alpha)

    for method in methods:
        axes[1].plot(t[zoom] - sim.disturbance_time_s,
                     (f_step[method] - system.f0)[zoom],
                     color=colors[method], linestyle=linestyles[method],
                     linewidth=settings.line_width("005", thick=True),
                     label=method)
    axes[1].axhline(0.3, color="k", linestyle="--", alpha=0.5,
                    label="Δf step (0.3 Hz)")
    axes[1].axhline(0.27, color="gray", linestyle=":", alpha=0.5,
                    label="90% target")
    axes[1].axvline(0, color="r", linestyle=":", alpha=0.7)
    axes[1].set_xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    axes[1].set_xticklabels(["0.00", "0.05", "$t - t_d$ [s]", "0.15", "0.20", "0.25"])
    axes[1].set_ylabel("$\\hat{\\Delta f}$ [Hz]")
    set_panel_title(axes[1], "Fig. 5b — Step injection ($\\Delta f$ = 0.3 Hz at $t_d$): "
                    "time-to-90%", "(b)", settings, fig="005")
    axes[1].legend(**legend_kwargs)
    axes[1].grid(True, alpha=grid_alpha)

    for method in methods:
        ratio = metrics[method]["pdf_ratio"]
        axes[2].semilogy(t[view], np.mean(pdf[method], axis=0)[view],
                         color=colors[method], linestyle=linestyles[method],
                         linewidth=settings.line_width("005"),
                         label=f"{method} ({format_sci3_math(ratio)}$\\times$)",
                         alpha=0.95)
    axes[2].axvline(sim.disturbance_time_s, color="r", linestyle=":", alpha=0.7)
    axes[2].axhspan(1e-2, 9e-2, color=settings.color("target_band", "yellow"),
                    alpha=0.12,
                    label="Op. target")
    axes[2].set_xticks([6, 7, 8, 9, 10, 11, 12])
    axes[2].set_xticklabels(["6", "7", "8", "Time [s]", "10", "11", "12"])
    axes[2].set_ylabel("$P_{\\Delta f}$ [Hz²] (log)")
    set_panel_title(axes[2], "Fig. 5c — $P_{\\Delta f}$ computed from each method's $\\hat f$ "
                    "(same sliding window). Post/pre ratio in legend.", "(c)", settings, fig="005")
    axes[2].legend(**legend_kwargs)
    axes[2].set_yticks(np.logspace(-8, -1, 8))
    axes[2].yaxis.set_minor_formatter(NullFormatter())
    axes[2].grid(False)
    axes[2].grid(True, alpha=grid_alpha, axis="x")
    axes[2].grid(True, alpha=grid_alpha, axis="y", which="major")

    x_pos = np.arange(len(methods))
    width = 0.27
    bars = [
        ([metrics[m]["rmse_mhz"] for m in methods], "RMSE post [mHz]", "steelblue", -width),
        ([10 * metrics[m]["noise_mhz"] for m in methods], "Noise floor [0.1 mHz]", "gray", 0),
        ([metrics[m]["latency_ms"] for m in methods], "Latency t90 [ms]", "salmon", width),
    ]
    bar_handles = []
    for values, label, color, offset in bars:
        bar_handles.append(axes[3].bar(x_pos + offset, values, width, label=label, color=color))
    axes[3].set_xticks(x_pos)
    axes[3].set_xticklabels(methods)
    axes[3].set_ylim(0, 170)
    axes[3].set_ylabel("metric value")
    set_panel_title(axes[3], "Fig. 5d — Quantitative comparison: accuracy, noise floor, latency",
                    "(d)", settings, fig="005")
    axes[3].legend(**legend_kwargs)
    axes[3].grid(True, alpha=grid_alpha, axis="y")
    bar_label_size = settings.style("bar_label_size", 9)
    fig_006 = settings.figure("005")
    if "bar_label_size" in fig_006:
        bar_label_size = fig_006["bar_label_size"]
    for bars_obj, values, fmt in zip(bar_handles,
                                     [[metrics[m]["rmse_mhz"] for m in methods],
                                      [10 * metrics[m]["noise_mhz"] for m in methods],
                                      [metrics[m]["latency_ms"] for m in methods]],
                                     ["{:.1f}", "{:.1f}", "{:.0f}"]):
        for bar, value in zip(bars_obj, values):
            if not np.isnan(value):
                axes[3].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             fmt.format(value), ha="center", va="bottom",
                             fontsize=bar_label_size)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "005-comparison_methods",
        settings,
        {
            "figure_number": "005",
            "latex_label": "fig:methods",
            "short_title": "Frequency-estimator benchmark",
            "caption": (
                "Quantitative benchmark of Hilbert, STFT, zero-crossing, and PLL "
                "frequency estimators under the same stochastic disturbance scenario. "
                f"(a) Ensemble-mean instantaneous frequency "
                f"(N={len(next(iter(df_est.values())))}, SNR=50 dB, stochastic $\\Delta f$); "
                "(b) step injection ($\\Delta f$ = 0.3 Hz at $t_d$) used to evaluate "
                "time-to-90%; "
                "(c) $P_{\\Delta f}$ computed from each method's $\\hat f$ with the "
                "same sliding window (post/pre ratio in legend); "
                "(d) quantitative comparison of accuracy, noise floor, and latency."
            ),
            "notes": "For the final paper, a compact table may replace or accompany the full four-panel figure.",
        },
    )
    plt.close(fig)


def plot_noise_resilience(output_dir: Path, sim: SimulationConfig,
                          methods: list[str], colors: dict[str, str],
                          snr_db: np.ndarray,
                          freq_noise_mhz: dict[str, np.ndarray],
                          pdf_floor_hz2: dict[str, np.ndarray],
                          settings: PlotSettings):
    """Figure 007 — estimator noise resilience versus voltage SNR."""
    fig, axes = plt.subplots(2, 1, figsize=settings.figsize("007", (7.16, 4.2)),
                             sharex=True)
    grid_alpha = settings.style("grid_alpha", 0.3)
    method_keys = {"Hilbert": "hilbert", "STFT": "stft",
                   "Zero-Cross": "zero_cross", "PLL": "pll"}

    for method in methods:
        linestyle = settings.linestyle(method_keys.get(method, ""), "-")
        values = freq_noise_mhz[method]
        mean = np.mean(values, axis=1)
        p10 = np.percentile(values, 10, axis=1)
        p90 = np.percentile(values, 90, axis=1)
        axes[0].plot(snr_db, mean, color=colors[method], linestyle=linestyle,
                     marker="o", linewidth=settings.line_width("007"),
                     markersize=settings.style("marker_size", 6), label=method)
        axes[0].fill_between(snr_db, p10, p90, color=colors[method], alpha=0.10)

        floor = pdf_floor_hz2[method]
        floor_mean = np.mean(floor, axis=1)
        floor_p10 = np.percentile(floor, 10, axis=1)
        floor_p90 = np.percentile(floor, 90, axis=1)
        axes[1].semilogy(snr_db, floor_mean, color=colors[method],
                         linestyle=linestyle, marker="o",
                         linewidth=settings.line_width("007"),
                         markersize=settings.style("marker_size", 6),
                         label=method)
        axes[1].fill_between(snr_db, floor_p10, floor_p90,
                             color=colors[method], alpha=0.10)

    axes[0].set_ylabel("Freq. noise floor [mHz]")
    axes[0].set_yscale("log")
    set_panel_title(axes[0], "Fig. 7a — pre-event estimator noise versus voltage SNR",
                    "(a)", settings, fig="007")
    axes[0].grid(True, alpha=grid_alpha, which="both")
    axes[0].legend(loc="upper right", fontsize=settings.style("legend_font_size", 8),
                   frameon=False)

    axes[1].set_xlabel("Voltage SNR [dB]")
    axes[1].set_ylabel("$P_{\\Delta f}$ floor [Hz$^2$]")
    set_panel_title(axes[1], "Fig. 7b — induced $P_{\\Delta f}$ baseline floor",
                    "(b)", settings, fig="007")
    axes[1].grid(True, alpha=grid_alpha, which="both")
    axes[1].set_xticks(snr_db)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "007-noise_resilience",
        settings,
        {
            "figure_number": "007",
            "latex_label": "fig:noise",
            "short_title": "Noise resilience versus SNR",
            "caption": (
                "Noise-resilience sweep over voltage SNR. "
                "(a) Pre-event estimator frequency noise floor versus SNR; "
                "(b) corresponding pre-event $P_{\\Delta f}$ baseline floor."
            ),
            "metrics": {
                "snr_db": snr_db.tolist(),
                "freq_noise_mhz_mean": {
                    method: np.mean(freq_noise_mhz[method], axis=1).tolist()
                    for method in methods
                },
                "pdf_floor_hz2_mean": {
                    method: np.mean(pdf_floor_hz2[method], axis=1).tolist()
                    for method in methods
                },
            },
        },
    )
    plt.close(fig)


def plot_mc_convergence(output_dir: Path, sim: SimulationConfig,
                        p_hilbert: np.ndarray, settings: PlotSettings):
    """Figure 008 — Monte Carlo convergence of Hilbert P_Delta_f estimates."""
    t = sim.t
    mask_post = (t > sim.disturbance_time_s + 0.5) & (t < 12.0)
    mask_pre = (t > 6.5) & (t < sim.disturbance_time_s - 0.1)
    post_values = np.mean(p_hilbert[:, mask_post], axis=1)
    pre_values = np.mean(p_hilbert[:, mask_pre], axis=1)
    n = np.arange(1, len(post_values) + 1)

    def running_stats(values: np.ndarray):
        csum = np.cumsum(values)
        mean = csum / n
        csum2 = np.cumsum(values ** 2)
        var = np.maximum(csum2 / n - mean ** 2, 0.0)
        sem = np.sqrt(var / n)
        return mean, sem

    post_mean, post_sem = running_stats(post_values)
    pre_mean, pre_sem = running_stats(pre_values)

    fig, axes = plt.subplots(2, 1, figsize=settings.figsize("008", (7.16, 4.2)),
                             sharex=True)
    grid_alpha = settings.style("grid_alpha", 0.3)
    color = settings.color("hilbert", "m")

    axes[0].plot(n, post_mean, color=color,
                 linewidth=settings.line_width("008", thick=True),
                 label="running mean")
    axes[0].fill_between(n, post_mean - 1.96 * post_sem,
                         post_mean + 1.96 * post_sem,
                         color=color, alpha=settings.style("band_alpha", 0.15),
                         label="95% SEM band")
    axes[0].axhline(post_mean[-1], color="k", linestyle="--", alpha=0.45,
                    linewidth=0.9, label=f"final = {post_mean[-1]:.3f} Hz$^2$")
    axes[0].set_ylabel("Plateau $P_{\\Delta f}$ [Hz$^2$]")
    set_panel_title(axes[0], "Fig. 8a — Monte Carlo convergence of post-event plateau",
                    "(a)", settings, fig="008")
    axes[0].grid(True, alpha=grid_alpha)
    axes[0].legend(loc="best", fontsize=settings.style("legend_font_size", 8),
                   frameon=False)

    axes[1].semilogy(n, pre_mean, color=color,
                     linewidth=settings.line_width("008", thick=True),
                     label="running mean")
    axes[1].fill_between(n, np.maximum(pre_mean - 1.96 * pre_sem, 1e-15),
                         pre_mean + 1.96 * pre_sem,
                         color=color, alpha=settings.style("band_alpha", 0.15),
                         label="95% SEM band")
    axes[1].axhline(pre_mean[-1], color="k", linestyle="--", alpha=0.45,
                    linewidth=0.9, label=f"final = {pre_mean[-1]:.2e} Hz$^2$")
    axes[1].set_xlabel("$N_{MC}$")
    axes[1].set_ylabel("Pre-event floor [Hz$^2$]")
    set_panel_title(axes[1], "Fig. 8b — Monte Carlo convergence of baseline floor",
                    "(b)", settings, fig="008")
    axes[1].grid(True, alpha=grid_alpha, which="both")
    axes[1].legend(loc="best", fontsize=settings.style("legend_font_size", 8),
                   frameon=False)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "008-mc_convergence",
        settings,
        {
            "figure_number": "008",
            "latex_label": "fig:mc",
            "short_title": "Monte Carlo convergence",
            "caption": (
                "Monte Carlo convergence of the Hilbert-based disturbance-power "
                "estimate. "
                "(a) Running ensemble mean and 95% standard-error band for the "
                "post-event $P_{\\Delta f}$ plateau; "
                "(b) same for the pre-event baseline floor."
            ),
            "metrics": {
                "n_mc": int(len(post_values)),
                "post_plateau_final_hz2": float(post_mean[-1]),
                "pre_floor_final_hz2": float(pre_mean[-1]),
            },
        },
    )
    plt.close(fig)


def _event_level_rate(p_window: np.ndarray, threshold: float, m_samples: int) -> float:
    """Frazione di realizzazioni (righe) con almeno una corsa di m_samples
    campioni consecutivi sopra `threshold` nella finestra `p_window` (N_mc, N_samples).

    Implementa la regola di allarme event-level: una realizzazione triggera se
    P_Df > theta per M finestre P_Df consecutive. Sotto m_samples=1 coincide
    con il rate sample-level OR-aggregato a livello realizzazione.
    """
    if p_window.size == 0 or p_window.shape[1] < m_samples:
        return float("nan")
    above = (p_window > threshold).astype(np.int32)
    # Somma scorrevole di lunghezza m_samples; un valore == m_samples ⇒ run completo.
    if m_samples <= 1:
        per_row = above.any(axis=1)
    else:
        kernel = np.ones(m_samples, dtype=np.int32)
        # Convoluzione 1D riga per riga via stride trick (più veloce di un loop).
        windows = sliding_window_view(above, m_samples, axis=1)
        run_sums = windows.sum(axis=-1)  # shape (N_mc, N_samples - m + 1)
        per_row = (run_sums >= m_samples).any(axis=1)
    return float(100.0 * np.mean(per_row))


def plot_threshold_calibration(output_dir: Path, sim: SimulationConfig,
                               p_hilbert: np.ndarray, settings: PlotSettings):
    """Figure 009 — calibrazione soglie sample-level + regola event-level."""
    t = sim.t
    mask_pre = (t > 6.5) & (t < sim.disturbance_time_s - 0.1)
    mask_post = (t > sim.disturbance_time_s + 0.5) & (t < 12.0)
    pre_2d = p_hilbert[:, mask_pre]
    post_2d = p_hilbert[:, mask_post]
    pre = pre_2d.ravel()
    post = post_2d.ravel()
    pre = pre[np.isfinite(pre) & (pre > 0)]
    post = post[np.isfinite(post) & (post > 0)]

    mu = float(np.mean(pre))
    sigma = float(np.std(pre))
    levels = np.array([3.0, 6.0, 10.0])
    thresholds = mu + levels * sigma
    false_alarm = np.array([100.0 * np.mean(pre > th) for th in thresholds])
    detection = np.array([100.0 * np.mean(post > th) for th in thresholds])

    # ---- Event-level: regola "M campioni P_Df consecutivi sopra theta" ----
    # M espresso in millisecondi e convertito in campioni con sim.fs.
    m_ms_grid = np.array([10.0, 25.0, 50.0, 100.0, 200.0])
    m_samples_grid = np.maximum(1, np.round(m_ms_grid * sim.fs / 1000.0).astype(int))
    far_event = np.zeros((len(levels), len(m_samples_grid)))
    det_event = np.zeros((len(levels), len(m_samples_grid)))
    for i, th in enumerate(thresholds):
        for j, m_s in enumerate(m_samples_grid):
            far_event[i, j] = _event_level_rate(pre_2d, th, int(m_s))
            det_event[i, j] = _event_level_rate(post_2d, th, int(m_s))

    fig, axes = plt.subplots(3, 1, figsize=settings.figsize("009", (7.16, 7.2)))
    grid_alpha = settings.style("grid_alpha", 0.3)
    bins = np.logspace(
        np.floor(np.log10(min(np.min(pre), np.min(post)))),
        np.ceil(np.log10(max(np.max(pre), np.max(post)))),
        70,
    )

    # Only the pre-event/normal regime is plotted. The post-event distribution
    # sits at P_Df values orders of magnitude above the calibration range and
    # would either crush the gaussian or fall off-axis: the relevant statement
    # for threshold calibration is "post-event values lie far above mu_n + k*sigma_n",
    # which is captured in the figure caption and metrics.
    axes[0].hist(pre, bins=bins, density=True, alpha=0.55, color="gray")
    for level, threshold in zip(levels, thresholds):
        axes[0].axvline(threshold, color=settings.color("threshold", "r"),
                        linestyle="--", linewidth=1.0, alpha=0.75)
        axes[0].text(threshold, 0.94, f"$\\theta_{{{int(level)}}}$",
                     transform=axes[0].get_xaxis_transform(),
                     rotation=90, va="top", ha="right",
                     fontsize=settings.style("legend_font_size", 8),
                     color=settings.color("threshold", "r"))
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-9, 1e-5)
    # x-label inlined as the tick label at 1e-7 to recover vertical room.
    axes[0].set_xticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    axes[0].set_xticklabels(["$10^{-9}$", "$10^{-8}$",
                             "$P_{\\Delta f}$ [Hz$^2$]",
                             "$10^{-6}$", "$10^{-5}$"])
    axes[0].set_ylabel("density\n(a.u.)\n \n ")
    # Suppress numeric y-ticks and the 1e6 offset notation: density values are
    # in arbitrary units (shape + threshold positions are what matters here).
    axes[0].set_yticks([])
    set_panel_title(axes[0], "Fig. 9a — empirical threshold calibration",
                    "(a)", settings, fig="009")
    axes[0].grid(True, alpha=grid_alpha, which="both")

    # Detection is 100% at every candidate threshold (post-event values are
    # far above mu_n + k*sigma_n for k in {3,6,10}); only false-alarm rate
    # varies, so only the false-alarm bars are plotted. Detection figures are
    # reported in the caption and metrics. Per-threshold color encoding is
    # shared with panel (c).
    threshold_colors = [plt.cm.viridis(t) for t in (0.20, 0.50, 0.80)]
    x_pos = np.arange(len(levels))
    bars_fa = axes[1].bar(x_pos, false_alarm, 0.55,
                          color=threshold_colors)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"$\\mu + {int(k)}\\sigma$" for k in levels])
    axes[1].set_ylabel("empirical\nrate [%]\n ")
    axes[1].set_ylim(0, 2.2)
    set_panel_title(axes[1], "Fig. 9b — sample-level operating point",
                    "(b)", settings, fig="009")
    axes[1].grid(True, alpha=grid_alpha, axis="y")
    bar_label_size = settings.style("bar_label_size", 9)
    # Tall bars: label inside near the top with a white minibox so it doesn't
    # collide with the panel ceiling. Short bars: label above as usual.
    y_top = axes[1].get_ylim()[1]
    for bar, value in zip(bars_fa, false_alarm):
        h = bar.get_height()
        if h > 0.5 * y_top:
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         h - 0.08, f"{value:.2g}",
                         ha="center", va="top", fontsize=bar_label_size,
                         bbox={"facecolor": "white", "edgecolor": "none",
                               "alpha": 0.85, "pad": 1.5})
        else:
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         h + 0.04, f"{value:.2g}",
                         ha="center", va="bottom", fontsize=bar_label_size)

    # ---- Panel (c): event-level FAR vs run length M ----
    # Detection is 100% at every (theta, M) combination, so only FAR is plotted;
    # threshold encoding (color + linestyle) matches panel (b).
    linestyles = ["-", "--", ":"]
    for i, level in enumerate(levels):
        axes[2].plot(m_ms_grid, far_event[i], linestyle=linestyles[i],
                     color=threshold_colors[i], marker="o", markersize=4,
                     label=f"FAR, $\\theta_{{{int(level)}}}$")
    axes[2].set_ylabel("event-level\nrate [%]")
    axes[2].set_ylim(-3, 105)
    axes[2].set_xscale("log")
    # x-label inlined as the tick label at M = 40 ms (between 25 and 50 ms data
    # points) to recover vertical room; "10" / "100" replace the verbose
    # "$10^1$" / "$10^2$" defaults.
    axes[2].set_xticks([10, 40, 100])
    axes[2].set_xticklabels(["10",
                             "run length $M$ above $\\theta$ [ms]",
                             "100"])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    set_panel_title(axes[2], "Fig. 9c — event-level operating point",
                    "(c)", settings, fig="009")
    axes[2].grid(True, alpha=grid_alpha, which="both")
    axes[2].legend(loc="center right",
                   fontsize=settings.style("legend_font_size", 8),
                   frameon=False)

    fig.tight_layout(h_pad=0.4)
    save_figure(
        fig,
        output_dir,
        "009-threshold_calibration",
        settings,
        {
            "figure_number": "009",
            "latex_label": "fig:thresholds",
            "short_title": "Threshold calibration",
            "caption": (
                "Empirical calibration of $P_{\\Delta f}$ thresholds from "
                "normal pre-event windows. Candidate thresholds are defined as "
                "$\\theta_k = \\mu_n + k\\sigma_n$. "
                "(a) Pre-event $P_{\\Delta f}$ distribution with the three "
                "candidate thresholds; the post-event distribution sits orders "
                "of magnitude above the calibration range "
                "(see (b)-(c) for separation) and is therefore omitted from "
                "this panel. "
                "(b) Sample-level operating point: empirical false-alarm "
                "rate per threshold (detection rate is 100\\% at every "
                "candidate threshold and is reported in the metrics). "
                "(c) Event-level operating point: false-alarm rate versus the "
                "run length $M$ for each candidate threshold "
                "(detection rate is 100\\% at every $(\\theta, M)$ pair "
                "and is reported in the metrics) — a realization triggers "
                "if $P_{\\Delta f} > \\theta$ for $M$ consecutive "
                "$P_{\\Delta f}$ samples. "
                "Threshold color/linestyle encoding is shared between (b) and (c)."
            ),
            "notes": (
                "Sample-level rates (b) overcount because consecutive "
                "$P_{\\Delta f}$ samples share most of the underlying sliding "
                "window. The event-level rule (c) is the deployment-relevant "
                "operating point: tune $(\\theta, M)$ to keep FAR low while "
                "preserving full detection."
            ),
            "metrics": {
                "mu_normal_hz2": mu,
                "sigma_normal_hz2": sigma,
                "thresholds_hz2": {
                    f"mu_plus_{int(level)}sigma": float(threshold)
                    for level, threshold in zip(levels, thresholds)
                },
                "sample_level": {
                    "false_alarm_percent": {
                        f"mu_plus_{int(level)}sigma": float(value)
                        for level, value in zip(levels, false_alarm)
                    },
                    "detection_percent": {
                        f"mu_plus_{int(level)}sigma": float(value)
                        for level, value in zip(levels, detection)
                    },
                },
                "event_level": {
                    "run_length_ms": [float(m) for m in m_ms_grid],
                    "run_length_samples": [int(m) for m in m_samples_grid],
                    "false_alarm_percent": {
                        f"mu_plus_{int(level)}sigma": [float(v) for v in far_event[i]]
                        for i, level in enumerate(levels)
                    },
                    "detection_percent": {
                        f"mu_plus_{int(level)}sigma": [float(v) for v in det_event[i]]
                        for i, level in enumerate(levels)
                    },
                },
            },
        },
    )
    plt.close(fig)


def plot_roc_detection(output_dir: Path, sim: SimulationConfig,
                       snr_sweep: np.ndarray, detectors: list[str],
                       threshold_grids: dict, roc_far: dict, roc_det: dict,
                       detection_at_far: dict, m_event_ms: float,
                       far_target_pct: float, settings: PlotSettings):
    """Figure 010 — ROC event-level + detection vs SNR (Estensione 1.H).

    Pannelli:
      (a) ROC log-FAR/log-Detection per il caso peggiore (SNR minimo) con i
          tre detector P_Df-Hilbert, P_Df-STFT, ROCOF_RMS-Hilbert, regola
          event-level con M = `m_event_ms` ms calibrata in 1.M.
      (b) Detection rate al vincolo FAR ≤ `far_target_pct`% in funzione del
          SNR di tensione.
    """
    detector_labels = {
        "P_df_Hilbert": "$P_{\\Delta f}$ — Hilbert",
        "P_df_STFT": "$P_{\\Delta f}$ — STFT",
        "ROCOF_RMS_Hilbert": "ROCOF$_{\\mathrm{RMS}}$ — Hilbert",
    }
    detector_colors = {
        "P_df_Hilbert": settings.color("hilbert", "m"),
        "P_df_STFT": settings.color("stft", "b"),
        "ROCOF_RMS_Hilbert": settings.color("rocof", "darkorange"),
    }
    detector_linestyles = {
        "P_df_Hilbert": "-",
        "P_df_STFT": "--",
        "ROCOF_RMS_Hilbert": ":",
    }

    fig, axes = plt.subplots(1, 2, figsize=settings.figsize("010", (10.5, 4.2)))
    grid_alpha = settings.style("grid_alpha", 0.3)

    # Pannello (a): ROC nel caso peggiore (SNR minimo).
    # Detectors that maintain ~100% detection across the entire plottable FAR
    # range carry no ROC shape information and are omitted from the panel;
    # they are mentioned in the caption instead. Typically this hides both
    # P_Df backends and only ROCOF_RMS shows a non-trivial curve.
    snr_idx_worst = int(np.argmin(snr_sweep))
    snr_worst = float(snr_sweep[snr_idx_worst])
    omitted = []
    for d in detectors:
        far = roc_far[d][snr_idx_worst]
        det = roc_det[d][snr_idx_worst]
        # Sort by FAR; drop FAR=0 points (not representable on a log axis).
        order = np.argsort(far)
        far_sorted = far[order]
        det_sorted = det[order]
        mask = far_sorted > 0
        if mask.sum() == 0:
            continue
        if det_sorted[mask].min() >= 99.999:
            omitted.append(d)
            continue
        axes[0].plot(far_sorted[mask], det_sorted[mask],
                     color=detector_colors[d],
                     linestyle=detector_linestyles[d],
                     marker="o", markersize=3,
                     linewidth=settings.line_width("010"),
                     label=detector_labels[d])
    axes[0].axvline(far_target_pct, color="black", linestyle="-.", linewidth=0.8,
                    alpha=0.6, label=f"FAR = {far_target_pct:g}%")
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.8, 110)
    axes[0].set_ylim(0, 105)
    # Plain-number tick labels (avoid the verbose 10^k rendering for k>=0).
    axes[0].set_xticks([1e0, 1e1, 1e2])
    axes[0].set_xticklabels(["1", "10", "100"])
    axes[0].set_xlabel("False alarm rate [%, event-level]")
    axes[0].set_ylabel("Detection rate [%, event-level]")
    set_panel_title(axes[0],
                    f"Fig. 10a — ROC at SNR = {snr_worst:.0f} dB (worst case)",
                    "(a)", settings, fig="010")
    axes[0].grid(True, alpha=grid_alpha, which="both")
    axes[0].legend(loc="lower right", fontsize=settings.style("legend_font_size", 8),
                   frameon=False)

    # Pannello (b): detection @ FAR target vs SNR.
    # Filter detectors that are always at ~100% across all SNRs (uninformative
    # bars); they're noted in the caption instead.
    omitted_b = [d for d in detectors
                 if np.all(np.asarray(detection_at_far[d]) >= 99.999)]
    detectors_plot = [d for d in detectors if d not in omitted_b]

    # Trim trailing SNRs whose values are all identical to the previous index
    # (across plotted detectors) — once we hit saturation the next bars add
    # no narrative.
    last_idx = len(snr_sweep) - 1
    if detectors_plot:
        for i in range(len(snr_sweep) - 1, 0, -1):
            same = all(
                abs(detection_at_far[d][i] - detection_at_far[d][i - 1]) < 1e-6
                for d in detectors_plot
            )
            if same:
                last_idx = i - 1
            else:
                break
    snr_sweep_plot = snr_sweep[: last_idx + 1]
    x_pos = np.arange(len(snr_sweep_plot))
    n_plot = max(len(detectors_plot), 1)
    width = 0.27 if n_plot > 1 else 0.55
    offsets = (np.arange(n_plot) - (n_plot - 1) / 2.0) * width
    for off, d in zip(offsets, detectors_plot):
        values = detection_at_far[d][: last_idx + 1]
        bars = axes[1].bar(x_pos + off, values, width,
                           color=detector_colors[d], alpha=0.85,
                           label=detector_labels[d])
        for bar, value in zip(bars, values):
            label = f"{value:.0f}" if np.isfinite(value) else "—"
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         min(bar.get_height() + 2.0, 102.0),
                         label, ha="center", va="bottom",
                         fontsize=settings.style("bar_label_size", 8))
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"{int(s)} dB" for s in snr_sweep_plot])
    axes[1].set_xlabel("Voltage SNR")
    axes[1].set_ylabel(f"Detection @ FAR ≤ {far_target_pct:g}% [%]")
    axes[1].set_ylim(0, 110)
    set_panel_title(axes[1],
                    f"Fig. 10b — operating point at FAR ≤ {far_target_pct:g}%, "
                    f"M = {m_event_ms:.0f} ms",
                    "(b)", settings, fig="010")
    axes[1].grid(True, alpha=grid_alpha, axis="y")
    if len(detectors_plot) > 1:
        axes[1].legend(loc="lower right",
                       fontsize=settings.style("legend_font_size", 8),
                       frameon=False)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "010-roc_detection",
        settings,
        {
            "figure_number": "010",
            "latex_label": "fig:roc",
            "short_title": "ROC event-level + detection vs SNR",
            "caption": (
                "Event-level detection performance under additive measurement "
                "noise. Panel (a) shows ROC curves at "
                f"SNR = {snr_worst:.0f} dB (worst case in the sweep), using the "
                f"deployment rule from Ext. 1.M ($M$ = {m_event_ms:.0f} ms "
                "consecutive $P_{\\Delta f}$ samples above $\\theta$); "
                + (
                    ("Detectors omitted from (a) because they maintain "
                     "$\\sim$100\\% detection across the full plotted FAR "
                     "range: " + ", ".join(detector_labels[d] for d in omitted) + ". ")
                    if omitted else ""
                ) +
                f"Panel "
                f"(b) reports the maximum detection rate compatible with "
                f"FAR $\\leq$ {far_target_pct:g}\\% as a function of voltage "
                "SNR. "
                + (
                    ("Detectors omitted from (b) because they keep 100\\% "
                     "detection across the entire SNR sweep: "
                     + ", ".join(detector_labels[d] for d in omitted_b) + ". ")
                    if omitted_b else ""
                )
                + (
                    f"SNRs above {snr_sweep_plot[-1]:.0f} dB are not shown "
                    "in (b): all plotted detectors are already at saturation "
                    "by that point."
                    if last_idx < len(snr_sweep) - 1 else ""
                )
            ),
            "notes": (
                "Thresholds are swept on a log-spaced grid covering [q0.001, "
                "q0.999] of the pooled pre/post score distribution. Detection "
                "@ FAR target is the maximum detection over thresholds that "
                "respect the FAR constraint; NaN means no admissible threshold "
                "exists at that SNR."
            ),
            "metrics": {
                "snr_db": [float(s) for s in snr_sweep],
                "m_event_ms": float(m_event_ms),
                "far_target_pct": float(far_target_pct),
                "detection_at_far_target": {
                    d: [float(v) for v in detection_at_far[d]]
                    for d in detectors
                },
                "thresholds_grid": {
                    d: [float(v) for v in threshold_grids[d]]
                    for d in detectors
                },
                "far_percent": {
                    d: roc_far[d].tolist() for d in detectors
                },
                "detection_percent": {
                    d: roc_det[d].tolist() for d in detectors
                },
            },
        },
    )
    plt.close(fig)
