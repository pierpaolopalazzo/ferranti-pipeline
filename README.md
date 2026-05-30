# Ferranti pipeline

Simulation and detection of Ferranti-effect transients on transmission lines,
with instantaneous-frequency estimators (Hilbert/IQ, STFT, zero-crossing, PLL)
and associated metrics (P_Δf, ROCOF, RMSE, latency, event-level ROC).

Companion code for the paper *Ferranti-effect transient detection* (in
preparation, 2026).

## Citation

A DOI-archived release will be deposited on Zenodo upon paper finalization.
In the meantime the repository can be cited via commit hash; see
`CITATION.cff`. After the Zenodo release, `CITATION.cff` will be updated
with the final DOI.

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Design goals

- reduce global state;
- separate simulation, estimators, metrics, and figure rendering;
- make runs configurable from the CLI;
- speed up the STFT estimator via vectorized frames;
- write outputs to dedicated subfolders under `figures/v3/`.

## Layout

- `ferranti_pipeline_v3.py` — CLI entrypoint and run orchestration.
- `ferranti_v3/config.py` — system/simulation dataclasses and TOML settings.
- `ferranti_v3/simulation.py` — Ferranti line, filters, synthetic signals.
- `ferranti_v3/estimators.py` — Hilbert, STFT, zero-crossing, PLL.
- `ferranti_v3/metrics.py` — P_Δf, RMS, ROCOF, RMSE, noise floor, latency.
- `ferranti_v3/plots.py` — figure rendering and JSON metadata.
- `ferranti_v3/workflow.py` — reusable ensemble workflows.
- `settings-ieee.toml` — IEEE paper figure preset (and schema for future presets).
- `requirements.txt` — runtime dependencies.

## Usage

### Output convention: hardcoded default + presets by nickname

| Mode                  | Command                                                       | Output             | Purpose                                                 |
|-----------------------|---------------------------------------------------------------|--------------------|---------------------------------------------------------|
| hardcoded / default   | `python ferranti_pipeline_v3.py`                              | `figures/v3/`      | screen review, default Matplotlib rcParams              |
| `ieee` preset (paper) | `python ferranti_pipeline_v3.py --config settings-ieee.toml`  | `figures/v3/ieee/` | IEEEtran double-column, 300 dpi, titles off             |

**Preset naming convention.** Each settings file is named
`settings-<nickname>.toml` and its output goes automatically to
`figures/v3/<nickname>/`. The nickname is derived from the filename (see
`PlotSettings.output_dir` in `config.py`); it is **not** a key inside the
TOML, so the name is immediately readable from the filesystem and two
distinct presets can never map to the same output folder.

To add a new preset (e.g. single-column layout, B/W palette, poster), just
create a new `settings-<nick>.toml` next to `settings-ieee.toml` and run
the pipeline with `--config`. When a new figure is added (e.g. `008-...`),
register it in the `[figures.NNN]` section of the active presets; in
hardcoded mode the fallbacks in `plots.py` are used (sized for screen
review).

### What a `settings-*.toml` file controls

- `[output]` — `dir`, `dpi`, `formats` *(nickname is not here: it comes from the filename)*
- `[ieee]` — `single_column_width`, `double_column_width` (inches)
- `[style]` — font, line/marker width, alpha, panel label, `show_titles`
- `[colors]` — palette for estimators and overlays
- `[linestyles]` — linestyles for estimators (useful for B/W print)
- `[figures.NNN]` — per-figure overrides (`width`, `height`, line widths)

The CLI option `--output-dir` remains an explicit override that bypasses
the filename-derived nickname.

### Output (both modes)

- `001-frequency_and_pdf.png`
- `002-inertia_sensitivity.png`
- `003-coherent_vs_stochastic.png`
- `004-comparison_pdf_rocof.png`
- `005-comparison_methods.png` *(cross-estimator benchmark)*
- `006-latency_breakdown.png` *(end-to-end latency breakdown)*
- `007-noise_resilience.png` *(SNR sweep 10–50 dB)*
- `008-mc_convergence.png` *(Monte Carlo convergence)*
- `009-threshold_calibration.png` *(sample-level threshold calibration + event-level rule "M consecutive samples above θ")*
- `010-roc_detection.png` *(event-level ROC and detection@FAR≤1% vs SNR, P_Δf-Hilbert vs P_Δf-STFT vs ROCOF_RMS)*

Hardcoded/default mode only saves PNG. Paper-ready presets such as
`settings-ieee.toml` also save JSON sidecars with figure number, LaTeX
label, caption, notes, and quantitative metrics when available.

### Thresholds: detection vs severity

The `0.04 Hz²` threshold shown in the main figures is an operational /
severity threshold. The statistical calibration introduced in Fig. 009
instead uses the normal-regime noise floor to define much lower detection
thresholds: `θ_det = μ_n + k σ_n`. For the current run:

- `μ_n = 2.47e-7 Hz²`
- `σ_n = 2.49e-7 Hz²`
- `θ_det,1 = 9.94e-7 Hz²`
- `θ_det,2 = 1.74e-6 Hz²`
- `θ_det,3 = 2.74e-6 Hz²`

Detection thresholds signal departure from normal noise; the operational
threshold classifies the energetic severity of the event.

### Event-level rule and ROC

Sample-level FAR/Detection rates overestimate, because consecutive P_Δf
samples share the same sliding window. The deployment rule is event-level:
a realization triggers if `P_Δf > θ` for `M` consecutive P_Δf samples.
Clean operating point at N_MC=30 from the base ensemble:
`(θ = μ_n + 6σ_n, M = 50 ms)` → FAR = 0%, detection = 100%.

Validation under measurement noise (Fig. 010, N=100, 5 SNR levels):

| SNR [dB] | P_Δf-Hilbert | P_Δf-STFT | ROCOF_RMS-Hilbert |
|---|---|---|---|
| 10 | 100% | 100% |   4% |
| 20 | 100% | 100% |  15% |
| 30 | 100% | 100% |  95% |
| 40 | 100% | 100% | 100% |
| 50 | 100% | 100% | 100% |

(Maximum event-level detection compatible with FAR ≤ 1%, M = 50 ms.)
P_Δf, with either the Hilbert/IQ or STFT back-end, keeps 100% detection
with 0/100 false alarms down to 10 dB SNR — a ≈ 25–30 dB margin over
ROCOF_RMS computed on the same `f_i(t)`.

## License

MIT — see `LICENSE`.
