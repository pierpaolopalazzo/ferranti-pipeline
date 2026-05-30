import tomllib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SystemConfig:
    line_length_m: float = 300e3
    r_per_m: float = 0.03e-3
    l_per_m: float = 0.9e-6
    c_per_m: float = 12e-12
    g_per_m: float = 0.0
    f0: float = 50.0
    vs_rms: float = 400e3


@dataclass(frozen=True)
class SimulationConfig:
    fs: int = 10_000
    duration_s: float = 20.0
    disturbance_time_s: float = 9.0
    window_s: float = 0.040
    bp_bw_hz: float = 2.0
    bp_order: int = 2
    load_mod_pct: float = 0.035
    subhz_low: float = 0.05
    subhz_high: float = 1.0
    snr_db: float = 50.0

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def n_win(self) -> int:
        return int(round(self.window_s * self.fs))

    @property
    def n_samples(self) -> int:
        return int(round(self.duration_s * self.fs))

    @property
    def t(self):
        import numpy as np

        return np.arange(self.n_samples) / self.fs

    @property
    def df_std_target(self) -> float:
        return self.load_mod_pct * 16.0


@dataclass(frozen=True)
class LineState:
    gamma: complex
    z0: complex
    ferranti_ratio: complex
    v_r: complex
    v_peak: float


@dataclass(frozen=True)
class Filters:
    subhz_b: object
    subhz_a: object
    bp_b: object
    bp_a: object


@dataclass(frozen=True)
class PlotSettings:
    data: dict
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "PlotSettings":
        path = Path(path)
        with path.open("rb") as handle:
            return cls(tomllib.load(handle), source_path=path)

    def section(self, name: str) -> dict:
        return self.data.get(name, {})

    def figure(self, number: str) -> dict:
        return self.data.get("figures", {}).get(number, {})

    def color(self, key: str, default: str) -> str:
        return self.section("colors").get(key, default)

    def linestyle(self, key: str, default: str = "-") -> str:
        return self.section("linestyles").get(key, default)

    def style(self, key: str, default, fig: str | None = None):
        if fig is not None:
            per_fig = self.figure(fig)
            if key in per_fig:
                return per_fig[key]
        return self.section("style").get(key, default)

    def output_dir(self, override: str | None = None) -> Path:
        """Resolve the output directory.

        Convention: when this PlotSettings was loaded from a file named
        `settings-<nickname>.toml`, output is routed to `<dir>/<nickname>/`.
        When no source file is attached (hardcoded run), output is `<dir>/`
        flat. Explicit `--output-dir` always wins.

        Relative `dir` paths in a settings file are anchored to the project
        root (the first ancestor of the settings file that contains a `src/`
        directory; otherwise the settings file's own directory). This keeps
        output stable regardless of which CWD the pipeline is launched from.
        """
        if override:
            return Path(override)
        base_dir = Path(self.section("output").get("dir", "figures/v3"))
        if self.source_path is None:
            return base_dir
        if not base_dir.is_absolute():
            anchor = self.source_path.resolve().parent
            for ancestor in anchor.parents:
                if (ancestor / "src").is_dir():
                    anchor = ancestor
                    break
            else:
                if anchor.name == "src":
                    anchor = anchor.parent
            base_dir = anchor / base_dir
        stem = self.source_path.stem  # e.g. "settings-ieee"
        nickname = stem.removeprefix("settings-") if stem.startswith("settings-") else stem
        return base_dir / nickname

    def dpi(self) -> int:
        return int(self.section("output").get("dpi", 150))

    def figsize(self, number: str, default: tuple[float, float]) -> tuple[float, float]:
        fig_cfg = self.figure(number)
        if not fig_cfg:
            return default
        width = fig_cfg.get("width", default[0])
        if width == "single":
            width_in = self.section("ieee").get("single_column_width", 3.50)
        elif width == "double":
            width_in = self.section("ieee").get("double_column_width", 7.16)
        else:
            width_in = float(width)
        return float(width_in), float(fig_cfg.get("height", default[1]))

    def line_width(self, number: str | None = None, thick: bool = False) -> float:
        key = "thick_line_width" if thick else "line_width"
        if number is not None and key in self.figure(number):
            return float(self.figure(number)[key])
        return float(self.style(key, 1.8 if thick else 1.2))


def apply_matplotlib_style(settings: PlotSettings) -> None:
    style = settings.section("style")
    plt.rcParams.update({
        "font.family": style.get("font_family", "serif"),
        "font.size": style.get("font_size", 8),
        "axes.labelsize": style.get("axes_label_size", 8),
        "axes.titlesize": style.get("axes_title_size", 8),
        "legend.fontsize": style.get("legend_font_size", 7),
        "xtick.labelsize": style.get("tick_label_size", 7),
        "ytick.labelsize": style.get("tick_label_size", 7),
        "svg.fonttype": "none",
        "svg.hashsalt": "ferranti",
    })
