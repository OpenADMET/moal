"""LiveDashboard: real-time updating matplotlib figure for campaign monitoring.

Three subplots update after every active learning iteration:
  1. Cumulative Actives Curve  — x: cumulative cost ($), y: actives found
  2. Cumulative Cost Curve     — stacked bar per iteration (DRC + PS) with total line
  3. Model Performance Curve   — x: iteration, y: configurable metric on test set
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from moal.evaluation import ModelMetric
from moal.types import LabelRecord

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Suppress matplotlib's own internal chatter.
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# Colour palette
_COLOUR_DRC = "#E07B39"  # orange
_COLOUR_PS = "#4C9BE8"  # blue
_COLOUR_ACT = "#2CA02C"  # green
_COLOUR_MET = "#9467BD"  # purple


class LiveDashboard:
    """Three-panel live-updating campaign dashboard.

    Args:
        n_iterations: Total planned iterations (used to pre-size x-axes).
        model_metric: Metric to display in the model performance panel.
        save_dir: If set, PNG snapshots are written here after every update.
        figsize: Overall figure size (width, height) in inches.
        show: If True, attempt interactive ``plt.ion()`` mode. If False (or if
            the display is unavailable), fall back to file-save-only mode.
    """

    def __init__(
        self,
        n_iterations: int,
        model_metric: ModelMetric = ModelMetric.MAE,
        save_dir: str | Path | None = None,
        figsize: tuple[int, int] = (15, 4),
        show: bool = True,
    ) -> None:
        self.n_iterations = n_iterations
        self.model_metric = model_metric
        self.save_dir = Path(save_dir) if save_dir else None
        self._interactive = False
        self._update_count = 0

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Use a non-interactive backend when show=False or when we detect
        # there is no display environment.
        if not show:
            matplotlib.use("Agg")

        self._fig, (self._ax1, self._ax2, self._ax3) = plt.subplots(
            1, 3, figsize=figsize
        )
        self._fig.suptitle(
            "Active Learning Campaign Dashboard", fontsize=11, fontweight="bold"
        )
        self._fig.tight_layout(pad=2.5)

        self._init_axes()

        if show:
            try:
                plt.ion()
                plt.pause(0.05)
                self._interactive = True
            except (OSError, RuntimeError):
                logger.debug(
                    "Interactive display unavailable; using file-save-only mode."
                )
                self._interactive = False

        # Data accumulators
        self._cum_costs: list[float] = []
        self._cum_actives: list[int] = []
        self._iter_drc_costs: list[float] = []
        self._iter_ps_costs: list[float] = []
        self._model_metric_values: list[float] = []

        # Twin y-axis for the cost panel — created once, reused on every update.
        self._ax2r: Axes = self._ax2.twinx()
        self._ax2r.set_ylabel("Cumulative Cost ($)", fontsize=8)
        self._ax2r.tick_params(axis="y", labelsize=7)

    # ------------------------------------------------------------------
    # Axis initialisation
    # ------------------------------------------------------------------

    def _init_axes(self) -> None:
        self._ax1.set_title("Cumulative Actives", fontsize=9)
        self._ax1.set_xlabel("Cumulative Cost ($)")
        self._ax1.set_ylabel("Confirmed Actives Found")
        self._ax1.grid(True, linestyle="--", alpha=0.4)

        self._ax2.set_title("Per-Iteration Cost Breakdown", fontsize=9)
        self._ax2.set_xlabel("Iteration")
        self._ax2.set_ylabel("Cost ($)")
        self._ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

        metric_label = self.model_metric.value.upper().replace("_", " ")
        self._ax3.set_title(f"Model Performance ({metric_label})", fontsize=9)
        self._ax3.set_xlabel("Iteration")
        self._ax3.set_ylabel(metric_label)
        self._ax3.grid(True, linestyle="--", alpha=0.4)

        # If no test set, render a placeholder annotation.
        self._no_test_annotation = self._ax3.text(
            0.5,
            0.5,
            "No test set provided",
            transform=self._ax3.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="grey",
            fontstyle="italic",
        )

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(
        self,
        labeled_records: list[LabelRecord],
        activity_threshold: float,
        iter_drc_cost: float,
        iter_ps_cost: float,
        model_metric_value: float | None = None,
    ) -> None:
        """Redraw all three panels with the latest campaign state.

        Args:
            labeled_records: All oracle-labeled records accumulated so far
                (ordered by acquisition time).
            activity_threshold: pEC50 threshold defining a confirmed active.
            iter_drc_cost: Total DRC cost incurred in the *current* iteration.
            iter_ps_cost: Total PS cost incurred in the *current* iteration.
            model_metric_value: Metric value from ``evaluate_model()`` for this
                iteration, or None if no test set is available.
        """
        self._update_count += 1

        # Accumulate per-iteration cost data.
        self._iter_drc_costs.append(iter_drc_cost)
        self._iter_ps_costs.append(iter_ps_cost)

        # Build cumulative actives/cost from records.
        cum_cost = 0.0
        cum_active = 0
        self._cum_costs = []
        self._cum_actives = []
        for rec in labeled_records:
            cum_cost += rec.cost
            if self._record_is_active(rec, activity_threshold):
                cum_active += 1
            self._cum_costs.append(cum_cost)
            self._cum_actives.append(cum_active)

        if model_metric_value is not None:
            self._model_metric_values.append(model_metric_value)

        self._draw_actives_panel()
        self._draw_cost_panel()
        self._draw_metric_panel()

        self._fig.tight_layout(pad=2.5)

        if self._interactive:
            try:
                self._fig.canvas.draw_idle()
                plt.pause(0.05)
            except (OSError, RuntimeError):
                self._interactive = False

        if self.save_dir:
            self._save_snapshot()

    # ------------------------------------------------------------------
    # Panel drawing
    # ------------------------------------------------------------------

    def _draw_actives_panel(self) -> None:
        ax: Axes = self._ax1
        ax.cla()
        ax.set_title("Cumulative Actives", fontsize=9)
        ax.set_xlabel("Cumulative Cost ($)")
        ax.set_ylabel("Confirmed Actives Found")
        ax.grid(True, linestyle="--", alpha=0.4)

        if not self._cum_costs:
            return

        xs = np.array(self._cum_costs)
        ys = np.array(self._cum_actives)
        ax.fill_between(xs, ys, color=_COLOUR_ACT, zorder=2, alpha=0.2)
        ax.plot(xs, ys, color=_COLOUR_ACT, linewidth=2, zorder=3, marker="")
        # ax.scatter(xs[:-1], ys[:-1], s=18, color=_COLOUR_ACT, alpha=0.5, zorder=3)
        # # Highlight the most recent point.
        # ax.scatter([xs[-1]], [ys[-1]], s=60, color=_COLOUR_ACT, zorder=4,
        #            edgecolors="white", linewidths=1.2)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    def _draw_cost_panel(self) -> None:
        ax: Axes = self._ax2
        ax2r: Axes = self._ax2r
        ax.cla()
        ax2r.cla()
        ax.set_title("Per-Iteration Cost Breakdown", fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost ($)")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax2r.set_ylabel("Cumulative Cost ($)", fontsize=8)
        ax2r.tick_params(axis="y", labelsize=7)
        ax2r.yaxis.set_label_position("right")
        ax2r.yaxis.set_ticks_position("right")

        n = len(self._iter_drc_costs)
        if n == 0:
            return

        iters = np.arange(1, n + 1)
        drc_arr = np.array(self._iter_drc_costs)
        ps_arr = np.array(self._iter_ps_costs)

        ax.bar(iters, drc_arr, color=_COLOUR_DRC, label="DRC", zorder=2)
        ax.bar(iters, ps_arr, bottom=drc_arr, color=_COLOUR_PS, label="PS", zorder=2)

        cum_total = np.cumsum(drc_arr + ps_arr)
        ax2r.plot(
            iters,
            cum_total,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label="Cumulative",
            zorder=3,
        )

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2r.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, fontsize=7, loc="upper left")
        ax.set_xlim(0.5, max(n + 0.5, self.n_iterations + 0.5))
        ax.set_ylim(bottom=0)

    def _draw_metric_panel(self) -> None:
        ax: Axes = self._ax3
        ax.cla()
        metric_label = self.model_metric.value.upper().replace("_", " ")
        ax.set_title(f"Model Performance ({metric_label})", fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric_label)
        ax.grid(True, linestyle="--", alpha=0.4)

        if not self._model_metric_values:
            ax.text(
                0.5,
                0.5,
                "No test set provided",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="grey",
                fontstyle="italic",
            )
            return

        iters = np.arange(1, len(self._model_metric_values) + 1)
        vals = np.array(self._model_metric_values)
        ax.plot(
            iters,
            vals,
            color=_COLOUR_MET,
            linewidth=1.5,
            marker="o",
            markersize=5,
            zorder=2,
        )
        # Highlight latest.
        ax.scatter(
            [iters[-1]],
            [vals[-1]],
            s=60,
            color=_COLOUR_MET,
            zorder=3,
            edgecolors="white",
            linewidths=1.2,
        )
        ax.set_xlim(0.5, max(len(iters) + 0.5, self.n_iterations + 0.5))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_is_active(rec: LabelRecord, threshold: float) -> bool:
        from moal.types import CensoringType

        if rec.censoring_type == CensoringType.EXACT:
            return rec.value >= threshold
        if rec.censoring_type == CensoringType.INTERVAL:
            return rec.value >= threshold
        return False

    def _save_snapshot(self) -> None:
        path = self.save_dir / f"dashboard_{self._update_count:04d}.png"  # type: ignore[operator]
        try:
            self._fig.savefig(path, dpi=100, bbox_inches="tight")
        except Exception as exc:
            logger.warning("Could not save dashboard snapshot: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Explicitly save the current figure to a file."""
        self._fig.savefig(path, dpi=120, bbox_inches="tight")

    def save_gif(
        self,
        path: str | Path,
        frame_duration_ms: int = 500,
        last_frame_duration_ms: int = 5000,
    ) -> None:
        """Assemble all saved PNG snapshots into an animated GIF.

        Reads every ``dashboard_XXXX.png`` written by :meth:`_save_snapshot`
        in sorted (chronological) order and encodes them into a looping GIF.
        This method is a no-op (with a warning) when ``save_dir`` was not set
        at construction time or when no snapshots have been written yet.

        Args:
            path: Destination file path for the GIF.
            frame_duration_ms: Display duration of each frame in milliseconds.
                Defaults to 500 (half a second per iteration frame).
            last_frame_duration_ms: Display duration of the final frame in
                milliseconds, allowing the viewer to read the completed state
                before the animation loops. Defaults to 5000 (5 seconds).
        """
        if not self.save_dir:
            logger.warning("save_gif called but save_dir is not set; skipping")
            return

        frame_paths = sorted(self.save_dir.glob("dashboard_*.png"))
        if not frame_paths:
            logger.warning(
                "No dashboard snapshots found in %s; skipping GIF", self.save_dir
            )
            return

        try:
            from PIL import Image  # Pillow is a declared dependency

            frames = [Image.open(p).convert("RGB") for p in frame_paths]
            # convert("P") gives an 8-bit palette GIF; quantize reduces colour
            # depth without visible banding at typical dashboard colour counts
            palette_frames = [f.convert("P", dither=Image.Dither.NONE) for f in frames]
            # Per-frame durations: hold the last frame longer so viewers can
            # read the final campaign state before the animation loops
            durations = [frame_duration_ms] * len(palette_frames)
            durations[-1] = last_frame_duration_ms
            palette_frames[0].save(
                path,
                format="GIF",
                save_all=True,
                append_images=palette_frames[1:],
                duration=durations,
                loop=0,  # 0 = loop forever
                optimize=False,
            )
            logger.info(
                "Dashboard animation (%d frames) saved to %s",
                len(frame_paths),
                path,
            )
        except Exception as exc:
            logger.warning("Could not save dashboard GIF: %s", exc)

    def close(self) -> None:
        """Close the matplotlib figure and release resources."""
        plt.close(self._fig)
