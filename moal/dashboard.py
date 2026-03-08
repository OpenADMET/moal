"""LiveDashboard: real-time updating matplotlib figure for campaign monitoring.

Four subplots (2×2) update after every active learning iteration:
  1. Cumulative Actives Curve  — x: cumulative cost ($), y: actives found
  2. Cumulative Cost Curve     — stacked bar per iteration (DRC + PS) with total line
  3. Model Performance Curve   — x: iteration, y: configurable metric on test set
  4. Compound Status           — bar per category (PS-only, DRC, Unqueried) showing
                                 current pool state; DRC bar is stacked to show
                                 PS→DRC upgrades
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

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
_COLOUR_UPGRADE = (
    "#9B59B6"  # purple-magenta (PS→DRC upgrades, matches terminal [magenta])
)
_COLOUR_UNQUERIED = "#D3D3D3"  # light gray


class LiveDashboard:
    """Four-panel live-updating campaign dashboard (2×2 grid).

    Parameters
    ----------
    n_iterations : int
        Total planned iterations (used to pre-size x-axes).
    n_compounds : int, optional
        Total number of compounds in the pool (used to compute the unqueried
        count in the compound status panel). Default is 0.
    model_metric : ModelMetric, optional
        Metric to display in the model performance panel. Default is MAE.
    figsize : tuple[int, int], optional
        Overall figure size (width, height) in inches. Default is (14, 8).
    show : bool, optional
        If True, attempt interactive ``plt.ion()`` mode. If False (or if the
        display is unavailable), fall back to file-save-only mode.
        Default is True.
    """

    def __init__(
        self,
        n_iterations: int,
        n_compounds: int = 0,
        model_metric: ModelMetric = ModelMetric.MAE,
        figsize: tuple[int, int] = (14, 8),
        show: bool = True,
    ) -> None:
        self.n_iterations = n_iterations
        self.n_compounds = n_compounds
        self.model_metric = model_metric
        self._interactive = False
        self._update_count = 0
        # In-memory PNG frames captured after every update, used by save_gif.
        self._frames: list[bytes] = []

        # Use a non-interactive backend when show=False or when we detect
        # there is no display environment.
        if not show:
            matplotlib.use("Agg")

        self._fig, _axes = plt.subplots(2, 2, figsize=figsize, dpi=150)
        self._ax1: Axes = _axes[0, 0]
        self._ax2: Axes = _axes[0, 1]
        self._ax3: Axes = _axes[1, 0]
        self._ax4: Axes = _axes[1, 1]
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
        self._iter_upgrade_costs: list[float] = []
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

        self._ax4.set_title("Compound Status", fontsize=9)
        self._ax4.set_ylabel("Compounds")
        self._ax4.grid(True, linestyle="--", alpha=0.4, axis="y")

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(
        self,
        labeled_records: list[LabelRecord],
        activity_threshold: float,
        iter_drc_cost: float,
        iter_ps_cost: float,
        iter_upgrade_cost: float = 0.0,
        model_metric_value: float | None = None,
    ) -> None:
        """Redraw all panels with the latest campaign state.

        Parameters
        ----------
        labeled_records : list[LabelRecord]
            All oracle-labeled records accumulated so far, ordered by
            acquisition time.
        activity_threshold : float
            pEC50 threshold defining a confirmed active.
        iter_drc_cost : float
            Total DRC cost incurred in the *current* iteration (includes
            upgrade costs).
        iter_ps_cost : float
            Total PS cost incurred in the *current* iteration.
        iter_upgrade_cost : float, optional
            Portion of ``iter_drc_cost`` that came from PS→DRC upgrades.
            Defaults to 0.0 (no upgrade distinction).
        model_metric_value : float, optional
            Metric value from ``evaluate_model()`` for this iteration, or None
            if no test set is available.
        """
        self._update_count += 1

        # Accumulate per-iteration cost data.
        self._iter_drc_costs.append(iter_drc_cost)
        self._iter_ps_costs.append(iter_ps_cost)
        self._iter_upgrade_costs.append(iter_upgrade_cost)

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
        self._draw_compound_status_panel(labeled_records)

        self._fig.tight_layout(pad=2.5)

        if self._interactive:
            try:
                self._fig.canvas.draw_idle()
                plt.pause(0.05)
            except (OSError, RuntimeError):
                self._interactive = False

        self._capture_frame()

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
        upgrade_arr = np.array(self._iter_upgrade_costs)
        # First-pass DRC is total DRC minus the upgrade portion
        drc_new_arr = drc_arr - upgrade_arr

        ax.bar(iters, drc_new_arr, color=_COLOUR_DRC, label="DRC", zorder=2)
        ax.bar(
            iters,
            upgrade_arr,
            bottom=drc_new_arr,
            color=_COLOUR_UPGRADE,
            label="PS→DRC",
            zorder=2,
        )
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
        # Force integer-only tick positions so labels never display as floats
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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
        # Force integer-only tick positions so labels never display as floats
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _draw_compound_status_panel(self, labeled_records: list[LabelRecord]) -> None:
        ax: Axes = self._ax4
        ax.cla()
        ax.set_title("Compound Status", fontsize=9)
        ax.set_ylabel("Compounds")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

        from moal.types import QueryType

        ps_smiles = {
            r.canonical_smiles
            for r in labeled_records
            if r.fidelity == QueryType.PRIMARY_SCREEN
        }
        drc_smiles = {
            r.canonical_smiles
            for r in labeled_records
            if r.fidelity == QueryType.DOSE_RESPONSE
        }
        n_upgrades = len(ps_smiles & drc_smiles)
        n_ps_only = len(ps_smiles) - n_upgrades
        n_drc_new = len(drc_smiles) - n_upgrades
        # Unique queried = PS-only + all DRC (upgrades are counted in DRC, not PS-only)
        n_queried = n_ps_only + len(drc_smiles)
        n_unqueried = max(self.n_compounds - n_queried, 0)
        n_all = n_ps_only + n_drc_new + n_upgrades + n_unqueried

        categories = ["PS", "DRC", "Unqueried"]
        x = np.arange(len(categories))

        ax.bar([x[0]], [n_ps_only], color=_COLOUR_PS, zorder=2, label="PS")
        ax.bar(
            [x[0]],
            [n_upgrades],
            bottom=[n_ps_only],
            color=_COLOUR_UPGRADE,
            zorder=2,
            # label="PS→DRC",
        )

        # DRC bar: first-pass DRC on the bottom, upgrades stacked on top
        ax.bar([x[1]], [n_drc_new], color=_COLOUR_DRC, zorder=2, label="DRC")
        ax.bar(
            [x[1]],
            [n_upgrades],
            bottom=[n_drc_new],
            color=_COLOUR_UPGRADE,
            zorder=2,
            label="PS→DRC",
        )
        ax.bar(
            [x[2]], [n_unqueried], color=_COLOUR_UNQUERIED, zorder=2, label="Unqueried"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(bottom=0, top=n_all * 1.05)
        ax.legend(fontsize=7, loc="upper right")

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

    def _capture_frame(self) -> None:
        """Render the current figure into an in-memory PNG buffer."""
        buf = io.BytesIO()
        try:
            self._fig.savefig(buf, format="png", dpi=600, bbox_inches="tight")
            self._frames.append(buf.getvalue())
        except Exception as exc:
            logger.warning("Could not capture dashboard frame: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the current figure to a file.

        Parameters
        ----------
        path : str or Path
            Destination file path (format inferred from extension).
        """
        self._fig.savefig(path, dpi=600, bbox_inches="tight")

    def save_gif(
        self,
        path: str | Path,
        frame_duration_ms: int = 500,
        last_frame_duration_ms: int = 5000,
    ) -> None:
        """Assemble all captured frames into an animated GIF.

        Uses the in-memory PNG frames captured by :meth:`_capture_frame` after
        every :meth:`update` call. This method is a no-op (with a warning) when
        no updates have been made yet.

        Parameters
        ----------
        path : str or Path
            Destination file path for the GIF.
        frame_duration_ms : int, optional
            Display duration of each frame in milliseconds. Default is 500
            (half a second per iteration frame).
        last_frame_duration_ms : int, optional
            Display duration of the final frame in milliseconds, allowing the
            viewer to read the completed state before the animation loops.
            Default is 5000 (5 seconds).
        """
        if not self._frames:
            logger.warning("No frames captured; skipping GIF")
            return

        try:
            from PIL import Image

            frames = [Image.open(io.BytesIO(f)).convert("RGB") for f in self._frames]
            palette_frames = [f.convert("P", dither=Image.Dither.NONE) for f in frames]
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
                len(self._frames),
                path,
            )
        except Exception as exc:
            logger.warning("Could not save dashboard GIF: %s", exc)

    def close(self) -> None:
        """Close the matplotlib figure and release resources."""
        plt.close(self._fig)
