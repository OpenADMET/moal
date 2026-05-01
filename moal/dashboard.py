"""LiveDashboard: Plotly + Dash live-updating campaign dashboard.

Four subplots (2×2) update after every active learning iteration:
  1. Cumulative Actives Curve  — x: cumulative cost ($), y: actives found
  2. Per-Iteration Cost Breakdown — stacked bars (DRC new, PS→DRC upgrades, PS)
                                    with cumulative cost line on secondary y-axis
  3. Model Performance Curve   — x: iteration, y: configurable metric on test set
  4. Compound Status           — bar per category (PS-only, DRC, Unqueried) with
                                 PS→DRC upgrades stacked on top of each relevant bar

The Dash server runs in a background daemon thread for live in-browser viewing.
After simulation completes, call ``save_html()`` to export the final animated
figure with a scrub slider and play/pause controls, then ``close()`` to stop the server.
"""

from __future__ import annotations

import io
import itertools
import logging
import math
import threading
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import Locator, MaxNLocator
from PIL import Image
from plotly.subplots import make_subplots
from werkzeug.serving import make_server

from moal.evaluation import ModelMetric
from moal.types import CensoringType, LabelRecord, QueryType

logger = logging.getLogger(__name__)

# Suppress noisy Werkzeug HTTP access logs during simulation
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# Colour palette (kept consistent with previous terminal palette)
_COLOUR_DRC = "#E07B39"  # orange
_COLOUR_PS = "#4C9BE8"  # blue
_COLOUR_ACT = "#2CA02C"  # green
_COLOUR_MET = "#D62728"  # red
_COLOUR_UPGRADE = "#9B59B6"  # purple-magenta (PS→DRC upgrades)
_COLOUR_UNQUERIED = "#888888"  # mid-grey

# DPI used for matplotlib GIF/PNG frame rendering — controls output pixel density
_GIF_RENDER_DPI: int = 150

# Best-effort HTML background colours keyed by Plotly template name
_THEME_BG: dict[str, str] = {
    "plotly_dark": "#111111",
    "plotly": "#ffffff",
    "plotly_white": "#ffffff",
    "ggplot2": "#e5e5e5",
    "seaborn": "#eaeaf2",
    "simple_white": "#ffffff",
}
_EASING = "circle"

# JS injected after Plotly initialisation in the HTML export to provide a single
# play/pause toggle button; uses querySelector so it avoids {plot_id} substitution issues
_PLAY_PAUSE_SCRIPT = r"""
(function() {
  var gd = document.querySelector('.js-plotly-plot');
  if (!gd) return;
  var playing = false;
  var playToken = 0;
  var toolbar = document.createElement('div');
  toolbar.style.cssText = 'margin:4px 0;padding-left:4px;';
  var btn = document.createElement('button');
  btn.innerHTML = '\u25B6 Play';
  btn.style.cssText = 'padding:5px 14px;font-size:13px;cursor:pointer;'
    + 'background:#555;color:#fff;border:1px solid #888;border-radius:4px;';
  toolbar.appendChild(btn);
  gd.insertAdjacentElement('beforebegin', toolbar);

  function setPlaying(isPlaying) {
    playing = isPlaying;
    btn.innerHTML = isPlaying ? '\u23F8 Pause' : '\u25B6 Play';
  }

  function getFrameNames() {
    var frames = (gd._transitionData && gd._transitionData._frames) || [];
    return frames.map(function(frame) { return frame.name; }).filter(Boolean);
  }

  function getStartIndex(frameNames) {
    var sliders = gd._fullLayout && gd._fullLayout.sliders;
    var active = sliders && sliders.length ? sliders[0].active : 0;
    if (typeof active !== 'number' || !isFinite(active) || active < 0) {
      return 0;
    }
    if (active >= frameNames.length - 1) {
      return 0;
    }
    return active;
  }

  function stopAnimation() {
    playToken += 1;
    setPlaying(false);
    return Plotly.animate(gd, [null], {
      mode: 'immediate',
      frame: {duration: 0, redraw: false},
      transition: {duration: 0, ordering: 'layout first'}
    });
  }

  function playAnimation() {
    var frameNames = getFrameNames();
    if (!frameNames.length) return Promise.resolve();

    var startIndex = getStartIndex(frameNames);
    var token = playToken + 1;
    playToken = token;
    setPlaying(true);

    return Plotly.animate(gd, [frameNames[startIndex]], {
      mode: 'immediate',
      frame: {duration: 0, redraw: false},
      transition: {duration: 0, ordering: 'layout first'}
    }).then(function() {
      if (token !== playToken) return;

      var remainingFrames = frameNames.slice(startIndex + 1);
      if (!remainingFrames.length) return;

      return Plotly.animate(gd, remainingFrames, {
        mode: 'afterall',
        frame: {duration: 700, redraw: false},
        transition: {
          duration: 300,
          easing: '__EASING__',
          ordering: 'layout first'
        }
      });
    }).catch(function() {
      return;
    }).then(function() {
      if (token === playToken) {
        setPlaying(false);
      }
    });
  }

  btn.addEventListener('click', function() {
    if (!playing) {
      playAnimation();
    } else {
      stopAnimation();
    }
  });

  playAnimation();
})();
"""
_PLAY_PAUSE_SCRIPT = _PLAY_PAUSE_SCRIPT.replace("__EASING__", _EASING)

# Axis index constants for the 2×2 subplot with secondary_y on (1,2)
# Row 1 col 1 → x, y1 | Row 1 col 2 → x2, y2 (primary), y3 (secondary)
# Row 2 col 1 → x3, y4 | Row 2 col 2 → x4, y5
_SUBPLOT_SPECS = [[{}, {"secondary_y": True}], [{}, {}]]


class _OneDPLocator(Locator):
    """Tick locator that restricts steps to the 1-2-5 sequence with step >= 0.1.

    This prevents the model performance y-axis from ever displaying ticks that
    require more than one decimal place (e.g. 0.35, 0.40, 0.45), which occurs
    when a narrow metric range causes MaxNLocator to choose step=0.05 or smaller.
    Falls back to showing the single mid-point value when the span is near zero.
    """

    _STEPS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    _MAX_TICKS = 6

    def __call__(self) -> list[float]:
        """Return tick positions for the current axis view interval.

        Delegates to :meth:`tick_values` using the axis view interval obtained
        from ``self.axis.get_view_interval()``.

        Returns
        -------
        list[float]
            Tick positions restricted to the 1-2-5 step sequence with
            step >= 0.1.
        """
        if self.axis is None:
            raise RuntimeError("_OneDPLocator.__call__ requires a bound axis")
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        """Compute tick positions for the given axis range.

        Selects the smallest step from the 1-2-5 sequence (starting at 0.1)
        such that at most ``_MAX_TICKS - 1`` steps fit within ``[vmin, vmax]``.
        Falls back to a single midpoint value when the span is near zero.

        Parameters
        ----------
        vmin : float
            Lower bound of the axis range.
        vmax : float
            Upper bound of the axis range.

        Returns
        -------
        list[float]
            Tick positions rounded to ten decimal places.
        """
        span = vmax - vmin
        if span < 1e-12:
            return [round((vmin + vmax) / 2, 10)]
        for step in self._STEPS:
            if span / step <= self._MAX_TICKS - 1:
                start = np.floor(vmin / step) * step
                end = vmax + step * 1e-9
                ticks = np.arange(start, end, step)
                return [round(float(t), 10) for t in ticks]
        # Safety fallback: use first and last value only
        return [round(vmin, 10), round(vmax, 10)]


class LiveDashboard:
    """Four-panel live-updating campaign dashboard backed by Plotly + Dash.

    Parameters
    ----------
    n_iterations : int
        Total planned iterations (used to label the x-axis upper bound).
    n_compounds : int, optional
        Total pool size (used to compute the unqueried count). Default is 0.
    model_metric : ModelMetric, optional
        Metric to show in the model performance panel. Default is MAE.
    port : int, optional
        Local port for the Dash server (bound to 127.0.0.1 only). Default is 8050.
    export_width : int, optional
        Pixel width used when exporting PNG frames and the HTML animation.
        Default is 1400.
    export_height : int, optional
        Pixel height used when exporting PNG frames and the HTML animation.
        Default is 800.
    theme : str, optional
        Plotly template applied to all figure renders. Any valid Plotly template
        name is accepted (e.g., ``"plotly"``, ``"plotly_white"``, ``"ggplot2"``).
        Default is ``"plotly_dark"``.
    """

    def __init__(
        self,
        n_iterations: int,
        n_compounds: int = 0,
        model_metric: ModelMetric = ModelMetric.MAE,
        port: int = 8050,
        export_width: int = 1400,
        export_height: int = 800,
        theme: str = "plotly_dark",
    ) -> None:
        self.n_iterations = n_iterations
        self.n_compounds = n_compounds
        self.model_metric = model_metric
        self._port = port
        self._export_width = export_width
        self._export_height = export_height
        self._theme = theme
        self._lock = threading.Lock()
        # One dict per update() call; consumed by the Dash callback and save_html()
        self._iterations: list[dict] = []
        # Pre-rendered PNG bytes captured at each update() for fast GIF assembly
        self._frames: list[bytes] = []
        self._metric_label = model_metric.value.upper().replace("_", " ")

        self._app = Dash(__name__, suppress_callback_exceptions=True)
        # Silence the Flask/Dash internal logger; warnings surface via moal's logger
        self._app.logger.setLevel(logging.ERROR)

        bg = _THEME_BG.get(theme.lower(), "#ffffff")

        self._app.layout = html.Div(
            [
                # Seed with the loading splash so there is no flash of a default blank figure
                # before the first Interval callback round-trip completes
                dcc.Graph(id="live-graph", figure=self._build_figure([])),
                dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
            ],
            style={"backgroundColor": bg},
        )

        @self._app.callback(
            Output("live-graph", "figure"),
            Input("interval-component", "n_intervals"),
        )
        def _refresh(_n: int) -> go.Figure:
            """Rebuild the live figure from the latest iteration snapshots.

            Parameters
            ----------
            _n : int
                Interval tick counter supplied by the Dash ``Interval``
                component; the value itself is unused — the callback fires on
                every tick regardless.

            Returns
            -------
            plotly.graph_objects.Figure
                Up-to-date four-panel figure built from all accumulated
                iterations.
            """
            with self._lock:
                iterations = list(self._iterations)
            return self._build_figure(iterations)

        # Attempt to bind the port; log a warning and continue without a server on failure
        self._server_active = False
        try:
            self._werkzeug_server = make_server("127.0.0.1", port, self._app.server)
            self._thread = threading.Thread(target=self._werkzeug_server.serve_forever, daemon=True)
            self._thread.start()
            self._server_active = True
            url = f"http://127.0.0.1:{port}"
            logger.info("Live dashboard available at %s", url)
            webbrowser.open(url)
        except OSError as exc:
            logger.warning(
                "Could not start dashboard server on port %d (%s); "
                "live browser view will be unavailable — HTML export will still work",
                port,
                exc,
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
        iter_upgrade_cost: float = 0.0,
        iter_n_drc_new: int = 0,
        iter_n_upgrades: int = 0,
        iter_n_ps: int = 0,
        model_metric_value: float | None = None,
    ) -> None:
        """Append iteration data for live display and deferred GIF/HTML export.

        Parameters
        ----------
        labeled_records : list[LabelRecord]
            All oracle-labeled records accumulated so far.
        activity_threshold : float
            pEC50 threshold defining a confirmed active.
        iter_drc_cost : float
            Total DRC cost incurred in the current iteration (includes upgrades).
        iter_ps_cost : float
            Total PS cost incurred in the current iteration.
        iter_upgrade_cost : float, optional
            Portion of ``iter_drc_cost`` attributable to PS→DRC upgrades.
        iter_n_drc_new : int, optional
            Number of new (first-pass) DRC queries issued this iteration.
        iter_n_upgrades : int, optional
            Number of PS→DRC upgrade queries issued this iteration.
        iter_n_ps : int, optional
            Number of PS queries issued this iteration.
        model_metric_value : float, optional
            Held-out test-set metric for this iteration, or None if unavailable.
        """
        ps_smiles = {
            r.canonical_smiles for r in labeled_records if r.fidelity == QueryType.PRIMARY_SCREEN
        }
        drc_smiles = {
            r.canonical_smiles for r in labeled_records if r.fidelity == QueryType.DOSE_RESPONSE
        }

        n_upgrades = len(ps_smiles & drc_smiles)
        n_ps_only = len(ps_smiles) - n_upgrades
        n_drc_new = len(drc_smiles) - n_upgrades
        n_queried = n_ps_only + len(drc_smiles)
        n_unqueried = max(self.n_compounds - n_queried, 0)

        cum_actives = sum(
            1 for r in labeled_records if self._record_is_active(r, activity_threshold)
        )
        cum_cost = sum(r.cost for r in labeled_records)

        snapshot = {
            "cum_cost": cum_cost,
            "cum_actives": cum_actives,
            "iter_drc_cost": iter_drc_cost,
            "iter_ps_cost": iter_ps_cost,
            "iter_upgrade_cost": iter_upgrade_cost,
            "iter_n_drc_new": iter_n_drc_new,
            "iter_n_upgrades": iter_n_upgrades,
            "iter_n_ps": iter_n_ps,
            "model_metric_value": model_metric_value,
            "n_ps_only": n_ps_only,
            "n_drc_new": n_drc_new,
            "n_upgrades": n_upgrades,
            "n_unqueried": n_unqueried,
        }

        with self._lock:
            self._iterations.append(snapshot)

        # Pre-render at update time so save_gif() is pure PIL assembly (no external renderer)
        self._capture_frame()

    # ------------------------------------------------------------------
    # Persistence and export
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the current figure as a static PNG.

        Uses the same matplotlib renderer as GIF export — no kaleido required.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        with self._lock:
            iterations = list(self._iterations)
        try:
            png = self._render_matplotlib_frame(iterations)
            Path(path).write_bytes(png)
            logger.info("Static dashboard PNG saved to %s", path)
        except Exception as exc:
            logger.warning("Could not save static dashboard PNG: %s", exc)

    def save_gif(
        self,
        path: str | Path,
        frame_duration_ms: int = 500,
        last_frame_duration_ms: int = 5000,
    ) -> None:
        """Assemble all captured PNG frames into an animated GIF.

        Frames are pre-rendered by matplotlib's Agg backend at each update()
        call, so this method is pure PIL assembly with no external renderer.

        Parameters
        ----------
        path : str or Path
            Destination file path for the GIF.
        frame_duration_ms : int, optional
            Display duration of each frame in milliseconds. Default is 500.
        last_frame_duration_ms : int, optional
            Display duration of the final frame in milliseconds. Default is 5000.
        """
        with self._lock:
            frames = list(self._frames)

        if not frames:
            logger.warning("No frames captured; skipping GIF export")
            return

        try:
            pil_frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in frames]
            palette_frames = [f.convert("P", dither=Image.Dither.NONE) for f in pil_frames]
            durations = [frame_duration_ms] * len(palette_frames)
            durations[-1] = last_frame_duration_ms
            palette_frames[0].save(
                path,
                format="GIF",
                save_all=True,
                append_images=palette_frames[1:],
                duration=durations,
                loop=0,
                optimize=False,
            )
            logger.info(
                "GIF animation (%d frames) saved to %s",
                len(frames),
                path,
            )
        except Exception as exc:
            logger.warning("Could not save dashboard GIF: %s", exc)

    def save_html(self, path: str | Path, *, use_cdn: bool = False) -> None:
        """Export the animated figure as a standalone HTML file.

        The exported file embeds an iteration slider and play/pause buttons so
        the user can scrub through all simulation iterations offline, without a
        running Dash server.

        Parameters
        ----------
        path : str or Path
            Destination file path (should end in ``.html``).
        use_cdn : bool, optional
            When ``True``, the Plotly JS bundle is loaded from the Plotly CDN
            instead of being embedded in the file.  This reduces the file size
            from ~3 MB to a few KB but requires an internet connection to view.
            Default is ``False`` (fully self-contained file).
        """
        animated_fig = self._build_animated_figure()
        animated_fig.update_layout(width=self._export_width, height=self._export_height)
        animated_fig.write_html(
            str(path),
            include_plotlyjs="cdn" if use_cdn else True,
            post_script=_PLAY_PAUSE_SCRIPT,
            auto_play=False,
        )
        logger.info("Animated HTML dashboard saved to %s", path)

    def close(self) -> None:
        """Shut down the Werkzeug server and release the background thread.

        A no-op when the server was never successfully started (e.g., because
        the requested port was already in use at construction time).
        """
        if not self._server_active:
            return
        try:
            self._werkzeug_server.shutdown()
            self._thread.join(timeout=5.0)
        except Exception as exc:
            logger.warning("Error shutting down dashboard server: %s", exc)

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _capture_frame(self) -> None:
        """Render the current iteration state to PNG bytes and append to ``_frames``.

        Acquires the iteration lock, delegates rendering to
        :meth:`_render_matplotlib_frame`, and appends the result to
        ``self._frames``. Exceptions are caught and logged as warnings so a
        rendering failure does not abort the active-learning loop.
        """
        with self._lock:
            iterations = list(self._iterations)
        try:
            self._frames.append(self._render_matplotlib_frame(iterations))
        except Exception as exc:
            logger.warning("Could not capture dashboard frame: %s", exc)

    def _render_matplotlib_frame(self, iterations: list[dict]) -> bytes:
        """Build a 2×2 matplotlib figure from iteration snapshots and return PNG bytes.

        Uses ``FigureCanvasAgg`` directly so no display environment or pyplot
        state is required — safe to call from background threads and headless CI.

        Parameters
        ----------
        iterations : list[dict]
            Accumulated iteration snapshots as produced by :meth:`update`.
            Each dict contains the keys ``cum_cost``, ``cum_actives``,
            ``iter_drc_cost``, ``iter_upgrade_cost``, ``iter_ps_cost``,
            ``model_metric_value``, ``n_ps_only``, ``n_drc_new``,
            ``n_upgrades``, and ``n_unqueried``.

        Returns
        -------
        bytes
            Raw PNG image bytes rendered at ``_GIF_RENDER_DPI`` resolution.
        """
        w = self._export_width / _GIF_RENDER_DPI
        h = self._export_height / _GIF_RENDER_DPI
        fig = Figure(figsize=(w, h), dpi=_GIF_RENDER_DPI)
        FigureCanvasAgg(fig)  # attaches the Agg backend so fig.savefig() works headlessly

        ax1, ax2, ax3, ax4 = fig.subplots(2, 2).flatten()

        cum_costs = [0.0, *[it["cum_cost"] for it in iterations]]
        cum_actives = [0, *[it["cum_actives"] for it in iterations]]
        iter_nums = list(range(1, len(iterations) + 1))
        iter_drc_new = [it["iter_drc_cost"] - it["iter_upgrade_cost"] for it in iterations]
        iter_upgrades = [it["iter_upgrade_cost"] for it in iterations]
        iter_ps = [it["iter_ps_cost"] for it in iterations]
        cum_total_costs = list(
            itertools.accumulate(it["iter_drc_cost"] + it["iter_ps_cost"] for it in iterations)
        )
        cum_total_costs_k = [c / 1000 for c in cum_total_costs]
        metric_iters = [
            i + 1 for i, it in enumerate(iterations) if it["model_metric_value"] is not None
        ]
        metric_vals = [
            it["model_metric_value"] for it in iterations if it["model_metric_value"] is not None
        ]
        last = (
            iterations[-1]
            if iterations
            else {
                "n_ps_only": 0,
                "n_drc_new": 0,
                "n_upgrades": 0,
                "n_unqueried": self.n_compounds,
            }
        )
        # Floor at 1.05 guarantees 0 and 1 are labelled even before any data arrives
        actives_y_max = max(1.05, max(cum_actives) * 1.05) if cum_actives else 1.05
        cost_k_y_max = max(1.05, max(cum_total_costs_k) * 1.05) if cum_total_costs_k else 1.05

        # Panel 1: Cumulative Actives
        ax1.plot(
            cum_costs,
            cum_actives,
            color=_COLOUR_ACT,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax1.set_title("Cumulative Actives", fontsize=9)
        ax1.set_xlabel("Cumulative Cost ($)", fontsize=8)
        ax1.set_ylabel("Actives Found", fontsize=8)
        # Explicit range ensures 0 and 1 are always labelled even before any actives are found
        ax1.set_ylim(-0.05, actives_y_max)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=1))
        # Integer y-ticks only — actives are whole numbers; min_n_ticks=1 prevents
        # fallback to float ticks when fewer than two integers are within the view
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=1))

        # Panel 2: Per-Iteration Cost Breakdown — stacked bars + cumulative cost line
        # ax2_r is always created so it can hold its label and default range even with no data
        ax2_r = ax2.twinx()
        ax2_r.set_ylabel("Cumulative Cost ($k)", fontsize=8)
        ax2_r.tick_params(axis="y", labelsize=7)
        # Explicit range ensures 0 and 1 are always labelled even on the first frame
        ax2_r.set_ylim(0, cost_k_y_max)
        # nbins=5 caps tick density; integer=True + min_n_ticks=1 prevents fractional
        # $k labels when fewer than two integers are within the view
        ax2_r.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=1))
        if iter_nums:
            ax2.bar(iter_nums, iter_drc_new, color=_COLOUR_DRC, label="DRC")
            ax2.bar(
                iter_nums,
                iter_upgrades,
                bottom=iter_drc_new,
                color=_COLOUR_UPGRADE,
                label="PS→DRC",
            )
            ps_bottoms = [d + u for d, u in zip(iter_drc_new, iter_upgrades, strict=False)]
            ax2.bar(iter_nums, iter_ps, bottom=ps_bottoms, color=_COLOUR_PS, label="PS")
            # Scale to thousands so tick values stay compact whole-number integers
            ax2_r.plot(
                iter_nums,
                cum_total_costs_k,
                color="#555555",
                linewidth=2,
                linestyle="--",
            )

        ax2.set_title("Per-Iteration Cost Breakdown", fontsize=9)
        ax2.set_xlabel("Iteration", fontsize=8)
        ax2.set_ylabel("Iteration Cost ($)", fontsize=8)
        ax2.set_xlim(0.5, max(1, len(iterations)) + 0.5)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=1))

        # Panel 3: Model Performance
        if metric_iters:
            ax3.plot(
                metric_iters,
                metric_vals,
                color=_COLOUR_MET,
                linewidth=2,
                marker="o",
                markersize=4,
            )
        else:
            ax3.text(
                0.5,
                0.5,
                "No test set provided",
                transform=ax3.transAxes,
                ha="center",
                va="center",
                color="grey",
                fontsize=9,
            )
        ax3.set_title(f"Model Performance ({self._metric_label})", fontsize=9)
        ax3.set_xlabel("Iteration", fontsize=8)
        ax3.set_ylabel(self._metric_label, fontsize=8)
        # _OneDPLocator restricts steps to 1-2-5 sequence with step >= 0.1,
        # preventing ticks that need more than one decimal place on any frame.
        # The explicit ylim from _metric_axis_params guarantees the view always
        # spans at least one full step, so _OneDPLocator sees >= 2 tick positions
        ax3.yaxis.set_major_locator(_OneDPLocator())
        metric_ymin, metric_ymax, _, _ = self._metric_axis_params(metric_vals)
        ax3.set_ylim(metric_ymin, metric_ymax)
        ax3.set_xlim(0.5, max(1, len(iterations)) + 0.5)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=1))

        # Panel 4: Compound Status — current pool state only (last snapshot)
        categories = ["Unqueried", "PS", "DRC"]
        base_vals = [last["n_unqueried"], last["n_ps_only"], last["n_drc_new"]]
        base_colors = [_COLOUR_UNQUERIED, _COLOUR_PS, _COLOUR_DRC]
        upgrade_vals = [0, last["n_upgrades"], last["n_upgrades"]]
        ax4.bar(categories, base_vals, color=base_colors, label=categories)
        ax4.bar(
            categories,
            upgrade_vals,
            bottom=base_vals,
            color=_COLOUR_UPGRADE,
            label="PS→DRC",
        )
        ax4.set_title("Compound Status", fontsize=9)
        ax4.set_xlabel("Category", fontsize=8)
        ax4.set_ylabel("Compounds", fontsize=8)
        ax4.set_ylim(0, 1.05 * max(self.n_compounds, 1))

        ax4.legend(loc="upper right", fontsize=7)

        fig.suptitle("Active Learning Campaign Dashboard", fontsize=11, fontweight="bold")
        fig.tight_layout(pad=2.0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_GIF_RENDER_DPI)
        return buf.getvalue()

    @staticmethod
    def _nice_dtick(max_val: float, max_ticks: int = 5) -> int:
        """Return the smallest integer step that keeps tick count at or below ``max_ticks``.

        Guarantees non-repeating integer labels on any axis range. The candidate
        list covers values from 1 to 10 000, which is sufficient for actives counts
        and $k-scaled cumulative costs on typical campaigns.

        Parameters
        ----------
        max_val : float
            Maximum axis value to accommodate.
        max_ticks : int, optional
            Upper bound on the number of tick marks. Default is 5.

        Returns
        -------
        int
            Tick step size chosen from the standard 1-2-5 sequence.
        """
        if max_val <= 0:
            return 1
        raw_step = max_val / max_ticks
        for candidate in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
            if candidate >= raw_step:
                return candidate
        return max(1, int(max_val))

    @staticmethod
    def _metric_axis_params(
        metric_vals: list[float],
    ) -> tuple[float, float, float, float]:
        """Return ``(ymin, ymax, tick0, dtick)`` for the model performance y-axis.

        Guarantees that at least two tick positions (multiples of ``dtick`` starting
        from ``tick0``) fall within ``[ymin, ymax]``, without requiring labels with
        more than one decimal place. The step is chosen from the 1-2-5 sequence
        based on the span of the data so the panel scales sensibly across both
        narrow early-iteration ranges and large final-iteration ranges.

        Parameters
        ----------
        metric_vals : list[float]
            Metric values collected so far. An empty list produces a default
            range of ``[0.0, 0.2]`` with ``dtick=0.1``.

        Returns
        -------
        tuple[float, float, float, float]
            A ``(ymin, ymax, tick0, dtick)`` tuple suitable for configuring a
            Plotly or matplotlib y-axis.
        """
        _steps = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        _max_ticks = 6

        if not metric_vals:
            return (0.0, 0.2, 0.0, 0.1)

        vmin_data = min(metric_vals)
        vmax_data = max(metric_vals)
        span = vmax_data - vmin_data

        step = _steps[-1]
        for s in _steps:
            if span < 1e-12 or span / s <= _max_ticks - 1:
                step = s
                break

        t_lo = math.floor(vmin_data / step) * step
        t_hi = math.ceil(vmax_data / step) * step

        # Guarantee at least two distinct tick positions
        if t_hi <= t_lo + step * 0.5:
            t_hi = t_lo + step

        margin = step * 0.05
        return (t_lo - margin, t_hi + margin, t_lo, step)

    def _build_figure(self, iterations: list[dict]) -> go.Figure:
        """Build the 2×2 Plotly subplot figure from a list of iteration snapshots.

        Returns a loading-splash figure when ``iterations`` is empty, and a
        fully populated four-panel figure otherwise.

        Parameters
        ----------
        iterations : list[dict]
            Accumulated iteration snapshots as produced by :meth:`update`.

        Returns
        -------
        plotly.graph_objects.Figure
            Four-panel Plotly figure with cumulative actives, per-iteration
            cost breakdown, model performance, and compound status subplots.
        """
        if not iterations:
            bg = _THEME_BG.get(self._theme.lower(), "#ffffff")
            fig = go.Figure()
            fig.add_annotation(
                text="Waiting for first iteration\u2026",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color="#aaaaaa"),
            )
            fig.update_layout(
                template=self._theme,
                paper_bgcolor=bg,
                plot_bgcolor=bg,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=self._export_height,
                width=self._export_width,
            )
            return fig

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=_SUBPLOT_SPECS,
            subplot_titles=[
                "Cumulative Actives",
                "Per-Iteration Cost Breakdown",
                f"Model Performance ({self._metric_label})",
                "Compound Status",
            ],
            vertical_spacing=0.18,
            horizontal_spacing=0.12,
        )

        cum_costs = [0.0, *[it["cum_cost"] for it in iterations]]
        cum_actives = [0, *[it["cum_actives"] for it in iterations]]
        iter_nums = list(range(1, len(iterations) + 1))
        iter_drc_new = [it["iter_drc_cost"] - it["iter_upgrade_cost"] for it in iterations]
        iter_upgrades = [it["iter_upgrade_cost"] for it in iterations]
        iter_ps = [it["iter_ps_cost"] for it in iterations]
        iter_n_drc_new = [it.get("iter_n_drc_new", 0) for it in iterations]
        iter_n_upgrades = [it.get("iter_n_upgrades", 0) for it in iterations]
        iter_n_ps = [it.get("iter_n_ps", 0) for it in iterations]
        # Scale to thousands so secondary y-axis ticks stay compact whole-number integers
        cum_total_costs = [
            c / 1000
            for c in itertools.accumulate(
                it["iter_drc_cost"] + it["iter_ps_cost"] for it in iterations
            )
        ]

        metric_iters = [
            i + 1 for i, it in enumerate(iterations) if it["model_metric_value"] is not None
        ]
        metric_vals = [
            it["model_metric_value"] for it in iterations if it["model_metric_value"] is not None
        ]

        last = iterations[-1]

        # Panel 1: Cumulative Actives line
        fig.add_trace(
            go.Scatter(
                x=cum_costs,
                y=cum_actives,
                mode="lines+markers",
                name="Actives",
                line=dict(color=_COLOUR_ACT, width=2),
                marker=dict(size=6),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Panel 2: Stacked cost bars — legend entries suppressed; panel 4 owns all legend items
        fig.add_trace(
            go.Bar(
                x=iter_nums,
                y=iter_drc_new,
                name="DRC",
                marker_color=_COLOUR_DRC,
                customdata=iter_n_drc_new,
                hovertemplate="<b>DRC</b><br>%{customdata} queries<br>$%{y:.0f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=iter_nums,
                y=iter_upgrades,
                name="PS→DRC",
                marker_color=_COLOUR_UPGRADE,
                customdata=iter_n_upgrades,
                hovertemplate="<b>PS→DRC</b><br>%{customdata} upgrades<br>$%{y:.0f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=iter_nums,
                y=iter_ps,
                name="PS",
                marker_color=_COLOUR_PS,
                customdata=iter_n_ps,
                hovertemplate="<b>PS</b><br>%{customdata} queries<br>$%{y:.0f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Cumulative cost line on secondary y-axis for panel 2 (scaled to $k)
        line_color = "#FFFFFF" if self._theme == "plotly_dark" else "#555555"
        fig.add_trace(
            go.Scatter(
                x=iter_nums,
                y=cum_total_costs,
                mode="lines",
                name="Cumulative Cost ($k)",
                line=dict(color=line_color, width=2, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=2,
            secondary_y=True,
        )

        # Panel 3: Model metric line
        fig.add_trace(
            go.Scatter(
                x=metric_iters,
                y=metric_vals,
                mode="lines+markers",
                name=self._metric_label,
                line=dict(color=_COLOUR_MET, width=2),
                marker=dict(size=6),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Panel 4: Compound status — source of truth for the unified legend
        # (canonical order via legendrank)
        for cat_name, y_val, color, rank in [
            ("Unqueried", last["n_unqueried"], _COLOUR_UNQUERIED, 4),
            ("PS", last["n_ps_only"], _COLOUR_PS, 3),
            ("DRC", last["n_drc_new"], _COLOUR_DRC, 1),
        ]:
            fig.add_trace(
                go.Bar(
                    x=[cat_name],
                    y=[y_val],
                    name=cat_name,
                    marker_color=color,
                    legendrank=rank,
                ),
                row=2,
                col=2,
            )
        # PS→DRC upgrade overlay stacked on top of PS and DRC bars
        fig.add_trace(
            go.Bar(
                x=["PS", "DRC", "Unqueried"],
                y=[last["n_upgrades"], last["n_upgrades"], 0],
                name="PS→DRC",
                marker_color=_COLOUR_UPGRADE,
                legendrank=2,
            ),
            row=2,
            col=2,
        )

        # "No test set" annotation when no metric data is available
        if not metric_vals:
            fig.add_annotation(
                text="No test set provided",
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color="grey", size=12),
                row=2,
                col=1,
            )

        # Pre-compute data maxima for dynamic integer dtick on actives and $k axes;
        # floor at 1.05 ensures both 0 and 1 are always visible even with no data
        max_actives = max(cum_actives, default=0)
        max_cost_k = max(cum_total_costs, default=0)
        actives_y_max = max(1.05, max_actives * 1.05)
        cost_k_y_max = max(1.05, max_cost_k * 1.05)
        # Metric axis: explicit range + tick0/dtick computed from data span so at
        # least two tick labels are always visible even on narrow early-iteration ranges
        metric_ymin, metric_ymax, metric_tick0, metric_dtick = self._metric_axis_params(metric_vals)

        fig.update_layout(
            title_text="Active Learning Campaign Dashboard",
            barmode="stack",
            template=self._theme,
            width=self._export_width,
            height=self._export_height,
            legend=dict(
                orientation="h",
                x=0.98,
                y=1.05,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(size=10),
                itemclick=False,
                itemdoubleclick=False,
            ),
        )
        # Y-axes: primary labels and floor; disable secondary gridlines to prevent clashing
        fig.update_yaxes(
            title_text="Actives Found",
            row=1,
            col=1,
            # Explicit range ensures 0 and 1 are visible even before any actives are found
            range=[0, actives_y_max],
            # Integer-only ticks; dtick computed from data so labels never repeat
            tickmode="linear",
            tick0=0,
            dtick=self._nice_dtick(max_actives),
        )
        fig.update_yaxes(title_text="Iteration Cost ($)", secondary_y=False, row=1, col=2)
        fig.update_yaxes(
            title_text="Cumulative Cost ($k)",
            secondary_y=True,
            row=1,
            col=2,
            showgrid=False,
            # Explicit range ensures 0 and 1 are visible even on the first frame;
            # integer ticks computed from data so labels never repeat across frames
            range=[0, cost_k_y_max],
            tickmode="linear",
            tick0=0,
            dtick=self._nice_dtick(max_cost_k),
        )
        fig.update_yaxes(
            title_text=self._metric_label,
            row=2,
            col=1,
            # Explicit range from _metric_axis_params guarantees >= 2 tick positions
            # are always visible. tick0/dtick are span-based (not max-based) so they
            # stay consistent with the range on every frame, including narrow early ones.
            # tickformat ".3~g" trims trailing zeros (e.g. 1.0 → "1", 0.5 → "0.5")
            range=[metric_ymin, metric_ymax],
            tickmode="linear",
            tick0=metric_tick0,
            dtick=metric_dtick,
            tickformat=".3~g",
        )
        fig.update_yaxes(
            title_text="Compounds",
            range=[0, 1.05 * max(self.n_compounds, 1)],
            row=2,
            col=2,
        )
        # X-axis labels — integer ticks on all numerical axes to avoid fractional iteration labels
        fig.update_xaxes(
            title_text="Cumulative Cost ($)",
            row=1,
            col=1,
            tickformat=",.0f",
            range=self._cumulative_cost_x_range(cum_costs),
        )
        fig.update_xaxes(
            title_text="Iteration",
            row=1,
            col=2,
            tickformat="d",
            tickmode="array",
            tickvals=self._iteration_x_tickvals(len(iterations)),
            range=self._iteration_x_range(len(iterations), padding=0.5),
        )
        fig.update_xaxes(
            title_text="Iteration",
            row=2,
            col=1,
            tickformat="d",
            tickmode="array",
            tickvals=self._iteration_x_tickvals(len(iterations)),
            range=self._iteration_x_range(len(iterations)),
        )
        fig.update_xaxes(
            title_text="Category",
            categoryorder="array",
            categoryarray=["Unqueried", "PS", "DRC"],
            row=2,
            col=2,
        )

        return fig

    def _build_animated_figure(self) -> go.Figure:
        """Construct an animated Plotly figure with per-iteration frames, slider, and play/pause.

        Acquires the iteration lock and builds one Plotly ``Frame`` per
        iteration so the HTML slider can scrub through cumulative history.
        Falls back to the loading-splash figure when no iterations have been
        recorded yet.

        Returns
        -------
        plotly.graph_objects.Figure
            Animated figure containing per-iteration frames, a full-width
            scrub slider, and an empty ``updatemenus`` list (play/pause is
            injected via ``_PLAY_PAUSE_SCRIPT`` in :meth:`save_html`).
        """
        with self._lock:
            iterations = list(self._iterations)

        if not iterations:
            return self._build_figure([])

        initial_fig = self._build_figure(iterations[:1])

        # Build one frame per iteration so the slider steps through cumulative history
        frames = [
            self._build_animation_frame(iterations[: i + 1], i + 1) for i in range(len(iterations))
        ]

        animated_fig = go.Figure(data=initial_fig.data, layout=initial_fig.layout, frames=frames)

        steps = [
            {
                "args": [
                    [str(i + 1)],
                    {
                        "frame": {"duration": 700, "redraw": False},
                        "mode": "immediate",
                        "transition": {
                            "duration": 300,
                            "easing": _EASING,
                            "ordering": "layout first",
                        },
                    },
                ],
                "label": str(i + 1),
                "method": "animate",
            }
            for i in range(len(iterations))
        ]

        animated_fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "prefix": "Iteration: ",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "pad": {"b": 10, "t": 50},
                    # Full-width slider; play/pause is provided via injected DOM button
                    "len": 1.0,
                    "x": 0.0,
                    "y": 0,
                    "steps": steps,
                    # Hide per-step tick labels; the currentvalue readout is sufficient
                    "tickcolor": "rgba(0,0,0,0)",
                    "font": {"color": "rgba(0,0,0,0)", "size": 1},
                }
            ],
            # No Plotly updatemenus — play/pause toggle is injected via post_script in save_html
            updatemenus=[],
        )

        return animated_fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_animation_frame(self, iterations: list[dict], iteration_index: int) -> go.Frame:
        """Build a Plotly animation frame with both data and per-frame axis layout.

        Parameters
        ----------
        iterations : list[dict]
            Cumulative iteration snapshots up to and including the current
            iteration (i.e., a prefix of the full history).
        iteration_index : int
            One-based iteration number used as the frame name for slider
            step addressing.

        Returns
        -------
        plotly.graph_objects.Frame
            Plotly frame whose ``data`` and ``layout`` reflect the cumulative
            state at ``iteration_index``.
        """
        frame_fig = self._build_figure(iterations)
        return go.Frame(
            data=list(frame_fig.data),
            layout=self._build_animation_frame_layout(frame_fig, iterations),
            name=str(iteration_index),
            traces=list(range(len(frame_fig.data))),
        )

    def _build_animation_frame_layout(
        self, frame_fig: go.Figure, iterations: list[dict]
    ) -> go.Layout:
        """Return layout updates needed for exported HTML animation frame scrubbing.

        Reconstructs per-axis range settings so that x-axes rescale correctly
        as the slider advances through iteration frames in the exported HTML.

        Parameters
        ----------
        frame_fig : plotly.graph_objects.Figure
            Fully rendered figure for the current iteration prefix, used as
            the source of truth for all axis attributes not computed here.
        iterations : list[dict]
            Cumulative iteration snapshots up to and including the current
            iteration.

        Returns
        -------
        plotly.graph_objects.Layout
            Partial layout containing updated ``xaxis``/``yaxis`` entries for
            all five axes in the 2×2 subplot grid.
        """
        n_iterations = len(iterations)
        cum_costs = [it["cum_cost"] for it in iterations]

        return go.Layout(
            xaxis={
                **frame_fig.layout.xaxis.to_plotly_json(),
                "range": self._cumulative_cost_x_range(cum_costs),
            },
            xaxis2={
                **frame_fig.layout.xaxis2.to_plotly_json(),
                "range": self._iteration_x_range(n_iterations, padding=0.5),
                "tickmode": "array",
                "tickvals": self._iteration_x_tickvals(n_iterations),
            },
            xaxis3={
                **frame_fig.layout.xaxis3.to_plotly_json(),
                "range": self._iteration_x_range(n_iterations),
                "tickmode": "array",
                "tickvals": self._iteration_x_tickvals(n_iterations),
            },
            xaxis4=frame_fig.layout.xaxis4.to_plotly_json(),
            yaxis=frame_fig.layout.yaxis.to_plotly_json(),
            yaxis2=frame_fig.layout.yaxis2.to_plotly_json(),
            yaxis3=frame_fig.layout.yaxis3.to_plotly_json(),
            yaxis4=frame_fig.layout.yaxis4.to_plotly_json(),
            yaxis5=frame_fig.layout.yaxis5.to_plotly_json(),
        )

    @staticmethod
    def _cumulative_cost_x_range(cum_costs: list[float]) -> list[float]:
        """Return an explicit x-range for the cumulative-cost subplot.

        Pads the maximum observed cumulative cost by 5 % and floors the upper
        bound at 1.0 so the axis is never empty before the first data point.

        Parameters
        ----------
        cum_costs : list[float]
            Sequence of cumulative costs, one per recorded iteration.

        Returns
        -------
        list[float]
            Two-element list ``[0.0, upper]`` suitable for Plotly's
            ``range`` axis parameter.
        """
        max_cost = max(cum_costs, default=0.0)
        return [0.0, max(1.0, max_cost * 1.05)]

    @staticmethod
    def _iteration_x_tickvals(n_iterations: int) -> list[int]:
        """Return one tick value per completed iteration with no duplicates.

        Using ``tickmode="array"`` with these explicit values prevents Plotly's
        auto-tick algorithm from placing multiple ticks at the same integer when
        the axis range is narrow (e.g. at iteration 1 or 2).

        Parameters
        ----------
        n_iterations : int
            Number of iterations recorded so far.

        Returns
        -------
        list[int]
            ``[1, 2, ..., n_iterations]``, or ``[1]`` when *n_iterations* is 0
            so the axis always has at least one labelled tick.
        """
        return list(range(1, max(n_iterations, 1) + 1))

    @staticmethod
    def _iteration_x_range(n_iterations: int, padding: float = 0.25) -> list[float]:
        """Return an explicit x-range for iteration-indexed subplots.

        Parameters
        ----------
        n_iterations : int
            Number of iterations recorded so far.
        padding : float, optional
            Symmetric padding added to either side of the ``[1, n_iterations]``
            range so bars and markers are not clipped at the axis edges.
            Default is 0.25.

        Returns
        -------
        list[float]
            Two-element list ``[1 - padding, n_iterations + padding]``.
        """
        upper = max(float(n_iterations), 1.0)
        return [1.0 - padding, upper + padding]

    @staticmethod
    def _record_is_active(rec: LabelRecord, threshold: float) -> bool:
        """Return ``True`` when a labeled record's value meets the activity threshold.

        Only ``EXACT`` and ``INTERVAL``-censored records can be confirmed
        active. ``LEFT``-censored records (PS inactive labels, i.e. pEC50
        below the PS threshold) are always inactive and return ``False``.

        Parameters
        ----------
        rec : LabelRecord
            A single oracle-labeled compound record.
        threshold : float
            pEC50 threshold at or above which a compound is considered active.

        Returns
        -------
        bool
            ``True`` if ``rec.value >= threshold`` for an ``EXACT`` or
            ``INTERVAL`` record; ``False`` for a ``LEFT``-censored record.
        """
        if rec.censoring_type == CensoringType.EXACT:
            return rec.value >= threshold
        if rec.censoring_type == CensoringType.INTERVAL:
            return rec.value >= threshold
        return False
