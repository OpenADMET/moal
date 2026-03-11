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
import threading
import webbrowser
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
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

# Best-effort HTML background colours keyed by Plotly template name
_THEME_BG: dict[str, str] = {
    "plotly_dark": "#111111",
    "plotly": "#ffffff",
    "plotly_white": "#ffffff",
    "ggplot2": "#e5e5e5",
    "seaborn": "#eaeaf2",
    "simple_white": "#ffffff",
}

# JS injected after Plotly initialisation in the HTML export to provide a single
# play/pause toggle button; uses querySelector so it avoids {plot_id} substitution issues
_PLAY_PAUSE_SCRIPT = r"""
(function() {
  var gd = document.querySelector('.js-plotly-plot');
  if (!gd) return;
  var playing = false;
  var toolbar = document.createElement('div');
  toolbar.style.cssText = 'margin:4px 0;padding-left:4px;';
  var btn = document.createElement('button');
  btn.innerHTML = '\u25B6 Play';
  btn.style.cssText = 'padding:5px 14px;font-size:13px;cursor:pointer;background:#555;color:#fff;border:1px solid #888;border-radius:4px;';
  toolbar.appendChild(btn);
  gd.insertAdjacentElement('beforebegin', toolbar);
  gd.on('plotly_animated', function() {
    playing = false;
    btn.innerHTML = '\u25B6 Play';
  });
  btn.addEventListener('click', function() {
    if (!playing) {
      Plotly.animate(gd, null, {frame:{duration:500,redraw:true},fromcurrent:true,transition:{duration:200}});
      btn.innerHTML = '\u23F8 Pause';
      playing = true;
    } else {
      Plotly.animate(gd, [null], {frame:{duration:0,redraw:false},mode:'immediate',transition:{duration:0}});
      btn.innerHTML = '\u25B6 Play';
      playing = false;
    }
  });
})();
"""

# Axis index constants for the 2×2 subplot with secondary_y on (1,2)
# Row 1 col 1 → x, y1 | Row 1 col 2 → x2, y2 (primary), y3 (secondary)
# Row 2 col 1 → x3, y4 | Row 2 col 2 → x4, y5
_SUBPLOT_SPECS = [[{}, {"secondary_y": True}], [{}, {}]]


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
        Local port for the Dash server. Default is 8050.
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
        # Warn only once when kaleido is absent or fails
        self._kaleido_warned = False
        self._metric_label = model_metric.value.upper().replace("_", " ")

        self._app = Dash(__name__, suppress_callback_exceptions=True)
        # Silence the Flask/Dash internal logger; warnings surface via moal's logger
        self._app.logger.setLevel(logging.ERROR)

        bg = _THEME_BG.get(theme.lower(), "#ffffff")
        fg = "#ffffff" if "dark" in theme.lower() else "#000000"

        self._app.layout = html.Div(
            [
                html.H3(
                    "Active Learning Campaign Dashboard",
                    style={
                        "textAlign": "center",
                        "color": fg,
                        "backgroundColor": bg,
                        "padding": "12px",
                        "margin": "0",
                    },
                ),
                # No fixed height/width — the figure layout dimensions from config control size
                dcc.Graph(id="live-graph"),
                dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
            ],
            style={"backgroundColor": bg},
        )

        @self._app.callback(
            Output("live-graph", "figure"),
            Input("interval-component", "n_intervals"),
        )
        def _refresh(_n: int) -> go.Figure:
            with self._lock:
                iterations = list(self._iterations)
            return self._build_figure(iterations)

        # Attempt to bind the port; log a warning and continue without a server on failure
        self._server_active = False
        try:
            self._werkzeug_server = make_server("127.0.0.1", port, self._app.server)
            self._thread = threading.Thread(
                target=self._werkzeug_server.serve_forever, daemon=True
            )
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
        model_metric_value : float, optional
            Held-out test-set metric for this iteration, or None if unavailable.
        """
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
            "model_metric_value": model_metric_value,
            "n_ps_only": n_ps_only,
            "n_drc_new": n_drc_new,
            "n_upgrades": n_upgrades,
            "n_unqueried": n_unqueried,
        }

        with self._lock:
            self._iterations.append(snapshot)

    # ------------------------------------------------------------------
    # Persistence and export
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the current figure as a static PNG. Requires kaleido.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        try:
            import plotly.io as pio

            with self._lock:
                iterations = list(self._iterations)
            fig = self._build_figure(iterations)
            pio.write_image(
                fig,
                str(path),
                format="png",
                width=self._export_width,
                height=self._export_height,
            )
            logger.info("Static dashboard PNG saved to %s", path)
        except Exception as exc:
            logger.warning("Could not save static dashboard PNG: %s", exc)

    def save_gif(
        self,
        path: str | Path,
        frame_duration_ms: int = 500,
        last_frame_duration_ms: int = 5000,
    ) -> None:
        """Render all iteration snapshots to PNG via kaleido and assemble an animated GIF.

        Frames are rendered in a single batch at export time so kaleido opens and
        closes Chromium only once regardless of how many iterations were run.

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
            iterations = list(self._iterations)

        if not iterations:
            logger.warning("No iterations recorded; skipping GIF export")
            return

        try:
            import plotly.io as pio
            from PIL import Image

            frame_bytes: list[bytes] = []
            for i, _ in enumerate(iterations):
                fig = self._build_figure(iterations[: i + 1])
                png = pio.to_image(
                    fig,
                    format="png",
                    width=self._export_width,
                    height=self._export_height,
                )
                frame_bytes.append(png)

            frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in frame_bytes]
            palette_frames = [f.convert("P", dither=Image.Dither.NONE) for f in frames]
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
                "Dashboard animation (%d frames) saved to %s",
                len(frame_bytes),
                path,
            )
        except Exception as exc:
            if not self._kaleido_warned:
                logger.warning(
                    "GIF export failed (kaleido may not be installed or Chrome unavailable): %s",
                    exc,
                )
                self._kaleido_warned = True

    def save_html(self, path: str | Path) -> None:
        """Export the animated figure as a standalone HTML file.

        The exported file embeds an iteration slider and play/pause buttons so
        the user can scrub through all simulation iterations offline, without a
        running Dash server.

        Parameters
        ----------
        path : str or Path
            Destination file path (should end in ``.html``).
        """
        animated_fig = self._build_animated_figure()
        animated_fig.update_layout(width=self._export_width, height=self._export_height)
        animated_fig.write_html(
            str(path), include_plotlyjs=True, post_script=_PLAY_PAUSE_SCRIPT
        )
        logger.info("Animated HTML dashboard saved to %s", path)

    def close(self) -> None:
        """Shut down the Werkzeug server and release the background thread."""
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

    def _build_figure(self, iterations: list[dict]) -> go.Figure:
        """Build the 2×2 Plotly subplot figure from a list of iteration snapshots."""
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

        cum_costs = [it["cum_cost"] for it in iterations]
        cum_actives = [it["cum_actives"] for it in iterations]
        iter_nums = list(range(1, len(iterations) + 1))
        iter_drc_new = [
            it["iter_drc_cost"] - it["iter_upgrade_cost"] for it in iterations
        ]
        iter_upgrades = [it["iter_upgrade_cost"] for it in iterations]
        iter_ps = [it["iter_ps_cost"] for it in iterations]
        cum_total_costs = list(
            itertools.accumulate(
                it["iter_drc_cost"] + it["iter_ps_cost"] for it in iterations
            )
        )

        metric_iters = [
            i + 1
            for i, it in enumerate(iterations)
            if it["model_metric_value"] is not None
        ]
        metric_vals = [
            it["model_metric_value"]
            for it in iterations
            if it["model_metric_value"] is not None
        ]

        if iterations:
            last = iterations[-1]
        else:
            last = {
                "n_ps_only": 0,
                "n_drc_new": 0,
                "n_upgrades": 0,
                "n_unqueried": self.n_compounds,
            }

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
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Cumulative cost line on secondary y-axis for panel 2 (trace kept, label suppressed)
        fig.add_trace(
            go.Scatter(
                x=iter_nums,
                y=cum_total_costs,
                mode="lines",
                name="Cumulative Cost ($)",
                line=dict(color="#FFFFFF", width=1.5, dash="dot"),
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

        # Panel 4: Compound status — source of truth for the unified legend (canonical order via legendrank)
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
        fig.update_yaxes(rangemode="tozero", title_text="Actives Found", row=1, col=1)
        fig.update_yaxes(
            title_text="Iteration Cost ($)", secondary_y=False, row=1, col=2
        )
        fig.update_yaxes(title_text="", secondary_y=True, row=1, col=2, showgrid=False)
        fig.update_yaxes(title_text=self._metric_label, row=2, col=1)
        fig.update_yaxes(
            title_text="Compounds",
            range=[0, 1.05 * max(self.n_compounds, 1)],
            row=2,
            col=2,
        )
        # X-axis labels
        fig.update_xaxes(title_text="Cumulative Cost ($)", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_xaxes(
            title_text="Category",
            categoryorder="array",
            categoryarray=["Unqueried", "PS", "DRC"],
            row=2,
            col=2,
        )

        return fig

    def _build_animated_figure(self) -> go.Figure:
        """Construct an animated Plotly figure with per-iteration frames, slider, and play/pause."""
        with self._lock:
            iterations = list(self._iterations)

        if not iterations:
            return self._build_figure([])

        final_fig = self._build_figure(iterations)

        # Build one frame per iteration so the slider steps through cumulative history
        frames = [
            go.Frame(
                data=list(self._build_figure(iterations[: i + 1]).data),
                name=str(i + 1),
            )
            for i in range(len(iterations))
        ]

        animated_fig = go.Figure(
            data=final_fig.data, layout=final_fig.layout, frames=frames
        )

        steps = [
            {
                "args": [
                    [str(i + 1)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
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
                    "active": len(iterations) - 1,
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

    @staticmethod
    def _record_is_active(rec: LabelRecord, threshold: float) -> bool:
        """Return True when a labeled record's value meets the activity threshold."""
        if rec.censoring_type == CensoringType.EXACT:
            return rec.value >= threshold
        if rec.censoring_type == CensoringType.INTERVAL:
            return rec.value >= threshold
        return False
