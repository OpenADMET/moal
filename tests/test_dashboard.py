"""Tests for LiveDashboard — Plotly + Dash implementation."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from moal.dashboard import LiveDashboard
from moal.evaluation import ModelMetric
from moal.types import CensoringType, LabelRecord, QueryType

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_werkzeug_server(monkeypatch):
    """Prevent real Werkzeug server from binding a port in every test,
    and suppress browser auto-open calls."""
    server = MagicMock()
    monkeypatch.setattr("moal.dashboard.make_server", lambda *a, **kw: server)
    monkeypatch.setattr("moal.dashboard.webbrowser.open", lambda *a, **kw: None)
    return server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    iteration: int,
    value: float,
    censoring_type: CensoringType,
    fidelity: QueryType,
    cost: float,
    smiles: str = "C",
) -> LabelRecord:
    """Construct a single LabelRecord for dashboard tests."""
    return LabelRecord(
        smiles=smiles,
        canonical_smiles=smiles,
        value=value,
        upper_bound=value if censoring_type == CensoringType.EXACT else 11.0,
        censoring_type=censoring_type,
        fidelity=fidelity,
        cost=cost,
        iteration=iteration,
    )


def _make_records(n: int = 6) -> list[LabelRecord]:
    """Produce n alternating DRC/PS LabelRecords spanning multiple iterations."""
    records = []
    for i in range(n):
        ct = CensoringType.EXACT if i % 2 == 0 else CensoringType.INTERVAL
        fid = QueryType.DOSE_RESPONSE if i % 2 == 0 else QueryType.PRIMARY_SCREEN
        v = 5.0 + i * 0.5
        records.append(
            _make_record(
                i // 2,
                v,
                ct,
                fid,
                cost=10.0 if fid == QueryType.DOSE_RESPONSE else 1.0,
                smiles=str(i),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Construction and server lifecycle
# ---------------------------------------------------------------------------


class TestDashboardInit:
    """Tests for dashboard construction and background server lifecycle."""

    def test_iterations_empty_on_construction(self):
        """No iteration data should exist before the first update() call."""
        db = LiveDashboard(n_iterations=5, n_compounds=20)
        assert db._iterations == []
        db.close()

    def test_frames_empty_on_construction(self):
        """No pre-rendered frames should exist before the first update() call."""
        db = LiveDashboard(n_iterations=5, n_compounds=20)
        assert db._frames == []
        db.close()

    def test_close_shuts_down_server(self, mock_werkzeug_server):
        """close() must call shutdown() on the Werkzeug server."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        db.close()
        mock_werkzeug_server.shutdown.assert_called_once()

    def test_constructor_uses_configured_port(self, monkeypatch):
        """The Werkzeug server must be created with the port supplied to the constructor."""
        captured: list[tuple] = []
        monkeypatch.setattr(
            "moal.dashboard.make_server",
            lambda host, port, app: captured.append((host, port)) or MagicMock(),
        )
        db = LiveDashboard(n_iterations=3, n_compounds=20, port=9999)
        db.close()
        assert captured[0] == ("127.0.0.1", 9999)

    def test_port_conflict_logs_warning_not_raises(self, monkeypatch, caplog):
        """When the port is already in use, construction must warn and succeed without a server."""
        import logging

        monkeypatch.setattr(
            "moal.dashboard.make_server",
            lambda *a, **kw: (_ for _ in ()).throw(
                OSError(48, "Address already in use")
            ),
        )
        with caplog.at_level(logging.WARNING, logger="moal.dashboard"):
            db = LiveDashboard(n_iterations=3, n_compounds=20, port=8050)

        assert not db._server_active
        assert "Could not start dashboard server" in caplog.text
        # close() must be a no-op when no server was started
        db.close()  # must not raise

    def test_close_is_noop_when_server_not_active(self, mock_werkzeug_server):
        """close() must not call shutdown when _server_active is False."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        db._server_active = False
        db.close()
        mock_werkzeug_server.shutdown.assert_not_called()


# ---------------------------------------------------------------------------
# Update accumulation
# ---------------------------------------------------------------------------


class TestDashboardUpdate:
    """Tests for data accumulation via update()."""

    def test_single_update_appends_one_snapshot(self):
        """Each update() call must append exactly one entry to _iterations."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=10.0, iter_ps_cost=2.0)
        assert len(db._iterations) == 1
        db.close()

    def test_multiple_updates_accumulate(self):
        """Three calls to update() must result in three snapshots."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        for i in range(3):
            db.update(
                records,
                activity_threshold=7.0,
                iter_drc_cost=10.0 * (i + 1),
                iter_ps_cost=float(i + 1),
            )
        assert len(db._iterations) == 3
        db.close()

    def test_snapshot_keys_are_present(self):
        """Each snapshot dict must contain all required data keys."""
        required_keys = {
            "cum_cost",
            "cum_actives",
            "iter_drc_cost",
            "iter_ps_cost",
            "iter_upgrade_cost",
            "model_metric_value",
            "n_ps_only",
            "n_drc_new",
            "n_upgrades",
            "n_unqueried",
        }
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
        assert required_keys.issubset(db._iterations[0].keys())
        db.close()

    def test_model_metric_stored_correctly(self):
        """model_metric_value must be stored as supplied, including None."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=5.0,
            iter_ps_cost=1.0,
            model_metric_value=None,
        )
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=5.0,
            iter_ps_cost=1.0,
            model_metric_value=0.75,
        )
        assert db._iterations[0]["model_metric_value"] is None
        assert db._iterations[1]["model_metric_value"] == pytest.approx(0.75)
        db.close()

    def test_upgrade_cost_defaults_to_zero(self):
        """Calling update() without iter_upgrade_cost must record 0.0 in the snapshot."""
        db = LiveDashboard(n_iterations=2, n_compounds=20)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=10.0, iter_ps_cost=2.0)
        assert db._iterations[0]["iter_upgrade_cost"] == 0.0
        db.close()


# ---------------------------------------------------------------------------
# Metric and cost history
# ---------------------------------------------------------------------------


class TestDashboardMetricHistory:
    """Tests for internal history accumulation across multiple updates."""

    def test_cost_and_metric_stacks_accumulated(self):
        """After two updates all cost and metric lists must have two entries each."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=10.0,
            iter_ps_cost=2.0,
            iter_upgrade_cost=3.0,
            model_metric_value=2.0,
        )
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=20.0,
            iter_ps_cost=4.0,
            iter_upgrade_cost=6.0,
            model_metric_value=1.5,
        )
        drc_costs = [it["iter_drc_cost"] for it in db._iterations]
        ps_costs = [it["iter_ps_cost"] for it in db._iterations]
        upgrade_costs = [it["iter_upgrade_cost"] for it in db._iterations]
        metric_vals = [it["model_metric_value"] for it in db._iterations]
        assert drc_costs == [10.0, 20.0]
        assert ps_costs == [2.0, 4.0]
        assert upgrade_costs == [3.0, 6.0]
        assert metric_vals == [2.0, 1.5]
        db.close()

    @pytest.mark.parametrize("metric", list(ModelMetric))
    def test_all_model_metrics_accepted(self, metric):
        """Every ModelMetric enum value must be accepted without error."""
        db = LiveDashboard(n_iterations=3, n_compounds=20, model_metric=metric)
        records = _make_records(4)
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=5.0,
            iter_ps_cost=1.0,
            model_metric_value=0.5,
        )
        db.close()


# ---------------------------------------------------------------------------
# Compound status counts
# ---------------------------------------------------------------------------


class TestCompoundStatusPanel:
    """Tests that update() derives correct compound counts from labeled records."""

    def test_bar_counts_with_mixed_records(self):
        """PS-only, DRC-new, upgrades, and unqueried are correctly partitioned."""

        def _rec(smi, fidelity, value=6.0, cost=1.0):
            ct = (
                CensoringType.EXACT
                if fidelity == QueryType.DOSE_RESPONSE
                else CensoringType.INTERVAL
            )
            return LabelRecord(
                smiles=smi,
                canonical_smiles=smi,
                value=value,
                upper_bound=value if ct == CensoringType.EXACT else 11.0,
                censoring_type=ct,
                fidelity=fidelity,
                cost=cost,
                iteration=0,
            )

        # A = PS-only, B = PS+DRC upgrade, C = DRC first-pass
        records = [
            _rec("A", QueryType.PRIMARY_SCREEN),
            _rec("B", QueryType.PRIMARY_SCREEN),
            _rec("B", QueryType.DOSE_RESPONSE),
            _rec("C", QueryType.DOSE_RESPONSE),
        ]
        db = LiveDashboard(n_iterations=2, n_compounds=10)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)

        snap = db._iterations[-1]
        assert snap["n_ps_only"] == 1, f"Expected n_ps_only=1, got {snap['n_ps_only']}"
        assert snap["n_upgrades"] == 1, (
            f"Expected n_upgrades=1, got {snap['n_upgrades']}"
        )
        assert snap["n_drc_new"] == 1, f"Expected n_drc_new=1, got {snap['n_drc_new']}"
        assert snap["n_unqueried"] == 7, (
            f"Expected n_unqueried=7, got {snap['n_unqueried']}"
        )
        db.close()

    def test_empty_records_does_not_raise(self):
        """update() with no records must not raise even with a non-zero compound pool."""
        db = LiveDashboard(n_iterations=3, n_compounds=50)
        db.update([], activity_threshold=7.0, iter_drc_cost=0.0, iter_ps_cost=0.0)
        assert db._iterations[-1]["n_unqueried"] == 50
        db.close()

    def test_unqueried_clamped_to_zero(self):
        """When n_compounds=0 the unqueried count must not go negative."""
        db = LiveDashboard(n_iterations=3)
        records = _make_records(6)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
        assert db._iterations[-1]["n_unqueried"] == 0
        db.close()


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------


class TestFrameCapture:
    """Tests for matplotlib-based PNG frame pre-capture at update() time."""

    def test_frame_appended_per_update(self):
        """Each update() call must append exactly one pre-rendered PNG frame."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        for i in range(3):
            db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
            assert len(db._frames) == i + 1
        db.close()

    def test_frames_are_valid_png_bytes(self):
        """Pre-captured frames must be valid PNG byte payloads."""
        from PIL import Image

        db = LiveDashboard(n_iterations=2, n_compounds=20)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
        db.close()

        assert db._frames
        img = Image.open(io.BytesIO(db._frames[0]))
        assert img.format == "PNG"

    def test_capture_failure_is_silent(self, monkeypatch, caplog):
        """A matplotlib render error in _capture_frame() must warn and not raise."""
        import logging

        monkeypatch.setattr(
            "moal.dashboard.LiveDashboard._render_matplotlib_frame",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("render error")),
        )
        db = LiveDashboard(n_iterations=2, n_compounds=20)
        records = _make_records(4)
        with caplog.at_level(logging.WARNING, logger="moal.dashboard"):
            db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)

        assert db._frames == []
        assert "Could not capture dashboard frame" in caplog.text
        db.close()

    def test_gif_skipped_when_no_frames_captured(self, monkeypatch, tmp_path, caplog):
        """If all _capture_frame() calls silently fail, save_gif() must warn and skip."""
        import logging

        monkeypatch.setattr(
            "moal.dashboard.LiveDashboard._render_matplotlib_frame",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("render error")),
        )
        db = LiveDashboard(n_iterations=2, n_compounds=20)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)

        gif_path = tmp_path / "fail.gif"
        with caplog.at_level(logging.WARNING, logger="moal.dashboard"):
            db.save_gif(gif_path)

        assert not gif_path.exists()
        assert "No frames captured" in caplog.text
        db.close()


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------


class TestSaveGif:
    """Tests for LiveDashboard.save_gif."""

    def test_gif_created_with_correct_frame_count(self, tmp_path):
        """A GIF produced from N updates must contain N frames."""
        from PIL import Image

        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        for _ in range(3):
            db.update(
                records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0
            )

        gif_path = tmp_path / "animation.gif"
        db.save_gif(gif_path)
        db.close()

        assert gif_path.exists(), "GIF file was not created"
        with Image.open(gif_path) as img:
            assert img.format == "GIF"
            frame_count = 0
            try:
                while True:
                    frame_count += 1
                    img.seek(frame_count)
            except EOFError:
                pass
        assert frame_count == 3

    def test_gif_skipped_when_no_frames(self, tmp_path):
        """No file should be created when no updates have been made."""
        db = LiveDashboard(n_iterations=2, n_compounds=20)
        gif_path = tmp_path / "should_not_exist.gif"
        db.save_gif(gif_path)
        db.close()
        assert not gif_path.exists()

    def test_last_frame_held_longer(self, tmp_path):
        """The final frame must carry last_frame_duration_ms, not frame_duration_ms."""
        from PIL import Image

        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        for _ in range(3):
            db.update(
                records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0
            )

        gif_path = tmp_path / "animation.gif"
        db.save_gif(gif_path, frame_duration_ms=500, last_frame_duration_ms=5000)
        db.close()

        with Image.open(gif_path) as img:
            durations = []
            try:
                while True:
                    durations.append(img.info.get("duration"))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

        assert durations[-1] == 5000
        assert all(d == 500 for d in durations[:-1])

    def test_single_frame_gif_uses_last_frame_duration(self, tmp_path):
        """A single-frame GIF should apply last_frame_duration_ms to that only frame."""
        from PIL import Image

        db = LiveDashboard(n_iterations=1, n_compounds=20)
        records = _make_records(2)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)

        gif_path = tmp_path / "single.gif"
        db.save_gif(gif_path, frame_duration_ms=500, last_frame_duration_ms=3000)
        db.close()

        with Image.open(gif_path) as img:
            assert img.info.get("duration") == 3000


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


class TestSaveHtml:
    """Tests for LiveDashboard.save_html."""

    def test_html_file_created(self, tmp_path):
        """save_html() must create a non-empty HTML file."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        for i in range(3):
            db.update(
                records,
                activity_threshold=7.0,
                iter_drc_cost=10.0 * (i + 1),
                iter_ps_cost=float(i + 1),
            )
        html_path = tmp_path / "dashboard.html"
        db.save_html(html_path)
        db.close()

        assert html_path.exists()
        assert html_path.stat().st_size > 0

    def test_html_contains_plotly_markers(self, tmp_path):
        """The exported HTML must contain Plotly JS and animation markers."""
        db = LiveDashboard(n_iterations=2, n_compounds=20)
        records = _make_records(4)
        for _ in range(2):
            db.update(
                records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0
            )

        html_path = tmp_path / "dashboard.html"
        db.save_html(html_path)
        db.close()

        content = html_path.read_text()
        assert "plotly" in content.lower()
        # Play/pause toggle is injected via post_script; Plotly.animate must appear there
        assert "Plotly.animate" in content

    def test_html_created_with_no_updates(self, tmp_path):
        """save_html() on an empty dashboard must create a valid (empty) HTML file."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        html_path = tmp_path / "empty.html"
        db.save_html(html_path)
        db.close()

        assert html_path.exists()
        assert html_path.stat().st_size > 0

    def test_html_slider_steps_match_iteration_count(self, tmp_path):
        """The slider step count embedded in the HTML must match the update count."""

        n = 4
        db = LiveDashboard(n_iterations=n, n_compounds=20)
        records = _make_records(4)
        for i in range(n):
            db.update(
                records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0
            )

        html_path = tmp_path / "dashboard.html"
        db.save_html(html_path)
        db.close()

        content = html_path.read_text()
        # Slider steps use plain numeric labels; verify the final iteration label is present
        assert f'"{n}"' in content


# ---------------------------------------------------------------------------
# Figure building (unit-level)
# ---------------------------------------------------------------------------


class TestBuildFigure:
    """Tests for the internal _build_figure method."""

    def test_empty_iterations_returns_figure(self):
        """_build_figure([]) must return a go.Figure without raising."""
        import plotly.graph_objects as go

        db = LiveDashboard(n_iterations=5, n_compounds=20)
        fig = db._build_figure([])
        assert isinstance(fig, go.Figure)
        db.close()

    def test_figure_has_correct_trace_count(self):
        """A figure built from non-empty iterations must have exactly 10 traces."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
        with db._lock:
            iters = list(db._iterations)
        fig = db._build_figure(iters)
        # 1 actives + 3 cost bars + 1 cum-cost line + 1 metric + 3 status bars + 1 PS→DRC overlay = 10
        assert len(fig.data) == 10
        db.close()

    def test_no_metric_adds_annotation(self):
        """_build_figure must add a 'No test set' annotation when metric data is absent."""
        db = LiveDashboard(n_iterations=3, n_compounds=20)
        records = _make_records(4)
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=5.0,
            iter_ps_cost=1.0,
            model_metric_value=None,
        )
        with db._lock:
            iters = list(db._iterations)
        fig = db._build_figure(iters)
        annotation_texts = [a.text for a in fig.layout.annotations if a.text]
        assert any("No test set" in t for t in annotation_texts)
        db.close()

    def test_export_dimensions_used_in_render(self, tmp_path):
        """export_width and export_height must control the PNG dimensions produced by save()."""
        from PIL import Image

        db = LiveDashboard(
            n_iterations=2, n_compounds=10, export_width=400, export_height=300
        )
        records = _make_records(2)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
        png_path = tmp_path / "out.png"
        db.save(png_path)
        db.close()

        assert png_path.exists()
        with Image.open(png_path) as img:
            w, h = img.size
        # matplotlib tight_layout can adjust dimensions slightly; allow ±20%
        assert abs(w - 400) / 400 < 0.2
        assert abs(h - 300) / 300 < 0.2


# ---------------------------------------------------------------------------
# _metric_axis_params
# ---------------------------------------------------------------------------


class TestMetricAxisParams:
    """Unit tests for LiveDashboard._metric_axis_params.

    The method must return (ymin, ymax, tick0, dtick) such that at least two
    multiples of dtick starting from tick0 fall within [ymin, ymax], and all
    ticks land on multiples of 0.1 or larger (so 1 decimal place is sufficient
    to distinguish every label).
    """

    @staticmethod
    def _count_ticks_in_range(ymin, ymax, tick0, dtick):
        """Count how many tick positions tick0 + k*dtick lie within [ymin, ymax]."""
        import math

        if dtick <= 0:
            return 0
        k_start = math.ceil((ymin - tick0) / dtick)
        k_end = math.floor((ymax - tick0) / dtick)
        return max(0, k_end - k_start + 1)

    def test_empty_gives_two_ticks(self):
        """No data should return a default range with >= 2 visible ticks."""
        ymin, ymax, tick0, dtick = LiveDashboard._metric_axis_params([])
        assert self._count_ticks_in_range(ymin, ymax, tick0, dtick) >= 2

    def test_single_point_gives_two_ticks(self):
        """A single metric value should produce a range with exactly 2 ticks."""
        ymin, ymax, tick0, dtick = LiveDashboard._metric_axis_params([0.6])
        assert self._count_ticks_in_range(ymin, ymax, tick0, dtick) >= 2

    def test_two_close_points_same_step_bucket(self):
        """Two points in the same 0.1-step bucket should still give >= 2 ticks."""
        ymin, ymax, tick0, dtick = LiveDashboard._metric_axis_params([0.60, 0.65])
        assert self._count_ticks_in_range(ymin, ymax, tick0, dtick) >= 2

    def test_wide_range_gives_multiple_ticks(self):
        """A wide range should produce multiple ticks and never exceed MAX_TICKS."""
        ymin, ymax, tick0, dtick = LiveDashboard._metric_axis_params([0.5, 2.5])
        n = self._count_ticks_in_range(ymin, ymax, tick0, dtick)
        assert n >= 2
        assert n <= 6

    def test_step_always_at_least_0_1(self):
        """dtick must never be smaller than 0.1 regardless of data span."""
        for vals in [[], [0.6], [0.60, 0.61], [0.0, 100.0]]:
            _, _, _, dtick = LiveDashboard._metric_axis_params(vals)
            assert dtick >= 0.1 - 1e-12, f"dtick={dtick} for vals={vals}"

    def test_data_values_within_range(self):
        """All input values must lie within the returned [ymin, ymax]."""
        vals = [0.45, 0.72, 1.1]
        ymin, ymax, _, _ = LiveDashboard._metric_axis_params(vals)
        for v in vals:
            assert ymin <= v <= ymax, f"value {v} outside [{ymin}, {ymax}]"

    def test_near_zero_value(self):
        """A value at 0.0 should not produce a negative ymin that breaks log scales."""
        ymin, ymax, tick0, dtick = LiveDashboard._metric_axis_params([0.0])
        assert self._count_ticks_in_range(ymin, ymax, tick0, dtick) >= 2

    def test_large_values(self):
        """Large metric values (e.g. RMSE ~10) should scale gracefully."""
        ymin, ymax, tick0, dtick = LiveDashboard._metric_axis_params([8.5, 9.5])
        assert self._count_ticks_in_range(ymin, ymax, tick0, dtick) >= 2
        assert dtick >= 0.1
