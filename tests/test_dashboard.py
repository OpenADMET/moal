"""Tests for LiveDashboard — headless mode (file output only)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from moal.dashboard import LiveDashboard
from moal.evaluation import ModelMetric
from moal.types import CensoringType, LabelRecord, QueryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    iteration: int,
    value: float,
    censoring_type: CensoringType,
    fidelity: QueryType,
    cost: float,
) -> LabelRecord:
    return LabelRecord(
        smiles="C",
        canonical_smiles="C",
        value=value,
        upper_bound=value if censoring_type == CensoringType.EXACT else 11.0,
        censoring_type=censoring_type,
        fidelity=fidelity,
        cost=cost,
        iteration=iteration,
    )


def _make_records(n: int = 6) -> list[LabelRecord]:
    records = []
    for i in range(n):
        ct = CensoringType.EXACT if i % 2 == 0 else CensoringType.INTERVAL
        fid = QueryType.DOSE_RESPONSE if i % 2 == 0 else QueryType.PRIMARY_SCREEN
        v = 5.0 + i * 0.5
        records.append(_make_record(i // 2, v, ct, fid, cost=10.0 if fid == QueryType.DOSE_RESPONSE else 1.0))
    return records


# ---------------------------------------------------------------------------
# Headless dashboard tests
# ---------------------------------------------------------------------------

class TestDashboardHeadless:
    def test_creates_figure_with_3_axes(self, tmp_path):
        db = LiveDashboard(n_iterations=5, show=False)
        # 3 primary axes + 1 twin axis for the cost panel = 4 total.
        assert len(db._fig.get_axes()) == 4
        db.close()

    def test_axis_count_stable_across_many_updates(self, tmp_path):
        """twinx() must not accumulate new axes on each update call."""
        db = LiveDashboard(n_iterations=10, show=False)
        records = _make_records(6)
        initial_count = len(db._fig.get_axes())
        for i in range(10):
            db.update(records, activity_threshold=7.0,
                      iter_drc_cost=10.0, iter_ps_cost=2.0,
                      model_metric_value=float(i))
        assert len(db._fig.get_axes()) == initial_count
        db.close()

    def test_update_captures_frame(self, tmp_path):
        db = LiveDashboard(n_iterations=3, show=False)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=10.0, iter_ps_cost=3.0)
        db.update(records, activity_threshold=7.0, iter_drc_cost=20.0, iter_ps_cost=5.0)
        db.close()
        assert len(db._frames) == 2

    def test_three_updates_three_frames(self, tmp_path):
        db = LiveDashboard(n_iterations=3, show=False)
        records = _make_records(6)
        for i in range(3):
            db.update(records[:i+2], activity_threshold=7.0,
                      iter_drc_cost=10.0, iter_ps_cost=2.0,
                      model_metric_value=float(i + 0.5))
        db.close()
        assert len(db._frames) == 3

    def test_no_updates_no_frames(self, tmp_path):
        db = LiveDashboard(n_iterations=3, show=False)
        db.close()
        assert db._frames == []

    def test_explicit_save(self, tmp_path):
        db = LiveDashboard(n_iterations=3, show=False)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
        out = tmp_path / "explicit.png"
        db.save(out)
        db.close()
        assert out.exists() and out.stat().st_size > 0


class TestDashboardNoTestSet:
    def test_no_metric_shows_annotation(self, tmp_path):
        """Model performance panel should show annotation when no test set data."""
        db = LiveDashboard(n_iterations=3, show=False)
        records = _make_records(4)
        # No model_metric_value → panel should contain annotation text.
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=2.0,
                  model_metric_value=None)
        # Check that panel 3 has text artists (the annotation).
        texts = [t.get_text() for t in db._ax3.texts]
        assert any("No test set" in t for t in texts)
        db.close()

    def test_with_metric_value_draws_line(self, tmp_path):
        db = LiveDashboard(n_iterations=3, show=False)
        records = _make_records(6)
        for val in [1.0, 0.8, 0.6]:
            db.update(records, activity_threshold=7.0,
                      iter_drc_cost=10.0, iter_ps_cost=2.0,
                      model_metric_value=val)
        assert len(db._model_metric_values) == 3
        db.close()


class TestDashboardMetricHistory:
    def test_metric_values_accumulated(self, tmp_path):
        db = LiveDashboard(n_iterations=4, show=False)
        records = _make_records(6)
        for v in [2.0, 1.5, 1.2]:
            db.update(records, activity_threshold=7.0,
                      iter_drc_cost=10.0, iter_ps_cost=1.0,
                      model_metric_value=v)
        assert db._model_metric_values == [2.0, 1.5, 1.2]
        db.close()

    def test_cost_stacks_accumulated(self, tmp_path):
        db = LiveDashboard(n_iterations=3, show=False)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=10.0, iter_ps_cost=2.0)
        db.update(records, activity_threshold=7.0, iter_drc_cost=20.0, iter_ps_cost=4.0)
        assert db._iter_drc_costs == [10.0, 20.0]
        assert db._iter_ps_costs  == [2.0, 4.0]
        db.close()

    @pytest.mark.parametrize("metric", list(ModelMetric))
    def test_all_model_metrics_accepted(self, metric, tmp_path):
        db = LiveDashboard(n_iterations=3, model_metric=metric, show=False)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0,
                  iter_drc_cost=5.0, iter_ps_cost=1.0,
                  model_metric_value=0.5)
        db.close()


class TestSaveGif:
    """Tests for LiveDashboard.save_gif."""

    def test_gif_created_with_correct_frame_count(self, tmp_path):
        """A GIF produced from N iteration snapshots must contain N frames."""
        from PIL import Image

        db = LiveDashboard(n_iterations=3, show=False)
        records = _make_records(4)
        for _ in range(3):
            db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)

        gif_path = tmp_path / "animation.gif"
        db.save_gif(gif_path)
        db.close()

        assert gif_path.exists(), "GIF file was not created"
        with Image.open(gif_path) as img:
            # Count frames by seeking through the GIF
            frame_count = 0
            try:
                while True:
                    frame_count += 1
                    img.seek(frame_count)
            except EOFError:
                pass
        assert frame_count == 3

    def test_gif_is_valid_gif_format(self, tmp_path):
        """The output file must be a valid GIF that Pillow can open."""
        from PIL import Image

        db = LiveDashboard(n_iterations=2, show=False)
        records = _make_records(2)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
        db.update(records, activity_threshold=7.0, iter_drc_cost=8.0, iter_ps_cost=2.0)

        gif_path = tmp_path / "animation.gif"
        db.save_gif(gif_path)
        db.close()

        with Image.open(gif_path) as img:
            assert img.format == "GIF"

    def test_gif_skipped_when_no_frames(self, tmp_path):
        """No file should be created and no exception raised when no updates have been made."""
        db = LiveDashboard(n_iterations=2, show=False)
        gif_path = tmp_path / "should_not_exist.gif"
        db.save_gif(gif_path)  # must not raise
        db.close()
        assert not gif_path.exists()

    def test_last_frame_held_longer_than_other_frames(self, tmp_path):
        """The final frame must carry the last_frame_duration_ms delay, not frame_duration_ms."""
        from PIL import Image

        db = LiveDashboard(n_iterations=3, show=False)
        records = _make_records(4)
        for _ in range(3):
            db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)

        gif_path = tmp_path / "animation.gif"
        db.save_gif(gif_path, frame_duration_ms=500, last_frame_duration_ms=5000)
        db.close()

        with Image.open(gif_path) as img:
            frame_durations = []
            try:
                while True:
                    frame_durations.append(img.info.get("duration"))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

        assert len(frame_durations) == 3
        assert frame_durations[-1] == 5000, "Last frame must use last_frame_duration_ms"
        assert all(d == 500 for d in frame_durations[:-1]), "Non-final frames must use frame_duration_ms"

    def test_single_frame_gif_uses_last_frame_duration(self, tmp_path):
        """A single-frame GIF should still apply last_frame_duration_ms to that frame."""
        from PIL import Image

        db = LiveDashboard(n_iterations=1, show=False)
        records = _make_records(2)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)

        gif_path = tmp_path / "single.gif"
        db.save_gif(gif_path, frame_duration_ms=500, last_frame_duration_ms=3000)
        db.close()

        with Image.open(gif_path) as img:
            assert img.info.get("duration") == 3000
