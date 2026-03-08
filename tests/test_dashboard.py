"""Tests for LiveDashboard — headless mode (file output only)."""

from __future__ import annotations

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
    """Helper that constructs a single LabelRecord for dashboard rendering tests."""
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
    """Helper that produces a list of n alternating DRC/PS LabelRecords spanning multiple iterations."""
    records = []
    for i in range(n):
        ct = CensoringType.EXACT if i % 2 == 0 else CensoringType.INTERVAL
        fid = QueryType.DOSE_RESPONSE if i % 2 == 0 else QueryType.PRIMARY_SCREEN
        v = 5.0 + i * 0.5
        records.append(
            _make_record(
                i // 2, v, ct, fid, cost=10.0 if fid == QueryType.DOSE_RESPONSE else 1.0
            )
        )
    return records


# ---------------------------------------------------------------------------
# Headless dashboard tests
# ---------------------------------------------------------------------------


class TestDashboardHeadless:
    """Tests that LiveDashboard operates correctly in headless mode (show=False, file output only)."""

    def test_creates_figure_with_4_axes(self, tmp_path):
        """The figure must have 4 primary axes plus 1 twin axis for cost at construction time."""
        db = LiveDashboard(n_iterations=5, n_compounds=20, show=False)
        # 4 primary axes + 1 twin axis for the cost panel = 5 total.
        assert len(db._fig.get_axes()) == 5
        db.close()

    def test_axis_count_stable_across_many_updates(self, tmp_path):
        """twinx() must not accumulate new axes on each update call."""
        db = LiveDashboard(n_iterations=10, n_compounds=20, show=False)
        records = _make_records(6)
        initial_count = len(db._fig.get_axes())
        for i in range(10):
            db.update(
                records,
                activity_threshold=7.0,
                iter_drc_cost=10.0,
                iter_ps_cost=2.0,
                model_metric_value=float(i),
            )
        assert len(db._fig.get_axes()) == initial_count
        db.close()

    @pytest.mark.parametrize("n_updates", [2, 3])
    def test_update_captures_frames(self, tmp_path, n_updates):
        """Each call to update() must append one frame to _frames so that save_gif() has the correct frame count."""
        db = LiveDashboard(n_iterations=3, n_compounds=20, show=False)
        records = _make_records(4)
        for i in range(n_updates):
            db.update(
                records,
                activity_threshold=7.0,
                iter_drc_cost=10.0 * (i + 1),
                iter_ps_cost=float(i + 1),
            )
        db.close()
        assert len(db._frames) == n_updates

    def test_no_updates_no_frames(self, tmp_path):
        """A dashboard that was never updated must have an empty frames list, so save_gif() skips file creation."""
        db = LiveDashboard(n_iterations=3, n_compounds=20, show=False)
        db.close()
        assert db._frames == []

    def test_explicit_save(self, tmp_path):
        """Calling save() after an update must write a non-empty PNG file to the given path."""
        db = LiveDashboard(n_iterations=3, n_compounds=20, show=False)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
        out = tmp_path / "explicit.png"
        db.save(out)
        db.close()
        assert out.exists() and out.stat().st_size > 0


class TestDashboardNoTestSet:
    """Tests for dashboard rendering when no model evaluation test set is configured."""

    def test_no_metric_shows_annotation(self, tmp_path):
        """Model performance panel should show annotation when no test set data."""
        db = LiveDashboard(n_iterations=3, n_compounds=20, show=False)
        records = _make_records(4)
        # No model_metric_value → panel should contain annotation text.
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=5.0,
            iter_ps_cost=2.0,
            model_metric_value=None,
        )
        # Check that panel 3 has text artists (the annotation).
        texts = [t.get_text() for t in db._ax3.texts]
        assert any("No test set" in t for t in texts)
        db.close()

    def test_with_metric_value_draws_line(self, tmp_path):
        """When model_metric_value is provided, the metric history list must grow with each update."""
        db = LiveDashboard(n_iterations=3, n_compounds=20, show=False)
        records = _make_records(6)
        for val in [1.0, 0.8, 0.6]:
            db.update(
                records,
                activity_threshold=7.0,
                iter_drc_cost=10.0,
                iter_ps_cost=2.0,
                model_metric_value=val,
            )
        assert len(db._model_metric_values) == 3
        db.close()


class TestDashboardMetricHistory:
    """Tests for internal history accumulation (cost stacks and model metric values) across multiple updates."""

    def test_metric_and_cost_stacks_accumulated(self, tmp_path):
        """After two updates, both cost-stack lists and the model-metric-value list must contain exactly two entries each."""
        db = LiveDashboard(n_iterations=3, n_compounds=20, show=False)
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
        assert db._model_metric_values == [2.0, 1.5]
        assert db._iter_drc_costs == [10.0, 20.0]
        assert db._iter_ps_costs == [2.0, 4.0]
        assert db._iter_upgrade_costs == [3.0, 6.0]
        db.close()

    @pytest.mark.parametrize("metric", list(ModelMetric))
    def test_all_model_metrics_accepted(self, metric, tmp_path):
        """Every ModelMetric enum value must be accepted by LiveDashboard without raising, ensuring forward compatibility."""
        db = LiveDashboard(
            n_iterations=3, n_compounds=20, model_metric=metric, show=False
        )
        records = _make_records(4)
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=5.0,
            iter_ps_cost=1.0,
            model_metric_value=0.5,
        )
        db.close()


class TestCostPanelUpgradeBar:
    """Tests for the PS→DRC upgrade segment in the Per-Iteration Cost Breakdown panel."""

    def test_upgrade_bar_present_when_upgrade_cost_nonzero(self, tmp_path):
        """Cost panel must render 3 bar containers (DRC, PS→DRC upgrade, PS) when upgrades occur."""
        db = LiveDashboard(n_iterations=2, n_compounds=20, show=False)
        records = _make_records(4)
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=15.0,
            iter_ps_cost=2.0,
            iter_upgrade_cost=5.0,
        )
        # ax2 should have 3 BarContainers: DRC new, PS→DRC upgrade, PS
        assert len(db._ax2.containers) == 3
        db.close()

    def test_upgrade_bar_absent_when_upgrade_cost_zero(self, tmp_path):
        """Cost panel must still render 3 bar containers even with zero upgrade cost (bar has zero height)."""
        db = LiveDashboard(n_iterations=2, n_compounds=20, show=False)
        records = _make_records(4)
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=15.0,
            iter_ps_cost=2.0,
            iter_upgrade_cost=0.0,
        )
        # Upgrade bar is always drawn; height is just 0 when there are no upgrades
        assert len(db._ax2.containers) == 3
        upgrade_height = db._ax2.containers[1][0].get_height()
        assert upgrade_height == 0.0
        db.close()

    def test_drc_new_plus_upgrade_equals_total_drc(self, tmp_path):
        """The first-pass DRC bar height plus upgrade bar height must equal total iter_drc_cost."""
        db = LiveDashboard(n_iterations=2, n_compounds=20, show=False)
        records = _make_records(4)
        db.update(
            records,
            activity_threshold=7.0,
            iter_drc_cost=20.0,
            iter_ps_cost=3.0,
            iter_upgrade_cost=8.0,
        )
        drc_new_height = db._ax2.containers[0][0].get_height()
        upgrade_height = db._ax2.containers[1][0].get_height()
        assert drc_new_height + upgrade_height == pytest.approx(20.0)
        db.close()

    def test_upgrade_cost_defaults_to_zero(self, tmp_path):
        """Calling update() without iter_upgrade_cost must not raise and must record 0.0."""
        db = LiveDashboard(n_iterations=2, n_compounds=20, show=False)
        records = _make_records(4)
        db.update(records, activity_threshold=7.0, iter_drc_cost=10.0, iter_ps_cost=2.0)
        assert db._iter_upgrade_costs == [0.0]
        db.close()


class TestSaveGif:
    """Tests for LiveDashboard.save_gif."""

    def test_gif_created_with_correct_frame_count_and_format(self, tmp_path):
        """A GIF produced from N iteration snapshots must contain N frames and be a valid GIF."""
        from PIL import Image

        db = LiveDashboard(n_iterations=3, n_compounds=20, show=False)
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
        """No file should be created and no exception raised when no updates have been made."""
        db = LiveDashboard(n_iterations=2, n_compounds=20, show=False)
        gif_path = tmp_path / "should_not_exist.gif"
        db.save_gif(gif_path)  # must not raise
        db.close()
        assert not gif_path.exists()

    def test_last_frame_held_longer_than_other_frames(self, tmp_path):
        """The final frame must carry the last_frame_duration_ms delay, not frame_duration_ms."""
        from PIL import Image

        db = LiveDashboard(n_iterations=3, n_compounds=20, show=False)
        records = _make_records(4)
        for _ in range(3):
            db.update(
                records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0
            )

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
        assert all(d == 500 for d in frame_durations[:-1]), (
            "Non-final frames must use frame_duration_ms"
        )

    def test_single_frame_gif_uses_last_frame_duration(self, tmp_path):
        """A single-frame GIF should still apply last_frame_duration_ms to that frame."""
        from PIL import Image

        db = LiveDashboard(n_iterations=1, n_compounds=20, show=False)
        records = _make_records(2)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)

        gif_path = tmp_path / "single.gif"
        db.save_gif(gif_path, frame_duration_ms=500, last_frame_duration_ms=3000)
        db.close()

        with Image.open(gif_path) as img:
            assert img.info.get("duration") == 3000


# ---------------------------------------------------------------------------
# Compound status panel tests
# ---------------------------------------------------------------------------


class TestCompoundStatusPanel:
    """Tests for the compound status bar panel (PS-only, DRC-new, upgrades, unqueried)."""

    def test_bar_counts_with_mixed_records(self, tmp_path):
        """PS-only, DRC-new, and upgrades are correctly partitioned."""
        from moal.types import CensoringType, QueryType

        # Craft records with known fidelity composition:
        #   - smiles "A": PS-only
        #   - smiles "B": PS then DRC (upgrade)
        #   - smiles "C": DRC first-pass
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

        records = [
            _rec("A", QueryType.PRIMARY_SCREEN),
            _rec("A", QueryType.DOSE_RESPONSE),
            _rec("B", QueryType.PRIMARY_SCREEN),
            _rec("B", QueryType.DOSE_RESPONSE),
            _rec("C", QueryType.DOSE_RESPONSE),
        ]
        # n_compounds=10: A(PS), B(upgrade), C(DRC), 7 unqueried
        db = LiveDashboard(n_iterations=2, n_compounds=10, show=False)
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)

        bars = db._ax4.containers
        # Expect 5 bar containers: PS-only, upgrade, DRC-new, upgrade, unqueried.
        # Use the public Rectangle.get_height() API rather than the internal
        # BarContainer.datavalues attribute.
        heights = [c[0].get_height() for c in bars]

        # PS-only: A → 1
        assert heights[0] == 1, f"Expected PS-only=1, got {heights[0]}"
        # Upgrades: A → 1 (top of PS bar)
        assert heights[1] == 1, f"Expected PS→DRC=1, got {heights[1]}"
        # DRC-new: C → 1 (bottom of DRC bar)
        assert heights[2] == 1, f"Expected DRC-new=1, got {heights[2]}"
        # Upgrades: B → 1 (top of DRC bar)
        assert heights[3] == 1, f"Expected PS→DRC=1, got {heights[3]}"
        # Unqueried: 10 - 1(A) - 2(B,C) = 7
        assert heights[4] == 7, f"Expected unqueried=7, got {heights[4]}"

        db.close()

    def test_empty_records_renders_without_error(self, tmp_path):
        """Panel must draw cleanly when no records have been labeled yet."""
        db = LiveDashboard(n_iterations=3, n_compounds=50, show=False)
        db.update([], activity_threshold=7.0, iter_drc_cost=0.0, iter_ps_cost=0.0)
        db.close()

    def test_unqueried_clamped_to_zero_when_n_compounds_not_set(self, tmp_path):
        """When n_compounds=0 (default), unqueried must not go negative."""
        db = LiveDashboard(n_iterations=3, show=False)
        records = _make_records(6)
        # Should not raise even though n_compounds < actual labeled count
        db.update(records, activity_threshold=7.0, iter_drc_cost=5.0, iter_ps_cost=1.0)
        db.close()
