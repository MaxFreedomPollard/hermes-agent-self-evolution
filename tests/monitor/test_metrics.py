"""Tests for the performance history store.

Entirely offline: a JSONL file in tmp_path, an injected clock, and fixed
timestamps. Nothing here reads the wall clock, so a trend that is declining
today is still declining when this suite runs in a year.
"""

import json

import pytest

from evolution.monitor.metrics import (
    BENCHMARK_SCORE,
    HIGHER_IS_BETTER,
    SECONDS_PER_DAY,
    SKILL_SUCCESS_RATE,
    TOOL_SELECTION_ACCURACY,
    TRACKED_METRICS,
    USER_CORRECTION,
    MetricPoint,
    MetricStore,
    TrendDirection,
    compute_trend,
    summarize,
)

# A fixed instant to hang every fixture off: 2023-11-14T22:13:20Z.
T0 = 1_700_000_000.0

DAY = SECONDS_PER_DAY


def point(metric, target, value, days_ago=0.0, samples=1, source="test", **metadata):
    return MetricPoint(
        metric=metric,
        target=target,
        value=value,
        timestamp=T0 - days_ago * DAY,
        samples=samples,
        source=source,
        metadata=dict(metadata),
    )


@pytest.fixture
def store(tmp_path):
    """A store whose clock is frozen at T0."""
    return MetricStore(tmp_path / "history" / "metrics.jsonl", clock=lambda: T0)


class TestMetricPoint:
    def test_round_trips_through_a_dict(self):
        original = point(SKILL_SUCCESS_RATE, "arxiv", 0.75, samples=12, note="hello")
        restored = MetricPoint.from_dict(original.to_dict())
        assert restored == original

    def test_round_trips_through_a_json_line(self):
        original = point(BENCHMARK_SCORE, "tblite", 0.62, source="benchmark")
        restored = MetricPoint.from_json_line(original.to_json_line())
        assert restored.metric == BENCHMARK_SCORE
        assert restored.target == "tblite"
        assert restored.value == pytest.approx(0.62)
        assert restored.source == "benchmark"

    def test_serialised_line_carries_a_human_readable_timestamp(self):
        blob = json.loads(point(SKILL_SUCCESS_RATE, "arxiv", 1.0).to_json_line())
        assert blob["at"].startswith("2023-11-14T")
        assert blob["timestamp"] == pytest.approx(T0)

    def test_can_be_rebuilt_from_the_iso_field_alone(self):
        blob = point(SKILL_SUCCESS_RATE, "arxiv", 0.5).to_dict()
        del blob["timestamp"]
        assert MetricPoint.from_dict(blob).timestamp == pytest.approx(T0)

    def test_empty_metric_is_rejected(self):
        with pytest.raises(ValueError):
            MetricPoint(metric="  ", target="arxiv", value=1.0, timestamp=T0)

    def test_empty_target_is_rejected(self):
        with pytest.raises(ValueError):
            MetricPoint(metric=SKILL_SUCCESS_RATE, target="", value=1.0, timestamp=T0)

    def test_negative_sample_count_is_rejected(self):
        with pytest.raises(ValueError):
            MetricPoint(
                metric=SKILL_SUCCESS_RATE,
                target="arxiv",
                value=1.0,
                timestamp=T0,
                samples=-1,
            )

    def test_values_are_coerced_to_numbers(self):
        p = MetricPoint(
            metric=SKILL_SUCCESS_RATE,
            target="arxiv",
            value="0.5",
            timestamp="1700000000",
            samples="4",
        )
        assert isinstance(p.value, float) and p.value == 0.5
        assert isinstance(p.samples, int) and p.samples == 4
        assert p.timestamp == pytest.approx(T0)

    def test_corrections_are_the_only_signal_where_higher_is_worse(self):
        assert HIGHER_IS_BETTER[USER_CORRECTION] is False
        for metric in TRACKED_METRICS:
            if metric != USER_CORRECTION:
                assert HIGHER_IS_BETTER[metric] is True


class TestStoreWriteAndReload:
    def test_missing_file_reads_as_empty_history(self, store):
        assert store.load() == []

    def test_append_creates_parent_directories(self, store):
        store.append(point(SKILL_SUCCESS_RATE, "arxiv", 0.9))
        assert store.path.exists()
        assert store.path.parent.is_dir()

    def test_record_uses_the_injected_clock(self, store):
        written = store.record(SKILL_SUCCESS_RATE, "arxiv", 0.9)
        assert written.timestamp == pytest.approx(T0)

    def test_explicit_timestamp_beats_the_clock(self, store):
        written = store.record(
            SKILL_SUCCESS_RATE, "arxiv", 0.9, timestamp=T0 - 5 * DAY
        )
        assert written.timestamp == pytest.approx(T0 - 5 * DAY)

    def test_append_and_reload_round_trip(self, store):
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.4, samples=8, metadata={"run": 1})
        store.record(TOOL_SELECTION_ACCURACY, "read_file", 0.95, samples=200)

        reloaded = store.load()
        assert [p.metric for p in reloaded] == [
            SKILL_SUCCESS_RATE,
            TOOL_SELECTION_ACCURACY,
        ]
        assert reloaded[0].metadata == {"run": 1}
        assert reloaded[1].samples == 200

    def test_extend_writes_every_point(self, store):
        store.extend(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, days_ago=n) for n in range(4)]
        )
        assert len(store.load()) == 4

    def test_a_truncated_line_does_not_blind_the_rest(self, store):
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.4)
        with store.path.open("a", encoding="utf-8") as handle:
            handle.write('{"metric": "skill_suc\n')
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.6)

        points = store.load()
        assert len(points) == 2
        assert store.skipped_lines == 1

    def test_blank_lines_are_not_counted_as_damage(self, store):
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.4)
        with store.path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n")
        assert len(store.load()) == 1
        assert store.skipped_lines == 0


class TestQueries:
    @pytest.fixture
    def filled(self, store):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.9, days_ago=40, samples=10),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.7, days_ago=10, samples=10),
                point(SKILL_SUCCESS_RATE, "debugging", 0.5, days_ago=5, samples=30),
                point(TOOL_SELECTION_ACCURACY, "read_file", 0.8, days_ago=2, samples=90),
                point(BENCHMARK_SCORE, "tblite", 0.61, days_ago=1, source="benchmark"),
            ]
        )
        return store

    def test_filter_by_metric(self, filled):
        assert len(filled.query(metric=SKILL_SUCCESS_RATE)) == 3

    def test_filter_by_several_metrics(self, filled):
        found = filled.query(metric=[BENCHMARK_SCORE, TOOL_SELECTION_ACCURACY])
        assert {p.metric for p in found} == {BENCHMARK_SCORE, TOOL_SELECTION_ACCURACY}

    def test_filter_by_target(self, filled):
        assert len(filled.query(target="arxiv")) == 2

    def test_filter_by_source(self, filled):
        assert len(filled.query(source="benchmark")) == 1

    def test_results_come_back_oldest_first(self, filled):
        stamps = [p.timestamp for p in filled.query()]
        assert stamps == sorted(stamps)

    def test_since_is_inclusive(self, filled):
        cutoff = T0 - 10 * DAY
        found = filled.query(metric=SKILL_SUCCESS_RATE, since=cutoff)
        assert len(found) == 2

    def test_until_excludes_later_points(self, filled):
        found = filled.query(until=T0 - 30 * DAY)
        assert len(found) == 1

    def test_window_drops_history_older_than_the_window(self, filled):
        recent = filled.window(30, now=T0, metric=SKILL_SUCCESS_RATE, target="arxiv")
        assert len(recent) == 1
        assert recent[0].value == pytest.approx(0.7)

    def test_latest_returns_the_newest_point(self, filled):
        assert filled.latest(SKILL_SUCCESS_RATE, "arxiv").value == pytest.approx(0.7)

    def test_latest_is_none_for_an_unknown_series(self, filled):
        assert filled.latest(SKILL_SUCCESS_RATE, "nonexistent") is None

    def test_pairs_lists_every_series(self, filled):
        assert (SKILL_SUCCESS_RATE, "debugging") in filled.pairs()
        assert len(filled.pairs()) == 4

    def test_targets_can_be_scoped_to_one_metric(self, filled):
        assert filled.targets(metric=SKILL_SUCCESS_RATE) == ["arxiv", "debugging"]


class TestAggregation:
    def test_weighted_mean_respects_sample_counts(self):
        points = [
            point(SKILL_SUCCESS_RATE, "arxiv", 1.0, days_ago=2, samples=1),
            point(SKILL_SUCCESS_RATE, "arxiv", 0.0, days_ago=1, samples=99),
        ]
        aggregate = summarize(SKILL_SUCCESS_RATE, "arxiv", points)
        assert aggregate.mean == pytest.approx(0.5)
        assert aggregate.weighted_mean == pytest.approx(0.01)
        assert aggregate.samples == 100
        assert aggregate.count == 2

    def test_last_value_is_chronological_not_file_order(self):
        points = [
            point(SKILL_SUCCESS_RATE, "arxiv", 0.2, days_ago=1),
            point(SKILL_SUCCESS_RATE, "arxiv", 0.9, days_ago=9),
        ]
        assert summarize(SKILL_SUCCESS_RATE, "arxiv", points).last_value == pytest.approx(0.2)

    def test_empty_aggregate_is_flagged(self):
        aggregate = summarize(SKILL_SUCCESS_RATE, "arxiv", [])
        assert aggregate.empty
        assert aggregate.samples == 0

    def test_store_summarize_honours_the_window(self, store):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.1, days_ago=60, samples=5),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.9, days_ago=1, samples=5),
            ]
        )
        aggregate = store.summarize(SKILL_SUCCESS_RATE, "arxiv", window_days=30, now=T0)
        assert aggregate.count == 1
        assert aggregate.weighted_mean == pytest.approx(0.9)


class TestTrends:
    def _series(self, values, target="arxiv", metric=SKILL_SUCCESS_RATE, step_days=7):
        return [
            point(metric, target, value, days_ago=(len(values) - 1 - i) * step_days)
            for i, value in enumerate(values)
        ]

    def test_rising_series_is_rising(self):
        trend = compute_trend(self._series([0.4, 0.6, 0.8]))
        assert trend.direction is TrendDirection.RISING
        assert trend.slope_per_day > 0
        assert not trend.significant

    def test_declining_series_is_declining_and_significant(self):
        trend = compute_trend(self._series([0.9, 0.8, 0.7]))
        assert trend.direction is TrendDirection.DECLINING
        assert trend.change == pytest.approx(-0.2)
        assert trend.significant
        assert trend.is_deterioration

    def test_flat_series_is_flat(self):
        trend = compute_trend(self._series([0.8, 0.8, 0.8]))
        assert trend.direction is TrendDirection.FLAT
        assert trend.slope_per_day == pytest.approx(0.0)
        assert not trend.significant

    def test_a_tiny_wobble_stays_flat(self):
        trend = compute_trend(self._series([0.80, 0.81, 0.79]))
        assert trend.direction is TrendDirection.FLAT

    def test_too_few_points_is_unknown_not_a_guess(self):
        trend = compute_trend(self._series([0.9, 0.1]))
        assert trend.direction is TrendDirection.UNKNOWN
        assert not trend.significant
        assert "need 3 points" in trend.note

    def test_slope_is_per_day_not_per_point(self):
        # Ten points, one a day, dropping 0.05 a day.
        values = [1.0 - 0.05 * i for i in range(10)]
        trend = compute_trend(self._series(values, step_days=1))
        assert trend.slope_per_day == pytest.approx(-0.05, abs=1e-9)
        assert trend.span_days == pytest.approx(9.0)

    def test_a_small_decline_is_not_significant(self):
        trend = compute_trend(self._series([0.85, 0.83, 0.82]))
        assert trend.direction is TrendDirection.DECLINING
        assert trend.change == pytest.approx(-0.03)
        assert not trend.significant

    def test_rising_corrections_count_as_deterioration(self):
        trend = compute_trend(
            self._series([1.0, 3.0, 6.0], target="read_file", metric=USER_CORRECTION)
        )
        assert trend.direction is TrendDirection.RISING
        assert trend.is_deterioration
        assert trend.significant

    def test_simultaneous_points_report_change_without_a_slope(self):
        points = [
            point(SKILL_SUCCESS_RATE, "arxiv", value, days_ago=3)
            for value in (0.9, 0.5, 0.4)
        ]
        trend = compute_trend(points)
        assert trend.slope_per_day == pytest.approx(0.0)
        assert trend.change == pytest.approx(-0.5)
        assert trend.direction is TrendDirection.DECLINING

    def test_describe_is_human_readable(self):
        trend = compute_trend(self._series([0.9, 0.8, 0.7]))
        described = trend.describe()
        assert "declining" in described
        assert "/day" in described

    def test_trend_serialises(self):
        blob = compute_trend(self._series([0.9, 0.8, 0.7])).to_dict()
        assert blob["direction"] == "declining"
        assert blob["significant"] is True

    def test_store_trend_reads_from_disk(self, store):
        store.extend(self._series([0.9, 0.8, 0.7]))
        trend = store.trend(SKILL_SUCCESS_RATE, "arxiv", window_days=60, now=T0)
        assert trend.direction is TrendDirection.DECLINING
        assert trend.n == 3


class TestArchiving:
    def test_old_points_move_to_the_archive_and_nothing_is_lost(self, store):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.4, days_ago=200),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.5, days_ago=150),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.6, days_ago=2),
            ]
        )
        moved = store.archive_before(T0 - 100 * DAY)

        assert moved == 2
        assert len(store.load()) == 1
        archived = store.archive_path.read_text().strip().splitlines()
        assert len(archived) == 2

    def test_archiving_nothing_is_a_no_op(self, store):
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.6, timestamp=T0)
        assert store.archive_before(T0 - 100 * DAY) == 0
        assert not store.archive_path.exists()
        assert len(store.load()) == 1

    def test_archiving_an_empty_store_is_safe(self, store):
        assert store.archive_before(T0) == 0
