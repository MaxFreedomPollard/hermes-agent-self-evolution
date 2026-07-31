"""Tests for the auto-triage ranker.

Pure computation over fixed points at fixed timestamps, so every ranking here
is reproducible. The arithmetic is asserted directly rather than through
approximate ordering, because the ranking is what decides where money gets
spent on an optimizer run.
"""

import pytest

from evolution.monitor.metrics import (
    BENCHMARK_SCORE,
    SECONDS_PER_DAY,
    SKILL_SUCCESS_RATE,
    TOOL_SELECTION_ACCURACY,
    USER_CORRECTION,
    MetricPoint,
    MetricStore,
)
from evolution.monitor.triage import (
    AutoTriage,
    TargetType,
    TriageConfig,
    rank_points,
)

T0 = 1_700_000_000.0
DAY = SECONDS_PER_DAY


def point(metric, target, value, days_ago=1.0, samples=1, **metadata):
    return MetricPoint(
        metric=metric,
        target=target,
        value=value,
        timestamp=T0 - days_ago * DAY,
        samples=samples,
        source="test",
        metadata=dict(metadata),
    )


def corrections(target, count, days_ago=1.0, **metadata):
    return [
        point(USER_CORRECTION, target, 1.0, days_ago=days_ago + i, **metadata)
        for i in range(count)
    ]


def by_target(entries):
    return {entry.target: entry for entry in entries}


class TestRankingOrder:
    def test_usage_frequency_breaks_a_tie_on_potential_improvement(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy", 0.5, samples=100),
                point(SKILL_SUCCESS_RATE, "quiet", 0.5, samples=10),
            ],
            now=T0,
        )
        assert [e.target for e in entries] == ["busy", "quiet"]
        assert entries[0].score == pytest.approx(0.5)
        assert entries[1].score == pytest.approx(0.05)

    def test_potential_improvement_breaks_a_tie_on_usage(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "healthy", 0.9, samples=100),
                point(SKILL_SUCCESS_RATE, "struggling", 0.4, samples=100),
            ],
            now=T0,
        )
        assert [e.target for e in entries] == ["struggling", "healthy"]

    def test_the_score_is_improvement_times_frequency(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "alpha", 0.6, samples=50),
                point(SKILL_SUCCESS_RATE, "beta", 0.9, samples=100),
            ],
            now=T0,
        )
        ranked = by_target(entries)
        assert ranked["alpha"].potential_improvement == pytest.approx(0.4)
        assert ranked["alpha"].usage_weight == pytest.approx(0.5)
        assert ranked["alpha"].score == pytest.approx(0.4 * 0.5)

    def test_a_busy_healthy_target_loses_to_a_quiet_broken_one_only_if_the_gap_is_big(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy_ok", 0.95, samples=1000),
                point(SKILL_SUCCESS_RATE, "quiet_bad", 0.10, samples=100),
            ],
            now=T0,
        )
        ranked = by_target(entries)
        assert ranked["busy_ok"].score == pytest.approx(0.05)
        assert ranked["quiet_bad"].score == pytest.approx(0.09)
        assert entries[0].target == "quiet_bad"

    def test_ordering_is_deterministic_for_identical_scores(self):
        points = [
            point(SKILL_SUCCESS_RATE, "zeta", 0.5, samples=10),
            point(SKILL_SUCCESS_RATE, "alpha", 0.5, samples=10),
        ]
        first = [e.target for e in rank_points(points, now=T0)]
        second = [e.target for e in rank_points(list(reversed(points)), now=T0)]
        assert first == second == ["alpha", "zeta"]

    def test_the_weighted_mean_drives_the_score_not_the_last_reading(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.2, days_ago=3, samples=99),
                point(SKILL_SUCCESS_RATE, "arxiv", 1.0, days_ago=1, samples=1),
            ],
            now=T0,
        )
        assert entries[0].current_value == pytest.approx(0.208)


class TestExplanations:
    def test_every_entry_names_both_ranking_factors(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0
        )[0]
        names = [f.name for f in entry.factors]
        assert names[:2] == ["potential improvement", "usage frequency"]

    def test_explain_states_the_arithmetic(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0
        )[0]
        text = entry.explain()
        assert "score 0.500" in text
        assert "potential improvement" in text
        assert " x " in text

    def test_explain_flags_a_trigger(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.2, samples=50)], now=T0
        )[0]
        assert "TRIGGERED" in entry.explain()

    def test_details_describe_each_factor(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0
        )[0]
        joined = " ".join(entry.details())
        assert "leaving 0.50" in joined
        assert "20 observations" in joined

    def test_entries_serialise_with_their_explanation(self):
        blob = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0
        )[0].to_dict()
        assert blob["target"] == "arxiv"
        assert blob["target_type"] == "skill"
        assert blob["actionable"] is True
        assert "explanation" in blob
        assert blob["factors"][0]["name"] == "potential improvement"


class TestTrendPressure:
    def _declining(self, target, values, samples=50):
        return [
            point(SKILL_SUCCESS_RATE, target, value, days_ago=21 - 7 * i, samples=samples)
            for i, value in enumerate(values)
        ]

    def test_a_declining_target_outranks_an_equally_bad_stable_one(self):
        entries = rank_points(
            self._declining("eroding", [0.9, 0.8, 0.7])
            + self._declining("steady", [0.8, 0.8, 0.8]),
            now=T0,
        )
        ranked = by_target(entries)
        assert ranked["eroding"].score == pytest.approx(0.2 * 1.0 * 1.5)
        assert ranked["steady"].score == pytest.approx(0.2)
        assert entries[0].target == "eroding"

    def test_the_decline_shows_up_as_a_named_factor(self):
        entry = by_target(rank_points(self._declining("eroding", [0.9, 0.8, 0.7]), now=T0))[
            "eroding"
        ]
        names = [f.name for f in entry.factors]
        assert "declining trend" in names
        assert entry.trend.significant

    def test_a_significant_decline_triggers_even_below_the_failure_threshold(self):
        entry = rank_points(self._declining("slipping", [0.98, 0.93, 0.88]), now=T0)[0]
        assert entry.potential_improvement < 0.3
        assert entry.triggered
        assert "significant decline" in entry.trigger_reason

    def test_an_improving_target_gets_no_boost(self):
        entry = rank_points(self._declining("improving", [0.5, 0.6, 0.7]), now=T0)[0]
        assert [f.name for f in entry.factors] == [
            "potential improvement",
            "usage frequency",
        ]
        assert not entry.trend.significant


class TestCorrections:
    def test_corrections_boost_a_measured_target(self):
        base = [point(TOOL_SELECTION_ACCURACY, "search_files", 0.8, samples=100)]
        quiet = [point(TOOL_SELECTION_ACCURACY, "read_file", 0.8, samples=100)]
        entries = rank_points(base + quiet + corrections("search_files", 5), now=T0)

        ranked = by_target(entries)
        assert ranked["search_files"].score == pytest.approx(0.2 * 1.25)
        assert ranked["read_file"].score == pytest.approx(0.2)
        assert entries[0].target == "search_files"

    def test_correction_pressure_saturates(self):
        entries = rank_points(
            [point(TOOL_SELECTION_ACCURACY, "search_files", 0.8, samples=100)]
            + corrections("search_files", 40),
            now=T0,
        )
        # 40 corrections cannot buy more than the saturated 1.5x multiplier.
        assert entries[0].score == pytest.approx(0.2 * 1.5)

    def test_a_target_known_only_from_corrections_still_ranks(self):
        entries = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.9, samples=100)]
            + corrections("web_search", 6),
            now=T0,
        )
        entry = by_target(entries)["web_search"]
        assert entry.metric == USER_CORRECTION
        assert entry.current_value is None
        assert entry.corrections == 6
        assert entry.score == pytest.approx(0.6 * 0.06)

    def test_enough_corrections_alone_fire_a_trigger(self):
        entry = rank_points(corrections("web_search", 5), now=T0)[0]
        assert entry.triggered
        assert "5 user corrections" in entry.trigger_reason

    def test_a_couple_of_corrections_do_not_fire_a_trigger(self):
        entry = rank_points(corrections("web_search", 2), now=T0)[0]
        assert not entry.triggered


class TestThresholdTriggering:
    def test_a_failure_rate_exactly_at_the_threshold_fires(self):
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.75, samples=20)], config, now=T0
        )[0]
        assert entry.potential_improvement == pytest.approx(0.25)
        assert entry.triggered
        assert "at or above threshold" in entry.trigger_reason

    def test_just_below_the_threshold_does_not_fire(self):
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.80, samples=20)], config, now=T0
        )[0]
        assert not entry.triggered
        assert entry.trigger_reason == ""

    def test_a_thin_sample_cannot_fire_a_trigger(self):
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.10, samples=3)], config, now=T0
        )[0]
        assert entry.potential_improvement == pytest.approx(0.9)
        assert not entry.triggered

    def test_the_sample_floor_is_inclusive(self):
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.10, samples=5)], config, now=T0
        )[0]
        assert entry.triggered

    def test_a_custom_threshold_is_honoured(self):
        points = [point(SKILL_SUCCESS_RATE, "arxiv", 0.85, samples=50)]
        assert not rank_points(points, TriageConfig(failure_threshold=0.3), now=T0)[0].triggered
        assert rank_points(points, TriageConfig(failure_threshold=0.1), now=T0)[0].triggered

    def test_triggers_helper_returns_only_fired_entries(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "broken", 0.2, samples=50),
                point(SKILL_SUCCESS_RATE, "fine", 0.99, samples=50),
            ]
        )
        fired = AutoTriage(store, TriageConfig()).triggers(now=T0)
        assert [e.target for e in fired] == ["broken"]


class TestFilteringAndScope:
    def test_history_outside_the_window_is_ignored(self):
        entries = rank_points(
            [point(SKILL_SUCCESS_RATE, "ancient", 0.1, days_ago=90, samples=100)],
            TriageConfig(window_days=30),
            now=T0,
        )
        assert entries == []

    def test_weak_candidates_are_dropped(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy", 0.5, samples=1000),
                point(SKILL_SUCCESS_RATE, "negligible", 0.99, samples=5),
            ],
            now=T0,
        )
        assert [e.target for e in entries] == ["busy"]

    def test_a_triggered_candidate_survives_the_score_floor(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy", 0.99, samples=100_000),
                point(SKILL_SUCCESS_RATE, "rare_but_broken", 0.05, samples=5),
            ],
            now=T0,
        )
        rare = by_target(entries)["rare_but_broken"]
        assert rare.score < TriageConfig().min_score
        assert rare.triggered

    def test_limit_truncates_the_ranking(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, f"skill{i}", 0.5, samples=10 * (i + 1))
                for i in range(4)
            ]
        )
        assert len(AutoTriage(store).rank(now=T0, limit=2)) == 2

    def test_autotriage_uses_the_store_clock_when_now_is_absent(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.5, days_ago=2, samples=20)])
        assert [e.target for e in AutoTriage(store).rank()] == ["arxiv"]


class TestTargetTyping:
    def test_skill_metrics_produce_skill_targets(self):
        entry = rank_points([point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0)[0]
        assert entry.target_type is TargetType.SKILL
        assert entry.actionable

    def test_tool_metrics_produce_tool_targets(self):
        entry = rank_points(
            [point(TOOL_SELECTION_ACCURACY, "read_file", 0.5, samples=20)], now=T0
        )[0]
        assert entry.target_type is TargetType.TOOL

    def test_benchmarks_are_ranked_but_never_actionable(self):
        entry = rank_points([point(BENCHMARK_SCORE, "tblite", 0.4, samples=20)], now=T0)[0]
        assert entry.target_type is TargetType.BENCHMARK
        assert not entry.actionable
        assert "advisory" in entry.explain()

    def test_metadata_declares_the_target_type(self):
        entry = rank_points(
            corrections("MEMORY_GUIDANCE", 6, target_type="prompt"), now=T0
        )[0]
        assert entry.target_type is TargetType.PROMPT
        assert entry.actionable

    def test_an_unrecognised_declared_type_is_ignored(self):
        entry = rank_points(corrections("read_file", 6, target_type="nonsense"), now=T0)[0]
        assert entry.target_type is TargetType.TOOL

    def test_a_correction_inherits_the_type_seen_earlier_in_history(self):
        # The success-rate reading is older than the window, so only the
        # corrections rank - but the target is still a skill, not a tool.
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.9, days_ago=80, samples=40)]
            + corrections("arxiv", 6),
            TriageConfig(window_days=30),
            now=T0,
        )[0]
        assert entry.target == "arxiv"
        assert entry.target_type is TargetType.SKILL

    def test_an_extra_metric_can_declare_its_own_type(self):
        config = TriageConfig(extra_metric_types={"tool_crash_free_rate": TargetType.CODE})
        entry = rank_points(
            [point("tool_crash_free_rate", "file_tools", 0.4, samples=30)],
            config,
            now=T0,
        )[0]
        assert entry.target_type is TargetType.CODE

    def test_actionable_only_hides_advisory_entries(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend(
            [
                point(BENCHMARK_SCORE, "tblite", 0.4, samples=20),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.4, samples=20),
            ]
        )
        entries = AutoTriage(store).rank(now=T0, actionable_only=True)
        assert [e.target for e in entries] == ["arxiv"]

    def test_declining_helper_surfaces_deteriorating_series(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "eroding", value, days_ago=21 - 7 * i, samples=50)
                for i, value in enumerate([0.9, 0.8, 0.7])
            ]
            + [point(SKILL_SUCCESS_RATE, "steady", 0.7, samples=50)]
        )
        assert [e.target for e in AutoTriage(store).declining(now=T0)] == ["eroding"]


class TestBookkeepingIsNotATarget:
    def test_the_loops_own_records_do_not_become_candidates(self):
        entries = rank_points(
            [
                point("optimization_run", "arxiv", 1.0, samples=1),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20),
            ],
            now=T0,
        )
        assert [(e.metric, e.target) for e in entries] == [(SKILL_SUCCESS_RATE, "arxiv")]

    def test_an_empty_history_ranks_nothing(self):
        assert rank_points([], now=T0) == []
