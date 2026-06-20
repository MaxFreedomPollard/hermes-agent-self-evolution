"""Tests for evolution config environment handling."""

from pathlib import Path

import pytest

from evolution.core.config import (
    EvolutionConfig,
    _default_lm_max_tokens,
    _default_tblite_baseline_score,
    resolve_hermes_agent_path,
)


def test_evolution_config_can_construct_without_repo(monkeypatch):
    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)

    config = EvolutionConfig()

    assert config.hermes_agent_path is None or isinstance(config.hermes_agent_path, Path)


def test_resolve_prefers_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_REPO", str(tmp_path / "env-repo"))
    explicit = tmp_path / "explicit-repo"

    assert resolve_hermes_agent_path(str(explicit)) == explicit


def test_api_base_reads_openrouter_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    assert EvolutionConfig().api_base == "https://openrouter.ai/api/v1"


def test_lm_max_tokens_validation(monkeypatch):
    monkeypatch.setenv("HERMES_EVOLUTION_MAX_TOKENS", "4096")
    assert _default_lm_max_tokens() == 4096

    monkeypatch.setenv("HERMES_EVOLUTION_MAX_TOKENS", "nope")
    with pytest.raises(ValueError):
        _default_lm_max_tokens()


def test_tblite_baseline_score_reads_env(monkeypatch):
    monkeypatch.setenv("HERMES_TBLITE_BASELINE_SCORE", "0.91")

    assert _default_tblite_baseline_score() == pytest.approx(0.91)

    monkeypatch.setenv("HERMES_TBLITE_BASELINE_SCORE", "not-a-number")
    with pytest.raises(ValueError):
        _default_tblite_baseline_score()
