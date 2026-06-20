"""Configuration and hermes-agent repo discovery."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _default_api_base() -> Optional[str]:
    """Return the configured OpenAI-compatible API base, if any."""

    return (
        os.getenv("OPENROUTER_BASE_URL")
        or os.getenv("DSPY_API_BASE")
        or os.getenv("OPENAI_API_BASE")
    )


def _default_lm_max_tokens() -> Optional[int]:
    """Return an optional LM max_tokens cap from the environment."""

    raw = os.getenv("HERMES_EVOLUTION_MAX_TOKENS")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("HERMES_EVOLUTION_MAX_TOKENS must be an integer") from None
    if value <= 0:
        raise ValueError("HERMES_EVOLUTION_MAX_TOKENS must be positive")
    return value


def _default_tblite_baseline_score() -> Optional[float]:
    """Return an optional precomputed TBLite baseline score from the environment."""

    raw = os.getenv("HERMES_TBLITE_BASELINE_SCORE")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError("HERMES_TBLITE_BASELINE_SCORE must be a number") from None


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path. Discovered lazily and non-fatally so the config
    # can be constructed even when no repo is present (e.g. unit tests, or
    # callers that pass an explicit path). Use resolve_hermes_agent_path() when
    # an explicit override should win, or get_hermes_agent_path() to require one.
    hermes_agent_path: Optional[Path] = field(default_factory=lambda: _discover_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration
    optimizer_model: str = "openai/gpt-4.1"  # Model for GEPA reflections
    eval_model: str = "openai/gpt-4.1-mini"  # Student/eval model
    judge_model: str = "openai/gpt-4.1"  # LLM-as-judge + dataset generation model
    api_base: Optional[str] = field(default_factory=_default_api_base)
    lm_max_tokens: Optional[int] = field(default_factory=_default_lm_max_tokens)

    # Constraints
    max_skill_size: int = 15_000  # Default cap; evolve_skill raises this to baseline+growth when needed.
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.2  # 20% max growth over baseline

    # Promotion gates
    min_skill_improvement: float = 0.10  # Phase-1 gate: >=10% relative score lift
    allow_heuristic_fallback: bool = False  # LLM-judge failures should fail closed by default

    # Eval dataset
    eval_dataset_size: int = 20  # Total examples to generate
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    holdout_ratio: float = 0.25

    # Benchmark gating
    run_pytest: bool = True
    run_tblite: bool = False  # Expensive — opt-in
    tblite_regression_threshold: float = 0.02  # Max 2% absolute regression allowed
    tblite_command: Optional[str] = field(default_factory=lambda: os.getenv("HERMES_TBLITE_COMMAND"))
    tblite_baseline_score: Optional[float] = field(default_factory=_default_tblite_baseline_score)

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True


def _discover_hermes_agent_path() -> Optional[Path]:
    """Best-effort hermes-agent repo discovery that never raises.

    Returns the discovered path, or None when no repo can be found. Used as the
    EvolutionConfig default so construction never crashes; callers that truly
    require the repo should use get_hermes_agent_path().
    """

    try:
        return get_hermes_agent_path()
    except FileNotFoundError:
        return None


def get_hermes_agent_path() -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. ../hermes-agent (sibling directory)
    """

    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p.resolve()

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path.resolve()

    sibling_path = Path(__file__).parent.parent.parent / "hermes-agent"
    if sibling_path.exists():
        return sibling_path.resolve()

    raise FileNotFoundError(
        "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
        "or ensure it exists at ~/.hermes/hermes-agent"
    )


def resolve_hermes_agent_path(hermes_repo: Optional[str] = None) -> Path:
    """Return the hermes-agent repo path, honoring an explicit override.

    An explicit path (for example from ``--hermes-repo``) is expanded and used
    as-is, taking precedence over auto-discovery. This lets callers point at a
    repo in a non-default location without the tool crashing just because
    ``~/.hermes/hermes-agent`` happens to be absent.

    When no override is given, falls back to :func:`get_hermes_agent_path`.
    """

    if hermes_repo:
        return Path(hermes_repo).expanduser()
    return get_hermes_agent_path()
