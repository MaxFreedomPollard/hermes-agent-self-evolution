"""Shared DSPy language-model construction helpers."""

from __future__ import annotations

from typing import Any, Optional

import dspy

from evolution.core.config import EvolutionConfig


def make_lm(model: str, config: Optional[EvolutionConfig] = None, **kwargs: Any) -> dspy.LM:
    """Construct a DSPy LM honoring configured OpenAI-compatible settings.

    The self-evolution repo supports OpenAI-style providers such as OpenRouter
    by passing ``api_base`` through to DSPy. Keeping this in one helper prevents
    the optimizer, dataset builder, and judge from silently using different
    providers or token caps.
    """

    params = dict(kwargs)
    if config is not None:
        if config.api_base and "api_base" not in params:
            params["api_base"] = config.api_base
        if config.lm_max_tokens and "max_tokens" not in params:
            params["max_tokens"] = config.lm_max_tokens
    return dspy.LM(model, **params)
