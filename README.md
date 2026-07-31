# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► PR against hermes-agent
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# See what this installation can find, optimize, and gate on
hermes-evolve status

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb
```

Run `hermes-evolve status` first. It reports whether a hermes-agent checkout is
reachable, how many targets each phase finds in it, and which validation gates can
actually run, which is otherwise something you learn by watching a run fail.

## Command Reference

Every phase has a `hermes-evolve` subcommand and an equivalent `python -m` form.

```bash
hermes-evolve status                      # what is reachable and gateable
hermes-evolve skill   --skill NAME        # Phase 1: skill files
hermes-evolve tools                       # Phase 2: tool descriptions
hermes-evolve prompt  --all-sections      # Phase 3: system prompt sections
hermes-evolve code    --tool file_tools   # Phase 4: tool implementation code
hermes-evolve monitor --once              # Phase 5: triage and propose
```

Phases 2 and 3 default to `--no-write`: they measure and report without touching your
hermes-agent checkout. Pass `--write` when you want the evolved text applied to a branch.
Phase 4 never writes to your working state at all; it produces a branch and a diff.

```bash
# Phase 2: evolve one toolset, keeping the repo untouched
hermes-evolve tools --toolset file --iterations 10

# Phase 3: zero benchmark regression tolerance, as PLAN.md requires
hermes-evolve prompt --section MEMORY_GUIDANCE --strict-gates

# Phase 5: print an installable weekly schedule (installs nothing)
hermes-evolve monitor --emit-cron
```

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | ✅ Implemented |
| **Phase 3** | System prompt sections | DSPy + GEPA | ✅ Implemented |
| **Phase 4** | Tool implementation code | Darwinian Evolver | ✅ Implemented |
| **Phase 5** | Continuous improvement loop | Automated pipeline | ✅ Implemented |

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Every evolved variant must pass:
1. **Full test suite** — `pytest tests/ -q` must pass 100%
2. **Size limits** — Skills ≤15KB, tool descriptions ≤500 chars
3. **Caching compatibility** — No mid-conversation changes
4. **Semantic preservation** — Must not drift from original purpose
5. **PR review** — All changes go through human review, never direct commit

Two of these are enforced mechanically rather than by convention:

**Structure is frozen.** Tool schemas and prompt constants are rewritten by replacing
the exact source span of a single string literal, then re-parsing and diffing the schema
skeleton. Any candidate that moves a parameter name, type, enum, default, or required
list is rejected before it reaches disk. Only description text can change.

**A missing gate is not a passing gate.** A benchmark that is not installed reports
`unavailable`, never `passed`. Runs are permissive by default so the pipeline is usable
today; `--strict-gates` turns any unavailable gate into a hard failure for a release
process that needs every gate to have actually run.

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
