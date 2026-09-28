# SPADE-inspired MetaWorld Environment Designer

This project implements only the **Environment Designer loop**. It does not train,
fine-tune, or evaluate a policy. Random or lightweight goal-directed actions are used
only to verify that generated MetaWorld environments can reset and step safely.

## Loop

```text
structured proposal (JSON-compatible Pydantic model)
  -> schema + static complexity + diversity checks
  -> real MetaWorld MT1 construction
  -> reset/step rollout sanity checks
  -> feedback + append-only JSONL memory
  -> next, strictly more complex proposal
```

The default designer is deterministic and rule-based, so the experiment works without
an LLM key. `RuleBasedDesigner.propose()` is the replaceable seam for an LLM designer:
an LLM must return the same `EnvironmentSpec` JSON contract and receives validator
feedback from rejected proposals. The exported machine-readable contract is at
`schemas/environment_spec.schema.json`.

## Complexity definition

The score is an explicit curriculum proxy, not a claim about learned-policy performance:

- **task family**: `reach < push < pick-place < door-open < assembly` (20 points/stage)
- **goal/task variation**: `5 * log2(goal_variations + 1)`
- **time pressure**: `(500 - episode_horizon) / 25` (shorter is harder)
- **constraints**: reduced action scale, nonzero variance requirement, and early success
  termination add small increments

Every accepted candidate must have a new fingerprint and score strictly greater than
the previous accepted candidate. MuJoCo XML is never generated or modified.

## Install

MetaWorld currently supports Python 3.10-3.13. With `uv`:

```bash
uv python install 3.12
uv sync --python 3.12 --extra dev
```

## Run

```bash
uv run pytest
uv run spade-metaworld --config configs/minimal.yaml
uv run python scripts/export_schema.py
```

Outputs are written under `results/latest/`:

- `history.jsonl`: every proposal, including rejected attempts and feedback
- `summary.json`: accepted specs, validation checks, metrics, and complexity breakdowns
- `resolved_config.yaml`: exact experiment configuration

The repository also includes `results/example_seed_20260927/`, produced by the default
configuration on Python 3.12.13 with MetaWorld 3.1.1. It accepts all four rounds and
covers `reach-v3`, `push-v3`, `pick-place-v3`, and `door-open-v3`.

To reproduce the checked-in result, use the same config and seed. MuJoCo physics may
produce tiny floating-point differences across CPU/OS versions, while specifications,
fingerprints, acceptance decisions, and complexity scores remain deterministic.

## Validation layers

1. Pydantic rejects unknown fields, invalid task families, and inconsistent horizons.
2. Static validation enforces monotonically increasing complexity and diversity.
3. MetaWorld must construct the requested MT1 environment and expose valid spaces.
4. Multiple seeded rollouts check finite/bounded observations, declared observation
   space membership, trajectory variance, rewards, termination, and truncation.
5. All outcomes enter append-only memory and failed checks become designer feedback.

## Scope and extensions

This minimal version fixes a safe whitelist of task families and changes only supported
MetaWorld constructor inputs. Useful extensions are an LLM-backed proposer, Pareto-style
multi-axis complexity (instead of one scalar), richer task-specific scripted sanity
controllers, custom multi-task benchmarks, rendered evidence, and repeated runs across
platforms. None of those requires policy training.
