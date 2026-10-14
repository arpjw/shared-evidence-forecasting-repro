"""Seeded synthetic questions with exactly calibrated marginal forecasts.

For a fair binary outcome, S_i | Y has mean a(2Y-1) and variance one,
so the Gaussian likelihood ratio gives log odds 2*a*S_i. Within a source
group the conditional covariance is overlap; distinct groups are independent
conditional on Y. Unconditional signal/forecast correlations differ.

Seed derivation is versioned and independent of Python's randomized hash().
Replays use the standard-library Random generator and gauss implementation;
record the Python version for archival reproducibility across runtimes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass

GroupID = str | int


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def derive_seed(seed: int, *parts: str | int) -> int:
    """Derive a stream seed from an integer root and typed component names."""
    _integer(seed, "seed")
    if any(isinstance(p, bool) or not isinstance(p, (str, int)) for p in parts):
        raise ValueError("seed components must be strings or integers")
    payload = json.dumps(["shared-evidence-v1", seed, *parts], ensure_ascii=True,
                         separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


@dataclass(frozen=True)
class SimulationConfig:
    """Equal-strength signals, with optional independent source groups.

    ``group_ids`` assigns one label to each forecaster. Labels identify random
    streams, so renaming a group changes its realized common noise. Omitting
    the labels uses a single exchangeable group. At overlap one, every member
    of a group receives precisely the same signal.
    """

    n_forecasters: int
    signal_strength: float
    overlap: float
    group_ids: tuple[GroupID, ...] | None = None

    def __post_init__(self) -> None:
        _integer(self.n_forecasters, "n_forecasters", 1)
        for name, value in (("signal_strength", self.signal_strength),
                            ("overlap", self.overlap)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.overlap > 1:
            raise ValueError("overlap must be <= 1")
        if self.group_ids is not None:
            if isinstance(self.group_ids, (str, bytes)):
                raise ValueError("group_ids must be a sequence of group labels")
            groups = tuple(self.group_ids)
            if len(groups) != self.n_forecasters:
                raise ValueError("group_ids must have one entry per forecaster")
            if any(isinstance(g, bool) or not isinstance(g, (str, int)) for g in groups):
                raise ValueError("group labels must be strings or integers")
            object.__setattr__(self, "group_ids", groups)

    @property
    def groups(self) -> tuple[GroupID, ...]:
        return self.group_ids if self.group_ids is not None else (0,) * self.n_forecasters


@dataclass(frozen=True)
class Question:
    question_id: int
    outcome: int
    signals: tuple[float, ...]
    log_odds: tuple[float, ...]
    group_ids: tuple[GroupID, ...]


def simulate_question(
    config: SimulationConfig, *, seed: int, question_id: int = 0,
    split: str = "default",
) -> Question:
    """Generate one question independently of batch size and call order.

    Distinct split names create distinct streams. Reusing a split, seed, and
    question ID deliberately reuses latent random draws across configurations
    for paired comparisons; it must never stand in for a held-out split.
    """
    _integer(question_id, "question_id")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")
    base = derive_seed(seed, split, question_id)
    outcome = int(random.Random(derive_seed(base, "outcome")).random() < 0.5)
    group_noise = {
        g: random.Random(derive_seed(base, "group", g)).gauss(0.0, 1.0)
        for g in dict.fromkeys(config.groups)
    }
    private = random.Random(derive_seed(base, "private"))
    common_scale = math.sqrt(config.overlap)
    private_scale = math.sqrt(1.0 - config.overlap)
    mean = config.signal_strength * (2 * outcome - 1)
    signals = tuple(
        mean + common_scale * group_noise[group] + private_scale * private.gauss(0, 1)
        for group in config.groups
    )
    log_odds = tuple(2.0 * config.signal_strength * signal for signal in signals)
    if not all(math.isfinite(value) for value in (*signals, *log_odds)):
        raise ValueError("signal_strength produced nonfinite signals or log odds")
    return Question(question_id, outcome, signals, log_odds, config.groups)


def simulate_questions(
    config: SimulationConfig, count: int, *, seed: int,
    split: str = "default", start: int = 0,
) -> tuple[Question, ...]:
    """Generate a replayable batch; ``start`` permits deterministic sharding."""
    _integer(count, "count")
    _integer(start, "start")
    _integer(seed, "seed")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")
    return tuple(simulate_question(config, seed=seed, split=split, question_id=i)
                 for i in range(start, start + count))
