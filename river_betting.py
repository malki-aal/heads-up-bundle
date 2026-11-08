"""Estimate the chip value of opening river bets and learn how they get answered."""

import numpy as np

from equity import river_distribution
from game import Action, HandResult, Observation, Street

BET_SIZES = (Action.RAISE_HALF, Action.RAISE_POT)
PRIOR_RESPONSES = 24


def _size_bucket(size: int, pot: int) -> int:
    return int(size >= 0.7 * pot)


def record_opening(obs: Observation, action: Action, memory: dict) -> None:
    pending = memory.get("river_open")
    if pending and pending[0] == obs.hand_index and obs.to_call > 0:
        _record_response(memory, pending[1], "raise")
    if obs.to_call == 0 and action in BET_SIZES:
        size = min(obs.raise_to[action] - obs.my_bet, obs.opp_stack)
        memory["river_open"] = (obs.hand_index, _size_bucket(size, obs.pot))


def finish_opening(result: HandResult, memory: dict) -> None:
    pending = memory.get("river_open")
    if pending and pending[0] == result.hand_index:
        response = "call" if result.went_to_showdown else "fold"
        _record_response(memory, pending[1], response)


def _record_response(memory: dict, bucket: int, response: str) -> None:
    stats = memory.setdefault("river_responses", {}).setdefault(bucket, {})
    stats[response] = stats.get(response, 0) + 1
    memory.pop("river_open", None)


def opening_values(obs: Observation, memory: dict) -> dict[Action, float]:
    """Value legal check, half-pot and pot bets in chips from this decision."""
    shares, strength, frequency = river_distribution(obs.hole_cards, obs.board)
    prior_actions = [
        event for event in obs.action_history
        if not event.is_me and event.street in (Street.FLOP, Street.TURN)
    ]
    aggression = sum(event.action in BET_SIZES for event in prior_actions)
    paid_calls = sum(event.action == Action.CHECK_CALL and event.amount > 0 for event in prior_actions)
    value_mix = min(0.72, 0.13 + 0.14 * aggression + 0.10 * paid_calls)
    strong = frequency * strength ** 3
    weights = (1 - value_mix) * frequency / frequency.sum() + value_mix * strong / strong.sum()

    # Compute percentiles in the range that reached this river. Keep ties together.
    _, groups = np.unique(strength, return_inverse=True)
    mass = np.bincount(groups, weights=weights)
    rank = (np.cumsum(mass) - mass / 2)[groups]

    pot = obs.pot
    values = {Action.CHECK_CALL: float(weights @ shares) * pot}
    for action in BET_SIZES:
        if action not in obs.legal_actions:
            continue
        size = min(obs.raise_to[action] - obs.my_bet, obs.opp_stack)
        stats = memory.get("river_responses", {}).get(_size_bucket(size, pot), {})
        total = PRIOR_RESPONSES + sum(stats.values())
        fold_prior = 0.30 + 0.32 * min(1.0, size / pot)
        fold_rate = (fold_prior * PRIOR_RESPONSES + stats.get("fold", 0)) / total
        fold = np.clip(0.5 + (fold_rate - rank) / 0.22, 0, 1)

        effective_stack = min(obs.my_stack, obs.opp_stack)
        raise_rate = (0.05 * PRIOR_RESPONSES + stats.get("raise", 0)) / total
        raised = np.zeros_like(rank)
        if size < effective_stack:
            # Give the strongest hands most of the raise probability.
            raised = np.minimum(1 - fold, raise_rate * rank ** 9 / (weights @ (rank ** 9)))
        called = 1 - fold - raised
        call_ev = shares * (pot + 2 * size) - size

        raise_size = min(effective_stack, 3 * size)
        raise_ev = shares * (pot + 2 * raise_size) - raise_size
        raise_value = max(float(weights @ (raised * raise_ev)), -size * float(weights @ raised))
        values[action] = float(weights @ (fold * pot + called * call_ev)) + raise_value
    return values


def choose_opening(obs: Observation, memory: dict, fallback: Action, margin: float) -> Action:
    values = opening_values(obs, memory)
    candidate = max(values, key=values.__getitem__)
    return candidate if values[candidate] > values[fallback] + margin * obs.pot else fallback
