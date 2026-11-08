"""Conservative action-value planning from public betting history."""

from dataclasses import dataclass
import os

import numpy as np

from game import Action, AgentConfig, HandResult, Observation, Street
from sampling import draw_worlds, showdown_scores, showdown_shares

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def _context(street, facing, due, pot, button, raises=0):
    return street, facing, int(due / (pot + due) >= 0.27) if facing else 0, button, raises >= 2


def _events(obs: Observation, config: AgentConfig):
    bets = [config.small_blind, config.big_blind] if obs.is_button else [config.big_blind, config.small_blind]
    stacks = [config.starting_stack - value for value in bets]
    pot = sum(bets)
    street, raises = Street.PREFLOP, 0
    for event in obs.action_history:
        if event.street != street:
            street, raises, bets = event.street, 0, [0, 0]
        player = 0 if event.is_me else 1
        due = max(0, bets[1 - player] - bets[player])
        if not event.is_me:
            allowed = raises < config.raise_cap and stacks[player] > due and stacks[1 - player] > 0
            yield _context(street, due > 0 and raises > 0, due, pot, not obs.is_button, raises), event.action, allowed
        bets[player] = event.street_total
        stacks[player] -= event.amount
        pot += event.amount
        raises += int(event.action in RAISES)


def _add_response(memory, context, action):
    counts = memory.setdefault('action_counts', {}).setdefault(context, [0, 0, 0])
    counts[0 if action == Action.FOLD else 2 if action in RAISES else 1] += 1


def finish_planning(result: HandResult, memory: dict) -> None:
    pending = memory.get('planning_pending')
    if pending is None or pending[0].hand_index != result.hand_index:
        return
    obs, action = memory.pop('planning_pending')
    config = memory.get('config', AgentConfig())
    for context, response, allowed in _events(obs, config):
        if context[1] or allowed:
            _add_response(memory, context, response)
    if action not in RAISES:
        return
    paid = obs.raise_to[action] - obs.my_bet
    due = obs.raise_to[action] - obs.opp_bet
    context = _context(obs.street, True, due, obs.pot + paid, not obs.is_button, obs.raises_this_street + 1)
    if not result.went_to_showdown and result.winner == 'me' and result.board == obs.board:
        _add_response(memory, context, Action.FOLD)
    elif result.went_to_showdown and (
        obs.street == Street.RIVER or paid == obs.my_stack or due >= obs.opp_stack
    ):
        _add_response(memory, context, Action.CHECK_CALL)


def _rates(memory, context):
    prior = np.array([0.0, 0.58, 0.42])
    if context[1]:
        prior = np.array([0.15, 0.60, 0.25]) if context[4] else np.array([0.28, 0.62, 0.10])
    counts = np.asarray(memory.get('action_counts', {}).get(context, [0, 0, 0]))
    return (16 * prior + counts) / (16 + counts.sum())


def _percentiles(scores, weights):
    _, group = np.unique(scores, return_inverse=True)
    mass = np.bincount(group, weights=weights)
    return (mass.cumsum() - 0.5 * mass)[group] / mass.sum()


def _responses(percentile, rates, can_raise):
    fold_rate, _, raise_rate = rates
    folded = np.clip(fold_rate * (1.8 - 1.6 * percentile), 0.0, 0.95)
    raised = np.zeros_like(folded)
    if can_raise:
        raised = np.minimum(1 - folded, raise_rate * (0.2 + 3.2 * percentile ** 3))
    return folded, 1 - folded - raised, raised


@dataclass
class BoardSamples:
    shares: np.ndarray
    strengths: dict
    improvement: float


def _sample(obs, preflop_model):
    rng = np.random.default_rng((*obs.hole_cards, *obs.board, 3119))
    count = 192 if obs.street == Street.FLOP else 256
    opponents, boards = draw_worlds(obs.hole_cards, obs.board, count, rng)
    shares = showdown_shares(obs.hole_cards, opponents, boards)
    strengths = {Street.PREFLOP: preflop_model.estimate_many(opponents)}
    for street, size in ((Street.FLOP, 3), (Street.TURN, 4)):
        if size <= len(obs.board):
            visible = np.broadcast_to(obs.board[:size], (count, size))
            strengths[street] = showdown_scores(opponents, visible)
    made = showdown_scores(np.asarray(obs.hole_cards), np.asarray(obs.board))
    completed = showdown_scores(np.broadcast_to(obs.hole_cards, opponents.shape), boards)
    improvement = float(((completed >> 20) > (made >> 20)).mean())
    return BoardSamples(shares, strengths, improvement)


def _range_weights(obs, samples, memory):
    weights = np.ones(len(samples.shares))
    config = memory.get('config', AgentConfig())
    for context, action, allowed in _events(obs, config):
        if not context[1] and not allowed:
            continue
        percentile = _percentiles(samples.strengths[context[0]], weights)
        rates = _rates(memory, context)
        if context[1]:
            probabilities = _responses(percentile, rates, allowed)
            likelihood = probabilities[0 if action == Action.FOLD else 2 if action in RAISES else 1]
        else:
            rate = rates[2]
            bet_probability = np.clip(rate + 1.4 * rate * (1 - rate) * (2 * percentile - 1), 0.05, 0.95)
            likelihood = bet_probability if action in RAISES else 1 - bet_probability
        weights *= np.maximum(likelihood, 0.05)
        weights /= weights.mean()
    # Retain a broad component when a sequence of correlated actions is modeled.
    return (0.8 * weights + 0.2) / len(weights)


def _equity(shares, weights, obs, remaining, pot, improvement):
    probability = float(weights @ shares / weights.sum())
    if remaining <= 0 or obs.street == Street.RIVER:
        return probability
    exposure = remaining / (remaining + pot)
    position_cost = 0.025 if obs.is_button else 0.065
    return max(0.0, probability - position_cost * exposure * (1 - 0.5 * improvement))


def action_values(obs: Observation, memory: dict, preflop_model) -> dict:
    key = (obs.hand_index, obs.hole_cards, obs.board)
    cache = memory.get('planning_samples')
    if cache is None or cache[0] != key:
        cache = key, _sample(obs, preflop_model)
        memory['planning_samples'] = cache
    samples = cache[1]
    weights = _range_weights(obs, samples, memory)
    price = min(obs.to_call, obs.my_stack)
    pot = obs.pot + price
    remaining = min(obs.my_stack - price, obs.opp_stack)
    equity = _equity(samples.shares, weights, obs, remaining, pot, samples.improvement)
    values = {Action.CHECK_CALL: equity * pot - price}
    if Action.FOLD in obs.legal_actions:
        values[Action.FOLD] = 0.0
    percentile = _percentiles(samples.strengths[obs.street], weights)
    config = memory.get('config', AgentConfig())
    for action in RAISES:
        if action not in obs.legal_actions:
            continue
        target = obs.raise_to[action]
        paid = target - obs.my_bet
        due = min(obs.opp_stack, target - obs.opp_bet)
        pot = obs.pot + paid + due
        remaining = min(obs.my_stack - paid, obs.opp_stack - due)
        can_raise = remaining > 0 and obs.raises_this_street + 1 < config.raise_cap
        rates = _rates(memory, _context(obs.street, True, due, obs.pot + paid, not obs.is_button, obs.raises_this_street + 1))
        folded, called, raised = _responses(percentile, rates, can_raise)
        call_weights = weights * called
        equity = _equity(samples.shares, call_weights, obs, remaining, pot, samples.improvement)
        value = float(weights @ folded) * obs.pot + call_weights.sum() * (equity * pot - paid)
        if can_raise:
            raised_to = min(obs.opp_bet + obs.opp_stack, target + pot)
            extra = min(obs.my_stack - paid, raised_to - target)
            raised_pot = obs.pot + paid + raised_to - obs.opp_bet + extra
            raised_weights = weights * raised
            equity = _equity(samples.shares, raised_weights, obs, 0, raised_pot, samples.improvement)
            value += raised_weights.sum() * max(-paid, equity * raised_pot - paid - extra)
        values[action] = float(value)
    return values


def choose_planned(obs, memory, preflop_model, fallback):
    values = action_values(obs, memory, preflop_model)
    candidate = max(values, key=values.__getitem__)
    # Demand a material advantage over the existing policy before changing it.
    margin = (0.02 if obs.to_call else 0.06) * obs.pot
    if candidate in RAISES:
        samples = memory['planning_samples'][1]
        if samples.shares.mean() < 0.40:
            return fallback
    return candidate if values[candidate] > values[fallback] + margin else fallback


def frequent_flop_bettor(obs: Observation, memory: dict) -> bool:
    if obs.street != Street.FLOP or obs.raises_this_street != 1:
        return False
    if obs.is_button and os.environ.get('HEADS_UP_IP_DEFENSE', '1') != '1':
        return False
    context = _context(Street.FLOP, False, 0, 1, not obs.is_button)
    counts = memory.get('action_counts', {}).get(context, [0, 0, 0])
    seen = sum(counts)
    threshold = 0.60 if obs.is_button else 0.75
    return seen >= 24 and (counts[2] + 6) / (seen + 12) >= threshold


def protect_turn_call(obs, memory, preflop_model, fallback):
    if obs.street != Street.TURN or fallback != Action.CHECK_CALL or not obs.to_call:
        return fallback
    price = min(obs.my_stack, obs.to_call)
    all_in = obs.opp_stack == 0 or price == obs.my_stack
    if not all_in and price / (obs.pot + price) < 0.29:
        return fallback
    context = _context(Street.TURN, False, 0, 1, not obs.is_button)
    counts = memory.get('action_counts', {}).get(context, [0, 0, 0])
    if sum(counts) < 16 or _rates(memory, context)[2] > 0.55:
        return fallback
    values = action_values(obs, memory, preflop_model)
    return Action.FOLD if values[Action.CHECK_CALL] < -0.03 * obs.pot else fallback
