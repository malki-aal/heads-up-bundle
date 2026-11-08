"""Shared betting policy for the learned and rules-only versions."""

from equity import card_counts, made_score, river_range_equity
from evaluator import category_of, STRAIGHT
from game import Action, Observation, Street

# Rough equity estimates by made hand category, before draw and board adjustments.
CATEGORY_EQUITY = (0.13, 0.31, 0.51, 0.68, 0.78, 0.90, 0.94, 0.985, 0.995)


def opponent_range_scale(memory: dict) -> float:
    """Shrink a value-heavy range when observed opening bets are indiscriminate."""
    openers = memory.get("opponent_openers", {})
    river = openers.get(Street.RIVER, {})
    counts = river if river.get("seen", 0) >= 16 else openers.get(Street.FLOP, {})
    seen = counts.get("seen", 0)
    rate = (counts.get("bets", 0) + 1) / (seen + 2)
    confidence = min(1.0, seen / 16)
    selection = max(0.0, min(1.0, (0.9 - rate) / 0.3))
    return 1 - confidence * (1 - selection)


def river_response_equity(obs: Observation, value_scale: float = 1.0) -> float:
    """Condition river equity on price and the betting line visible so far."""
    price = min(obs.to_call, obs.my_stack)
    pot_odds = price / (obs.pot + price)
    prior_aggression = {
        event.street for event in obs.action_history
        if not event.is_me and event.street in (Street.FLOP, Street.TURN)
        and event.action in (Action.RAISE_HALF, Action.RAISE_POT)
    }
    river_reraise = obs.raises_this_street >= 2
    pressure = min(1.0, pot_odds * 1.5 + 0.20 * len(prior_aggression) + 0.25 * river_reraise)
    value_weight = (0.55 + 0.35 * pressure) * value_scale
    return river_range_equity(obs.hole_cards, obs.board, value_weight, 2 + 10 * pressure)


def preflop_equity(hole: tuple[int, ...]) -> float:
    ranks, suits = card_counts(hole)
    pairs = [rank for rank, count in enumerate(ranks) if count >= 2]
    high_cards = sum(ranks[6:])
    connected = max(sum(bool(count) for count in ranks[start:start + 4]) for start in range(6))
    return (
        0.35
        + 0.025 * high_cards
        + 0.045 * sum(count >= 2 for count in suits)
        + 0.017 * (connected - 2)
        + 0.035 * len(pairs)
        + 0.009 * max(pairs, default=0)
        - 0.06 * sum(count >= 3 for count in ranks)
    )


def rules_equity(hole: tuple[int, ...], board: tuple[int, ...]) -> float:
    score = made_score(hole, board)
    category = category_of(score)
    _, hole_suits = card_counts(hole)
    board_ranks, board_suits = card_counts(board)
    equity = CATEGORY_EQUITY[category]
    if category < STRAIGHT:
        equity += 0.08 * ((score >> 16 & 15) / 8 - 0.5)
    if category <= STRAIGHT and (max(board_suits) >= 3 or max(board_ranks) >= 2):
        equity -= 0.12
    if len(board) < 5 and any(h >= 2 and b == 2 for h, b in zip(hole_suits, board_suits)):
        equity += (5 - len(board)) * 0.06
    return equity


def select_action(
    obs: Observation, equity: float, raise_margin: float = 0.0, call_margin: float = 0.0,
    preflop_half: float = 0.56, preflop_pot: float = 0.61,
) -> Action:
    """Convert showdown equity into a legal bet, call, check, or fold."""
    legal = obs.legal_actions
    price = min(obs.to_call, obs.my_stack)
    pot_odds = price / (obs.pot + price) if price else 0.0
    all_in_call = obs.opp_stack == 0 or price == obs.my_stack

    if not obs.board:
        if equity >= preflop_pot + raise_margin and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if equity >= preflop_half + raise_margin and obs.raises_this_street == 0 and Action.RAISE_HALF in legal:
            return Action.RAISE_HALF
    else:
        # Require extra strength to re-raise into a range that has already bet.
        pressure = raise_margin + (0.07 if price else 0.0)
        if equity >= 0.84 + pressure and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if equity >= 0.68 + pressure and Action.RAISE_HALF in legal:
            return Action.RAISE_HALF

    if Action.FOLD in legal:
        # Pot odds give the break-even equity if calling ends the betting.
        # With more betting to come, allow for incomplete equity realization.
        cushion = 0.0
        if not all_in_call and len(obs.board) < 5:
            cushion = 0.04 + call_margin
            if obs.raises_this_street >= 2:
                cushion += 0.04
        if equity < pot_odds + cushion:
            return Action.FOLD
    return Action.CHECK_CALL
