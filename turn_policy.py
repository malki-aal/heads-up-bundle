"""Public turn features and a compact action-value model for Short Deck PLO5."""

from functools import lru_cache
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from equity import DECK
from evaluator import get_table
from game import Action, Observation, Street

TURN_WEIGHTS_PATH = Path(__file__).with_name("turn_value_weights.npz")
TURN_SAMPLES = 96
RANK_INDEX = np.array([[comb(card, size) for card in DECK] for size in range(1, 6)])
CARD_INDEX = np.arange(5)
HOLE_PAIRS = np.array(list(combinations(range(5), 2)))
BOARD_TRIPLES = {
    size: np.array(list(combinations(range(size), 3))) for size in (4, 5)
}


@lru_cache(maxsize=1)
def rank_lookup() -> np.ndarray:
    """Index the official five-card scores by the ranks of card combinations."""
    cards = np.array(list(combinations(DECK, 5)), dtype=np.int8)
    lookup = np.empty(len(cards), dtype=np.int32)
    lookup[RANK_INDEX[CARD_INDEX, cards].sum(axis=1)] = np.fromiter(
        get_table().values(), dtype=np.int32, count=len(cards),
    )
    lookup.flags.writeable = False
    return lookup


def _best(hands: np.ndarray, boards: np.ndarray) -> np.ndarray:
    pairs = hands[:, HOLE_PAIRS][:, :, None, :]
    triples = boards[:, BOARD_TRIPLES[boards.shape[1]]][:, None, :, :]
    shape = np.broadcast_shapes(pairs.shape[:-1], triples.shape[:-1])
    cards = np.sort(
        np.concatenate((np.broadcast_to(pairs, (*shape, 2)),
                        np.broadcast_to(triples, (*shape, 3))), axis=-1),
        axis=-1,
    )
    return rank_lookup()[RANK_INDEX[CARD_INDEX, cards].sum(axis=-1)].max(axis=(1, 2))


def turn_context(
    obs: Observation, equity: float, opener_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build visible features and sample compatible opposing hands and rivers."""
    unseen = np.array(sorted(set(DECK).difference(obs.hole_cards, obs.board)), dtype=np.int8)
    rng = np.random.default_rng((*obs.hole_cards, *obs.board))
    random_keys = rng.random((TURN_SAMPLES, len(unseen)))
    worlds = unseen[np.argpartition(random_keys, 5, axis=1)[:, :6]]
    hands = worlds[:, :5]
    turn = np.broadcast_to(obs.board, (TURN_SAMPLES, 4))
    rivers = np.column_stack((turn, worlds[:, 5]))
    hero = np.broadcast_to(obs.hole_cards, (TURN_SAMPLES, 5))
    opponent_made = _best(hands, turn)
    ordered = np.sort(opponent_made)
    lower = np.searchsorted(ordered, opponent_made, side="left")
    upper = np.searchsorted(ordered, opponent_made, side="right")
    strength = (lower + upper) / (2 * TURN_SAMPLES)

    raises = {Action.RAISE_HALF, Action.RAISE_POT}
    prior_bets = sum(
        not event.is_me and event.street == Street.FLOP and event.action in raises
        for event in obs.action_history
    )
    price = min(obs.to_call, obs.my_stack)
    pot_odds = price / (obs.pot + price)
    pressure = min(1.0, 1.6 * pot_odds + 0.14 * prior_bets + 0.12 * max(0, obs.raises_this_street - 1))
    value_mix = pressure * (1 - 0.70 * opener_rate)
    value_weights = strength ** (2 + 6 * pressure)
    weights = (1 - value_mix) / TURN_SAMPLES + value_mix * value_weights / value_weights.sum()

    ours, theirs = _best(hero, rivers), _best(hands, rivers)
    share = (ours > theirs) + 0.5 * (ours == theirs)
    made = int(_best(hero[:1], turn[:1])[0])
    made_category, top_rank = made >> 20, (made >> 16) & 15
    stack = min(obs.my_stack, obs.opp_stack)
    half_cost = obs.raise_to.get(Action.RAISE_HALF, obs.my_bet) - obs.my_bet
    pot_cost = obs.raise_to.get(Action.RAISE_POT, obs.my_bet) - obs.my_bet
    features = np.asarray([
        equity, share.mean(), weights @ share, made_category / 8, top_rank / 8,
        pot_odds, obs.pot / 80, stack / 80, float(obs.is_button),
        min(1.0, prior_bets / 3), obs.raises_this_street / 4, opener_rate,
        half_cost / obs.pot, pot_cost / obs.pot, float(price > 0),
        float(Action.RAISE_HALF in obs.legal_actions),
    ], dtype=np.float32)
    return features, worlds, weights


class TurnValueModel:
    """Predict incremental chip value relative to folding, in units of the pot."""

    def __init__(self, path: Path = TURN_WEIGHTS_PATH) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.mean = data["mean"]
            self.scale = data["scale"]
            self.layers = tuple(
                tuple((data[f"{action.name}_w{i}"], data[f"{action.name}_b{i}"]) for i in range(3))
                for action in (Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)
            )
        rank_lookup()

    def action_values(self, features: np.ndarray, legal: tuple[Action, ...]) -> dict[Action, float]:
        inputs = (features - self.mean) / self.scale
        result = {}
        for action in legal:
            if action == Action.FOLD:
                result[action] = 0.0
                continue
            x = inputs
            *hidden, output = self.layers[int(action) - 1]
            for weights, bias in hidden:
                x = np.maximum(x @ weights + bias, 0)
            weights, bias = output
            result[action] = float((x @ weights + bias)[0])
        return result

    def choose_action(
        self, obs: Observation, equity: float, opener_rate: float, fallback: Action, margin: float,
    ) -> Action:
        features, _, _ = turn_context(obs, equity, opener_rate)
        values = self.action_values(features, obs.legal_actions)
        candidate = max(values, key=values.__getitem__)
        return candidate if values[candidate] > values[fallback] + margin else fallback
