"""Card features and CPU inference for a learned showdown equity model."""

from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

from evaluator import get_table

WEIGHTS_PATH = Path(__file__).with_name("equity_weights.npz")
DECK = tuple(range(36))
CARD_MASKS = tuple(1 << card for card in DECK)
STRAIGHTS = tuple(frozenset(range(start, start + 5)) for start in range(5)) + (
    frozenset((8, 0, 1, 2, 3)),
)


def made_score(hole: Sequence[int], board: Sequence[int]) -> int:
    """Best legal hand using the currently visible board (at least 3 cards)."""
    table = get_table()
    pairs = [a | b for a, b in combinations((CARD_MASKS[c] for c in hole), 2)]
    triples = [a | b | c for a, b, c in combinations((CARD_MASKS[c] for c in board), 3)]
    return max(table[pair | triple] for pair in pairs for triple in triples)


def card_counts(cards: Sequence[int]) -> tuple[list[int], list[int]]:
    ranks = [0] * 9
    suits = [0] * 4
    for card in cards:
        ranks[card >> 2] += 1
        suits[card & 3] += 1
    return ranks, suits


def equity_features(hole: Sequence[int], board: Sequence[int]) -> np.ndarray:
    """Describe a flop, turn, or river using only cards the bot can see."""
    hole_ranks, hole_suits = card_counts(hole)
    board_ranks, board_suits = card_counts(board)
    board_rank_set = {rank for rank, count in enumerate(board_ranks) if count}
    hole_rank_set = {rank for rank, count in enumerate(hole_ranks) if count}
    score = made_score(hole, board)
    category = score >> 20

    # A straight needs 2 distinct hole ranks and 3 distinct board ranks.
    straight_shapes = [0] * 4
    for ranks in STRAIGHTS:
        for first, second in combinations(hole_rank_set & ranks, 2):
            matched = len((board_rank_set & ranks) - {first, second})
            straight_shapes[matched] += 1

    suit_profiles = sorted(zip(board_suits, hole_suits), reverse=True)
    flush_blockers = sorted(
        (
            board_suits[suit],
            hole_suits[suit],
            max((card >> 2 for card in hole if card & 3 == suit), default=-1),
        )
        for suit in range(4)
    )
    values = [
        float(len(board) == 4),
        float(len(board) == 5),
        *(count / 4 for count in hole_ranks + board_ranks),
        *(float(category == i) for i in range(9)),
        *((score >> shift & 15) / 15 for shift in (16, 12, 8, 4, 0)),
        *(count / 15 for count in straight_shapes),
        *(count / 5 for profile in suit_profiles for count in profile),
        *((rank + 1) / 9 for _, _, rank in flush_blockers),
    ]
    return np.asarray(values, dtype=np.float32)


class EquityModel:
    """Run the exported ReLU network using NumPy and safe array-only weights."""

    def __init__(self, path: Path = WEIGHTS_PATH) -> None:
        with np.load(path, allow_pickle=False) as arrays:
            self.layers = tuple((arrays[f"w{i}"], arrays[f"b{i}"]) for i in range(3))
        for weights, bias in self.layers:
            weights.flags.writeable = False
            bias.flags.writeable = False

    def estimate(self, hole: Sequence[int], board: Sequence[int]) -> float:
        values = equity_features(hole, board)
        for weights, bias in self.layers[:-1]:
            values = np.maximum(values @ weights + bias, 0)
        weights, bias = self.layers[-1]
        logit = float((values @ weights + bias)[0])
        return float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))
