"""Batch legal Omaha showdowns using scores supplied by the organizer."""

from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np

from equity import DECK
from turn_policy import RANK_INDEX, rank_lookup

PAIRS = np.array(list(combinations(range(5), 2)))
TRIPLES = {size: np.array(list(combinations(range(size), 3))) for size in (3, 4, 5)}


def combination_ids(cards: np.ndarray) -> np.ndarray:
    ordered = np.sort(cards, axis=-1)
    return RANK_INDEX[np.arange(cards.shape[-1]), ordered].sum(axis=-1)


@lru_cache(maxsize=1)
def score_values() -> np.ndarray:
    values = np.unique(np.append(rank_lookup(), np.int32(0)))
    assert len(values) < 65536
    values.flags.writeable = False
    return values


@lru_cache(maxsize=1)
def prepare_scores() -> np.ndarray:
    """Build a two-hole/three-board score table during server startup."""
    hands = np.array(list(combinations(DECK, 5)), dtype=np.int8)
    values = np.searchsorted(score_values(), rank_lookup()[combination_ids(hands)]).astype(np.uint16)
    scores = np.zeros((comb(36, 2), comb(36, 3)), dtype=np.uint16)
    for pair in combinations(range(5), 2):
        triple = [index for index in range(5) if index not in pair]
        scores[combination_ids(hands[:, pair]), combination_ids(hands[:, triple])] = values
    scores.flags.writeable = False
    return scores


def showdown_scores(hands: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Evaluate each row with exactly two hole cards and three board cards."""
    pairs = combination_ids(hands[..., PAIRS])
    triples = combination_ids(boards[..., TRIPLES[boards.shape[-1]]])
    ranks = prepare_scores()[pairs[..., :, None], triples[..., None, :]].max(axis=(-2, -1))
    return score_values()[ranks]


def draw_worlds(hole: tuple[int, ...], board: tuple[int, ...], count: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw disjoint opposing hands and runouts from visible information."""
    unseen = np.array(sorted(set(DECK).difference(hole, board)), dtype=np.int8)
    order = rng.random((count, len(unseen))).argsort(axis=1)
    cards = unseen[order[:, :10 - len(board)]]
    boards = np.column_stack((np.broadcast_to(board, (count, len(board))), cards[:, 5:]))
    return cards[:, :5], boards.astype(np.int8)


def showdown_shares(hole: tuple[int, ...], opponents: np.ndarray,
                    boards: np.ndarray) -> np.ndarray:
    hero = showdown_scores(np.broadcast_to(hole, opponents.shape), boards)
    villain = showdown_scores(opponents, boards)
    return (hero > villain).astype(float) + 0.5 * (hero == villain)


@lru_cache(maxsize=1)
def river_combinations():
    hands = np.array(list(combinations(range(26), 5)), dtype=np.int8)
    pairs = np.array(list(combinations(range(26), 2)), dtype=np.int8)
    hands.flags.writeable = pairs.flags.writeable = False
    return hands, pairs


@lru_cache(maxsize=1)
def river_pair_offsets():
    hands, _ = river_combinations()
    wide = hands.astype(np.uint16)
    offsets = 26 * wide[:, PAIRS[:, 0]] + wide[:, PAIRS[:, 1]]
    offsets.flags.writeable = False
    return offsets


def exact_river_distribution(hole, board):
    """Count every compatible five-card hand, grouped by its showdown score."""
    hands, pairs = river_combinations()
    unseen = np.array(sorted(set(DECK).difference(hole, board)), dtype=np.int8)
    pair_ids = combination_ids(unseen[pairs])
    triples = combination_ids(np.asarray(board)[TRIPLES[5]])
    pair_scores = prepare_scores()[pair_ids[:, None], triples[None, :]].max(axis=1)
    matrix = np.zeros((26, 26), dtype=np.uint16)
    matrix[pairs[:, 0], pairs[:, 1]] = pair_scores
    opponent = matrix.ravel()[river_pair_offsets()].max(axis=1)
    counts = np.bincount(opponent, minlength=len(score_values()))
    occupied = np.flatnonzero(counts)
    scores, frequency = score_values()[occupied], counts[occupied]
    hero = showdown_scores(np.asarray(hole), np.asarray(board))
    shares = (hero > scores).astype(float) + 0.5 * (hero == scores)
    strength = (frequency.cumsum() - 0.5 * frequency) / len(hands)
    for values in (shares, strength, frequency):
        values.flags.writeable = False
    return shares, strength, frequency
