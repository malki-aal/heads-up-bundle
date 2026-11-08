"""Suit-invariant starting-hand features and offline-trained CPU inference."""

from pathlib import Path

import numpy as np


def starting_features(hands: np.ndarray) -> np.ndarray:
    hands = np.atleast_2d(hands)
    ranks = hands >> 2
    suits = hands & 3
    counts = (ranks[..., None] == np.arange(9)).sum(axis=1)
    suit_counts = (suits[..., None] == np.arange(4)).sum(axis=1)
    connections = np.column_stack([
        (counts[:, start:start + 5] > 0).sum(axis=1) for start in range(5)
    ] + [(counts[:, [8, 0, 1, 2, 3]] > 0).sum(axis=1)])
    suited = []
    for suit in range(4):
        ordered = np.sort(np.where(suits == suit, ranks + 1, 0), axis=1)
        pair = 10 * ordered[:, -1] + ordered[:, -2]
        suited.append(np.where(suit_counts[:, suit] >= 2, pair, 0))
    suited = np.sort(np.stack(suited, axis=1), axis=1)
    return np.column_stack((
        counts / 4, counts == 2, counts >= 3,
        np.sort(suit_counts, axis=1) / 5,
        np.sort(connections, axis=1) / 5,
        suited / 99,
    )).astype(np.float32)


class PreflopModel:
    def __init__(self, path: Path = Path(__file__).with_name('preflop_weights.npz')):
        with np.load(path, allow_pickle=False) as data:
            self.mean = data['mean']
            self.scale = data['scale']
            self.raise_thresholds = data['raise_thresholds']
            self.layers = tuple((data[f'w{i}'], data[f'b{i}']) for i in range(3))

    def estimate(self, hole: tuple[int, ...]) -> float:
        return float(self.estimate_many(np.asarray(hole))[0])

    def estimate_many(self, hands: np.ndarray) -> np.ndarray:
        values = (starting_features(hands) - self.mean) / self.scale
        for weights, bias in self.layers[:-1]:
            values = np.maximum(values @ weights + bias, 0)
        weights, bias = self.layers[-1]
        return np.clip((values @ weights + bias)[..., 0], 0.05, 0.95)
