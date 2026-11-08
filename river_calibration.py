"""Calibrate river bet ranges using public cards, prices and action history."""
from pathlib import Path
import numpy as np
from equity import river_distribution
from game import Action, Street


def calibration_features(obs, memory, estimate):
    shares, _, frequency = river_distribution(obs.hole_cards, obs.board)
    uniform = float(np.average(shares, weights=frequency))
    def logit(value):
        value = np.clip(value, .001, .999)
        return np.log(value / (1 - value))
    raises = (Action.RAISE_HALF, Action.RAISE_POT)
    prior = [e for e in obs.action_history if not e.is_me]
    counts = memory.get('opponent_openers', {}).get(Street.RIVER, {})
    seen = counts.get('seen', 0)
    price = min(obs.my_stack, obs.to_call)
    return np.asarray([
        logit(estimate), logit(uniform), price / (obs.pot + price), float(obs.is_button),
        sum(e.street == Street.FLOP and e.action in raises for e in prior),
        sum(e.street == Street.TURN and e.action in raises for e in prior),
        obs.raises_this_street, (counts.get('bets', 0) + 1) / (seen + 2), min(1, seen / 32),
        np.log1p(obs.pot), min(obs.my_stack, obs.opp_stack) / 80,
        sum(e.street == Street.TURN and e.action == Action.CHECK_CALL and e.amount == 0 for e in prior),
    ], dtype=float)


class RiverCalibration:
    def __init__(self, path=None):
        path = path or Path(__file__).with_name('river_calibration_weights.npz')
        with np.load(path, allow_pickle=False) as data:
            self.mean, self.scale = data['mean'], data['scale']
            self.weights, self.bias = data['weights'], float(data['bias'])

    def estimate(self, obs, memory, estimate):
        shares, _, _ = river_distribution(obs.hole_cards, obs.board)
        if np.all(shares == shares[0]):
            return float(shares[0])
        inputs = (calibration_features(obs, memory, estimate) - self.mean) / self.scale
        logit = float(inputs @ self.weights + self.bias)
        return float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))
