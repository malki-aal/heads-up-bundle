"""Train the turn policy offline from counterfactual rewards on simulated hands."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from functools import lru_cache
from pathlib import Path

import numpy as np

from agent import MODEL, _choose_action
from game import Action, Deal, Game, Street, button_for_deal, make_deal
from strategy import preflop_equity, select_action
from turn_policy import TURN_WEIGHTS_PATH, rank_lookup, turn_context

ACTIONS = (Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)
PROFILES = ("regular", "pressure", "station", "pot", "regular", "pressure")


@lru_cache(maxsize=4096)
def estimate(hole, board):
    return MODEL.estimate(hole, board) if board else preflop_equity(hole)


def rollout_policy(obs, profile):
    """Play the rest of a hand using one of several visible-state policies."""
    if profile == "station":
        return Action.CHECK_CALL
    if profile == "pot":
        return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL

    equity = estimate(obs.hole_cards, obs.board)
    if obs.street == Street.RIVER and obs.to_call > 0:
        price = min(obs.to_call, obs.my_stack)
        equity -= (1 - equity) * (0.24 + price / (obs.pot + price))

    if profile == "pressure":
        return select_action(obs, equity, raise_margin=-0.10, call_margin=0.06)
    if obs.street in (Street.FLOP, Street.TURN):
        margin = -0.10 if obs.street == Street.FLOP else -0.08
        return select_action(obs, equity, raise_margin=margin, call_margin=0.10)
    return select_action(obs, equity)


def rewards(game, obs, worlds, weights, profile, rng, rollouts):
    hero = game.to_act
    legal = [action for action in ACTIONS if action in obs.legal_actions]
    accumulated = np.zeros(len(ACTIONS))
    for index in rng.choice(len(worlds), rollouts, p=weights):
        world = worlds[index]
        holes = [obs.hole_cards, obs.hole_cards]
        holes[1 - hero] = tuple(sorted(map(int, world[:5])))
        deal = Deal(tuple(holes), (*obs.board, int(world[5])))
        for action in legal:
            branch = copy(game)
            branch.deal = deal
            for field in ("stacks", "contributed", "bets", "acted", "history"):
                setattr(branch, field, getattr(game, field).copy())
            branch.apply(action)
            while not branch.is_terminal:
                seat = branch.to_act
                actor_profile = "regular" if seat == hero else profile
                branch.apply(rollout_policy(branch.observation(seat), actor_profile))
            # Folding would lose the chips already committed, a constant across actions.
            accumulated[int(action) - 1] += branch.result().nets[hero] + game.contributed[hero]
    target = np.full(len(ACTIONS), np.nan)
    for action in legal:
        index = int(action) - 1
        target[index] = accumulated[index] / (rollouts * obs.pot)
    return target


def generate(task):
    seed, batch, deals, rollouts = task
    profile = PROFILES[batch % len(PROFILES)]
    rng = np.random.default_rng([seed, batch])
    memory = {}
    features, targets = [], []
    for offset in range(deals):
        index = batch * deals + offset
        hero = offset % 2
        game = Game(make_deal(seed, index), button_for_deal(index // 2), index)
        while not game.is_terminal and game.street <= Street.TURN:
            seat = game.to_act
            obs = game.observation(seat)
            if seat != hero:
                game.apply(rollout_policy(obs, profile))
                continue
            if obs.street == Street.TURN:
                seen = memory.get("flop_openers", 0)
                bets = memory.get("flop_open_bets", 0)
                rate = (bets + 2) / (seen + 5)
                row, worlds, weights = turn_context(obs, estimate(obs.hole_cards, obs.board), rate)
                features.append(row)
                targets.append(rewards(game, obs, worlds, weights, profile, rng, rollouts))
            game.apply(_choose_action(obs, memory, use_turn_model=False))
    return features, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--deals-per-batch", type=int, default=112)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--seed", type=int, default=48203871)
    parser.add_argument("--output", type=Path, default=TURN_WEIGHTS_PATH)
    args = parser.parse_args()
    if MODEL is None:
        parser.error("training needs the ML equity model")
    if min(args.batches, args.deals_per_batch, args.rollouts, args.workers) < 1:
        parser.error("batch, deal, rollout, and worker counts must be positive")
    if args.output.suffix != ".npz":
        parser.error("output must end in .npz")

    rank_lookup()
    tasks = [(args.seed, i, args.deals_per_batch, args.rollouts) for i in range(args.batches)]
    features, targets = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch, (rows, values) in enumerate(pool.map(generate, tasks), start=1):
            features.extend(rows)
            targets.extend(values)
            if batch % 8 == 0:
                print(f"Batches {batch}/{args.batches}, turn spots {len(features)}", flush=True)

    # scikit-learn is needed only to train. The bot loads NumPy arrays at runtime.
    from sklearn.neural_network import MLPRegressor

    x, y = np.asarray(features), np.asarray(targets)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.01)
    x = (x - mean) / scale
    arrays = {"mean": mean, "scale": scale}
    for action, labels in zip(ACTIONS, y.T):
        valid = np.isfinite(labels)
        model = MLPRegressor(
            hidden_layer_sizes=(48, 24), alpha=0.005, max_iter=140, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=12, random_state=args.seed,
        )
        model.fit(x[valid], labels[valid])
        for i, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
            arrays[f"{action.name}_w{i}"] = weights.astype(np.float32)
            arrays[f"{action.name}_b{i}"] = bias.astype(np.float32)
        print(f"{action.name}: {valid.sum()} examples", flush=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
