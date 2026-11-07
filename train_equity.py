"""Train a small neural network on simulated Short Deck PLO5 showdowns."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path
import random

import numpy as np

from equity import DECK, WEIGHTS_PATH, equity_features
from evaluator import omaha_score, warmup


def simulate_state(seed: int, board_size: int, rollouts: int) -> tuple[np.ndarray, float]:
    rng = random.Random(seed)
    known = rng.sample(DECK, 5 + board_size)
    hole, board = known[:5], known[5:]
    remaining = [card for card in DECK if card not in known]
    cards_to_come = 5 - board_size
    points = 0.0
    hero_score = omaha_score(hole, board) if not cards_to_come else None

    for _ in range(rollouts):
        runout = rng.sample(remaining, cards_to_come + 5)
        full_board = board + runout[:cards_to_come]
        hero = hero_score if hero_score is not None else omaha_score(hole, full_board)
        opponent = omaha_score(runout[cards_to_come:], full_board)
        points += 1.0 if hero > opponent else 0.5 if hero == opponent else 0.0

    return equity_features(hole, board), points / rollouts


def generate(executor, start: int, count: int, rollouts: int) -> tuple[np.ndarray, np.ndarray]:
    seeds = range(start, start + count)
    boards = (3 + seed % 3 for seed in seeds)
    simulate = partial(simulate_state, rollouts=rollouts)
    samples = list(executor.map(simulate, seeds, boards, chunksize=96))
    features, labels = zip(*samples)
    return np.stack(features), np.asarray(labels, dtype=np.float32)


def train(args: argparse.Namespace) -> None:
    # Import PyTorch after the simulation workers have exited.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=warmup) as executor:
        x_train, y_train = generate(executor, args.seed, args.states, args.rollouts)
        print(f"Generated {len(y_train):,} training states", flush=True)
        x_valid, y_valid = generate(executor, args.seed + args.states, args.validation, args.validation_rollouts)
        print(f"Generated {len(y_valid):,} validation states", flush=True)

    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_valid, y_valid = torch.from_numpy(x_valid), torch.from_numpy(y_valid)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    loss_fn = nn.BCEWithLogitsLoss()
    best_error = float("inf")
    best_epoch = 0
    best_arrays = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in torch.randperm(len(y_train)).split(256):
            loss = loss_fn(model(x_train[batch]).squeeze(1), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x_valid).squeeze(1).sigmoid()
            rmse = ((prediction - y_valid).square().mean().sqrt()).item()
        if rmse < best_error:
            best_error, best_epoch = rmse, epoch
            best_arrays = {}
            for i, layer in enumerate((model[0], model[2], model[4])):
                best_arrays[f"w{i}"] = layer.weight.detach().numpy().T.copy()
                best_arrays[f"b{i}"] = layer.bias.detach().numpy().copy()

    np.savez_compressed(args.output, **best_arrays)
    constant_rmse = (y_valid.sub(0.5).square().mean().sqrt()).item()
    print(f"Epoch {best_epoch}: validation RMSE {best_error:.4f}; constant 50% RMSE {constant_rmse:.4f}")
    print(f"Saved {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=45000)
    parser.add_argument("--rollouts", type=int, default=80)
    parser.add_argument("--validation", type=int, default=3000)
    parser.add_argument("--validation-rollouts", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument("--output", type=Path, default=WEIGHTS_PATH)
    args = parser.parse_args()
    if min(args.states, args.rollouts, args.validation, args.validation_rollouts, args.epochs, args.workers) < 1:
        parser.error("counts, rollouts, epochs and workers must be positive")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    train(args)


if __name__ == "__main__":
    main()
