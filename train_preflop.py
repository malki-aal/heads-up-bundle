"""Train starting-hand equity offline on independently sampled legal showdowns."""

import argparse
from pathlib import Path

import numpy as np

from preflop import starting_features
from sampling import prepare_scores, showdown_scores
from strategy import preflop_equity


def generate(count, rollouts, seed):
    rng = np.random.default_rng(seed)
    hands = rng.random((count, 36)).argsort(axis=1)[:, :5].astype(np.int8)
    labels = []
    for start in range(0, count, 32):
        batch = hands[start:start + 32]
        available = np.arange(36)[None, :] != batch[:, :, None]
        available = np.broadcast_to(np.arange(36), (len(batch), 36))[available.all(axis=1)].reshape(len(batch), 31)
        positions = rng.random((len(batch), rollouts, 31)).argsort(axis=-1)[..., :10]
        worlds = np.take_along_axis(available[:, None, :], positions, axis=-1)
        opponents, boards = worlds[..., :5], worlds[..., 5:]
        hero = showdown_scores(np.broadcast_to(batch[:, None, :], opponents.shape), boards)
        villain = showdown_scores(opponents, boards)
        labels.extend(((hero > villain) + 0.5 * (hero == villain)).mean(axis=1))
    return hands, np.asarray(labels, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=int, default=18000)
    parser.add_argument('--rollouts', type=int, default=192)
    parser.add_argument('--validation', type=int, default=1500)
    parser.add_argument('--validation-rollouts', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('preflop_weights.npz'))
    args = parser.parse_args()
    prepare_scores()
    hands, target = generate(args.states, args.rollouts, 81037)
    print(f'Generated {len(hands)} training hands', flush=True)
    valid_hands, valid_target = generate(args.validation, args.validation_rollouts, 91081)
    features, valid_features = starting_features(hands), starting_features(valid_hands)
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 0.02)
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(48, 24), alpha=0.02, max_iter=180,
                         early_stopping=True, n_iter_no_change=15, random_state=7717)
    model.fit((features - mean) / scale, target)
    prediction = model.predict((valid_features - mean) / scale)
    baseline = np.array([preflop_equity(tuple(map(int, hand))) for hand in valid_hands])
    arrays = {'mean': mean, 'scale': scale}
    train_predictions = model.predict((features - mean) / scale)
    old_scores = np.array([preflop_equity(tuple(map(int, hand))) for hand in hands])
    arrays['raise_thresholds'] = np.quantile(train_predictions, [np.mean(old_scores < 0.56), np.mean(old_scores < 0.61)])
    for index, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        arrays[f'w{index}'] = weights.astype(np.float32)
        arrays[f'b{index}'] = bias.astype(np.float32)
    np.savez_compressed(args.output, **arrays)
    print(f'Validation RMSE: model {np.mean((prediction-valid_target)**2)**0.5:.4f}, original {np.mean((baseline-valid_target)**2)**0.5:.4f}', flush=True)


if __name__ == '__main__':
    main()
