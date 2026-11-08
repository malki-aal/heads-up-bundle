# Heads Up | EquityNet

EquityNet is a Short Deck PLO5 bot with a small neural network trained in PyTorch. The network estimates showdown equity from the flop onward. The preflop rules and the code that picks a legal action are shared with the rules-only comparison bot. The tournament server interface comes from the organizers' starter.

## Install in your fork

1. Fork the [official starter repo](https://github.com/tamu-datathon-org/heads-up-bundle) and clone your fork. Start from a clean copy of the starter on a branch in your team's repo.
2. Extract the ZIP into the **repo root**, next to `game.py`. This replaces the starter `agent.py` and `requirements.txt` and adds the model, training code, and benchmark. If you already modified either file, review and merge your changes before overwriting it.
3. Set `PARTICIPANT` in `agent.py` to your registered team name. The default mode is `ml`. The `equity_weights.npz` file must sit next to `equity.py`.
4. Run the following from your activated Python environment.

   ```bash
   python -m pip install -r requirements-dev.txt
   python -m pytest -q
   python agent.py
   ```

The server loads the included weights and uses NumPy to run predictions on CPU. PyTorch is needed only to retrain. The rules-only version is available through `HEADS_UP_MODE=rules`.

## Replay the comparison

Stop your manually started server if you no longer need it. The benchmark starts its own bots and restarts them between halves. It uses the official judge and fresh servers for both seats.

```bash
python benchmark_equity.py --seeds 461 967 --output benchmark_results.json
```

Each run compares ML and rules head-to-head and against the same three starter opponents. The default is 1,000 duplicate deals per match, or 2,000 hands. Set `--jobs 1` to run one match at a time on a slower machine. `--deals 200` is useful for quick local checks.

Before submitting, run `bash tools/check_build.sh` with Docker running. Commit the runtime files `agent.py`, `strategy.py`, `equity.py`, `equity_weights.npz`, and `requirements.txt`. Push to your team's public fork and use the full SHA of that commit on the submission form. The code was checked with the starter's tests and HTTP smoke test. The Docker image check still needs to be run on a machine with Docker.

## Retrain the model

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python train_equity.py
```

The included weights were trained on 45,000 simulated flop, turn, and river situations with 80 rollouts each. Each rollout uses the official evaluator on two hands with the same completed board. The target counts a win as 1, a tie as 0.5, and a loss as 0. Validation uses 3,000 separate simulated situations and 240 rollouts each. The training script selects the epoch with the lowest validation error and exports the weights. The model treats unseen opponent cards as equally likely. The betting policy adjusts its decisions for pot odds and raises already made, but it does not learn opponent ranges from betting history.

## Experimental turn action values

Set `HEADS_UP_TURN_MODEL=1` to try the turn model in ML mode. In PowerShell, run
`$env:HEADS_UP_TURN_MODEL = "1"` before starting the bot. This mode compares legal
actions when facing a turn bet. It keeps the current action unless the model
predicts a chip gain of more than 12% of the pot from switching. The default uses
the existing policy.

The included `turn_value_weights.npz` was trained on simulated turns by playing
out every legal action on the same sampled hands and rivers and rewarding
incremental chips. Runtime inference uses NumPy and visible information only.
To retrain offline, install scikit-learn and run `python train_turn_value.py`.

## Local benchmark results

The table reports the average big blinds won per duplicate deal across seeds `461` and `967`, with 1,000 duplicate deals in each match. The two versions used the same preflop and betting rules.

| Opponent | EquityNet | Rules-only bot |
| --- | ---: | ---: |
| Rules-only bot, head-to-head | +3.86 | — |
| Random | +7.28 | +7.35 |
| Calling station | +6.81 | +7.37 |
| Always pot | +7.98 | +4.96 |

EquityNet won both head-to-head matches (+3.93 and +3.79 bb/deal) and all six matches against the starter opponents. Both versions won all six starter-opponent matches. EquityNet gained the most against the always-pot opponent. The rules version averaged slightly more against the random opponent and more against the calling station. All 14 matches completed with zero errors, forfeits, or exhausted time banks.

On a separate set of 1,500 simulated situations with 300 rollouts each, the model's equity RMSE was 5.5 percentage points versus 19.4 for the rules estimate. That measures accuracy against randomly sampled hands. The match results cover the included opponents and two seeds. The competition field may play differently.
