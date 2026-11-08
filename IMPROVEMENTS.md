# EquityNet improvements

The starting point is `malki-aal/heads-up-bundle` commit
`21b22f6215fee270033b10df1e9e1a84e722e133`.

## Strategy changes

- Calls that end betting use break-even pot odds. Future-betting cushions apply
  only when chips remain behind and another betting street can occur. This
  removes a second penalty for aggression already included in river ranges.
- The bot counts observed opponent opening checks and bets separately on the
  flop, turn and river. Repeated observations of the same hand are counted once.
- A smoothed betting frequency and sample-size weight control how strongly a
  river bet implies a value-heavy hand range. Once there are 16 river samples,
  the river frequency takes precedence over the flop frequency.
- All learned counts live in `memory` and reset for each half. Calculations use
  visible cards and public actions. The existing evaluator supplies the exact
  short-deck rankings and two-hole/three-board Omaha construction.

## Open-source research

These projects informed the approach and evaluation. The implementation changes
are original Python code using the existing NumPy dependency.

| Project | License | Relevant ideas | Fit to this tournament |
| --- | --- | --- | --- |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | Apache-2.0 | Imperfect-information policies, action values, CFR and evaluation | Useful research reference. Its Hold'em examples require a custom game model for this short-deck PLO5 variant. |
| [RLCard](https://github.com/datamllab/rlcard) | MIT | Card-game agents, small-game CFR, reproducible evaluation | Useful agent and evaluation reference. Its Hold'em policies use different card and action assumptions. |
| [PokerRL](https://github.com/EricSteinberger/PokerRL) | MIT | Head-to-head policy evaluation, opponent exploitation and best-response evaluation | Motivates evaluating against several styles and the original bot. Full training frameworks are an offline extension. |
| [PokerKit](https://github.com/uoftcprg/pokerkit) | MIT | Explicit game variants and hand construction | Useful rules reference. This bot retains the organizer's evaluator to match the supplied ranking and legal actions. |

The runtime remains CPU-only, with the existing server endpoints, dependencies,
weights and port handling. Opponent identification is based on public actions.

## Reproduce evaluation

Run the original checkout as a reference using fresh servers for both halves:

```bash
git worktree add --detach ../heads-up-original 21b22f6215fee270033b10df1e9e1a84e722e133
python -m pytest -q
python benchmark_equity.py --reference ../heads-up-original --seeds 2719 3907 --deals 1000 --jobs 1 --output benchmark_results_improved.json
```

The benchmark includes direct play against the original and matched games
against all four supplied opponents, including `always_half`. The output records
the official judge's confidence intervals, errors, time usage and verdicts.

Use independent seeds for further validation. Results on these opponents do not
establish an equilibrium strategy or a guaranteed tournament result.

Run `bash tools/check_build.sh` with Docker available before submission.
