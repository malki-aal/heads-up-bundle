"""
Tests for YOUR bot (agent.py). Run them from the bundle folder:

    python -m pytest -q

They talk to your bot the same way the judge does (through its web routes,
but without starting a real server), so a passing run means the judge can
play with your bot. Add your own tests at the bottom!
"""

import importlib.util
import os
import random
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Works whether this file sits next to agent.py or inside a tests/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = HERE if os.path.isfile(os.path.join(HERE, "game.py")) else os.path.dirname(HERE)
sys.path.insert(0, BUNDLE)  # so agent.py can import game.py and evaluator.py

from evaluator import cards_from_str, omaha_score  # noqa: E402
from game import Action, AgentConfig, Deal, Game, Street, make_deal  # noqa: E402
from judge_engine import coerce_action  # noqa: E402  (reads answers exactly like the judge)
from river_betting import finish_opening, opening_values, record_opening  # noqa: E402
from turn_policy import TurnValueModel, _best, turn_context  # noqa: E402

TOKEN = "test-token"
AUTH = {"X-Agent-Token": TOKEN}


def load_agent():
    """Load a fresh copy of agent.py (its own MEMORY and globals)."""
    spec = importlib.util.spec_from_file_location("agent_under_test", os.path.join(BUNDLE, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot():
    """(agent module, test client) with the tournament's token check switched on."""
    module = load_agent()
    module.TOKEN = TOKEN
    return module, TestClient(module.app)


def start_half(client, half=1, seat=0):
    """POST /new-half the way the judge does."""
    body = {"half": half, "seat": seat, "config": AgentConfig().to_dict()}
    return client.post("/new-half", json=body, headers=AUTH)


def play_hands(client, num_hands, seed="test", raise_rate=0.3):
    """Play hands where YOUR bot is seat 0 and a random opponent is seat 1.

    Every decision goes through /act exactly like in a real match, and
    finished hands are delivered with the next /act (like the judge does).
    Returns the list of every action your bot chose.
    """
    rng = random.Random(seed)
    pending = []      # finished hands waiting to be delivered
    chosen = []
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200, f"/act failed: {r.json()}"
                reply = r.json()
                assert reply.get("hand_end_errors", []) == [], \
                    f"on_hand_end crashed: {reply['hand_end_errors'][0]['traceback']}"
                action = coerce_action(reply["action"])
                assert action in obs.legal_actions, \
                    f"choose_action returned {reply['action']!r}; legal were {[a.name for a in obs.legal_actions]}"
                chosen.append(action)
            else:
                # Random opponent that raises sometimes, so your bot faces bets.
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)  # end of half
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    return chosen


# ---------------------------------------------------------------------------
# Your bot plays correctly
# ---------------------------------------------------------------------------

def test_names_are_set(bot):
    _, client = bot
    info = client.get("/").json()
    assert isinstance(info["participant"], str) and info["participant"]
    assert isinstance(info["agent_name"], str) and info["agent_name"]


def test_new_half_works(bot):
    _, client = bot
    r = start_half(client)
    assert r.status_code == 200, f"new_half crashed: {r.json()}"


def test_every_decision_is_legal(bot):
    # 200 hands with a random opponent: every choose_action answer must be legal.
    module, client = bot
    start_half(client)
    play_hands(client, 200)
    assert module.MEMORY.get("fallbacks", 0) == 0


def test_works_in_both_seats_and_both_halves(bot):
    _, client = bot
    for half, seat in ((1, 0), (2, 1)):
        assert start_half(client, half, seat).status_code == 200
        play_hands(client, 30, seed=f"half{half}")


def test_handles_aggressive_opponent(bot):
    # An opponent that raises whenever it can: your bot must cope with
    # big bets and all-ins.
    _, client = bot
    start_half(client)
    play_hands(client, 100, seed="aggressive", raise_rate=1.0)


def test_memory_is_fresh_each_half(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 10)
    start_half(client, half=2, seat=1)
    after_reset = dict(module.MEMORY)
    fresh = load_agent()
    fresh.new_half(AgentConfig(), fresh.MEMORY)
    assert after_reset.keys() == fresh.MEMORY.keys(), "memory should only hold what new_half() puts there"


# ---------------------------------------------------------------------------
# The server plumbing still works (only fails if you edited it)
# ---------------------------------------------------------------------------

def test_token_is_checked(bot):
    _, client = bot
    body = {"half": 1, "seat": 0, "config": {}}
    assert client.post("/new-half", json=body).status_code == 401
    assert client.post("/new-half", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_bad_token_is_refused_before_the_body_is_read(bot):
    # Without the right token the bot must answer 401 without parsing the body,
    # so junk sent by anyone else can't slow your bot down.
    _, client = bot
    junk = b"this is not JSON" * 1000
    assert client.post("/act", content=junk).status_code == 401
    assert client.post("/act", content=junk, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_malformed_body_is_rejected(bot):
    _, client = bot
    start_half(client)
    assert client.post("/act", content=b"not json", headers=AUTH).status_code == 400
    assert client.post("/act", json=[1, 2, 3], headers=AUTH).status_code == 400


def test_act_before_new_half_is_refused(bot):
    _, client = bot
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    assert client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH).status_code == 409


def test_crashes_are_reported_to_the_judge(bot):
    # If choose_action raises, the judge must get a 500 with your traceback.
    module, client = bot

    def broken(obs, memory):
        raise ValueError("oops")

    module.choose_action = broken
    start_half(client)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH)
    assert r.status_code == 500
    assert "ValueError: oops" in r.json()["error"] and "Traceback" in r.json()["traceback"]


def test_reads_port_from_environment():
    source = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert 'os.environ.get("PORT"' in source, "agent.py must read its port from $PORT"
    assert '0.0.0.0' in source, "agent.py must listen on 0.0.0.0 so the judge can reach it"


# ---------------------------------------------------------------------------
# Add your own tests below, e.g. "with two aces preflop my bot raises".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hole", "other_hole", "board", "expected"),
    [
        ("8h 8s 6c Ad Qd", "6h 7d 8d Tc Qs", "8c 9d Js Kc Ac", Action.FOLD),
        ("Ah Kh 6c 7d 8s", "7h 8h 9c Td Ad", "Qh Jh Th 6s 7c", Action.RAISE_POT),
    ],
)
def test_equitynet_responds_to_river_pot_bet(
    monkeypatch, hole, other_hole, board, expected,
):
    monkeypatch.setenv("HEADS_UP_MODE", "ml")
    module = load_agent()
    module.TOKEN = TOKEN
    client = TestClient(module.app)
    assert start_half(client).status_code == 200
    deal = Deal(
        (tuple(cards_from_str(hole)), tuple(cards_from_str(other_hole))),
        tuple(cards_from_str(board)),
    )
    game = Game(deal, button_seat=0)
    while game.street < Street.RIVER:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    reply = client.post(
        "/act",
        json={"observation": game.observation(0).to_dict(), "hand_results": []},
        headers=AUTH,
    )
    assert reply.status_code == 200
    assert coerce_action(reply.json()["action"]) == expected


@pytest.mark.parametrize('street', [Street.FLOP, Street.TURN])
def test_turn_sampler_excludes_visible_cards_and_matches_judge_scores(street):
    deal = make_deal("turn-worlds", 0)
    game = Game(deal, button_seat=0)
    while game.street < street:
        game.apply(Action.CHECK_CALL)
    obs = game.observation(game.to_act)
    features, worlds, weights = turn_context(obs, equity=0.5, opener_rate=0.4)
    visible = set(obs.hole_cards + obs.board)
    assert all(len(set(world)) == 10 - len(obs.board) and not visible.intersection(world) for world in worlds)
    assert weights.sum() == pytest.approx(1.0)

    boards = np.column_stack((np.broadcast_to(obs.board, (len(worlds), len(obs.board))), worlds[:, 5:]))
    scores = _best(worlds[:, :5], boards)
    assert scores.tolist() == [
        omaha_score(tuple(map(int, world[:5])), tuple(map(int, board)))
        for world, board in zip(worlds, boards)
    ]
    values = TurnValueModel().action_values(features, obs.legal_actions)
    assert set(values) == set(obs.legal_actions)
    assert np.isfinite(list(values.values())).all()


@pytest.mark.parametrize(
    ("response", "label"),
    [(Action.FOLD, "fold"), (Action.CHECK_CALL, "call"), (Action.RAISE_POT, "raise")],
)
def test_river_opening_tracks_each_public_response_once(response, label):
    game = Game(make_deal("river-opener", 0), button_seat=0)
    while game.street < Street.RIVER:
        game.apply(Action.CHECK_CALL)
    hero = game.to_act
    memory = {}
    record_opening(game.observation(hero), Action.RAISE_HALF, memory)
    game.apply(Action.RAISE_HALF)
    game.apply(response)
    if response == Action.RAISE_POT:
        record_opening(game.observation(hero), Action.FOLD, memory)
        game.apply(Action.FOLD)
    result = game.hand_result(hero)
    finish_opening(result, memory)
    finish_opening(result, memory)
    assert memory == {"river_responses": {0: {label: 1}}}


def test_river_betting_values_both_sizes_with_the_nuts():
    deal = Deal(
        (tuple(cards_from_str("Ah Kh 6c 7d 8s")), tuple(cards_from_str("7h 8h 9c Td Ad"))),
        tuple(cards_from_str("Qh Jh Th 6s 7c")),
    )
    game = Game(deal, button_seat=1)
    while game.street < Street.RIVER:
        game.apply(Action.CHECK_CALL)
    obs = game.observation(0)
    memory = {"river_responses": {0: {"call": 40}, 1: {"call": 40}}}
    values = opening_values(obs, memory)
    assert set(values) == set(obs.legal_actions)
    assert values[Action.RAISE_POT] > values[Action.RAISE_HALF] > values[Action.CHECK_CALL]


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN, Street.RIVER])
def test_all_in_call_uses_break_even_pot_odds(street):
    from dataclasses import replace
    from strategy import select_action

    game = Game(make_deal("call-price", 0), button_seat=0)
    while game.street < street:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    obs = game.observation(game.to_act)
    obs = replace(
        obs, my_stack=obs.to_call, raises_this_street=2,
        legal_actions=(Action.FOLD, Action.CHECK_CALL), raise_to={},
    )
    break_even = obs.to_call / (obs.pot + obs.to_call)
    assert select_action(obs, break_even + 0.01, call_margin=0.10) == Action.CHECK_CALL
    assert select_action(obs, break_even - 0.01, call_margin=0.10) == Action.FOLD


def test_river_call_does_not_add_a_future_betting_cushion():
    from dataclasses import replace
    from strategy import select_action

    game = Game(make_deal("river-price", 0), button_seat=0)
    while game.street < Street.RIVER:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    obs = replace(game.observation(game.to_act), raises_this_street=2)
    break_even = obs.to_call / (obs.pot + obs.to_call)
    assert select_action(obs, break_even + 0.01) == Action.CHECK_CALL


def test_opponent_openers_are_counted_once_across_public_history():
    module = load_agent()
    game = Game(make_deal("openers", 0), button_seat=0)
    while game.street < Street.RIVER:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    obs = game.observation(0)
    memory = {}
    module.record_opponent_openers(obs, memory)
    module.record_opponent_openers(obs, memory)
    counts = memory["opponent_openers"]
    assert [counts[street]["seen"] for street in (Street.FLOP, Street.TURN, Street.RIVER)] == [1, 1, 1]
    assert [counts[street]["bets"] for street in (Street.FLOP, Street.TURN, Street.RIVER)] == [0, 0, 1]
    assert memory["flop_openers"] == 1
    assert memory["flop_open_bets"] == 0


def test_range_learning_respects_sample_size_and_street():
    from strategy import opponent_range_scale

    assert opponent_range_scale({}) == 1.0
    sparse = {"opponent_openers": {Street.FLOP: {"seen": 4, "bets": 4}}}
    assert 0.75 <= opponent_range_scale(sparse) <= 1.0
    frequent = {"opponent_openers": {Street.FLOP: {"seen": 32, "bets": 32}}}
    assert opponent_range_scale(frequent) == 0.0
    # Selective river betting overrides broad flop betting once enough is observed.
    frequent["opponent_openers"][Street.RIVER] = {"seen": 32, "bets": 4}
    assert opponent_range_scale(frequent) == 1.0


@pytest.mark.parametrize('board_size', [0, 3, 4, 5])
def test_batch_showdowns_exclude_known_cards_and_match_referee(board_size):
    from sampling import draw_worlds, showdown_scores, showdown_shares

    deal = make_deal('batch-check', board_size)
    hole, board = deal.hole_cards[0], deal.board[:board_size]
    opponents, boards = draw_worlds(hole, board, 32, np.random.default_rng(117))
    shares = showdown_shares(hole, opponents, boards)
    for index, (opponent, completed) in enumerate(zip(opponents, boards)):
        opponent, completed = tuple(map(int, opponent)), tuple(map(int, completed))
        assert len(set(hole + opponent + completed)) == 15
        assert completed[:board_size] == board
        hero_score = omaha_score(hole, completed)
        villain_score = omaha_score(opponent, completed)
        assert shares[index] == (1.0 if hero_score > villain_score else 0.5 if hero_score == villain_score else 0.0)
    assert showdown_scores(opponents, boards).tolist() == [
        omaha_score(tuple(map(int, hand)), tuple(map(int, completed)))
        for hand, completed in zip(opponents, boards)
    ]


def test_starting_model_is_invariant_to_card_order_and_suit_names():
    from preflop import PreflopModel

    model = PreflopModel()
    hand = (0, 5, 14, 27, 32)
    suits = (2, 0, 3, 1)
    renamed = tuple((card & ~3) | suits[card & 3] for card in hand)
    assert model.estimate(hand) == pytest.approx(model.estimate(hand[::-1]), abs=1e-6)
    assert model.estimate(hand) == pytest.approx(model.estimate(renamed), abs=1e-6)
    assert 0 < model.raise_thresholds[0] <= model.raise_thresholds[1] < 1


def test_planner_values_larger_bets_with_an_unbeatable_hand():
    from postflop import action_values
    from preflop import PreflopModel

    deal = Deal(
        (tuple(cards_from_str('Ah Kh 6c 7d 8s')), tuple(cards_from_str('7h 8h 9c Td Ad'))),
        tuple(cards_from_str('Qh Jh Th 6s 7c')),
    )
    game = Game(deal, button_seat=1)
    while game.street < Street.FLOP:
        game.apply(Action.CHECK_CALL)
    memory = {'action_counts': {
        (Street.FLOP, True, 0, True, False): [0, 200, 0],
        (Street.FLOP, True, 1, True, False): [0, 200, 0],
    }}
    values = action_values(game.observation(0), memory, PreflopModel())
    assert np.isfinite(list(values.values())).all()
    assert values[Action.RAISE_POT] > values[Action.RAISE_HALF] > values[Action.CHECK_CALL]


def test_planning_records_terminal_fold_once():
    from postflop import finish_planning

    game = Game(make_deal('planning-fold', 0), button_seat=0)
    while game.street < Street.RIVER:
        game.apply(Action.CHECK_CALL)
    hero = game.to_act
    obs = game.observation(hero)
    memory = {'planning_pending': (obs, Action.RAISE_HALF)}
    game.apply(Action.RAISE_HALF)
    game.apply(Action.FOLD)
    result = game.hand_result(hero)
    finish_planning(result, memory)
    counts = {key: tuple(value) for key, value in memory['action_counts'].items()}
    assert sum(value[0] for value in counts.values()) == 1
    finish_planning(result, memory)
    assert counts == {key: tuple(value) for key, value in memory['action_counts'].items()}


def test_policy_failure_folds_instead_of_paying_an_unknown_bet(monkeypatch):
    module = load_agent()
    game = Game(make_deal('fallback-price', 0), button_seat=0)
    obs = game.observation(0)
    monkeypatch.setattr(module, '_choose_action', lambda *_: 1 / 0)
    memory = {}
    assert module.choose_action(obs, memory) == Action.FOLD
    assert memory['fallbacks'] == 1


def test_exact_river_distribution_matches_all_compatible_hands():
    from sampling import exact_river_distribution, river_combinations, showdown_shares

    deal = make_deal('full-river', 17)
    hole, board = deal.hole_cards[0], deal.board
    shares, strength, frequency = exact_river_distribution(hole, board)
    hands, _ = river_combinations()
    unseen = np.array(sorted(set(range(36)).difference(hole, board)))
    opponents = unseen[hands]
    expected = showdown_shares(hole, opponents, np.broadcast_to(board, opponents.shape))
    assert frequency.sum() == 65780
    assert np.average(shares, weights=frequency) == pytest.approx(expected.mean())
    assert np.all(np.diff(strength) >= 0)


def test_wider_flop_defense_requires_evidence_in_the_correct_position():
    from dataclasses import replace
    from postflop import frequent_flop_bettor

    game = Game(make_deal('frequent-bets', 0), button_seat=1)
    while game.street < Street.FLOP:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_HALF)
    obs = game.observation(0)
    key = (Street.FLOP, False, 0, True, False)
    assert not frequent_flop_bettor(obs, {})
    assert not frequent_flop_bettor(obs, {'action_counts': {key: [0, 0, 4]}})
    assert not frequent_flop_bettor(obs, {'action_counts': {key: [0, 24, 8]}})
    memory = {'action_counts': {key: [0, 3, 29]}}
    assert frequent_flop_bettor(obs, memory)
    assert not frequent_flop_bettor(replace(obs, is_button=True), memory)
    assert not frequent_flop_bettor(replace(obs, raises_this_street=2), memory)


def test_turn_value_margin_has_the_same_units_as_the_model(monkeypatch):
    from dataclasses import replace
    import turn_policy

    game = Game(make_deal('turn-value-units', 0), button_seat=0)
    while game.street < Street.TURN:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    obs = game.observation(game.to_act)
    monkeypatch.setattr(turn_policy, 'turn_context', lambda *_: (np.zeros(16), None, None))
    model = TurnValueModel.__new__(TurnValueModel)
    model.mean = np.zeros(16)
    model.chip_scale = 80.0
    model.action_values = lambda *_: {Action.FOLD: 0.0, Action.CHECK_CALL: 0.05}
    # The same four-chip advantage clears a small-pot cushion, but not a
    # twelve-chip cushion. Values exported in pot units retain legacy behavior.
    assert model.choose_action(replace(obs, pot=20), .5, .4, Action.FOLD, .15) == Action.CHECK_CALL
    assert model.choose_action(replace(obs, pot=80), .5, .4, Action.FOLD, .15) == Action.FOLD
    model.chip_scale = None
    assert model.choose_action(replace(obs, pot=20), .5, .4, Action.FOLD, .15) == Action.FOLD


def test_richer_flop_features_use_only_visible_cards():
    game = Game(make_deal('rich-public-flop', 0), button_seat=0)
    while game.street < Street.FLOP:
        game.apply(Action.CHECK_CALL)
    obs = game.observation(game.to_act)
    base, _, _ = turn_context(obs, .5, .4)
    rich, _, _ = turn_context(obs, .5, .4, rich=True)
    assert rich.shape == (22,)
    assert rich[:16] == pytest.approx(base)
    assert np.all((rich[16:] >= 0) & (rich[16:] <= 1))


def test_action_advantages_are_converted_back_to_values():
    model = TurnValueModel.__new__(TurnValueModel)
    model.mean, model.scale = np.zeros(16), np.ones(16)
    model.relative_to_call = True
    model.layers = tuple(
        ((np.zeros((16, 1)), np.zeros(1)),
         (np.zeros((1, 1)), np.zeros(1)),
         (np.zeros((1, 1)), np.array([value])))
        for value in (.2, .1, -.1)
    )
    values = model.action_values(np.zeros(16), tuple(Action))
    assert values == pytest.approx({Action.FOLD: 0, Action.CHECK_CALL: .2,
                                   Action.RAISE_HALF: .3, Action.RAISE_POT: .1})


def test_planning_budget_remains_available_later_in_the_half(monkeypatch):
    from dataclasses import replace
    module = load_agent()
    monkeypatch.setattr(module, 'SPREAD_PLANNING', True)
    obs = Game(make_deal('spread-budget', 0), button_seat=0).observation(0)
    memory = {'config': AgentConfig(), 'planning_time': .5, 'thinking_time': 3.0}
    assert not module.planning_allowed(obs, memory)
    assert module.planning_allowed(replace(obs, hand_index=700), memory)
    memory['thinking_time'] = 7.0
    assert not module.planning_allowed(replace(obs, hand_index=999), memory)


def test_exported_tree_paths_keep_each_action_head_separate():
    from tree_policy import ArrayTrees
    shape = (3, 1, 3)
    data = {'tree_feature': np.zeros(shape, dtype=int),
            'tree_left': np.tile([1, 1, 2], (3, 1, 1)),
            'tree_right': np.tile([2, 1, 2], (3, 1, 1)),
            'tree_threshold': np.full(shape, .5),
            'tree_value': np.tile([0., 1., -1.], (3, 1, 1)),
            'tree_base': np.array([0., 1., 2.]), 'tree_depth': 3}
    model = ArrayTrees(data)
    assert model.predict(np.array([.25])) == pytest.approx([1, 2, 3])
    assert model.predict(np.array([.75])) == pytest.approx([-1, 0, 1])


def test_river_calibration_preserves_certain_showdowns():
    from river_calibration import RiverCalibration
    deal = Deal(tuple(map(tuple, [cards_from_str('Ah Kh 6c 7d 8s'),
                                 cards_from_str('7h 8h 9c Td Ad')])),
                tuple(cards_from_str('Qh Jh Th 6s 7c')))
    game = Game(deal, button_seat=1)
    while game.street < Street.RIVER:
        game.apply(Action.CHECK_CALL)
    model = RiverCalibration.__new__(RiverCalibration)
    # No learned parameters are needed for a hand that beats every compatible
    # opposing hand. Calibration must retain this exact certainty.
    assert model.estimate(game.observation(0), {}, 1.0) == 1.0



def test_turn_raise_guard_blocks_model_raises_vs_non_folder(monkeypatch):
    monkeypatch.setenv("HEADS_UP_RAISE_GUARD", "1")
    agent = load_agent()
    obs = None
    sticky = {"turn_raises_seen": 20, "turn_raise_folds": 1}
    folder = {"turn_raises_seen": 20, "turn_raise_folds": 12}
    assert agent.turn_raise_guard(obs, sticky, Action.FOLD, Action.RAISE_HALF) == Action.FOLD
    assert agent.turn_raise_guard(obs, folder, Action.FOLD, Action.RAISE_HALF) == Action.RAISE_HALF
    assert agent.turn_raise_guard(obs, {}, Action.FOLD, Action.RAISE_HALF) == Action.RAISE_HALF  # prior 0.4
    assert agent.turn_raise_guard(obs, sticky, Action.FOLD, Action.CHECK_CALL) == Action.CHECK_CALL


def test_turn_raise_guard_on_by_default(monkeypatch):
    monkeypatch.delenv("HEADS_UP_RAISE_GUARD", raising=False)
    assert load_agent().RAISE_GUARD is True


def test_turn_raise_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HEADS_UP_RAISE_GUARD", "0")
    assert load_agent().RAISE_GUARD is False
