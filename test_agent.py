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
    _, client = bot
    start_half(client)
    play_hands(client, 200)


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


def test_turn_sampler_excludes_visible_cards_and_matches_judge_scores():
    deal = make_deal("turn-worlds", 0)
    game = Game(deal, button_seat=0)
    while game.street < Street.TURN:
        game.apply(Action.CHECK_CALL)
    obs = game.observation(game.to_act)
    features, worlds, weights = turn_context(obs, equity=0.5, opener_rate=0.4)
    visible = set(obs.hole_cards + obs.board)
    assert all(len(set(world)) == 6 and not visible.intersection(world) for world in worlds)
    assert weights.sum() == pytest.approx(1.0)

    boards = np.column_stack((np.broadcast_to(obs.board, (len(worlds), 4)), worlds[:, 5]))
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
