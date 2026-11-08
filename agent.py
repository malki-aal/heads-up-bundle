"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback
from time import perf_counter

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from equity import EXACT_RIVER, EquityModel
from evaluator import warmup
from game import Action, AgentConfig, HandResult, Observation, Street, default_action
from postflop import choose_planned, finish_planning, frequent_flop_bettor, protect_turn_call
from preflop import PreflopModel
from river_betting import choose_opening, finish_opening, record_opening
from river_calibration import RiverCalibration
from strategy import opponent_range_scale, preflop_equity, river_response_equity, rules_equity, select_action
from turn_policy import TURN_WEIGHTS_PATH, TurnValueModel
from sampling import prepare_scores, river_combinations, river_pair_offsets

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# Set PARTICIPANT to the team name registered with the organizers.
PARTICIPANT = "datajohn"
BOT_MODE = os.environ.get("HEADS_UP_MODE", "ml").lower()
if BOT_MODE not in ("ml", "rules"):
    raise ValueError("HEADS_UP_MODE must be ml or rules")
AGENT_NAME = "EquityNet" if BOT_MODE == "ml" else "EquityRules"

# ===== YOUR CODE HERE =====

MODEL = EquityModel() if BOT_MODE == "ml" else None
BALANCED = MODEL is not None and os.environ.get('HEADS_UP_BALANCED', '1') == '1'
STARTING = PreflopModel() if BALANCED else None
STARTING_POLICY = BALANCED and os.environ.get('HEADS_UP_PREFLOP', '1') == '1'
PLANNING = BALANCED and os.environ.get('HEADS_UP_PLANNING', '1') == '1'
if PLANNING or (MODEL is not None and EXACT_RIVER):
    prepare_scores()
if MODEL is not None and EXACT_RIVER:
    river_combinations()
    river_pair_offsets()
TURN_MODEL = (
    TurnValueModel(os.environ.get('HEADS_UP_TURN_WEIGHTS', TURN_WEIGHTS_PATH))
    if MODEL is not None and os.environ.get("HEADS_UP_TURN_MODEL", "1") == "1"
    else None
)
TURN_MARGIN = float(os.environ.get('HEADS_UP_TURN_MARGIN', '0.12'))
FLOP_MODEL = (
    TurnValueModel(os.environ.get('HEADS_UP_FLOP_WEIGHTS', os.path.join(os.path.dirname(__file__), 'flop_value_weights.npz')))
    if MODEL is not None and os.environ.get('HEADS_UP_FLOP_MODEL', '0') == '1'
    else None
)
FLOP_MARGIN = float(os.environ.get('HEADS_UP_FLOP_MARGIN', '0.12'))
RIVER_BETTING = MODEL is not None and os.environ.get("HEADS_UP_RIVER_BETS", "1") == "1"
RIVER_BET_MARGIN = float(os.environ.get('HEADS_UP_RIVER_MARGIN', '0.15'))
LATE_GUARD = os.environ.get('HEADS_UP_LATE_GUARD', '1') == '1'
SPREAD_PLANNING = os.environ.get('HEADS_UP_SPREAD_PLANNING', '0') == '1'
RIVER_CALIBRATION = (
    RiverCalibration(os.environ.get('HEADS_UP_RIVER_CALIBRATION_WEIGHTS'))
    if MODEL is not None and os.environ.get('HEADS_UP_RIVER_CALIBRATION', '1') == '1'
    else None
)


def planning_allowed(obs: Observation, memory: dict) -> bool:
    config = memory.get('config', AgentConfig())
    budget = config.time_bank_seconds
    if not SPREAD_PLANNING:
        return memory.get('thinking_time', 0.0) < 0.35 * budget
    progress = min(1.0, (obs.hand_index + 100) / max(1, config.num_hands))
    return (memory.get('planning_time', 0.0) < 0.35 * budget * progress
            and memory.get('thinking_time', 0.0) < 0.70 * budget)


def run_planning(function, obs, memory, fallback):
    started = perf_counter()
    try:
        return function(obs, memory, STARTING, fallback)
    finally:
        memory['planning_time'] = memory.get('planning_time', 0.0) + perf_counter() - started


def new_half(config: AgentConfig, memory: dict) -> None:
    memory.clear()
    memory['config'] = config


def choose_action(obs: Observation, memory: dict) -> Action:
    started = perf_counter()
    try:
        if RAISE_GUARD:
            observe_turn_raise_response(obs, memory)
        action = _choose_action(obs, memory)
        if RAISE_GUARD and obs.street == Street.TURN and action in (Action.RAISE_HALF, Action.RAISE_POT):
            memory['turn_raise_pending'] = obs.hand_index
        if action not in obs.legal_actions:
            raise ValueError('Policy selected an illegal action')
        if RIVER_BETTING and obs.street == Street.RIVER:
            record_opening(obs, action, memory)
        if PLANNING:
            memory['planning_pending'] = obs, action
        return action
    except Exception:
        memory["fallbacks"] = memory.get("fallbacks", 0) + 1
        memory.pop('planning_pending', None)
        return default_action(obs.to_call)
    finally:
        memory['thinking_time'] = memory.get('thinking_time', 0.0) + perf_counter() - started


def _choose_action(obs: Observation, memory: dict, *, use_turn_model: bool = True) -> Action:
    if MODEL is not None:
        record_opponent_openers(obs, memory)
        seen = memory.get("flop_openers", 0)
        bets = memory.get("flop_open_bets", 0)
        if obs.street == Street.RIVER and obs.to_call > 0:
            equity = river_response_equity(obs, opponent_range_scale(memory))
            if RIVER_CALIBRATION is not None:
                equity = RIVER_CALIBRATION.estimate(obs, memory, equity)
            return select_action(obs, equity)

    # The estimate is constant across repeat decisions on the same board.
    key = (obs.hand_index, tuple(obs.hole_cards), tuple(obs.board))
    if memory.get("equity_key") != key:
        _, hole, board = key
        if not board:
            estimate = STARTING.estimate(hole) if STARTING_POLICY else preflop_equity(hole)
        elif MODEL is None:
            estimate = rules_equity(hole, board)
        else:
            estimate = MODEL.estimate(hole, board)
        memory.update(equity_key=key, equity=estimate)

    if not obs.board and STARTING_POLICY:
        half, pot = STARTING.raise_thresholds
        return select_action(obs, memory['equity'], preflop_half=half, preflop_pot=pot)

    if MODEL is not None and obs.street in (Street.FLOP, Street.TURN):
        # Chosen offline by net chips across duplicate matches and several styles.
        raise_margin = -0.10 if obs.street == Street.FLOP else -0.08
        equity = memory["equity"]
        fallback = select_action(obs, equity, raise_margin=raise_margin, call_margin=0.10)
        if FLOP_MODEL is not None and obs.street == Street.FLOP and obs.to_call > 0:
            opener_rate = (bets + 2) / (seen + 5)
            return FLOP_MODEL.choose_action(obs, equity, opener_rate, fallback, FLOP_MARGIN)
        if use_turn_model and TURN_MODEL is not None and obs.street == Street.TURN and obs.to_call > 0:
            opener_rate = (bets + 2) / (seen + 5)
            chosen = TURN_MODEL.choose_action(obs, equity, opener_rate, fallback, TURN_MARGIN)
            return turn_raise_guard(obs, memory, fallback, chosen)
        plan_here = obs.to_call == 0 or frequent_flop_bettor(obs, memory)
        if PLANNING and plan_here and planning_allowed(obs, memory):
            return run_planning(choose_planned, obs, memory, fallback)
        if PLANNING and LATE_GUARD and planning_allowed(obs, memory):
            return run_planning(protect_turn_call, obs, memory, fallback)
        return fallback

    fallback = select_action(obs, memory["equity"])
    if RIVER_BETTING and obs.street == Street.RIVER and obs.to_call == 0 and obs.raise_to:
        return choose_opening(obs, memory, fallback, RIVER_BET_MARGIN)
    return fallback


def record_opponent_openers(obs: Observation, memory: dict) -> None:
    # In position, both a check and a bet from the opponent are observable.
    # Replay the public history so a later decision also captures earlier streets.
    if not obs.is_button:
        return
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        counts = memory.setdefault("opponent_openers", {}).setdefault(street, {})
        if counts.get("last_hand") == obs.hand_index:
            continue
        opener = next((event for event in obs.action_history if event.street == street), None)
        if opener is None or opener.is_me:
            continue
        counts["last_hand"] = obs.hand_index
        counts["seen"] = counts.get("seen", 0) + 1
        raised = opener.action in (Action.RAISE_HALF, Action.RAISE_POT)
        counts["bets"] = counts.get("bets", 0) + int(raised)
        if street == Street.FLOP:
            # Preserve the counters used by the optional turn model and trainer.
            memory["last_flop_opener"] = obs.hand_index
            memory["flop_openers"] = counts["seen"]
            memory["flop_open_bets"] = counts["bets"]


# On by default (HEADS_UP_RAISE_GUARD=0 disables): block turn-model raises that the base rule would not make
# once this opponent has shown, within the half, that it rarely folds to our turn raises.
RAISE_GUARD = MODEL is not None and os.environ.get('HEADS_UP_RAISE_GUARD', '1') == '1'
RAISE_GUARD_PRIOR = 0.4      # prior fold rate to our turn raises
RAISE_GUARD_WEIGHT = 6.0     # prior strength in observations
RAISE_GUARD_MIN_FOLD = 0.33  # about break-even fold rate for a half-pot raise


def _record_turn_raise(memory: dict, folded: bool) -> None:
    memory['turn_raises_seen'] = memory.get('turn_raises_seen', 0) + 1
    memory['turn_raise_folds'] = memory.get('turn_raise_folds', 0) + int(folded)
    memory.pop('turn_raise_pending', None)


def observe_turn_raise_response(obs: Observation, memory: dict) -> None:
    """A later decision in the same hand means the opponent did not fold to our turn raise."""
    if memory.get('turn_raise_pending') == obs.hand_index and (
            obs.street > Street.TURN or obs.to_call > 0):
        _record_turn_raise(memory, folded=False)


def turn_raise_guard(obs: Observation, memory: dict, fallback: Action, chosen: Action) -> Action:
    if not RAISE_GUARD or chosen == fallback or chosen not in (Action.RAISE_HALF, Action.RAISE_POT):
        return chosen
    seen, folds = memory.get('turn_raises_seen', 0), memory.get('turn_raise_folds', 0)
    rate = (folds + RAISE_GUARD_WEIGHT * RAISE_GUARD_PRIOR) / (seen + RAISE_GUARD_WEIGHT)
    return fallback if rate < RAISE_GUARD_MIN_FOLD else chosen


def on_hand_end(result: HandResult, memory: dict) -> None:
    if RAISE_GUARD and memory.get('turn_raise_pending') == result.hand_index:
        # Hand ended on the turn without showdown and we won: the opponent folded.
        _record_turn_raise(memory, folded=(not result.went_to_showdown and result.winner == 'me'
                                           and len(result.board) == 4))
    if PLANNING:
        finish_planning(result, memory)
    if RIVER_BETTING:
        finish_opening(result, memory)


# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
