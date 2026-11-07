"""Compare both equity policies with the official duplicate-match judge."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
from secrets import token_urlsafe
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
BASELINES = ("random", "calling_station", "always_pot")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def play(task: tuple[int, str, str, str], deals: int) -> dict:
    index, left, right, seed = task
    ports = (15300 + 2 * index, 15301 + 2 * index)
    base_env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    tokens = (token_urlsafe(24), token_urlsafe(24))
    judge_env = {**base_env, "AGENT1_TOKEN": tokens[0], "AGENT2_TOKEN": tokens[1]}

    with tempfile.TemporaryDirectory() as workdir:
        tmp = Path(workdir)
        paths = [tmp / "half1.json", tmp / "half2.json"]
        for half, output in enumerate(paths, start=1):
            # Each half starts fresh servers, as it does in the tournament.
            with ExitStack() as stack:
                for port, mode, token in zip(ports, (left, right), tokens):
                    env = {**base_env, "PORT": str(port), "HEADS_UP_MODE": mode, "AGENT_TOKEN": token}
                    script = "agent.py" if mode in ("ml", "rules") else f"baseline-bots/baseline_{mode}.py"
                    log = stack.enter_context((tmp / f"{half}-{port}.log").open("w"))
                    process = subprocess.Popen(
                        [sys.executable, script], cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                    stack.callback(stop, process)
                run = subprocess.run(
                    [
                        sys.executable, "judge_engine.py",
                        "--agent1", f"http://127.0.0.1:{ports[0]}", "--agent2", f"http://127.0.0.1:{ports[1]}",
                        "--deals", str(deals), "--half", str(half),
                        "--seed", seed, "--json", str(output),
                    ],
                    cwd=ROOT, env=judge_env, capture_output=True, text=True, timeout=120,
                )
                if run.returncode or not output.exists():
                    logs = "\n".join(log.read_text() for log in tmp.glob(f"{half}-*.log"))
                    raise RuntimeError(f"{left} vs {right}, seed {seed}: {run.stdout}{run.stderr}\n{logs}")

        combined = tmp / "match.json"
        subprocess.run(
            [sys.executable, "judge_engine.py", "combine", *map(str, paths), "--json", str(combined)],
            cwd=ROOT, env=base_env, capture_output=True, check=True, timeout=15,
        )
        result = json.loads(combined.read_text())
    return {"left": left, "right": right, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", default=["461", "967"])
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args()
    pairings = [("ml", "rules"), *((mode, bot) for bot in BASELINES for mode in ("ml", "rules"))]
    cases = [(left, right, seed) for left, right in pairings for seed in args.seeds]
    tasks = [(i, *case) for i, case in enumerate(cases)]
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(play, task, args.deals) for task in tasks]
        for future in futures:
            result = future.result()
            results.append(result)
            print(
                f"{result['left']:5} vs {result['right']:15} seed {result['match_seed']:>4}: "
                f"{result['mean_bb_per_deal']:+.3f} bb/deal, {result['verdict']}",
                flush=True,
            )
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
