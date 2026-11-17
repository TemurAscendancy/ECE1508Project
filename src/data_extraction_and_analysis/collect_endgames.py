# we cant simply insert initial states into the simulator for some reason.
# instead, let the vgc-bench agents play each other until we get to a 2v2 endgame.
# Then, save the checkpoint and use this as a sample for our dataset.
#
# e.g. call: python collect_endgames.py --n_checkpoints 1000 --out_dir ~/vgc_endgames

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import asyncio
import json
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from local_sim import LocalBattle, SimBridge, simError, random_seed


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_VGC_BENCH_ROOT = Path(
    os.environ.get("VGC_BENCH_ROOT", PROJECT_ROOT.parent / "vgc-bench")
)
sys.path.insert(0, str(HERE.parent / "data_extraction_and_analysis"))
sys.path.insert(0, str(HERE))

AGENT_NAMES = ("bc", "sh", "mbp", "random")
SIDES = ("p1", "p2")


def parse_pool(text):
    # comma separated list of agent, probability pairs.
    # make random low otherwise this becomes noisy
    pool = {}
    for item in text.split(","):
        name, _, weight = item.partition("=")
        name = name.strip()
        if name not in AGENT_NAMES:
            raise argparse.ArgumentTypeError(f"unknown agent {name!r} ({AGENT_NAMES})")
        pool[name] = float(weight) if weight else 1.0
    res = {}
    for n, w in pool.items():
        if w < 0:
            raise argparse.ArgumentTypeError(f"negative weight for {n!r}")
        res[n] = w
    return res


def scan_existing(path):
    # this keeps crashing (my laptop doesnt have much memory D:)
    # make a way to resume :D
    count, max_idx = 0, -1
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line after a crash
                count += 1
                max_idx = max(max_idx, rec["battle_idx"])
    return count, max_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_checkpoints", type=int, required=True)
    parser.add_argument("--pool", type=parse_pool, default=parse_pool("bc=0.35,sh=0.35,mbp=0.2,random=0.1"))
    parser.add_argument("--reg", default="ma")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--num_teams", type=int, default=4)
    parser.add_argument("--take_from_end", action="store_true")
    parser.add_argument("--allow_mirror", action="store_true")
    parser.add_argument("--max_turns", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="~/vgc_endgames")
    parser.add_argument("--bc_path", default=None)
    parser.add_argument("--vgc_bench_root", default=str(DEFAULT_VGC_BENCH_ROOT))
    parser.add_argument("--node", default="node")
    parser.add_argument("--log_level", type=int, default=40)
    args = parser.parse_args()
    if not args.pool:
        parser.error("agent pool empty")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "endgames.jsonl"
    root = Path(args.vgc_bench_root).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)

    
    from vgc_bench.src.teams import RandomTeamBuilder
    from vgc_bench.src.utils import format_map
    from vgc_bench_agents import bc_spec, make_agent, setup_vgc_bench

    setup_vgc_bench(root)

    reg = args.reg
    if not (root / "teams" / f"reg_{reg}").is_dir():
        parser.error(f"no teams for regulation {reg!r} in {root / 'teams'}")
    formats = {reg: format_map[reg] }

    # io handling to adhere to pokemon showdown's async protocol.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    run = loop.run_until_complete
    bridge = SimBridge(root / "pokemon-showdown", node=args.node)

    # build one player per (side, agent); the BC network is loaded once and shared.
    # poke-env players must be constructed while `loop` is running (otherwise their
    # __init__ waits on the loop from outside and blocks forever), hence the coroutine
    any_fmt = next(iter(formats.values()))
    specs, agents = {}, {s: {} for s in SIDES}

    async def build_agents():
        shared_policy = None
        for name in args.pool:
            if name == "bc":
                spec = bc_spec(args.bc_path)
            else:
                spec = {"name": name}
            specs[name] = spec
            for side in SIDES:
                agent = make_agent(spec, side, any_fmt, loop, log_level=args.log_level, shared_policy=shared_policy)
                if name == "bc":
                    shared_policy = agent.policy
                agents[side][name] = agent

    run(build_agents())

    # (relative path, packed team)
    teams = {}
    builder = RandomTeamBuilder(args.run_id, args.num_teams, reg, take_from_end=args.take_from_end)
    res = []
    for p in builder._team_paths:
        res.append((str(p), builder._load_team(p)))
    teams[reg] = res

    collected, max_idx = scan_existing(out_path)
    idx = max_idx + 1
    if collected:
        print(f"resuming: {collected} checkpoints on disk, next battle {idx}")

    names = list(args.pool)
    res = []
    for n in names:
        res.append(args.pool[n])
    weights = res
    status_counts = Counter()
    init_time= time.time()
    out = open(out_path, "a", encoding="utf-8")
    try:
        while collected < args.n_checkpoints:
            rng = random.Random(f"{args.seed}:{idx}")
            reg = args.reg
            reg_teams = teams[reg]
            i1 = rng.randrange(len(reg_teams))
            i2 = rng.randrange(len(reg_teams))
            while i2 == i1 and not args.allow_mirror and len(reg_teams) > 1:
                i2 = rng.randrange(len(reg_teams))
            res = {}
            for s in SIDES:
                res[s] = rng.choices(names, weights)[0]
            agent_names = res
            seed = random_seed(rng)
            # agents draw from the global RNGs; seed them so each battle is reproducible
            agent_seed = rng.getrandbits(32)
            random.seed(agent_seed)
            np.random.seed(agent_seed)
            torch.manual_seed(agent_seed)

            res = {}
            for s in SIDES:
                res[s] = agents[s][agent_names[s]]
            players = res
            lb = LocalBattle(bridge, players, formats[reg], f"b{idx}")
            try:
                run(lb.start(reg_teams[i1][1], reg_teams[i2][1], seed))
                status = run(lb.run(LocalBattle.is_2v2_endgame, args.max_turns))
                if status == "stopped":
                    record = {
                        "id": f"{idx:09d}",
                        "battle_idx": idx,
                        "format": formats[reg],
                        "reg": reg,
                        "seed": seed,
                        "turn": lb.battle("p1").turn,
                        "agents": {s: specs[agent_names[s]] for s in SIDES},
                        "team_paths": {"p1": reg_teams[i1][0], "p2": reg_teams[i2][0]},
                        "endgame": lb.endgame_summary(),
                        "state_hash": lb.state_hash(),
                        "input_log": run(lb.input_log()),
                    }
                    out.write(json.dumps(record, separators=(",", ":")) + "\n")
                    out.flush()
                    collected += 1
                    status = "checkpoint"
            except Exception as e:
                status = "error"
                if status_counts["error"] < 5:
                    print(f"battle {idx} failed:", file=sys.stderr)
                    traceback.print_exc()
                if isinstance(e, simError) and bridge.proc.poll() is not None:
                    bridge.restart()
            finally:
                try:
                    run(lb.close())
                except Exception:
                    pass
            status_counts[status] += 1
            idx += 1

            battles = sum(status_counts.values())
            if battles % 25 == 0 or collected >= args.n_checkpoints:
                elapsed = time.time() - init_time
                print(
                    f"[{elapsed:7.0f}s] checkpoints {collected}/{args.n_checkpoints}"
                    f" | battles {battles} (yield {status_counts['checkpoint'] / battles:.2f})"
                    f" | {dict(status_counts)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        out.close()
        bridge.shutdown()
        loop.close()
    print(f"done: {collected} checkpoints in {out_path}")


if __name__ == "__main__":
    main()
