
# gym env wrapper for the 2v2 endgame checkpoints.
# reset() restores the random checkpoint with a fresh new one.
# the opponent is the agent that originally played on the other side
# basic flow:
#   env = EndgameEnv("~/vgc_endgames_ma", split="train")
#   obs, info = env.reset(seed=0)
#   while True:
#       action = policy(obs, info["action_mask"])
#       obs, reward, terminated, truncated, info = env.step(action)
#       if terminated or truncated:
#           obs, info = env.reset()
# reward is +1 win, -1 loss at the end.

import asyncio
import hashlib
import itertools
import json
import random


import sys
import threading
import zlib

from pathlib import Path
import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# encoding also puts data_extraction_and_analysis and vgc-bench on the path
from encoding import observation_length, ActionSpec, decode, encode_obs, legal_actions, vgc_bench_root
from local_sim import LocalBattle, SimBridge, random_seed, run_sync, sides, simError
from vgc_bench_agents import make_agent, resolve_spec

other_side = {"p1": "p2", "p2": "p1"}
env_ids = itertools.count()  # unique battle ids if there are a few envs in one process

class EndgameDataset:
    # checkpoint records from the *.jsonl files in a folder
    # kept zlib compressed in memory (~0.6 KB each instead of ~3 KB)
    # split: a fixed val_fraction of records goes to "val" (by a hash of the id), the rest to "train"
    def __init__(self, data_dir, split=None, val_fraction=0.05, max_records=None):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.blobs = []
        for path in sorted(self.data_dir.glob("*.jsonl")):
            with open(path, "rb") as fh:
                for line in fh:
                    if split is not None:
                        rec_id = json.loads(line)["id"]
                        h = int(hashlib.sha1(rec_id.encode()).hexdigest()[:8], 16) / 2**32
                        if (split == "val") != (h < val_fraction):
                            continue
                    self.blobs.append(zlib.compress(line))
                    if len(self.blobs) == max_records:
                        return
    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, i):
        return json.loads(zlib.decompress(self.blobs[i]))


class EndgameEnv(gym.Env):

    # data: EndgameDataset (or optionally the checkpoint folder)
    # learner_side is p1, p2 or random from the checkpoint
    # opponent: if you want to usse a different opponent from the one who played
    def __init__(self, data, split=None, learner_side="random", opponent=None,
                 max_turns=30, hp_shaping=0.0, ally_targets=True):
        super().__init__()
        if not isinstance(data, EndgameDataset):
            data = EndgameDataset(data, split)
        self.data = data
        self.learner_side = learner_side
        self.opponent = opponent
        self.max_turns = max_turns
        # might need denser reward, add hp shaping by
        self.hp_shaping = hp_shaping
        self.actions = ActionSpec(ally_targets)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_length,), np.float32)
        self.action_space = gym.spaces.Discrete(self.actions.n_actions)
        #print(self.observation_space)
        #print(self.action_space)
        # poke-env players need a running event loop, so run one in a background thread
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.bridge = SimBridge(str(vgc_bench_root / "pokemon-showdown"))
        self.uid = next(env_ids)
        self.episodes = 0
        fmt = data[0]["format"]
        # the learner never picks by itself, step() submits its orders
        self.learners = {}
        for s in sides:
            self.learners[s] = make_agent({"name": "passive"}, s, fmt, self.loop)
        self.opponents = {}
        self.bc_policy = None
        self.lb = None

    def get_opponent(self, side, spec, fmt):
        key = (side, spec["name"])
        if key not in self.opponents:
            spec = resolve_spec(spec)
            player = make_agent(spec, side, fmt, self.loop, shared_policy=self.bc_policy)
            if spec["name"] == "bc":
                self.bc_policy = player.policy  # one network for all bc opponents
            self.opponents[key] = player
        return self.opponents[key]

    @property
    def battle(self):
        # the learner's poke-env DoubleBattle
        return self.lb.battle(self.side)

    # asad todo: try training DQN w/ this signal, should help
    def hp_diff(self):
        own = sum(m.current_hp_fraction for m in self.battle.team.values())
        opp = sum(m.current_hp_fraction for m in self.battle.opponent_team.values())
        return own - opp

    async def advance(self):
        # play on until the learner has to pick moves again or the battle ends
        lb, side = self.lb, self.side
        await lb.send_pending()
        while not lb.ended:
            if lb.round_fresh:
                req = lb.requests()[side]
                if not req.get("wait"):
                    if not req.get("forceSwitch"):
                        self.tables, self.mask = legal_actions(lb.battle(side), self.actions)
                        if self.mask.any():
                            return
                        
                    # not a move choice (e.g. a switch after revival blessing), pick randomly
                    order = lb.players[side].choose_random_doubles_move(lb.battle(side))
                    lb.submit(side, order.message)
            if not await lb.send_pending():
                raise simError(f"battle stuck (record {self.record_id})")
        self.mask = np.zeros(self.actions.n_actions, dtype=np.int8)

    def info(self):
        return {
            "action_mask": self.mask,
            "record_id": self.record_id,
            "learner_side": self.side,
            "opponent": self.opp_name,
            "turn": self.battle.turn,
            "won": self.battle.won,
        }

    def reset(self, *, seed=None, options=None):
        # options: {"index": i, "side": "p1"} to pick a specific scenario (e.g. a fixed eval set)
        super().reset(seed=seed)
        options = options or {}
        if self.lb is not None:
            run_sync(self.lb.close(), self.loop)
        rng = self.np_random

        rec = self.data[options.get("index", int(rng.integers(len(self.data))))]
        side = options.get("side", self.learner_side)
        if side == "random":
            side = sides[int(rng.integers(2))]
        opp_side = other_side[side]

        opp_spec = self.opponent or rec["agents"][opp_side]
        players = {
            side: self.learners[side],
            opp_side: self.get_opponent(opp_side, opp_spec, rec["format"]),
        }

        self.side, self.record_id, self.opp_name = side, rec["id"], opp_spec["name"]
        self.episodes += 1

        # new random instance
        reseed = random_seed(random.Random(int(rng.integers(2**62))))
        battle_id = f"env{self.uid}n{self.episodes}"
        auto = {side: False, opp_side: True}
        self.lb = run_sync(LocalBattle.restore(self.bridge, players, rec, battle_id, reseed, auto), self.loop)

        run_sync(self.lb.prompt(opp_side), self.loop)  # opponent picks its move now

        self.tables, self.mask = legal_actions(self.battle, self.actions)
        self.start_turn = self.battle.turn
        self.last_hp = self.hp_diff()
        return encode_obs(self.battle), self.info()

    def step(self, action):
        order = decode(action, self.tables, self.battle, self.actions)
        self.lb.submit(self.side, order.message)
        run_sync(self.advance(), self.loop)

        b = self.battle
        terminated = self.lb.ended
        reward = 0.0

        if terminated:
            reward = 1.0 if b.won else -1.0 if b.lost else 0.0
            
        if self.hp_shaping:
            hp = self.hp_diff()
            reward += self.hp_shaping * (hp - self.last_hp)
            self.last_hp = hp
        truncated = not terminated and b.turn - self.start_turn >= self.max_turns
        return encode_obs(b), reward, terminated, truncated, self.info()

    def describe_action(self, action):
        return self.actions.describe(action)

    def close(self):
        if self.lb is not None:
            run_sync(self.lb.close(), self.loop)
        self.bridge.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop)
