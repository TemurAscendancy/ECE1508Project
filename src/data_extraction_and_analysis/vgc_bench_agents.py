# grab one of the agents from vgc-bench
# agent specs are dicts
from __future__ import annotations
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from poke_env.player import (
        MaxBasePowerPlayer,
        Player,
        RandomPlayer,
        SimpleHeuristicsPlayer,
    )
from poke_env.ps_client import AccountConfiguration

from local_sim import side_usernames
import torch
from vgc_bench.src.policy_player import PolicyPlayer

def setup_vgc_bench(vgc_bench_root_path):
    root = Path(vgc_bench_root_path).resolve()
    if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    os.chdir(root)
    return root

def download_bc_checkpoint():
    return Path(hf_hub_download(repo_id="cameronangliss/vgc-bench-models", filename="results/saves_bc/seed1/100.zip", repo_type="model"))

def bc_spec(path):
    if path is None:
        path = download_bc_checkpoint()
        spec = {"hf_repo": "cameronangliss/vgc-bench-models", "hf_file": "results/saves_bc/seed1/100.zip"}
    else:
        spec = {}
    path = Path(path).absolute()  # keep the readable HF snapshot path (a symlink)
    return {"name": "bc", "file": str(path), **spec}

def resolve_spec(spec):
    if spec["name"] == "bc" and not Path(spec["file"]).is_file():
        if "hf_repo" not in spec:
            raise FileNotFoundError(spec["file"])
        path = download_bc_checkpoint()
        spec = {**spec, "file": str(path)}
    return spec

def make_agent(spec, side, battle_formal, loop, log_level=40, shared_policy=None):
    args = dict(
        account_configuration=AccountConfiguration(side_usernames[side], None),
        battle_format=battle_formal,
        log_level=log_level,
        max_concurrent_battles=0,  # unlimited: battles we abandon never "finish"
        accept_open_team_sheet=True,
        start_listening=False,
        loop=loop,
    )
    name = spec["name"]
    if name == "random":
        return RandomPlayer(**args)
    if name == "mbp":
        return MaxBasePowerPlayer(**args)
    if name == "sh":
        return SimpleHeuristicsPlayer(**args)
    if name == "bc":

        player = PolicyPlayer(
            policy=shared_policy,
            deterministic=spec.get("deterministic", False),
            **args,
        )
        if shared_policy is None:
            player.set_policy(spec["file"], torch.device(spec.get("device", "cpu")))
        player.policy.eval()
        return player
    if name == "passive":
        # placeholder for a player that never chooses moves; the caller submits orders (e.g. a DQN learner)
        class PassivePlayer(Player):
            def choose_move(self, battle):
                raise RuntimeError("passive player: submit orders via LocalBattle")

        return PassivePlayer(**args)
    else:
        raise ValueError(f"unknown agent {name!r}")