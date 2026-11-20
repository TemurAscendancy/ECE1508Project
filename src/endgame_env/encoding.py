# observation and action encoding for the 2v2 endgames
# pokemon has a lot of information, because its cooler than chess
# state: 2170 values
# 0-27 : global (terrain, weather)
# 28-56: your side's pokemon, (tailwind/reflect, conditions, mega)
# 57-85: same as 28-56, but for opponents team
# 86-606: your slot a pokemon, ability, item, stats, typing, hp, status, etc.
# 607-1127: same, but slot 2
# 1128-1648: same, but opponent slot 1
# 1649-2169: same, but opponent slot 2

# vgc-bench handles this embedding, but we probably want to remove the redundant stuff
# at some point.
# ASAD TODO: remove redundant features, e.g. rank, openteamsheet, etc.

# action: 625 values (this needs to be reduced...)
# 25 possible actions per pokemon, 8 moves (4, with and without mega) * 3 targets + 1 (pass)
# for both mons this is 25*25=625. We can make it per pokemon, but you would not get
# double up strategies, but atleast youd learn something...

# legality comes from poke-env's DoublesEnv.get_action_mask_individual (pp, choice lock,
# encore, taunt, fainted targets, mega...) folded into the layout above

import os
import sys
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
vgc_bench_root = Path(os.environ.get("VGC_BENCH_ROOT", here.parents[2] / "vgc-bench")).resolve()
sys.path.insert(0, str(here.parent / "data_extraction_and_analysis"))
sys.path.insert(0, str(vgc_bench_root))

# vgc_bench reads data/*.json relative to the cwd when imported, so chdir just for the imports
cwd = os.getcwd()
os.chdir(vgc_bench_root)
from poke_env.environment import DoublesEnv
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.utils import abilities, glob_obs_len, items, moves, pokemon_obs_len, side_obs_len
os.chdir(cwd)

# observation
pokemon_start_index = glob_obs_len + 2 * side_obs_len
observation_length = pokemon_start_index + 4 * pokemon_obs_len
categorical_idx = np.array([pokemon_start_index + k * pokemon_obs_len + j for k in range(4) for j in range(6)])
vocab = {"ability": len(abilities), "item": len(items), "move": len(moves)}


def encode_obs(battle):
    blocks = [
        PolicyPlayer.embed_global(battle),
        PolicyPlayer.embed_side(battle, 2000),  # same fake ratings as vgc-bench
        PolicyPlayer.embed_side(battle, 0, opp=True),
    ]
    sides = [
        (battle.team, battle.active_pokemon, False),
        (battle.opponent_team, battle.opponent_active_pokemon, True),
    ]
    for team, active, opp in sides:
        for slot, mon in enumerate(active):
            if mon is None or mon.fainted:
                blocks.append(np.zeros(pokemon_obs_len, dtype=np.float32))
                continue
            team_pos = next(i for i, m in enumerate(team.values()) if m is mon)
            blocks.append(PolicyPlayer.embed_pokemon(
                mon, team_pos, from_opponent=opp, active_a=slot == 0, active_b=slot == 1
            ))
    return np.concatenate(blocks, dtype=np.float32)


# actions
class ActionSpec:
    def __init__(self, ally_targets=True):
        self.ally_targets = ally_targets
        self.n_cols = 3 if ally_targets else 2
        self.slot_actions = 1 + 2 * 4 * self.n_cols
        self.n_actions = self.slot_actions ** 2
        self.mega_start = 1 + 4 * self.n_cols

    def index(self, move, col, mega):
        return 1 + mega * 4 * self.n_cols + move * self.n_cols + col

    def describe_slot(self, idx):
        if idx == 0:
            return "pass"
        mega, rest = divmod(idx - 1, 4 * self.n_cols)
        move, col = divmod(rest, self.n_cols)
        return f"move{move + 1}->{('foe a', 'foe b', 'ally')[col]}" + (" +mega" if mega else "")

    def describe(self, action):
        a, b = divmod(int(action), self.slot_actions)
        return f"[a: {self.describe_slot(a)} | b: {self.describe_slot(b)}]"


def slot_table(battle, pos, spec):
    # map each legal slot action (our layout) -> poke-env's per-slot index
    # poke-env per slot: 0 pass, 1-6 switch, 7 + 20*gimmick + 5*move + (target + 2)
    # showdown targets: 1, 2 = foe slots, -1, -2 = own slots, 0 = no target
    mask = DoublesEnv.get_action_mask_individual(battle, pos)
    legal = []
    for i, v in enumerate(mask):
        if v:
            legal.append(i)
    self_t = -1 
    ally_t = -2
    if pos != 0:
        self_t = -2
        ally_t = -1

    ally_alive = False
    if battle.active_pokemon[1-pos]:
        ally_alive = True

    # (move, mega) -> {target: poke-env index}
    groups = {}
    for i in legal:
        if i < 7:
            continue
        gimmick, rest = divmod(i - 7, 20)
        if gimmick > 1:
            continue  # z-move / dynamax / tera, not in this format
        move, t = divmod(rest, 5)
        groups.setdefault((move, gimmick), {})[t - 2] = i

    if 0 in legal:
        table = {0:0}
    else:
        table = {}
    for (move, mega), targets in groups.items():
        cols = {}
        if 1 in targets:
            cols[0] = targets[1]
        if 2 in targets:
            cols[1] = targets[2]
        if spec.ally_targets and ally_alive and ally_t in targets:
            cols[2] = targets[ally_t]
        if 1 not in targets and 2 not in targets:
            # no foe to pick (spread / self / ally move): goes in col 0
            # without the ally col, ally only moves (helping hand) go here too
            if spec.ally_targets:
                options = (0, self_t)
            else:
                options = (0, ally_t, self_t)
            for t in options:
                if t in targets:
                    cols[0] = targets[t]
                    break
        if not cols:
            cols[0] = next(iter(targets.values()))  # e.g. helping hand with a fainted ally
        for col, i in cols.items():
            table[spec.index(move, col, mega)] = i
    return table


def legal_actions(battle, spec):
    # the mask is all zeros if there's no move to pick (e.g. a switch after revival blessing)
    tables = (slot_table(battle, 0, spec), slot_table(battle, 1, spec))
    mask = np.zeros(spec.n_actions, dtype=np.int8)
    for a in tables[0]:
        for b in tables[1]:
            if a >= spec.mega_start and b >= spec.mega_start:
                continue
            mask[a * spec.slot_actions + b] = 1
    return tables, mask


def decode(action, tables, battle, spec):
    # joint action -> poke-env order (action_to_order raises if it's illegal)
    a, b = divmod(int(action), spec.slot_actions)
    return DoublesEnv.action_to_order(np.array([tables[0][a], tables[1][b]]), battle)
