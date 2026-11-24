# eval of each players in the scenarios of the --play in the trained agents.
# this gives us a reference of how good the agent is, because simply playing against one
# another from the val states might be misleading. The endgame might already be lobsided.

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import inspect
import re
import time
from collections import Counter
#import
import numpy as np

players = ("bc", "sh", "mbp", "random")
player_names = {
    "bc": "BC",
    "sh": "SimpleHeuristics",
    "mbp": "MaxBasePower",
    "random": "Random (poke-env)",
    "random_policy": "Random legal action",
}
RESULT_RE = re.compile(r"^RESULT learner=(\S+) opponent=(\S+) win_rate=([0-9.]+)")


def agent_spec(name, data):
    # bc needs the checkpoint file from a record's spec, the others are just a name
    if name != "bc":
        return {"name": name}
    data_len = len(data)
    for i in range(data_len):
        for spec in data[i]["agents"].values():
            if spec["name"] == "bc":
                return spec
    raise ValueError("no bc spec found in the dataset")


def agent_order(env, name, spec):
    from local_sim import run_sync

    player = env.get_opponent(env.side, spec, env.data[0]["format"])
    order = player.choose_move(env.battle)
    if inspect.isawaitable(order):
        order = run_sync(order, env.loop)
    return order


def play(dql, learner, spec, n, rng):
    from local_sim import run_sync, simError
    env = dql.env
    agent = dql.agent
    out = Counter()
    for ep in range(1, n + 1):
        try:
            state, info = env.reset(seed=ep)
            done = False
            while not done:
                if learner in players:
                    order = agent_order(env, learner, spec)
                    env.lb.submit(env.side, order.message)
                    run_sync(env.advance(), env.loop)
                    b = env.battle
                    ended = env.lb.ended
                    truncated = not ended and b.turn - env.start_turn >= env.max_turns
                    info = env.info()
                else:
                    mask = info["action_mask"]
                    if learner == "random_policy":
                        action = int(rng.choice(np.flatnonzero(mask)))
                    else:
                        action = agent.greedy(dql.feature_representation(state), mask)
                    state, _, ended, truncated, info = env.step(action)
                done = ended or truncated
        except simError:
            out["error"] += 1
            continue

        if info["won"]:
            out["win"] += 1
        elif info["won"] is False:
            out["loss"] += 1
        elif truncated:
            out["trunc"] += 1
        else:
            out["tie"] += 1 # should never happen
    return out


def read_logs(paths):
    # {(learner, opponent): win_rate} from RESULT lines, in first-seen order
    results = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                m = RESULT_RE.match(line)
                if m:
                    results[(m.group(1), m.group(2))] = float(m.group(3))
    if not results:
        raise ValueError(f"no RESULT lines in {paths}")
    return results


def display_name(learner, labels):
    if learner in labels:
        return labels[learner]
    if learner in player_names:
        return player_names[learner]
    return f"{os.path.basename(os.path.dirname(learner))}/{os.path.splitext(os.path.basename(learner))[0]}"


def save_figure(results, path, labels, episodes):
    # heatmap of learner-side win rate; diverging around 0.5 (blue = learner wins more,
    # red = learner loses more, neutral gray = even), rows sorted by mean, a mean column
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb

    learners = list(dict.fromkeys(l for l, _ in results))
    opponents = [o for o in players if any(oo == o for _, oo in results)]
    grid = np.array([[results.get((l, o), np.nan) for o in opponents] for l in learners])
    means = np.nanmean(grid, axis=1)
    order = np.argsort(-means)
    grid, means = grid[order], means[order]
    learners = [learners[i] for i in order]
    dqn_rows = [l not in player_names for l in learners]  # trained policies, not baselines

    ink, ink2, muted, surface = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
    cmap = LinearSegmentedColormap.from_list("win", ["#e34948", "#f0efec", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=0.15, vcenter=0.5, vmax=0.85)

    def label_ink(value):
        r, g, b = to_rgb(cmap(norm(value)))
        return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else ink

    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots(figsize=(1.5 * (n_cols + 1) + 2.6, 0.55 * n_rows + 1.9), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    gap = 0.04  # surface gap between cells
    for i in range(n_rows):
        cells = list(enumerate(grid[i])) + [(n_cols + 0.25, means[i])]  # mean column set apart
        for j, v in cells:
            if np.isnan(v):
                continue
            ax.add_patch(plt.Rectangle((j + gap, i + gap), 1 - 2 * gap, 1 - 2 * gap,
                                       facecolor=cmap(norm(v)), edgecolor="none"))
            mirror = j < n_cols and learners[i] == opponents[j]
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color=label_ink(v), fontweight="bold" if j >= n_cols else "normal",
                    fontstyle="italic" if mirror else "normal")
    ax.set_xlim(0, n_cols + 1.25)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks([j + 0.5 for j in range(n_cols)] + [n_cols + 0.75])
    ax.set_xticklabels([player_names[o] for o in opponents] + ["Mean"], fontsize=9, color=ink2)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels([display_name(l, labels) for l in learners], fontsize=9, color=ink2)
    for tick, is_dqn in zip(ax.get_yticklabels(), dqn_rows):
        if is_dqn:
            tick.set_fontweight("bold")
            tick.set_color(ink)
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Opponent", fontsize=9, color=muted, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Policy on the learner's side", fontsize=9, color=muted, labelpad=8)
    fig.suptitle("Head-to-head win rate on the same endgames", x=0.02, ha="left",
                 fontsize=12, color=ink, fontweight="bold")
    fig.text(0.02, 0.005,
             f"Win rate of the learner-side policy, {episodes} identical val endgames per cell "
             f"(±~2.2 pp). Italic = mirror match. Blue = wins more than half, red = loses more.",
             fontsize=7.5, color=muted, ha="left", va="bottom")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks([0.15, 0.3, 0.5, 0.7, 0.85])
    cbar.ax.tick_params(labelsize=8, colors=muted, length=0)
    cbar.outline.set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--learners", nargs="+")
    p.add_argument("--opponents", nargs="+", default=list(players))
    p.add_argument("--data", default="~/vgc_endgames_ma")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--per_mon", action="store_true", help="the .pth learners use per-pokemon Q heads")
    p.add_argument("--figure", help="save the win-rate matrix as a heatmap to this path")
    p.add_argument("--from_logs", nargs="+", help="plot RESULT lines from these logs instead of playing")
    p.add_argument("--label", action="append", default=[], metavar="LEARNER=NAME",
                   help="display name for a learner in the figure, e.g. runs/x/final_weights.pth='Double DQN'")
    args = p.parse_args()
    labels = dict(kv.split("=", 1) for kv in args.label)

    if args.from_logs:
        if not args.figure:
            p.error("--from_logs needs --figure")
        save_figure(read_logs(args.from_logs), args.figure, labels, args.episodes)
        raise SystemExit
    if not args.learners:
        p.error("--learners is required unless --from_logs is given")

    import torch
    from Hyperparameters import Hyperparameters
    from DQL import DQL

    torch.set_num_threads(1)
    hp = Hyperparameters(data_dir=args.data)
    hp.change(per_mon=args.per_mon)
    dql = DQL(hp, train_mode=False)
    data = dql.env.data
    rng = np.random.default_rng(123)
    results = {}

    for learner in args.learners:
        spec = None
        if learner in players:
            spec = agent_spec(learner, data)
        elif learner != "random_policy":
            dql.agent.onlineDQN.load_state_dict(torch.load(learner, map_location=dql.agent.device))
            dql.agent.onlineDQN.eval()
        name = learner if spec or learner == "random_policy" else os.path.relpath(learner)
        for opp in args.opponents:
            dql.env.opponent = agent_spec(opp, data)
            t = time.time()
            out = play(dql, learner, spec, args.episodes, rng)
            n = sum(out.values()) - out["error"]
            print(f"RESULT learner={name} opponent={opp} win_rate={out['win'] / n:.3f} "
                  f"{dict(out)} ({time.time() - t:.0f}s)", flush=True)
            results[(name, opp)] = out["win"] / n
    dql.env.close()
    if args.figure:
        save_figure(results, args.figure, labels, args.episodes)
