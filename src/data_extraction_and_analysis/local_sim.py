# local sim of pokemon showdown battles
# localbattle plays one battle on the SimBridge
# it then talks to the players, reads their moves, writes back to bridge
# bridge then talks to the showdown sim and gives each player their information
# SimBridge basically acts like the server, in a separate node process that talks over pipes

import asyncio
import hashlib
import json
import random
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from poke_env.battle import DoubleBattle
from poke_env.player import Player

sides = ("p1", "p2")
bridge_path = Path(__file__).resolve().with_name("sim_bridge.js")
side_usernames = {"p1": "LocalP1", "p2": "LocalP2"}

class simError(RuntimeError):
    pass

async def _no_request_handling(*args: Any, **kwargs: Any):
    pass

def run_sync(coro, loop):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
# copy from prng.ts (vgc bench pokemon-showdown submofule)
def random_seed(rng):
        return "sodium," + "".join(f"{rng.getrandbits(8):02x}" for _ in range(16))

class SimBridge:
    # pipe to showdown (acts like the server)
    def __init__(self, showdown_root, node="node"):
        self.showdown_root =showdown_root
        self.node = node
        self.start()

    def start(self):
        self.proc = subprocess.Popen(
                    [self.node, str(bridge_path), self.showdown_root],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                )
    def shutdown(self):
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()

    def restart(self):
        self.shutdown()
        self.start()

    def call(self, op, data="", battle_id=""):
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps({"id": battle_id, "op": op, "data": data}))
        self.proc.stdin.write("\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise simError(f"sim bridge died (exit code {self.proc.poll()})")
        resp = json.loads(line)
        if "error" in resp:
            raise simError(resp["error"])
        return resp["events"]

    def new(self, battle_id):
        return self.call("new", battle_id=battle_id)
    def write(self, battle_id, data):
        return self.call("write", data=data, battle_id=battle_id)
    def close(self, battle_id):
        return self.call("close", battle_id=battle_id)

# hardcode specific error msgs
no_target_err_re = re.compile(r"\[Invalid choice\] Can't move: You can't choose a target for (.+)")
needs_target_err_re = re.compile(r"\[Invalid choice\] Can't move: (.+) needs a target")

def choice_error(events):
    for event in events:
        if event[0] == "side":
           for line in event[2].split("\n"):
                if line.startswith("|error|"):
                    return line[len("|error|") :] 
    return None

# some moves require special parsing

def strip_targ(choice_line, move_name):
    # moves like outrage and petal dance need special handling
    move_id = re.sub(r"[^a-z0-9]", "", move_name.lower())
    fixed = re.sub(rf"(move {move_id}) [+-]?\d+", r"\1", choice_line)
    if fixed != choice_line:
        return fixed
    else:
        return None

def restore_default(choice_line, move_name):
    # player left to default (e.g. struggle)
    move_id = re.sub(r"[^a-z0-9]", "", move_name.lower())
    fixed = re.sub(rf"move {move_id}(?= *(,|$))", "default", choice_line, count=1)
    if fixed != choice_line:
        return fixed
    else:
        return None
    
def repair_choice(choice_line, err):
    if m := no_target_err_re.match(err):
        return strip_targ(choice_line, m.group(1))
    if m := needs_target_err_re.match(err):
        return restore_default(choice_line, m.group(1))
    return None

class LocalBattle:
    # battle between two offline players on simbridge
    def __init__(self, bridge, players, format, battle_id, auto=None):
        self.bridge = bridge
        self.players = players
        #print(self.players)
        self.format = format
        self.battle_id = battle_id
        self.tag = f"battle-{format}-{battle_id}"
        #print(f"tag = {self.tag})
        if auto is not None:
            self.auto = dict(auto)
        else:
            self.auto = {}
            for s in sides:
                self.auto[s] = True
        self.last_request = {}
        for s in sides:
            self.last_request[s] = None
        self.pending = {}
        for s in sides:
            self.pending[s] = None
        self.ended = False
        self.end_info = None
        self._initialized = {}
        for s in sides:
            self._initialized[s] = False
        self.round_fresh = False
        for side, player in players.items():
            player._format = format # reg-a, since it has the least pokemon
            self._hook(side, player)
            self._set_request_handling(side, self.auto[side])
        bridge.new(battle_id)

    def _hook(self, side, player):
        async def capture(message, room="", message_2=None):
            if room != self.tag:
                #print(f"room = {room}, tag = {self.tag}")
                return
            if message.startswith("/choose "):
                self.pending[side] = message[8:]
            elif message.startswith("/team "):
                self.pending[side] = "team " + message[6:]
            elif message == "/forfeit":
                self.pending[side] = "forfeit"
            # rest is server only, ifnore


        player.ps_client.send_message = capture 

    def battle(self, side):
        battle = self.players[side]._battles[self.tag]
        return battle

    async def _feed(self, side, text):
        text_split = text.split("\n")
        if not self._initialized[side]:
            text_split = ["|init|battle"] + text_split
            self._initialized[side] = True
        split = [[f">{self.tag}"]] + [line.split("|") for line in text_split]
        await self.players[side]._handle_battle_message(split)

    async def _process(self, events):
        requested = set()
        for event in events:
            kind = event[0]
            if kind == "update":
                for side, text in zip(sides, event[1:3]):
                    if text:
                        await self._feed(side, text)
            elif kind == "side":
                side, text = event[1], event[2]
                for line in text.split("\n"):
                    if line.startswith("|request|"):
                        self.last_request[side] = line[len("|request|") :]
                        requested.add(side)
                await self._feed(side, text)
            elif kind == "end":
                self.ended = True
                if event[1]:
                    self.end_info = json.loads(event[1])
                else:
                    self.end_info = {}
            elif kind == "error":
                raise simError(event[1])
        if len(requested) == 2:
            self.round_fresh = True

    async def write(self, data):
        events = self.bridge.write(self.battle_id, data)
        await self._process(events)
        return events
    
    def _set_request_handling(self, side, enabled):
        player = self.players[side]
        if enabled:
            player.__dict__.pop("_handle_battle_request", None)
        else:
            player._handle_battle_request = _no_request_handling
    @contextmanager
    def passive(self):
        # ignore requests for replaying
        for side in sides:
            self._set_request_handling(side, False)
        try:
            yield
        finally:
            for side in sides:
                self._set_request_handling(side, self.auto[side])
    async def start(self, team_p1, team_p2, seed):
        start = {"formatid": self.format, "seed": seed}
        await self.write(f">start {json.dumps(start)}")
        for side, team in zip(sides, (team_p1, team_p2)):
            player = {"name": side_usernames[side], "team": team}
            await self.write(f">player {side} {json.dumps(player)}")
        await self.write(">show-openteamsheets")

    # rebuild a checkpointed battle, return LocalBattle object
    @classmethod
    async def restore(cls, bridge, players, record, battle_id, reseed=None, auto=None):
        lb = cls(bridge, players, record["format"], battle_id, auto=auto)
        with lb.passive():
            for line in record["input_log"].split("\n"):
                if not line.startswith(">") or line.startswith(">version"):
                    continue
                await lb._replay_line(line)
                if line.startswith(">player p2"):
                    await lb.write(">show-openteamsheets")
            for s in sides:
                lb.pending[s] = None
        lb._mark_teampreview(record["input_log"])
        if reseed is not None:
            await lb.write(f">reseed {reseed}")
        lb.round_fresh = True
        return lb
    
    async def _replay_line(self, line):
        # write one line of the input log back into the sim
        # showdown logs some choices in a form it rejects on replay, so try to repair them
        events = self.bridge.write(self.battle_id, line)
        for _ in range(4): 
            err = choice_error(events)
            if err is None:
                break
            fixed = repair_choice(line, err)
            if fixed is None:
                break
            line = fixed
            events = self.bridge.write(self.battle_id, line)
        err = choice_error(events)
        if err is not None:
            raise simError(f"replay of {self.battle_id} diverged at {line!r}: {err}")
        await self._process(events)

    def _mark_teampreview(self, input_log):
        # poke-env's Player.teampreview marks which mons were brought to the battle
        # during replay the players never ran teampreview, so do it by hand from the log
        # e.g. ">p1 team 1234" -> mons 1,2,3,4 were selected
        for line in input_log.split("\n"):
            for side in sides:
                prefix = f">{side} team "
                if line.startswith(prefix):
                    mons = list(self.battle(side).team.values())
                    for c in line[len(prefix) :].strip():
                        if c.isdigit() and 0 < int(c) <= len(mons):
                            mons[int(c) - 1]._selected_in_teampreview = True

    async def prompt(self, side):
        
        player = self.players[side]
        await type(player)._handle_battle_request(player, self.battle(side))

    def submit(self, side, order_message):
        # queue an order for a side by hand (e.g. from the DQN)
        self.pending[side] = order_message.removeprefix("/choose ").strip()

    async def send_pending(self):
        # send whatever orders are queued right now, returns whether anything was sent
        queued = []
        for s in sides:
            if self.pending[s] is not None:
                queued.append((s, self.pending[s]))
        for side, msg in queued:
            self.pending[side] = None
            self.round_fresh = False
            await self.write(f">{side} {msg}")
            if self.round_fresh or self.ended:
                break
        return len(queued) > 0

    async def run(self, stop=None, max_turns=100):
        # let both players play until:
        #   the battle ends-> "ended"
        #   stop(self) is True at start of a round -> "stopped"
        #   the turn cap is hit -> "turn_cap"
        #   nobody has anything to send -> "stuck"
        while not self.ended:
            if self.round_fresh and stop is not None and stop(self):
                return "stopped"
            if self.battle("p1").turn > max_turns:
                return "turn_cap"
            sent = await self.send_pending()
            if not sent:
                return "stuck"
        return "ended"

    async def input_log(self):
        events = self.bridge.write(self.battle_id, ">requestlog")
        logs = []
        for e in events:
            if e[0] == "log":
                logs.append(e[1])
        if not logs:
            raise simError("no input log returned")
        return logs[0]

    async def close(self):
        # free the sim battle and make both players forget it
        try:
            self.bridge.close(self.battle_id)
        finally:
            for side, player in self.players.items():
                self._set_request_handling(side, True)
                player._battles.pop(self.tag, None)
                # poke-env counts open battles in this queue, empty it so it doesn't fill up
                queue = player._battle_count_queue
                while not queue.empty():
                    queue.get_nowait()
                    queue.task_done()

    # endgame detection
    def requests(self):
        # latest request per side, as dicts
        out = {}
        for s, r in self.last_request.items():
            if r:
                out[s] = json.loads(r)
            else:
                out[s] = {}
        return out

    @staticmethod
    def _alive(mon):
        return not mon["condition"].endswith(" fnt") and mon["condition"] != "0"

    def is_2v2_endgame(self):
        # both sides have exactly 2 non-fainted mons, both on the field,
        # and both have to pick moves (no team preview / forced switch / waiting)
        for req in self.requests().values():
            if not req or req.get("wait") or req.get("teamPreview"):
                return False
            if req.get("forceSwitch"):
                return False
            alive = [m for m in req["side"]["pokemon"] if self._alive(m)]
            if len(alive) != 2:
                return False
            if not all(m["active"] for m in alive):
                return False
        return True

    def endgame_summary(self):
        # the remaining mons per side, saved with the checkpoint
        out = {}
        for side, req in self.requests().items():
            out[side] = []
            for m in req["side"]["pokemon"]:
                if self._alive(m):
                    out[side].append({
                        "ident": m["ident"],
                        "details": m["details"],
                        "condition": m["condition"],
                        "item": m.get("item"),
                        "terastallized": m.get("terastallized"),
                    })
        return out

    def state_hash(self):
        # hash of both sides' latest raw requests, should match after an exact restore
        h = hashlib.sha1()
        for side in sides:
            h.update((self.last_request[side] or "").encode())
        return h.hexdigest()