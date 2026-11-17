- We want a dataset of 2v2 endgames (each side has 2 non feinted pokemon left) to train a DQN/actor-critic/etc.

- Showdown cannot be initialized at an intermediate state just from logs, it needs a checkpoint from a locally played game. So we take 2 random agents from vgc-bench, let them play until we reach 2v2 and save the checkpoint.

- Checkpoint is history of moves made so far, which agents play (so we can recover the same agent when training our model), teams and seed (to reproduce)

Files:

- collect_endgames.py: main file to run data collection. Calls make_agent from vgc_bench_agents.py, loads teams from vgc-bench's RandomTeamBuilder and runs local battles using the LocalBattle class. Can also continue gathering checkpoints from an existing run. 

- local_sim.py: plays battle without a showdown server:
    - SimBridge: start() launches node sim_bridge.js. call is the read/write to the pipe used to communicate to sim_bridge.js.
    - LocalBattle: runs one battle on the SimBridge: writes the players' moves to the sim, feeds the sim's events back to the players, detects the 2v2 endgame and restores a battle from a checkpoint

- sim_bridge.js: node program that hosts the showdown simulation. Loads BattleStream and extractChannelMessages from showdown repo, reads stdin line by line to determine if it needs to make a BattleStream, delete it, or send it input and return the sim's output.

- vgc_bench_agents.py: builds the agents. An agent is described by a spec dict, which is saved in each checkpoint. "random" -> RandomPlayer, "mbp" -> MaxBasePowerPlayer, "sh" -> SimpleHeuristicsPlayer, "bc" -> vgc-bench's PolicyPlayer

AI usage:

- AI was used in understanding the vgc-bench codebase and the necessary protocols to interact with it (i.e. what messages are sent (text matches, the format, etc))

- AI was also used in making the large part of the sim_bridge.js file, as it required direct knowledge of the pokemon-showdown codebase, which would have taken lots of time to understand manually, and is unrelated to RL/course content anyways

- AI was used to help debug compile errors, specifically issues where imports would not resolve and we could only run the code from the vgc-bench repo. It added some file path instantiations and additions of files to library search paths to help us resolve this issue.
