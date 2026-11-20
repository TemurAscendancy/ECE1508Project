from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"


class Hyperparameters():
    def __init__(self, data_dir = "~/vgc_endgames_ma"):
        self.data_dir = data_dir
        self.run_dir = str(RUNS_DIR)
        self.RL_load_path = f'{self.run_dir}/final_weights.pth'
        self.save_path = f'{self.run_dir}/final_weights'
        self.learning_rate = 5e-4
        self.discount_factor = 0.99
        self.batch_size = 64
        self.targetDQN_update_rate = 500      # in environment steps
        self.num_episodes = 20000
        self.num_test_episodes = 500
        self.epsilon_decay = 0.9997           # per episode: 0.99 -> 0.01 after ~15k episodes
        self.buffer_size = 20000

        # project specific 
        self.learning_starts = 1000           # transitions in the buffer before SGD starts
        self.hp_shaping = 0.0                 # 0 = sparse +-1 reward (see EndgameEnv)
        self.max_turns = 30                   # truncate the episode after this many turns
        self.ally_targets = True              # 625 joint actions (False: 289)
        self.test_opponent = None             # e.g. {"name": "sh"}; None = record's original agent
        self.max_records = None               # cap on checkpoints loaded (None = all)
        self.checkpoint_every = 1000          # save ckpt_<episode>.pth every N episodes
        self.seed = 0

    def change(self, **kwargs):
        '''
        Change any hyperparameter by name, e.g.
        hp.change(num_episodes=5000, epsilon_decay=0.999, batch_size=32)
        '''
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise AttributeError(f"unknown hyperparameter: {name}")
            setattr(self, name, value)
        if "run_dir" in kwargs:
            self.RL_load_path = f'{self.run_dir}/final_weights.pth'
            self.save_path = f'{self.run_dir}/final_weights'
