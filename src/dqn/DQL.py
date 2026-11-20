import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from collections import Counter, deque
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Hyperparameters import Hyperparameters
from Agent import Agent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "endgame_env"))
from endgame_env import EndgameDataset, EndgameEnv  # noqa: E402
from local_sim import simError  # noqa: E402


class DQL():
    def __init__(self, hyperparameters:Hyperparameters, train_mode):

        torch.set_num_threads(1)

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters
        Path(self.hp.run_dir).mkdir(parents=True, exist_ok=True)

        # Load the environment
        # train: train split, opponent = the agent that originally played the other side
        # test:  held-out val split, optionally against a fixed opponent (hp.test_opponent)
        if train_mode:
            split, opponent = "train", None
        else:
            split, opponent = "val", self.hp.test_opponent
        data = EndgameDataset(self.hp.data_dir, split=split, max_records=self.hp.max_records)
        print(f"Loaded {len(data)} {split} checkpoints from {data.data_dir}")
        self.env = EndgameEnv(
            data,
            opponent=opponent,
            max_turns=self.hp.max_turns,
            hp_shaping=self.hp.hp_shaping,
            ally_targets=self.hp.ally_targets,
        )

        # Initiate the Agent
        self.agent = Agent(env = self.env, hyperparameters = self.hp)


    def feature_representation(self, state:np.ndarray):
        """
        represent the feature of the state
        The env already returns the encoded observation (see endgame_env/encoding.py);
        the categorical ids in it are embedded inside the DQN
        """
        return torch.as_tensor(state, dtype=torch.float32, device=self.agent.device)

    def train(self):
        """
        Traing the DQN via DQL
        """

        total_steps = 0
        self.collected_rewards = []
        self.outcomes = []
        last_100 = deque(maxlen=100)
        skipped = 0

        # Training loop
        for episode in range(1, self.hp.num_episodes+1):

            # sample a new state (a random checkpoint, side and sim seed)
            state, info = self.env.reset(seed=self.hp.seed if episode == 1 else None)
            state = self.feature_representation(state)
            mask = info["action_mask"]
            ended = False
            truncated = False
            step_size = 0
            episode_reward = 0

            try:
                while not ended and not truncated:
                    # find <action> via epsilon greedy (over legal actions)
                    action = self.agent.epsilon_greedy(state, mask)

                    # find nest state and reward
                    next_state, reward, ended, truncated, info = self.env.step(action)

                    # Find the feature of <next_state> using <self.feature_representation>
                    next_state = self.feature_representation(next_state)
                    next_mask = info["action_mask"]
                    # Put it into replay buffer (only a real battle end is terminal;
                    # a truncated episode still bootstraps from next_state)
                    self.agent.replay_buffer.store(state, action, next_state, reward, ended, next_mask)
                    total_steps += 1

                    if len(self.agent.replay_buffer) >= max(self.hp.learning_starts, self.hp.batch_size):

                        # use <self.agent.apply_SGD> implementation to update the online DQN
                        self.agent.apply_SGD(ended or truncated)

                        # Update target-network weights
                        if total_steps % self.hp.targetDQN_update_rate == 0:
                            # Copy the online DQN into the Target DQN
                            self.agent.update_target()

                    state = next_state
                    mask = next_mask
                    episode_reward += reward
                    step_size +=1
            except simError as e:
                # a rare stuck battle: drop the rest of this episode and move on
                skipped += 1
                print(f"Episode {episode}: skipped ({e})")
                continue

            outcome = "win" if info["won"] else "loss" if info["won"] is False else "trunc" if truncated else "tie"
            self.collected_rewards.append(episode_reward)
            self.outcomes.append(outcome)
            last_100.append(outcome == "win")

            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()

            # Print Results of the Episode
            printout = (f"Episode: {episode}, "
                      f"Total Time Steps: {total_steps}, "
                      f"Trajectory Length: {step_size}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, "
                      f"Outcome: {outcome}, "
                      f"Win Rate (last 100): {np.mean(last_100):.2f}, "
                      f"Epsilon: {self.agent.epsilon:.2f}")
            print(printout)

            if episode % self.hp.checkpoint_every == 0:
                self.agent.save(f"{self.hp.run_dir}/ckpt_{episode}.pth")

        if skipped:
            print(f"{skipped} episodes skipped because of simulator errors")
        self.agent.save(self.hp.save_path + '.pth')
        np.save(f"{self.hp.run_dir}/rewards.npy", np.array(self.collected_rewards))
        np.save(f"{self.hp.run_dir}/losses.npy", np.array(self.agent.loss_list))
        self.plot_learning_curves()
        self.env.close()


    def play(self):
        """
        play with the learned policy
        You can only run it if you already have trained the DQN and saved its weights as .pth file
        Plays greedily on the held-out val checkpoints and reports the win rate
        """

        # Load the trained DQN
        self.agent.onlineDQN.load_state_dict(torch.load(self.hp.RL_load_path, map_location=torch.device(self.agent.device)))
        self.agent.onlineDQN.eval()

        outcomes = Counter()
        # Playing
        for episode in range(1, self.hp.num_test_episodes+1):
            state, info = self.env.reset(seed=episode)
            mask = info["action_mask"]
            ended = False
            truncated = False
            step_size = 0
            episode_reward = 0

            while not ended and not truncated:
                # Find the feature of <state> using <self.feature_representation>
                # Act greedy (over legal actions) and find <action>
                state = self.feature_representation(state)
                action = self.agent.greedy(state, mask)

                next_state, reward, ended, truncated, info = self.env.step(action)

                state = next_state
                mask = info["action_mask"]
                episode_reward += reward
                step_size += 1

            outcome = "win" if info["won"] else "loss" if info["won"] is False else "trunc" if truncated else "tie"
            outcomes[outcome] += 1
            # Print Results of Episode
            printout = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, "
                      f"Outcome: {outcome}, "
                      f"Opponent: {info['opponent']}")
            print(printout)

        n = self.hp.num_test_episodes
        print(f"\nWin rate: {outcomes['win'] / n:.3f}  "
              f"(win {outcomes['win']}, loss {outcomes['loss']}, "
              f"trunc {outcomes['trunc']}, tie {outcomes['tie']}, of {n})")
        self.env.close()

    def plot_learning_curves(self):
        # Calculate the Moving Average over last 100 episodes
        moving_average = np.convolve(self.collected_rewards, np.ones(100)/100, mode='valid')

        plt.figure()
        plt.title("Reward")
        plt.plot(self.collected_rewards, label='Reward', color='gray')
        plt.plot(np.arange(99, 99 + len(moving_average)), moving_average, label='Moving Average', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        # Save the figure
        plt.savefig(f'{self.hp.run_dir}/Reward_vs_Episode.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.clf()
        plt.close()


        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_list, label='Loss', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Training Loss")

        # Save the figure
        plt.grid(True)
        plt.savefig(f'{self.hp.run_dir}/Learning_Curve.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.clf()
        plt.close()
