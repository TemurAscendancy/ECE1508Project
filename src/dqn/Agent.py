import sys
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from collections import deque

from Hyperparameters import Hyperparameters

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "endgame_env"))
from encoding import categorical_idx, observation_length, vocab  # noqa: E402


class Replay_Buffer():
    """
    Experience Replay Buffer to store experiences
    (same as assignment 3, plus the legal-action mask of the next state)
    """
    def __init__(self, size, device):


        self.device = device

        self.size = size # size of the buffer

        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.terminals = deque(maxlen=size)
        self.next_masks = deque(maxlen=size)


    def store(self, state, action, next_state, reward, terminal, next_mask):
        """
        Store experiences to their respective queues
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.next_masks.append(next_mask)


    def sample(self, batch_size):
        """
        Sample from the buffer
        """
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=self.device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor([self.terminals[i] for i in indices], dtype=torch.bool, device=self.device)
        next_masks = torch.as_tensor(np.stack([self.next_masks[i] for i in indices]), dtype=torch.bool, device=self.device)

        return states, actions, next_states, rewards, terminals, next_masks


    def __len__(self):
        return len(self.terminals)


class DQN(nn.Module):
    """
    The Deep Q-Network (DQN) model

    The observation holds 24 categorical ids (ability, item, move1-4 for each of
    the 4 active Pokemon, see encoding.categorical_idx). Those go through
    nn.Embedding lookup tables (the move table is shared by the 4 move slots);
    the embeddings are concatenated with the remaining continuous features and
    fed to an MLP that outputs one Q-value per joint action.
    """

    def __init__(self, num_actions, feature_size, embed_dim = 16):
        super(DQN, self).__init__()
        cont_idx = np.setdiff1d(np.arange(feature_size), categorical_idx)
        self.register_buffer("cat_idx", torch.as_tensor(categorical_idx, dtype=torch.long), persistent=False)
        self.register_buffer("cont_idx", torch.as_tensor(cont_idx, dtype=torch.long), persistent=False)

        self.ability_emb = nn.Embedding(vocab["ability"], embed_dim)
        self.item_emb = nn.Embedding(vocab["item"], embed_dim)
        self.move_emb = nn.Embedding(vocab["move"], embed_dim)

        in_size = len(categorical_idx) * embed_dim + len(cont_idx)
        self.network = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        # x: (feature_size,) or (batch, feature_size)
        ids = x[..., self.cat_idx].long().reshape(*x.shape[:-1], 4, 6)  # 4 mons x (ability, item, move1-4)
        emb = torch.cat([
            self.ability_emb(ids[..., 0:1]),
            self.item_emb(ids[..., 1:2]),
            self.move_emb(ids[..., 2:6]),
        ], dim=-2)                                                         # (..., 4, 6, embed_dim)
        emb = emb.flatten(start_dim=-3)                                    # (..., 24 * embed_dim)
        return self.network(torch.cat([emb, x[..., self.cont_idx]], dim=-1))


class Agent:
    """
    Implementing Agent DQL Algorithm
    """

    def __init__(self, env:gym.Env, hyperparameters:Hyperparameters, device = False):

        # Some Initializations
        if not device:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters


        self.epsilon = 0.99
        self.loss_list = []
        self.current_loss = 0
        self.episode_counts = 0

        self.action_space  = env.action_space
        self.feature_space = env.observation_space
        self.replay_buffer = Replay_Buffer(self.hp.buffer_size, device = self.device)

        # Initiate the online and Target DQNs
        # per_mon: one Q head per active pokemon (2 x 25 outputs) instead of one
        # Q-value per joint action (625 outputs), see joint_q / apply_SGD
        num_actions = self.action_space.n
        self.slot_actions = int(round(np.sqrt(num_actions)))
        if self.hp.per_mon:
            num_actions = 2 * self.slot_actions
        feature_size = self.feature_space.shape[0]
        self.onlineDQN = DQN(num_actions = num_actions, feature_size = feature_size)
        self.onlineDQN.to(self.device)
        self.targetDQN = DQN(num_actions = num_actions, feature_size = feature_size)
        self.targetDQN.to(self.device)
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())

        self.loss_function = nn.MSELoss()

        # set the optimizer to Adam and call it <self.optimizer>, i.e., self.optimizer = optim.Adam()
        self.optimizer = torch.optim.Adam(self.onlineDQN.parameters(), lr=self.hp.learning_rate)


    def joint_q(self, output):
        if not self.hp.per_mon:
            return output
        q = output.reshape(*output.shape[:-1], 2, self.slot_actions)
        return (q[..., 0, :, None] + q[..., 1, None, :]).flatten(start_dim=-2)

    def masked_argmax(self, q_values, mask):
        """
        argmax over the legal actions only (mask: int8/bool array, 1 = legal)
        """
        mask = torch.as_tensor(mask, dtype=torch.bool, device=q_values.device)
        return torch.argmax(q_values.masked_fill(~mask, -float("inf"))).item()


    def epsilon_greedy(self, state, mask):
        """
        Implement epsilon-greedy policy (over the legal actions)
        """
        # This function should return the action chosen by epsilon greedy algorithm #
        with torch.no_grad():
            output = self.joint_q(self.onlineDQN(state)) # should be (num_actions,)
            random_value = np.random.rand()
            if random_value < self.epsilon:
                # uniform over the legal actions
                return int(self.action_space.sample(mask=mask))
            else:
                # return the index of the max legal action value
                return self.masked_argmax(output, mask)

    def greedy(self, state, mask):
        """
        Implement greedy policy (over the legal actions)
        """
        # This function should return the action chosen by greedy algorithm #
        with torch.no_grad():
            output = self.joint_q(self.onlineDQN(state)) # should be (num_actions,)
            return self.masked_argmax(output, mask)



    def apply_SGD(self, ended):
        """
        Train DQN
            ended (bool): Indicates whether the episode meets a terminal state or not. If ended,
            calculate the loss of the episode.
        """

        # Sample from the replay buffer
        states, actions, next_states, rewards, terminals, next_masks = self.replay_buffer.sample(self.hp.batch_size)

        # Q heads: joint = 1 head over the 625 joint actions
        # per_mon = 2 heads (one per pokemon) over its 25 actions. Each pokemon is its own
        # Q-learner on the team reward, its ally's choice is just part of the environment
        if self.hp.per_mon:
            S = self.slot_actions
            actions = torch.stack([actions // S, actions % S], dim=1)                     # batch_size x 2
            joint_masks = next_masks.reshape(-1, S, S)
            next_masks = torch.stack([joint_masks.any(2), joint_masks.any(1)], dim=1)  # batch_size x 2 x S
        else:
            actions = actions.unsqueeze(1)                                               # batch_size x 1
            next_masks = next_masks.unsqueeze(1)                                         # batch_size x 1 x 625
        heads = lambda net, x: net(x).reshape(next_masks.shape)                          # batch_size x heads x actions per head
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)

        # Compute <Q_hat> using the online DQN
        # states should be batch_size x feature_size, rewards is batch_size x 1, actions is batch_size x heads
        Q_hat = heads(self.onlineDQN, states).gather(2, actions.unsqueeze(2)).squeeze(2) # should be batch_size x heads

        with torch.no_grad():
            # Compute the maximum Q-value (over legal next actions) for off-policy update and call it <next_target_q_value>
            next_target_q = heads(self.targetDQN, next_states) # should be batch_size x heads x actions per head
            if self.hp.double_dqn:
                # Double DQN: the online DQN picks the best legal next action and the
                # target DQN evaluates it (reduces the max-operator overestimation)
                next_online_q = heads(self.onlineDQN, next_states).masked_fill(~next_masks, -1e9)
                next_actions = torch.argmax(next_online_q, dim=2, keepdim=True)
                next_target_q_value = next_target_q.gather(2, next_actions).squeeze(2) # should be batch_size x heads
            else:
                next_target_q = next_target_q.masked_fill(~next_masks, -1e9) # illegal next actions can't be the max
                next_target_q_value, _ = torch.max(next_target_q, dim=2) # should be batch_size x heads


        next_target_q_value[terminals.expand_as(next_target_q_value)] = 0 # Set Q-value for terminal states to zero (their mask is all zeros)
        # Compute the Q-estimator and call it <y>
        y = rewards + self.hp.discount_factor * next_target_q_value
        loss = self.loss_function(Q_hat, y) # Compute the loss

        # Update the running loss and learned counts for logging and plotting
        self.current_loss += loss.item()
        self.episode_counts += 1

        if ended:
            episode_loss = self.current_loss / self.episode_counts # Average loss per episode
            # Track the loss for final graph
            self.loss_list.append(episode_loss)
            self.current_loss = 0
            self.episode_counts = 0

        # Apply backpropagation
        self.optimizer.zero_grad()
        loss.backward()



        # Clip the gradients
        # It's just to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(self.onlineDQN.parameters(), 2)

        # Update DQN by using the optimizer: <self.optimizer>
        self.optimizer.step()

    def update_target(self):
        """
        Update the target network
        """
        # Copy the online DQN into target DQN
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())


    def update_epsilon(self):
        """
        reduce epsilon by the decay factor
        """
        # Gradually reduce epsilon
        self.epsilon = max(0.01, self.epsilon * self.hp.epsilon_decay)


    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention
        This can be used for later test of the trained agent
        """
        torch.save(self.onlineDQN.state_dict(), path)
