import numpy as np
import random
from collections import deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Implements a learning agent to solve the environment

    Args:
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        train (bool, optional): whether the agent should be used in training mode
        device (str, optional): device to host model
        lin_features (int, optional): number of nodes in hidden layers
        bn (bool, optional): whether batch norm should be added after hidden layer
        dropout_prob (float, optional): dropout probability of hidden layers
        buffer_size (int, optional): replay buffer size
        batch_size (int, optional):
        lr (float, optional): learning rate
        gamma (float, optional): discount factor
        tau (float, optional): for soft update of target parameters
        update_freq (int, optional): number of steps between each update
    """

    def __init__(self, state_size, action_size, train=False, device=None,
                 lin_feats=64, bn=False, dropout_prob=0, buffer_size=1e5,
                 batch_size=64, lr=5e-4, gamma=0.99, tau=1e-3, update_freq=4):

        self.state_size = state_size
        self.action_size = action_size
        self.train = train
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size,
                                       lin_feats, bn, dropout_prob).to(self.device)

        # Training mode attributes
        if self.train:
            self.bs = batch_size
            self.gamma = gamma
            self.tau = tau
            self.update_freq = update_freq
            self.qnetwork_target = QNetwork(state_size, action_size,
                                            lin_feats, bn, dropout_prob).to(self.device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
            # Replay memory
            self.memory = ReplayBuffer(action_size, buffer_size, self.bs, self.device)
            # Initialize time step (for updating every UPDATE_EVERY steps)
            self.t_step = 0

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        Returns:
            int: selected action index
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        if self.train:
            self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if not self.train or random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """Let the agent perform a training step

        Args:
            state (array_like): current state
            action (int): action index
            reward (float): received reward
            next_state (array_like): next state
            done (bool): whether the episode is over
        """

        if not self.train:
            raise ValueError('agent cannot be trained if constructor argument train=False')
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.bs:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        if not self.train:
            raise ValueError('agent cannot be trained if constructor argument train=False')

        states, actions, rewards, next_states, dones = experiences

        # Get the target Q values
        Q_targets_next = self.qnetwork_target.forward(next_states).detach().max(dim=1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (torch.nn.Module): weights will be copied from
            target_model (torch.nn.Module): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # Inplace interpolation
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Args:
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        device (torch.device): device to use for tensor operations
    """

    def __init__(self, action_size, buffer_size, batch_size, device):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory.

        Returns:
            tuple: tuple of vectorized sampled experiences
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
