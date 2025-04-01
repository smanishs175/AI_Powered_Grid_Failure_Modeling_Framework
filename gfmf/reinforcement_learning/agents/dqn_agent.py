"""
Deep Q-Network (DQN) Implementation.

This module implements a DQN agent with several enhancements:
- Double DQN to reduce overestimation bias
- Prioritized Experience Replay for efficient learning
- Dueling network architecture for better value estimation
"""

import os
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import pickle
import copy

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Stores transitions and samples with priority based on TD error.
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialize buffer with given capacity and parameters.
        
        Args:
            capacity (int): Maximum capacity of the buffer
            alpha (float): How much prioritization to use (0=none, 1=full)
            beta_start (float): Initial value of beta for importance sampling
            beta_frames (int): Number of frames over which to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        self.frame = 1
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a new experience in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Default max priority for new experiences
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences based on priorities.
        
        Args:
            batch_size (int): Size of batch to sample
            
        Returns:
            tuple: (batch, indices, weights)
        """
        if len(self.buffer) < batch_size:
            return None, None, None
            
        # Calculate current beta
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample experiences and calculate importance sampling weights
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Separate experience components
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        weights = torch.from_numpy(weights).float()
        
        batch = (states, actions, rewards, next_states, dones)
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices (list): Indices of experiences to update
            priorities (list): New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network Architecture.
    
    Separates state value and advantage functions for better learning.
    """
    
    def __init__(self, state_dim, action_dim, hidden_layers=(128, 64, 32)):
        """
        Initialize the Q-Network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_layers (tuple): Sizes of hidden layers
        """
        super(DuelingQNetwork, self).__init__()
        
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
            nn.Linear(hidden_layers[2], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
            nn.Linear(hidden_layers[2], action_dim)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using dueling architecture formula
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class DQNAgent:
    """
    Deep Q-Network Agent implementation.
    
    Implements a DQN agent with Double DQN, Prioritized Experience Replay,
    and Dueling Network Architecture.
    """
    
    def __init__(self, state_dim, action_dim, config=None):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config (dict, optional): Configuration parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing DQN Agent")
        
        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default configuration
        default_config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'buffer_size': 100000,
            'batch_size': 64,
            'update_every': 4,
            'hidden_layers': [128, 64, 32],
            'tau': 0.001,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'prioritized_replay': True,
            'alpha': 0.6,
            'beta_start': 0.4,
            'beta_frames': 100000
        }
        
        # Update with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Set up device (CPU/GPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize Q-Networks
        self.qnetwork_local = DuelingQNetwork(
            state_dim, 
            action_dim, 
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        self.qnetwork_target = DuelingQNetwork(
            state_dim, 
            action_dim, 
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Initialize replay buffer
        if self.config['prioritized_replay']:
            self.memory = PrioritizedReplayBuffer(
                self.config['buffer_size'],
                alpha=self.config['alpha'],
                beta_start=self.config['beta_start'],
                beta_frames=self.config['beta_frames']
            )
        else:
            self.memory = deque(maxlen=self.config['buffer_size'])
            
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Initialize epsilon for epsilon-greedy policy
        self.epsilon = self.config['epsilon_start']
        
        # Training metrics
        self.metrics = {
            'losses': [],
            'rewards': [],
            'epsilons': []
        }
        
        self.logger.info(f"DQN Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def step(self, state, action, reward, next_state, done):
        """
        Update agent's knowledge after taking a step in the environment.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience in replay memory
        if self.config['prioritized_replay']:
            self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append(Experience(state, action, reward, next_state, done))
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.config['update_every']
        if self.t_step == 0 and len(self.memory) >= self.config['batch_size']:
            self._learn()
    
    def act(self, state, eval_mode=False):
        """
        Choose an action given the current state.
        
        Args:
            state: Current state
            eval_mode (bool): If True, use greedy policy instead of epsilon-greedy
            
        Returns:
            int: Selected action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Set networks to evaluation mode
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        # Set networks back to training mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_dim))
        else:
            return np.argmax(action_values.cpu().data.numpy())
    
    def _learn(self):
        """
        Update value parameters using given batch of experience tuples.
        """
        # Sample from replay buffer
        if self.config['prioritized_replay']:
            batch, indices, weights = self.memory.sample(self.config['batch_size'])
            if batch is None:
                return
                
            states, actions, rewards, next_states, dones = batch
            weights = weights.to(self.device)
        else:
            if len(self.memory) < self.config['batch_size']:
                return
                
            experiences = random.sample(self.memory, self.config['batch_size'])
            
            states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
            weights = torch.ones_like(rewards).to(self.device)
        
        # Double DQN Update
        # Get best actions for next states according to local model
        q_local_argmax = self.qnetwork_local(next_states).detach().argmax(dim=1, keepdim=True)
        
        # Get Q values for these actions from target model
        q_targets_next = self.qnetwork_target(next_states).gather(1, q_local_argmax)
        
        # Compute Q targets for current states
        q_targets = rewards + (self.config['gamma'] * q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute TD errors
        td_errors = torch.abs(q_targets - q_expected).detach().cpu().numpy()
        
        # Compute loss with importance sampling weights
        loss = (weights * F.mse_loss(q_expected, q_targets, reduction='none')).mean()
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients if needed
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        
        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.config['tau'])
        
        # Update priorities in replay buffer
        if self.config['prioritized_replay']:
            self.memory.update_priorities(indices, (td_errors + 1e-5).flatten())
        
        # Update epsilon
        self.epsilon = max(self.config['epsilon_end'], self.epsilon * self.config['epsilon_decay'])
        
        # Record metrics
        self.metrics['losses'].append(loss.item())
        self.metrics['epsilons'].append(self.epsilon)
    
    def _soft_update(self, local_model, target_model, tau):
        """
        Soft update of target network parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: Source model
            target_model: Target model
            tau (float): Interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save_policy(self, filepath):
        """
        Save the trained policy.
        
        Args:
            filepath (str): Path to save the policy
            
        Returns:
            str: Path where policy was saved
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save model weights and config
        state = {
            'qnetwork': self.qnetwork_local.state_dict(),
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'metrics': self.metrics,
            'epsilon': self.epsilon
        }
        
        torch.save(state, filepath)
        
        self.logger.info(f"Saved DQN policy to {filepath}")
        
        return filepath
    
    def load_policy(self, filepath):
        """
        Load a trained policy.
        
        Args:
            filepath (str): Path to the saved policy
            
        Returns:
            DQNAgent: Self with loaded policy
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Policy file not found: {filepath}")
        
        # Load on CPU explicitly for compatibility
        state = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Update instance variables
        self.config = state['config']
        self.state_dim = state['state_dim']
        self.action_dim = state['action_dim']
        self.metrics = state.get('metrics', self.metrics)
        self.epsilon = state.get('epsilon', self.config['epsilon_end'])
        
        # Recreate networks with loaded dimensions
        self.qnetwork_local = DuelingQNetwork(
            self.state_dim, 
            self.action_dim, 
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        self.qnetwork_target = DuelingQNetwork(
            self.state_dim, 
            self.action_dim, 
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Load weights
        self.qnetwork_local.load_state_dict(state['qnetwork'])
        self.qnetwork_target.load_state_dict(state['qnetwork'])
        
        # Set to evaluation mode
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        
        self.logger.info(f"Loaded DQN policy from {filepath}")
        
        return self
    
    def train(self, env, num_episodes=1000, max_steps=None, eval_freq=100):
        """
        Train the agent on the given environment.
        
        Args:
            env: OpenAI Gym environment
            num_episodes (int): Number of episodes to train
            max_steps (int): Maximum steps per episode
            eval_freq (int): How often to evaluate the agent
            
        Returns:
            dict: Training metrics
        """
        if max_steps is None:
            max_steps = env.max_steps if hasattr(env, 'max_steps') else 1000
            
        self.logger.info(f"Training DQN agent for {num_episodes} episodes, max {max_steps} steps per episode")
        
        rewards = []
        avg_rewards = []
        best_avg_reward = -float('inf')
        best_weights = None
        
        for episode in range(1, num_episodes+1):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                self.step(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
            avg_reward = np.mean(rewards[-100:])  # Moving average of last 100 episodes
            avg_rewards.append(avg_reward)
            
            # Record metrics
            self.metrics['rewards'].append(episode_reward)
            
            # Store best weights
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_weights = copy.deepcopy(self.qnetwork_local.state_dict())
            
            # Log progress
            if episode % 10 == 0:
                self.logger.info(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | "
                                f"Epsilon: {self.epsilon:.4f}")
            
            # Evaluate agent
            if episode % eval_freq == 0:
                eval_reward = self.evaluate(env, num_episodes=5)
                self.logger.info(f"Evaluation at episode {episode}: {eval_reward:.2f}")
        
        # Restore best weights
        if best_weights is not None:
            self.qnetwork_local.load_state_dict(best_weights)
            self.qnetwork_target.load_state_dict(best_weights)
            
        self.logger.info(f"Training completed. Best average reward: {best_avg_reward:.2f}")
        
        return {
            'rewards': rewards,
            'avg_rewards': avg_rewards,
            'best_avg_reward': best_avg_reward,
            'losses': self.metrics['losses'],
            'epsilons': self.metrics['epsilons']
        }
    
    def evaluate(self, env, num_episodes=10):
        """
        Evaluate the agent on the given environment.
        
        Args:
            env: OpenAI Gym environment
            num_episodes (int): Number of episodes to evaluate
            
        Returns:
            float: Average reward over evaluation episodes
        """
        rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.act(state, eval_mode=True)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards)
        self.logger.info(f"Evaluation over {num_episodes} episodes: {avg_reward:.2f}")
        
        return avg_reward
