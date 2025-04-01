"""
Soft Actor-Critic (SAC) Implementation.

This module implements a SAC agent with the following features:
- Off-policy algorithm with entropy regularization
- Automatic entropy coefficient tuning
- Twin Q-networks to mitigate overestimation bias
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
from collections import deque
import random
from gym import spaces
import time


class SACCritic(nn.Module):
    """
    Critic network for SAC.
    
    Implements twin Q-networks to reduce overestimation bias.
    """
    
    def __init__(self, state_dim, action_dim, hidden_layers=(128, 64)):
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_layers (tuple): Sizes of hidden layers
        """
        super(SACCritic, self).__init__()
        
        # First Q-network
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], 1)
        )
        
        # Second Q-network
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], 1)
        )
    
    def forward(self, state, action_probs):
        """
        Forward pass through both Q-networks.
        
        Args:
            state: State tensor
            action_probs: Action probability tensor
            
        Returns:
            tuple: (q1_values, q2_values)
        """
        # Concatenate state and action probabilities as input
        x = torch.cat([state, action_probs], dim=1)
        
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        
        return q1, q2
    
    def q1(self, state, action_probs):
        """
        Forward pass through just the first Q-network.
        
        Args:
            state: State tensor
            action_probs: Action probability tensor
            
        Returns:
            torch.Tensor: Q1 values
        """
        x = torch.cat([state, action_probs], dim=1)
        return self.q1_network(x)


class SACActor(nn.Module):
    """
    Actor network for SAC.
    
    Outputs a probability distribution over discrete actions.
    """
    
    def __init__(self, state_dim, action_dim, hidden_layers=(128, 64)):
        """
        Initialize the actor network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_layers (tuple): Sizes of hidden layers
        """
        super(SACActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.action_dim = action_dim
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            torch.Tensor: Action probabilities
        """
        action_probs = self.network(state)
        return action_probs
    
    def sample(self, state):
        """
        Sample an action from the policy.
        
        Args:
            state: State tensor
            
        Returns:
            tuple: (action, log_prob, entropy, action_probs)
        """
        action_probs = self.forward(state)
        
        # Create a categorical distribution
        dist = Categorical(action_probs)
        
        # Sample an action
        action = dist.sample()
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, action_probs


class ReplayBuffer:
    """
    Replay buffer for off-policy learning.
    
    Stores experiences and samples randomly for training.
    """
    
    def __init__(self, capacity):
        """
        Initialize buffer with given capacity.
        
        Args:
            capacity (int): Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
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
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        
        Args:
            batch_size (int): Size of batch to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        
        # Separate components
        states = np.vstack([e[0] for e in batch])
        actions = np.vstack([e[1] for e in batch])
        rewards = np.vstack([e[2] for e in batch])
        next_states = np.vstack([e[3] for e in batch])
        dones = np.vstack([e[4] for e in batch]).astype(np.uint8)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent.
    
    Implements SAC with automatic entropy adjustment for discrete action spaces.
    """
    
    def __init__(self, state_dim_or_env, action_dim=None, config=None):
        """
        Initialize the SAC agent.
        
        Can be initialized in two ways:
        1. With state_dim and action_dim directly
        2. With an environment object
        
        Args:
            state_dim_or_env: Either the dimension of the state space (int) or an environment object
            action_dim (int, optional): Dimension of the action space, required if state_dim_or_env is an int
            config (dict, optional): Configuration parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing SAC Agent")
        
        # Check if first argument is an environment
        if hasattr(state_dim_or_env, 'observation_space') and hasattr(state_dim_or_env, 'action_space'):
            # This is an environment
            self.env = state_dim_or_env
            self.config = config  # Store original config for later
            
            # Get state and action dimensions from environment
            # Get state dimension
            if isinstance(self.env.observation_space, spaces.Box):
                state_dim = int(np.prod(self.env.observation_space.shape))
            else:
                state_dim = self.env.observation_space.n
                
            # Get action dimension
            if isinstance(self.env.action_space, spaces.Discrete):
                action_dim = self.env.action_space.n
            elif isinstance(self.env.action_space, spaces.Box):
                action_dim = int(np.prod(self.env.action_space.shape))
            else:
                raise ValueError(f"Unsupported action space type: {type(self.env.action_space)}")
                
        else:
            # Direct initialization with dimensions
            self.env = None
            state_dim = state_dim_or_env
            if action_dim is None:
                raise ValueError("action_dim must be provided when state_dim is provided directly")
        
        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default configuration
        default_config = {
            'lr': 0.0003,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'auto_entropy_tuning': True,
            'target_entropy': -np.log(1.0/action_dim) * 0.98,  # Slightly lower than uniform
            'buffer_size': 100000,
            'batch_size': 64,
            'update_frequency': 1,
            'hidden_layers': [128, 64]
        }
        
        # Update with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Set up device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor network
        self.actor = SACActor(
            state_dim,
            action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Initialize critic networks
        self.critic = SACCritic(
            state_dim,
            action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Initialize target critic network
        self.critic_target = SACCritic(
            state_dim,
            action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Copy weights to target network
        self._hard_update(self.critic, self.critic_target)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config['lr']
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config['lr']
        )
        
        # Initialize entropy coefficient (alpha)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.config['alpha']
        
        if self.config['auto_entropy_tuning']:
            self.alpha = torch.exp(self.log_alpha).item()
            self.target_entropy = self.config['target_entropy']
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config['lr'])
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(self.config['buffer_size'])
        
        # Training step counter
        self.total_steps = 0
        self.updates = 0
        
        # Training metrics
        self.metrics = {
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'alpha_values': [],
            'q_values': [],
            'rewards': []
        }
        
        self.logger.info(f"SAC Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state, eval_mode=False):
        """
        Choose an action given the current state.
        
        Args:
            state: Current state
            eval_mode (bool): If True, use greedy policy
            
        Returns:
            int: Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if eval_mode:
                action_probs = self.actor(state)
                action = torch.argmax(action_probs, dim=1).item()
                return action
            else:
                action, _, _, _ = self.actor.sample(state)
                return action.item()
    
    def step(self, state, action, reward, next_state, done):
        """
        Store experience and update networks if needed.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience
        self.memory.push(state, action, reward, next_state, done)
        
        # Increment step counter
        self.total_steps += 1
        
        # Update networks if it's time
        if self.total_steps % self.config['update_frequency'] == 0:
            if len(self.memory) >= self.config['batch_size']:
                self._learn()
    
    def _learn(self):
        """
        Update actor and critic networks from experiences.
        """
        # Sample from replay buffer
        experiences = self.memory.sample(self.config['batch_size'])
        if experiences is None:
            return
            
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Create one-hot encoding for actions
        actions_one_hot = F.one_hot(actions.squeeze(1), num_classes=self.action_dim).float()
        
        # Get current action probabilities
        _, _, _, current_action_probs = self.actor.sample(states)
        
        # ----------------------------------------
        # Update critic networks
        # ----------------------------------------
        
        with torch.no_grad():
            # Sample actions from the target actor
            next_actions, next_log_probs, _, next_action_probs = self.actor.sample(next_states)
            
            # Get Q values from target critic
            next_q1, next_q2 = self.critic_target(next_states, next_action_probs)
            
            # Take the minimum of the two Q values
            next_q = torch.min(next_q1, next_q2)
            
            # Compute the target Q value (with entropy)
            target_q = rewards + (1 - dones) * self.config['gamma'] * (next_q - self.alpha * next_log_probs)
        
        # Get current Q values
        current_q1, current_q2 = self.critic(states, actions_one_hot)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ----------------------------------------
        # Update actor network
        # ----------------------------------------
        
        # Get Q value for current policy
        actions_pred, log_probs_pred, _, action_probs_pred = self.actor.sample(states)
        q1_pred = self.critic.q1(states, action_probs_pred)
        
        # Compute actor loss (with entropy)
        actor_loss = (self.alpha * log_probs_pred - q1_pred).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------------------------
        # Update alpha (entropy coefficient)
        # ----------------------------------------
        
        if self.config['auto_entropy_tuning']:
            # Sample new actions (since actor has been updated)
            _, log_probs, _, _ = self.actor.sample(states)
            
            # Compute alpha loss
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            # Update alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update alpha value
            self.alpha = torch.exp(self.log_alpha).item()
            
            # Record alpha loss
            self.metrics['alpha_losses'].append(alpha_loss.item())
            self.metrics['alpha_values'].append(self.alpha)
        
        # ----------------------------------------
        # Update target networks
        # ----------------------------------------
        
        self._soft_update(self.critic, self.critic_target, self.config['tau'])
        
        # Record metrics
        self.metrics['actor_losses'].append(actor_loss.item())
        self.metrics['critic_losses'].append(critic_loss.item())
        self.metrics['q_values'].append(current_q1.mean().item())
        
        self.updates += 1
    
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
    
    def _hard_update(self, local_model, target_model):
        """
        Hard update of target network parameters.
        θ_target = θ_local
        
        Args:
            local_model: Source model
            target_model: Target model
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
    
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
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'metrics': self.metrics,
            'alpha': self.alpha,
            'log_alpha': self.log_alpha.item() if self.config['auto_entropy_tuning'] else None
        }
        
        torch.save(state, filepath)
        
        self.logger.info(f"Saved SAC policy to {filepath}")
        
        return filepath
    
    def load_policy(self, filepath):
        """
        Load a trained policy.
        
        Args:
            filepath (str): Path to the saved policy
            
        Returns:
            SACAgent: Self with loaded policy
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
        self.alpha = state.get('alpha', self.config['alpha'])
        
        if self.config['auto_entropy_tuning'] and 'log_alpha' in state and state['log_alpha'] is not None:
            self.log_alpha = torch.tensor(state['log_alpha'], requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config['lr'])
        
        # Recreate networks with loaded dimensions
        self.actor = SACActor(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        self.critic = SACCritic(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        self.critic_target = SACCritic(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Load weights
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        
        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()
        
        # Recreate optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config['lr']
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config['lr']
        )
        
        self.logger.info(f"Loaded SAC policy from {filepath}")
        
        return self
    
    def train(self, env=None, total_timesteps=1000, eval_freq=100, n_eval_episodes=5):
        """
        Train the agent on the given environment.
        
        Args:
            env: OpenAI Gym environment (if not provided during initialization)
            total_timesteps (int): Total timesteps to train for
            eval_freq (int): How often to evaluate the agent (in timesteps)
            n_eval_episodes (int): Number of episodes to evaluate the agent for
            
        Returns:
            dict: Training metrics
        """
        # Ensure we have an environment - either from initialization or provided
        if env is None:
            if self.env is None:
                raise ValueError("Environment must be provided either during initialization or to the train method")
            env = self.env
            
        self.logger.info(f"Training SAC agent for {total_timesteps} total timesteps")
        
        rewards = []
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        best_avg_reward = -float('inf')
        best_weights = None
        
        # Initial reset
        state, info = env.reset()
        
        start_time = time.time()
        
        for step in range(1, total_timesteps + 1):
            # Select action
            action = self.act(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition and update networks
            self.step(state, action, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            current_episode_reward += reward
            current_episode_length += 1
            
            # Handle episode termination
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                rewards.append(current_episode_reward)
                
                # Reset for next episode
                state, info = env.reset()
                current_episode_reward = 0
                current_episode_length = 0
            
            # Evaluate agent
            if step % eval_freq == 0:
                eval_results = self.evaluate(env, n_eval_episodes)
                mean_reward = eval_results['mean_reward']
                std_reward = eval_results['std_reward']
                self.logger.info(f"Step {step}/{total_timesteps} | Eval reward: {mean_reward:.2f} ± {std_reward:.2f}")
                
                # Store best model weights
                if mean_reward > best_avg_reward:
                    best_avg_reward = mean_reward
                    best_weights = {
                        'actor': copy.deepcopy(self.actor.state_dict()),
                        'critic': copy.deepcopy(self.critic.state_dict()),
                        'critic_target': copy.deepcopy(self.critic_target.state_dict())
                    }
            
            # Log progress
            if step % 1000 == 0:
                avg_reward = np.mean(rewards[-min(100, len(rewards)):]) if rewards else 0
                self.logger.info(f"Step {step}/{total_timesteps} | Avg Episode Reward: {avg_reward:.2f} | Updates: {self.updates}")
        
        # Restore best weights
        if best_weights is not None:
            self.actor.load_state_dict(best_weights['actor'])
            self.critic.load_state_dict(best_weights['critic'])
            self.critic_target.load_state_dict(best_weights['critic_target'])
            
        elapsed_time = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed_time:.2f}s. Best average reward: {best_avg_reward:.2f}")
        
        return {
            'rewards': rewards,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'best_reward': best_avg_reward,
            'time_elapsed': elapsed_time
        }
    
    def evaluate(self, env=None, n_eval_episodes=10):
        """
        Evaluate the agent on the given environment.
        
        Args:
            env: OpenAI Gym environment (if not provided during initialization)
            n_eval_episodes (int): Number of episodes to evaluate
            
        Returns:
            tuple: (mean_reward, std_reward) over evaluation episodes
        """
        # Ensure we have an environment - either from initialization or provided
        if env is None:
            if self.env is None:
                raise ValueError("Environment must be provided either during initialization or to the evaluate method")
            env = self.env
        
        rewards = []
        
        for episode in range(n_eval_episodes):
            episode_reward = 0
            state, info = env.reset()
            done = False
            
            while not done:
                action = self.act(state, eval_mode=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                done = terminated or truncated
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        self.logger.info(f"Evaluation over {n_eval_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'rewards': rewards
        }
