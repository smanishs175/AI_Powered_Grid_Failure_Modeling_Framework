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
    
    def __init__(self, state_dim, action_dim, config=None):
        """
        Initialize the SAC agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config (dict, optional): Configuration parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing SAC Agent")
        
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
            
        self.logger.info(f"Training SAC agent for {num_episodes} episodes, max {max_steps} steps per episode")
        
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
                best_weights = {
                    'actor': copy.deepcopy(self.actor.state_dict()),
                    'critic': copy.deepcopy(self.critic.state_dict()),
                    'critic_target': copy.deepcopy(self.critic_target.state_dict())
                }
            
            # Log progress
            if episode % 10 == 0:
                self.logger.info(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | "
                                f"Alpha: {self.alpha:.4f} | Updates: {self.updates}")
            
            # Evaluate agent
            if episode % eval_freq == 0:
                eval_reward = self.evaluate(env, num_episodes=5)
                self.logger.info(f"Evaluation at episode {episode}: {eval_reward:.2f}")
        
        # Restore best weights
        if best_weights is not None:
            self.actor.load_state_dict(best_weights['actor'])
            self.critic.load_state_dict(best_weights['critic'])
            self.critic_target.load_state_dict(best_weights['critic_target'])
            
        self.logger.info(f"Training completed. Best average reward: {best_avg_reward:.2f}")
        
        return {
            'rewards': rewards,
            'avg_rewards': avg_rewards,
            'best_avg_reward': best_avg_reward,
            'actor_losses': self.metrics['actor_losses'],
            'critic_losses': self.metrics['critic_losses'],
            'alpha_values': self.metrics['alpha_values'] if self.config['auto_entropy_tuning'] else None
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
