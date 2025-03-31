"""
Twin Delayed DDPG (TD3) Implementation.

This module implements a TD3 agent with the following features:
- Double critic networks to reduce overestimation bias
- Delayed policy updates for stability
- Target policy smoothing for better exploration
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
import random
from collections import deque


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


class Actor(nn.Module):
    """
    Actor network for TD3.
    
    Outputs action probabilities for discrete action space.
    """
    
    def __init__(self, state_dim, action_dim, hidden_layers=(256, 256)):
        """
        Initialize the actor network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_layers (tuple): Sizes of hidden layers
        """
        super(Actor, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            torch.Tensor: Action probabilities
        """
        return self.actor(state)


class Critic(nn.Module):
    """
    Critic network for TD3.
    
    Implements twin Q-networks to reduce overestimation bias.
    """
    
    def __init__(self, state_dim, action_dim, hidden_layers=(256, 256)):
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_layers (tuple): Sizes of hidden layers
        """
        super(Critic, self).__init__()
        
        # Q1 network
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], 1)
        )
        
        # Q2 network
        self.critic2 = nn.Sequential(
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
            tuple: (q1_value, q2_value)
        """
        # Concatenate state and action probabilities
        sa_concat = torch.cat([state, action_probs], dim=1)
        
        q1 = self.critic1(sa_concat)
        q2 = self.critic2(sa_concat)
        
        return q1, q2
    
    def Q1(self, state, action_probs):
        """
        Forward pass through just the first Q-network.
        
        Args:
            state: State tensor
            action_probs: Action probability tensor
            
        Returns:
            torch.Tensor: Q1 value
        """
        sa_concat = torch.cat([state, action_probs], dim=1)
        return self.critic1(sa_concat)


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent.
    
    Implements TD3 with adaptation for discrete action spaces.
    """
    
    def __init__(self, state_dim, action_dim, config=None):
        """
        Initialize the TD3 agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config (dict, optional): Configuration parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing TD3 Agent")
        
        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default configuration
        default_config = {
            'lr': 0.0003,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_freq': 2,
            'buffer_size': 100000,
            'batch_size': 64,
            'exploration_noise': 0.1,
            'hidden_layers': [256, 256],
            'exploration_decay': 0.995,
            'min_exploration_noise': 0.01
        }
        
        # Update with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Set up device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor networks
        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        self.actor_target = Actor(
            state_dim,
            action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Initialize critic networks
        self.critic = Critic(
            state_dim,
            action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        self.critic_target = Critic(
            state_dim,
            action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Copy weights to target networks
        self._hard_update(self.actor, self.actor_target)
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
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(self.config['buffer_size'])
        
        # Training step counter
        self.total_steps = 0
        
        # Current exploration noise
        self.exploration_noise = self.config['exploration_noise']
        
        # Training metrics
        self.metrics = {
            'actor_losses': [],
            'critic_losses': [],
            'q_values': [],
            'rewards': []
        }
        
        self.logger.info(f"TD3 Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state, eval_mode=False):
        """
        Choose an action given the current state.
        
        Args:
            state: Current state
            eval_mode (bool): If True, use deterministic policy
            
        Returns:
            int: Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state)
            
            if eval_mode:
                # Use deterministic policy for evaluation
                action = torch.argmax(action_probs, dim=1).item()
            else:
                # Add exploration noise to action probabilities
                noise = torch.randn_like(action_probs) * self.exploration_noise
                noisy_action_probs = action_probs + noise
                # Ensure valid probability distribution
                noisy_action_probs = F.softmax(noisy_action_probs, dim=1)
                
                # Sample from distribution
                dist = Categorical(noisy_action_probs)
                action = dist.sample().item()
        
        return action
    
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
        
        # Update networks
        if len(self.memory) >= self.config['batch_size']:
            self._learn()
            
            # Decay exploration noise
            self.exploration_noise = max(
                self.config['min_exploration_noise'],
                self.exploration_noise * self.config['exploration_decay']
            )
    
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
        
        # -------------- Update Critic --------------
        
        # Get next action probabilities from target actor
        with torch.no_grad():
            next_action_probs = self.actor_target(next_states)
            
            # Add noise for target policy smoothing
            noise = torch.randn_like(next_action_probs) * self.config['policy_noise']
            noise = noise.clamp(-self.config['noise_clip'], self.config['noise_clip'])
            
            noisy_next_action_probs = next_action_probs + noise
            # Ensure valid probability distribution
            noisy_next_action_probs = F.softmax(noisy_next_action_probs, dim=1)
            
            # Get target Q values
            target_q1, target_q2 = self.critic_target(next_states, noisy_next_action_probs)
            
            # Use minimum of two Q values to reduce overestimation
            target_q = torch.min(target_q1, target_q2)
            
            # Compute target value with discount
            target_value = rewards + (1 - dones) * self.config['gamma'] * target_q
        
        # Get current Q values
        current_q1, current_q2 = self.critic(states, actions_one_hot)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Record metrics
        self.metrics['critic_losses'].append(critic_loss.item())
        self.metrics['q_values'].append(current_q1.mean().item())
        
        # -------------- Update Actor (delayed) --------------
        
        # Delayed policy updates
        if self.total_steps % self.config['policy_freq'] == 0:
            # Get action probabilities from current actor
            actor_action_probs = self.actor(states)
            
            # Compute actor loss (maximize Q value)
            actor_loss = -self.critic.Q1(states, actor_action_probs).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.critic, self.critic_target, self.config['tau'])
            self._soft_update(self.actor, self.actor_target, self.config['tau'])
            
            # Record metrics
            self.metrics['actor_losses'].append(actor_loss.item())
    
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
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'metrics': self.metrics,
            'exploration_noise': self.exploration_noise
        }
        
        torch.save(state, filepath)
        
        self.logger.info(f"Saved TD3 policy to {filepath}")
        
        return filepath
    
    def load_policy(self, filepath):
        """
        Load a trained policy.
        
        Args:
            filepath (str): Path to the saved policy
            
        Returns:
            TD3Agent: Self with loaded policy
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
        self.exploration_noise = state.get('exploration_noise', self.config['min_exploration_noise'])
        
        # Recreate networks
        self.actor = Actor(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        self.actor_target = Actor(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        self.critic = Critic(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        self.critic_target = Critic(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Load weights
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.critic_target.load_state_dict(state['critic_target'])
        
        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
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
        
        self.logger.info(f"Loaded TD3 policy from {filepath}")
        
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
            
        self.logger.info(f"Training TD3 agent for {num_episodes} episodes, max {max_steps} steps per episode")
        
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
                    'actor_target': copy.deepcopy(self.actor_target.state_dict()),
                    'critic_target': copy.deepcopy(self.critic_target.state_dict())
                }
            
            # Log progress
            if episode % 10 == 0:
                self.logger.info(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | "
                                f"Noise: {self.exploration_noise:.4f}")
            
            # Evaluate agent
            if episode % eval_freq == 0:
                eval_reward = self.evaluate(env, num_episodes=5)
                self.logger.info(f"Evaluation at episode {episode}: {eval_reward:.2f}")
        
        # Restore best weights
        if best_weights is not None:
            self.actor.load_state_dict(best_weights['actor'])
            self.critic.load_state_dict(best_weights['critic'])
            self.actor_target.load_state_dict(best_weights['actor_target'])
            self.critic_target.load_state_dict(best_weights['critic_target'])
            
        self.logger.info(f"Training completed. Best average reward: {best_avg_reward:.2f}")
        
        return {
            'rewards': rewards,
            'avg_rewards': avg_rewards,
            'best_avg_reward': best_avg_reward,
            'actor_losses': self.metrics['actor_losses'],
            'critic_losses': self.metrics['critic_losses'],
            'q_values': self.metrics['q_values']
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
