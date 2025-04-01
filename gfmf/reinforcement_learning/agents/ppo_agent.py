"""
Proximal Policy Optimization (PPO) Implementation.

This module implements a PPO agent with the following features:
- Clipped objective function to prevent destructive policy updates
- Generalized Advantage Estimation (GAE) for variance reduction
- Shared network architecture for policy and value functions
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


class PPONetwork(nn.Module):
    """
    Actor-Critic Network for PPO.
    
    Features shared layers between policy and value networks.
    """
    
    def __init__(self, state_dim, action_dim, hidden_layers=(128, 64)):
        """
        Initialize the network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_layers (tuple): Sizes of hidden layers
        """
        super(PPONetwork, self).__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_layers[1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_layers[1], 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            tuple: (action_probs, state_value)
        """
        features = self.shared_layers(state)
        
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)
        
        return action_probs, state_value
    
    def get_action(self, state, deterministic=False):
        """
        Select an action based on the state.
        
        Args:
            state: Input state tensor
            deterministic (bool): Whether to select action deterministically
            
        Returns:
            tuple: (action, log_prob, entropy, state_value)
        """
        action_probs, state_value = self.forward(state)
        
        # Create a categorical distribution over action probabilites
        dist = Categorical(action_probs)
        
        # Either sample from distribution or take most likely action
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        
        # Calculate log probability and entropy for the selected action
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, state_value
    
    def evaluate_action(self, state, action):
        """
        Evaluate an action for a given state.
        
        Args:
            state: Input state tensor
            action: Action to evaluate
            
        Returns:
            tuple: (log_prob, entropy, state_value)
        """
        action_probs, state_value = self.forward(state)
        
        # Create a categorical distribution over action probabilites
        dist = Categorical(action_probs)
        
        # Calculate log probability and entropy for the given action
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, state_value


class PPOMemory:
    """
    Memory buffer for PPO.
    
    Stores experiences collected during an episode.
    """
    
    def __init__(self):
        """Initialize empty memory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def push(self, state, action, reward, log_prob, value, done):
        """
        Store an experience.
        
        Args:
            state: Observed state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate
            done: Whether episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get_batch(self, gamma=0.99, gae_lambda=0.95, device=None):
        """
        Process experiences into training batch.
        
        Calculates returns and advantages using GAE.
        
        Args:
            gamma (float): Discount factor
            gae_lambda (float): GAE parameter
            device: Torch device
            
        Returns:
            tuple: (states, actions, log_probs, returns, advantages)
        """
        if device is None:
            device = torch.device("cpu")
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        
        # Calculate returns and advantages using GAE
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1 or self.dones[i]:
                next_value = 0
            else:
                next_value = values[i + 1]
                
            # Calculate TD target
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - values[i]
            
            # Calculate GAE
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            
            # Insert in reverse order
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, old_log_probs, returns, advantages
    
    def clear(self):
        """Clear memory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def __len__(self):
        """Return number of stored experiences."""
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    
    Implements PPO with clipped objective and Generalized Advantage Estimation.
    """
    
    def __init__(self, state_dim, action_dim, config=None):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config (dict, optional): Configuration parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing PPO Agent")
        
        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default configuration
        default_config = {
            'lr': 0.0003,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_param': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'epochs': 4,
            'batch_size': 64,
            'hidden_layers': [128, 64]
        }
        
        # Update with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Set up device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor-critic network
        self.network = PPONetwork(
            state_dim,
            action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config['lr']
        )
        
        # Initialize memory
        self.memory = PPOMemory()
        
        # Training metrics
        self.metrics = {
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': [],
            'rewards': []
        }
        
        self.logger.info(f"PPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state, deterministic=False):
        """
        Choose an action given the current state.
        
        Args:
            state: Current state
            deterministic (bool): Whether to act deterministically
            
        Returns:
            int: Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action(state, deterministic)
            
        return action.item(), log_prob.item(), value.item()
    
    def step(self, state, action, reward, log_prob, value, next_state, done):
        """
        Store experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.push(state, action, reward, log_prob, value, done)
    
    def update(self):
        """
        Update policy and value networks using collected experiences.
        
        Returns:
            dict: Update metrics
        """
        if len(self.memory) == 0:
            return None
            
        self.logger.debug(f"Updating PPO agent with {len(self.memory)} experiences")
        
        # Get training batch
        states, actions, old_log_probs, returns, advantages = self.memory.get_batch(
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            device=self.device
        )
        
        # Track metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        
        # Perform multiple epochs of updates
        for _ in range(self.config['epochs']):
            # Process in mini-batches
            batch_size = min(self.config['batch_size'], len(self.memory))
            indices = np.arange(len(self.memory))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(self.memory), batch_size):
                end_idx = min(start_idx + batch_size, len(self.memory))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions
                new_log_probs, entropies, new_values = self.network.evaluate_action(
                    batch_states, batch_actions
                )
                
                # Calculate ratios and surrogate objectives
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios,
                    1.0 - self.config['clip_param'],
                    1.0 + self.config['clip_param']
                ) * batch_advantages
                
                # Calculate policy loss (negative, since we're minimizing)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(new_values.squeeze(-1), batch_returns)
                
                # Calculate entropy loss
                entropy_loss = -entropies.mean()
                
                # Combine losses
                total_loss = (
                    policy_loss +
                    self.config['value_coef'] * value_loss +
                    self.config['entropy_coef'] * entropy_loss
                )
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config['max_grad_norm']
                )
                
                self.optimizer.step()
                
                # Record metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(total_loss.item())
        
        # Clear memory
        self.memory.clear()
        
        # Update metrics
        self.metrics['policy_losses'].extend(policy_losses)
        self.metrics['value_losses'].extend(value_losses)
        self.metrics['entropy_losses'].extend(entropy_losses)
        self.metrics['total_losses'].extend(total_losses)
        
        # Calculate averages
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy_loss = np.mean(entropy_losses)
        avg_total_loss = np.mean(total_losses)
        
        self.logger.debug(f"Update completed. Policy loss: {avg_policy_loss:.4f}, "
                         f"Value loss: {avg_value_loss:.4f}, "
                         f"Entropy loss: {avg_entropy_loss:.4f}")
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_total_loss
        }
    
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
            'network': self.network.state_dict(),
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'metrics': self.metrics
        }
        
        torch.save(state, filepath)
        
        self.logger.info(f"Saved PPO policy to {filepath}")
        
        return filepath
    
    def load_policy(self, filepath):
        """
        Load a trained policy.
        
        Args:
            filepath (str): Path to the saved policy
            
        Returns:
            PPOAgent: Self with loaded policy
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
        
        # Recreate network with loaded dimensions
        self.network = PPONetwork(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['hidden_layers']
        ).to(self.device)
        
        # Load weights
        self.network.load_state_dict(state['network'])
        
        # Set to evaluation mode
        self.network.eval()
        
        self.logger.info(f"Loaded PPO policy from {filepath}")
        
        return self
    
    def train(self, env, num_episodes=1000, max_steps=None, update_freq=2048):
        """
        Train the agent on the given environment.
        
        Args:
            env: OpenAI Gym environment
            num_episodes (int): Number of episodes to train
            max_steps (int): Maximum steps per episode
            update_freq (int): How many steps to collect before updating
            
        Returns:
            dict: Training metrics
        """
        if max_steps is None:
            max_steps = env.max_steps if hasattr(env, 'max_steps') else 1000
            
        self.logger.info(f"Training PPO agent for {num_episodes} episodes, max {max_steps} steps per episode")
        
        episode_rewards = []
        avg_rewards = []
        best_avg_reward = -float('inf')
        best_weights = None
        
        total_steps = 0
        updates = 0
        
        for episode in range(1, num_episodes+1):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Select action
                action, log_prob, value = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                self.step(state, action, reward, log_prob, value, next_state, done)
                
                state = next_state
                episode_reward += reward
                total_steps += 1
                
                # Update policy if enough steps have been collected
                if total_steps % update_freq == 0:
                    self.update()
                    updates += 1
                
                if done:
                    break
            
            # Record episode reward
            episode_rewards.append(episode_reward)
            self.metrics['rewards'].append(episode_reward)
            
            # Calculate moving average
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)
            
            # Store best weights
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_weights = copy.deepcopy(self.network.state_dict())
            
            # Log progress
            if episode % 10 == 0:
                self.logger.info(f"Episode {episode}/{num_episodes} | Reward: {episode_reward:.2f} | "
                                f"Avg Reward: {avg_reward:.2f} | Updates: {updates}")
        
        # Restore best weights
        if best_weights is not None:
            self.network.load_state_dict(best_weights)
            
        self.logger.info(f"Training completed. Best average reward: {best_avg_reward:.2f}")
        
        return {
            'rewards': episode_rewards,
            'avg_rewards': avg_rewards,
            'best_avg_reward': best_avg_reward,
            'updates': updates,
            'policy_losses': self.metrics['policy_losses'],
            'value_losses': self.metrics['value_losses'],
            'entropy_losses': self.metrics['entropy_losses'],
            'total_losses': self.metrics['total_losses']
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
                action, _, _ = self.act(state, deterministic=True)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards)
        self.logger.info(f"Evaluation over {num_episodes} episodes: {avg_reward:.2f}")
        
        return avg_reward
