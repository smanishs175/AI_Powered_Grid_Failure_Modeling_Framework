"""
Generative Adversarial Imitation Learning (GAIL) Implementation.

This module implements a GAIL agent with the following features:
- Discriminator network to distinguish expert from agent trajectories
- Policy network trained using discriminator signals
- PPO-based policy optimization
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
import pickle
from collections import defaultdict


class Discriminator(nn.Module):
    """
    Discriminator network for GAIL.
    
    Distinguishes between expert and agent trajectories.
    """
    
    def __init__(self, state_dim, action_dim, hidden_layers=(128, 64)):
        """
        Initialize the discriminator network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_layers (tuple): Sizes of hidden layers
        """
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], 1),
            nn.Sigmoid()  # Output probability in [0, 1]
        )
    
    def forward(self, state, action):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            action: Action tensor (one-hot encoded)
            
        Returns:
            torch.Tensor: Probability that the trajectory is from an expert
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class Policy(nn.Module):
    """
    Policy network for GAIL.
    
    Uses architecture similar to PPO but optimized using GAIL rewards.
    """
    
    def __init__(self, state_dim, action_dim, hidden_layers=(128, 64)):
        """
        Initialize the policy network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_layers (tuple): Sizes of hidden layers
        """
        super(Policy, self).__init__()
        
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
        
        return log_prob, entropy, state_value, action_probs


class GAILMemory:
    """
    Memory for GAIL.
    
    Stores trajectories from agent and expert.
    """
    
    def __init__(self):
        """Initialize empty memory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def push(self, state, action, reward, next_state, done, log_prob, value):
        """
        Store a transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: Value estimate
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get_batch(self, gamma=0.99, gae_lambda=0.95, device=None):
        """
        Process experiences into training batch.
        
        Calculates returns and advantages using GAE.
        
        Args:
            gamma (float): Discount factor
            gae_lambda (float): GAE parameter
            device: Torch device
            
        Returns:
            dict: Batch of experiences
        """
        if device is None:
            device = torch.device("cpu")
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(device)
        dones = torch.FloatTensor(np.array(self.dones)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        
        # Calculate returns and advantages using GAE
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1 or dones[i]:
                next_value = 0
            else:
                next_value = values[i + 1]
                
            # Calculate TD target
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            
            # Calculate GAE
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            
            # Insert in reverse order
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create one-hot encoding for actions
        actions_one_hot = F.one_hot(actions, num_classes=actions_one_hot.shape[1] if len(actions_one_hot.shape) > 1 else 3).float()
        
        return {
            'states': states,
            'actions': actions,
            'actions_one_hot': actions_one_hot,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'old_log_probs': old_log_probs,
            'returns': returns,
            'advantages': advantages,
            'values': values
        }
    
    def clear(self):
        """Clear memory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def __len__(self):
        """Return number of stored transitions."""
        return len(self.states)


class ExpertDataset:
    """
    Dataset for expert demonstrations.
    
    Provides access to expert trajectories for imitation learning.
    """
    
    def __init__(self, capacity=100):
        """
        Initialize the dataset.
        
        Args:
            capacity (int): Maximum number of expert trajectories to store
        """
        self.trajectories = []
        self.capacity = capacity
    
    def add_trajectory(self, trajectory):
        """
        Add an expert trajectory.
        
        Args:
            trajectory (dict): Expert trajectory with states and actions
        """
        if len(self.trajectories) >= self.capacity:
            self.trajectories.pop(0)
            
        self.trajectories.append(trajectory)
    
    def sample_batch(self, batch_size, device=None):
        """
        Sample a batch of state-action pairs from expert demonstrations.
        
        Args:
            batch_size (int): Number of samples to draw
            device: Torch device
            
        Returns:
            tuple: (states, actions)
        """
        if device is None:
            device = torch.device("cpu")
            
        # Flatten all trajectories into state-action pairs
        states = []
        actions = []
        
        for traj in self.trajectories:
            states.extend(traj['states'])
            actions.extend(traj['actions'])
        
        # Sample indices
        if len(states) <= batch_size:
            indices = np.arange(len(states))
        else:
            indices = np.random.choice(len(states), batch_size, replace=False)
        
        # Get samples
        batch_states = np.array([states[i] for i in indices])
        batch_actions = np.array([actions[i] for i in indices])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(batch_states).to(device)
        actions_tensor = torch.LongTensor(batch_actions).to(device)
        
        # Create one-hot encoding for actions
        action_dim = max(3, max(batch_actions) + 1)  # Ensure at least 3 actions
        actions_one_hot = F.one_hot(actions_tensor, num_classes=action_dim).float()
        
        return states_tensor, actions_tensor, actions_one_hot
    
    def load(self, filepath):
        """
        Load expert demonstrations from file.
        
        Args:
            filepath (str): Path to the demonstrations file
            
        Returns:
            bool: Success status
        """
        if not os.path.exists(filepath):
            return False
            
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            if isinstance(data, list):
                self.trajectories = data[:self.capacity]
                return True
            else:
                return False
        except Exception as e:
            return False
    
    def save(self, filepath):
        """
        Save expert demonstrations to file.
        
        Args:
            filepath (str): Path to save the demonstrations
            
        Returns:
            bool: Success status
        """
        try:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(filepath, 'wb') as f:
                pickle.dump(self.trajectories, f)
                
            return True
        except Exception as e:
            return False
    
    def __len__(self):
        """Return number of trajectories."""
        return len(self.trajectories)


class GAILAgent:
    """
    Generative Adversarial Imitation Learning (GAIL) agent.
    
    Implements a GAIL agent that learns to imitate expert trajectories.
    """
    
    def __init__(self, state_dim, action_dim, config=None):
        """
        Initialize the GAIL agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config (dict, optional): Configuration parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing GAIL Agent")
        
        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default configuration
        default_config = {
            'lr_policy': 0.0003,
            'lr_discriminator': 0.0001,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_param': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'ppo_epochs': 10,
            'minibatch_size': 64,
            'policy_hidden_layers': [128, 64],
            'discriminator_hidden_layers': [128, 64],
            'expert_trajectories_file': None,
            'reward_scale': 1.0
        }
        
        # Update with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Set up device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy network
        self.policy = Policy(
            state_dim,
            action_dim,
            hidden_layers=self.config['policy_hidden_layers']
        ).to(self.device)
        
        # Initialize discriminator network
        self.discriminator = Discriminator(
            state_dim,
            action_dim,
            hidden_layers=self.config['discriminator_hidden_layers']
        ).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config['lr_policy']
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['lr_discriminator']
        )
        
        # Initialize memory
        self.memory = GAILMemory()
        
        # Initialize expert dataset
        self.expert_dataset = ExpertDataset()
        
        # Load expert trajectories if available
        if self.config['expert_trajectories_file'] and os.path.exists(self.config['expert_trajectories_file']):
            self.expert_dataset.load(self.config['expert_trajectories_file'])
            self.logger.info(f"Loaded {len(self.expert_dataset)} expert trajectories")
        
        # Initialize training metrics
        self.metrics = defaultdict(list)
        
        self.logger.info(f"GAIL Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state, eval_mode=False):
        """
        Choose an action given the current state.
        
        Args:
            state: Current state
            eval_mode (bool): If True, use deterministic policy
            
        Returns:
            int: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action(
                state_tensor,
                deterministic=eval_mode
            )
        
        return action.item(), log_prob.item(), entropy.item(), value.item()
    
    def step(self, state, action, reward, next_state, done, log_prob, value):
        """
        Store experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: Value estimate
        """
        self.memory.push(state, action, reward, next_state, done, log_prob, value)
    
    def _compute_gail_reward(self, state, action):
        """
        Compute GAIL reward based on discriminator output.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            float: GAIL reward
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        action_one_hot = F.one_hot(action_tensor, num_classes=self.action_dim).float()
        
        # Get discriminator output
        with torch.no_grad():
            expert_prob = self.discriminator(state_tensor, action_one_hot)
        
        # Extract probability and compute reward
        prob = expert_prob.item()
        
        # log(D(s,a)) reward formulation
        reward = np.log(max(prob, 1e-10))
        
        # Scale reward
        reward *= self.config['reward_scale']
        
        return reward
    
    def _update_discriminator(self, expert_batch_size=64):
        """
        Update the discriminator network.
        
        Args:
            expert_batch_size (int): Number of expert samples to use
            
        Returns:
            float: Discriminator loss
        """
        # Check if expert dataset and memory have enough samples
        if len(self.expert_dataset) == 0 or len(self.memory) < expert_batch_size:
            return 0.0
        
        # Get agent experiences
        agent_batch = self.memory.get_batch(
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            device=self.device
        )
        
        # Sample expert trajectories
        expert_states, _, expert_actions_one_hot = self.expert_dataset.sample_batch(
            expert_batch_size,
            device=self.device
        )
        
        # Get discriminator outputs for expert
        expert_preds = self.discriminator(expert_states, expert_actions_one_hot)
        
        # Get discriminator outputs for agent
        agent_states = agent_batch['states']
        agent_actions_one_hot = agent_batch['actions_one_hot']
        agent_preds = self.discriminator(agent_states, agent_actions_one_hot)
        
        # Compute binary cross-entropy loss
        expert_loss = F.binary_cross_entropy(
            expert_preds,
            torch.ones_like(expert_preds)
        )
        
        agent_loss = F.binary_cross_entropy(
            agent_preds,
            torch.zeros_like(agent_preds)
        )
        
        # Total loss
        discriminator_loss = expert_loss + agent_loss
        
        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # Record metrics
        self.metrics['disc_loss'].append(discriminator_loss.item())
        self.metrics['expert_acc'].append((expert_preds > 0.5).float().mean().item())
        self.metrics['agent_acc'].append((agent_preds < 0.5).float().mean().item())
        
        return discriminator_loss.item()
    
    def _update_policy(self):
        """
        Update the policy network using PPO.
        
        Returns:
            dict: Update metrics
        """
        # Get batch of experiences
        batch = self.memory.get_batch(
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            device=self.device
        )
        
        # Unpack batch
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        returns = batch['returns']
        advantages = batch['advantages']
        
        # Training loop
        for _ in range(self.config['ppo_epochs']):
            # Generate random permutation for minibatch sampling
            indices = torch.randperm(states.size(0))
            
            # Process minibatches
            for start_idx in range(0, states.size(0), self.config['minibatch_size']):
                # Get minibatch indices
                idx = indices[start_idx:start_idx + self.config['minibatch_size']]
                
                # Get minibatch data
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Get policy outputs for minibatch
                new_log_probs, entropy, values, _ = self.policy.evaluate_action(mb_states, mb_actions)
                
                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Compute surrogate losses
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(
                    ratio,
                    1.0 - self.config['clip_param'],
                    1.0 + self.config['clip_param']
                ) * mb_advantages
                
                # Policy loss
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.config['value_coef'] * value_loss + 
                    self.config['entropy_coef'] * entropy_loss
                )
                
                # Update policy
                self.policy_optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config['max_grad_norm']
                )
                self.policy_optimizer.step()
                
                # Record metrics
                self.metrics['policy_loss'].append(policy_loss.item())
                self.metrics['value_loss'].append(value_loss.item())
                self.metrics['entropy'].append(entropy.mean().item())
        
        return {
            'policy_loss': np.mean(self.metrics['policy_loss'][-self.config['ppo_epochs']:]),
            'value_loss': np.mean(self.metrics['value_loss'][-self.config['ppo_epochs']:]),
            'entropy': np.mean(self.metrics['entropy'][-self.config['ppo_epochs']:]) 
        }
    
    def train_discriminator_and_policy(self):
        """
        Train both discriminator and policy.
        
        Returns:
            dict: Training metrics
        """
        if len(self.memory) == 0 or len(self.expert_dataset) == 0:
            return {
                'disc_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0
            }
        
        # Update discriminator
        disc_loss = self._update_discriminator()
        
        # Update policy
        policy_metrics = self._update_policy()
        
        # Clear memory
        self.memory.clear()
        
        return {
            'disc_loss': disc_loss,
            **policy_metrics
        }
    
    def train(self, env, num_episodes=1000, max_steps=None, update_interval=2048):
        """
        Train the agent on the given environment.
        
        Args:
            env: OpenAI Gym environment
            num_episodes (int): Maximum number of episodes
            max_steps (int): Maximum steps per episode
            update_interval (int): Steps before policy update
            
        Returns:
            dict: Training metrics
        """
        if max_steps is None:
            max_steps = env.max_steps if hasattr(env, 'max_steps') else 1000
            
        # Check if expert trajectories are available
        if len(self.expert_dataset) == 0:
            self.logger.warning("No expert trajectories available for GAIL training")
            return None
            
        self.logger.info(f"Training GAIL agent for {num_episodes} episodes, max {max_steps} steps per episode")
        
        episode_rewards = []
        episode_lengths = []
        
        total_steps = 0
        
        for episode in range(1, num_episodes+1):
            state = env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < max_steps:
                # Select action
                action, log_prob, _, value = self.act(state)
                
                # Environment step
                next_state, env_reward, done, info = env.step(action)
                
                # Compute GAIL reward
                gail_reward = self._compute_gail_reward(state, action)
                
                # Store experience
                self.step(state, action, gail_reward, next_state, done, log_prob, value)
                
                # Update state
                state = next_state
                episode_reward += env_reward  # Track the environment reward
                step_count += 1
                total_steps += 1
                
                # Update if enough steps collected
                if total_steps % update_interval == 0:
                    metrics = self.train_discriminator_and_policy()
                    self.logger.info(f"Update at step {total_steps}: Disc Loss = {metrics['disc_loss']:.4f}, "
                                     f"Policy Loss = {metrics['policy_loss']:.4f}")
                
                if done:
                    break
            
            # Record episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                self.logger.info(f"Episode {episode}/{num_episodes} | "
                                 f"Avg Reward: {avg_reward:.2f} | "
                                 f"Avg Length: {avg_length:.2f}")
            
            # Evaluate agent
            if episode % 100 == 0:
                eval_reward = self.evaluate(env, num_episodes=5)
                self.metrics['eval_rewards'].append(eval_reward)
                self.logger.info(f"Evaluation at episode {episode}: {eval_reward:.2f}")
        
        self.logger.info(f"Training completed after {total_steps} steps")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'eval_rewards': self.metrics['eval_rewards'],
            'disc_loss': self.metrics['disc_loss'],
            'policy_loss': self.metrics['policy_loss'],
            'value_loss': self.metrics['value_loss'],
            'entropy': self.metrics['entropy'],
            'expert_acc': self.metrics['expert_acc'],
            'agent_acc': self.metrics['agent_acc']
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
                action, _, _, _ = self.act(state, eval_mode=True)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards)
        self.logger.info(f"Evaluation over {num_episodes} episodes: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_policy(self, filepath):
        """
        Save the agent's policy.
        
        Args:
            filepath (str): Path to save the policy
            
        Returns:
            str: Path where policy was saved
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save model weights and configuration
        state = {
            'policy': self.policy.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'metrics': dict(self.metrics)
        }
        
        torch.save(state, filepath)
        
        self.logger.info(f"Saved GAIL policy to {filepath}")
        
        return filepath
    
    def load_policy(self, filepath):
        """
        Load a trained policy.
        
        Args:
            filepath (str): Path to the saved policy
            
        Returns:
            GAILAgent: Self with loaded policy
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Policy file not found: {filepath}")
        
        # Load on CPU explicitly for compatibility
        state = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Update instance variables
        self.config = state['config']
        self.state_dim = state['state_dim']
        self.action_dim = state['action_dim']
        
        if 'metrics' in state:
            self.metrics = defaultdict(list, state['metrics'])
        
        # Recreate networks
        self.policy = Policy(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['policy_hidden_layers']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            self.state_dim,
            self.action_dim,
            hidden_layers=self.config['discriminator_hidden_layers']
        ).to(self.device)
        
        # Load weights
        self.policy.load_state_dict(state['policy'])
        self.discriminator.load_state_dict(state['discriminator'])
        
        # Set to evaluation mode
        self.policy.eval()
        self.discriminator.eval()
        
        # Recreate optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config['lr_policy']
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['lr_discriminator']
        )
        
        self.logger.info(f"Loaded GAIL policy from {filepath}")
        
        return self
    
    def generate_demonstration(self, env, max_steps=1000):
        """
        Generate a demonstration trajectory using the current policy.
        
        Args:
            env: OpenAI Gym environment
            max_steps (int): Maximum steps per episode
            
        Returns:
            dict: Trajectory with states and actions
        """
        state = env.reset()
        states = []
        actions = []
        
        for step in range(max_steps):
            action, _, _, _ = self.act(state, eval_mode=True)
            
            states.append(state)
            actions.append(action)
            
            next_state, _, done, _ = env.step(action)
            state = next_state
            
            if done:
                break
        
        return {
            'states': states,
            'actions': actions
        }
    
    def add_expert_trajectory(self, trajectory):
        """
        Add an expert trajectory to the dataset.
        
        Args:
            trajectory (dict): Expert trajectory
        """
        self.expert_dataset.add_trajectory(trajectory)
    
    def save_expert_trajectories(self, filepath):
        """
        Save expert trajectories to file.
        
        Args:
            filepath (str): Path to save the trajectories
            
        Returns:
            bool: Success status
        """
        return self.expert_dataset.save(filepath)
    
    def load_expert_trajectories(self, filepath):
        """
        Load expert trajectories from file.
        
        Args:
            filepath (str): Path to the trajectories file
            
        Returns:
            bool: Success status
        """
        success = self.expert_dataset.load(filepath)
        if success:
            self.logger.info(f"Loaded {len(self.expert_dataset)} expert trajectories from {filepath}")
        else:
            self.logger.warning(f"Failed to load expert trajectories from {filepath}")
        return success
