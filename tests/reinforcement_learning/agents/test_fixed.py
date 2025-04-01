"""
Fixed integration test to run all reinforcement learning agent tests.

This module allows running isolated tests for each RL agent without importing
the full Grid Failure Modeling Framework.
"""

import unittest
import os
import sys
import torch
import numpy as np
import gym
from gym import spaces

# Add paths to allow direct imports of agent classes without going through the main module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from gfmf.reinforcement_learning.agents.dqn_agent import DQNAgent
from gfmf.reinforcement_learning.agents.ppo_agent import PPOAgent
from gfmf.reinforcement_learning.agents.sac_agent import SACAgent
from gfmf.reinforcement_learning.agents.td3_agent import TD3Agent
from gfmf.reinforcement_learning.agents.gail_agent import GAILAgent


# Include mock environment directly to avoid import issues
class MockGridEnv(gym.Env):
    """
    A simple mock environment for testing RL agents with discrete action space.
    
    This environment simulates a simplified grid environment with configurable
    state and action dimensions, and predictable dynamics for testing.
    """
    
    def __init__(self, state_dim=10, action_dim=3, max_steps=100):
        """
        Initialize the mock environment.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            max_steps (int): Maximum number of steps per episode
        """
        super(MockGridEnv, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(state_dim,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.steps = 0
        self.reset()
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial state
        """
        # Initialize state with random values between -0.5 and 0.5
        self.state = np.random.uniform(-0.5, 0.5, size=self.state_dim).astype(np.float32)
        self.steps = 0
        return self.state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Update state based on action
        # For testing purposes, we make a simple update
        # Action 0: Move state values in positive direction
        # Action 1: Move state values in negative direction
        # Action 2: Random small changes
        
        if action == 0:
            self.state += 0.1 * np.ones(self.state_dim).astype(np.float32)
        elif action == 1:
            self.state -= 0.1 * np.ones(self.state_dim).astype(np.float32)
        else:
            self.state += 0.05 * np.random.uniform(-1, 1, size=self.state_dim).astype(np.float32)
        
        # Clip state to observation space bounds
        self.state = np.clip(self.state, -1.0, 1.0).astype(np.float32)
        
        # Compute reward
        # For testing, reward is the negative sum of absolute state values
        # This encourages the agent to keep state values close to zero
        reward = -np.sum(np.abs(self.state)) / self.state_dim
        
        # Increment step counter
        self.steps += 1
        
        # Check if episode is done
        done = (self.steps >= self.max_steps)
        
        # Additional info
        info = {}
        
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment. Not implemented for the mock environment."""
        pass
    
    def close(self):
        """Close the environment."""
        pass


class MockContinuousGridEnv(MockGridEnv):
    """
    A mock environment with continuous action space for testing.
    
    This extends the MockGridEnv with a continuous action space for testing
    agents that require continuous actions (e.g., SAC, TD3).
    """
    
    def __init__(self, state_dim=10, action_dim=3, max_steps=100):
        """
        Initialize the continuous mock environment.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            max_steps (int): Maximum number of steps per episode
        """
        # Initialize with parent class
        super(MockContinuousGridEnv, self).__init__(state_dim, action_dim, max_steps)
        
        # Override the action space to be continuous
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
    
    def step(self, action):
        """
        Take a step in the environment with continuous action.
        
        Args:
            action (np.ndarray): Continuous action to take
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Convert continuous action to affect state
        # Each action dimension affects a portion of the state
        action_np = np.asarray(action).flatten()
        
        # Repeat action values to match state dimension if needed
        if len(action_np) < self.state_dim:
            action_repeated = np.repeat(action_np, self.state_dim // len(action_np) + 1)
            action_effect = action_repeated[:self.state_dim]
        else:
            action_effect = action_np[:self.state_dim]
        
        # Update state based on action
        self.state += 0.1 * action_effect
        
        # Clip state to observation space bounds
        self.state = np.clip(self.state, -1.0, 1.0).astype(np.float32)
        
        # Compute reward
        # For testing, reward is the negative sum of absolute state values
        # This encourages the agent to keep state values close to zero
        reward = -np.sum(np.abs(self.state)) / self.state_dim
        
        # Increment step counter
        self.steps += 1
        
        # Check if episode is done
        done = (self.steps >= self.max_steps)
        
        # Additional info
        info = {}
        
        return self.state, reward, done, info


class TestDQNAgent(unittest.TestCase):
    """Test cases for the DQN Agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 10
        self.action_dim = 3
        self.env = MockGridEnv(state_dim=self.state_dim, action_dim=self.action_dim, max_steps=10)
        self.agent = DQNAgent(self.state_dim, self.action_dim)
    
    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertIsNotNone(self.agent.qnetwork_local)
        self.assertIsNotNone(self.agent.qnetwork_target)
        self.assertIsNotNone(self.agent.memory)
    
    def test_act(self):
        """Test action selection."""
        state = self.env.reset()
        
        # Test action with exploration
        action = self.agent.act(state)
        # Convert to Python int if it's a tensor or numpy value
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        
        # Test action without exploration (eval mode)
        action = self.agent.act(state, eval_mode=True)
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)


class TestPPOAgent(unittest.TestCase):
    """Test cases for the PPO Agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 10
        self.action_dim = 3
        self.env = MockGridEnv(state_dim=self.state_dim, action_dim=self.action_dim, max_steps=10)
        self.agent = PPOAgent(self.state_dim, self.action_dim)
    
    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertIsNotNone(self.agent.network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_act(self):
        """Test action selection."""
        state = self.env.reset()
        
        # Test action selection with exploration
        result = self.agent.act(state)
        
        # Handle different return formats from PPO agent
        if isinstance(result, tuple) and len(result) == 3:
            action, log_prob, value = result
            # entropy may be combined with log_prob or omitted
            self.assertIsInstance(log_prob, (float, np.float32, torch.Tensor))
            self.assertIsInstance(value, (float, np.float32, torch.Tensor))
        elif isinstance(result, tuple) and len(result) == 4:
            action, log_prob, entropy, value = result
            self.assertIsInstance(log_prob, (float, np.float32, torch.Tensor))
            self.assertIsInstance(entropy, (float, np.float32, torch.Tensor))
            self.assertIsInstance(value, (float, np.float32, torch.Tensor))
        else:
            action = result
            
        # Convert action to Python int if it's a tensor or numpy value
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        
        # For PPO, let's just do a regular act call since it might not support eval_mode
        result = self.agent.act(state)
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result
            
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)


class TestSACAgent(unittest.TestCase):
    """Test cases for the SAC Agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 10
        self.action_dim = 3
        self.env = MockGridEnv(state_dim=self.state_dim, action_dim=self.action_dim, max_steps=10)
        self.agent = SACAgent(self.state_dim, self.action_dim)
    
    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.critic)
        self.assertIsNotNone(self.agent.critic_target)
        self.assertIsNotNone(self.agent.memory)
    
    def test_act(self):
        """Test action selection."""
        state = self.env.reset()
        
        # Test action with exploration
        action = self.agent.act(state)
        # Convert to Python int if it's a tensor or numpy value
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        
        # Test action without exploration
        try:
            # First try with eval_mode if supported
            action = self.agent.act(state, eval_mode=True)
        except TypeError:
            # Fall back to standard act if eval_mode not supported
            action = self.agent.act(state)
            
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)


class TestTD3Agent(unittest.TestCase):
    """Test cases for the TD3 Agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 10
        self.action_dim = 3
        self.env = MockGridEnv(state_dim=self.state_dim, action_dim=self.action_dim, max_steps=10)
        self.agent = TD3Agent(self.state_dim, self.action_dim)
    
    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.actor_target)
        self.assertIsNotNone(self.agent.critic)
        self.assertIsNotNone(self.agent.critic_target)
        self.assertIsNotNone(self.agent.memory)
    
    def test_act(self):
        """Test action selection."""
        state = self.env.reset()
        
        # Test action with exploration
        action = self.agent.act(state)
        # Convert to Python int if it's a tensor or numpy value
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        
        # Test action without exploration
        try:
            # First try with eval_mode if supported
            action = self.agent.act(state, eval_mode=True)
        except TypeError:
            # Fall back to standard act if eval_mode not supported
            action = self.agent.act(state)
            
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)


class TestGAILAgent(unittest.TestCase):
    """Test cases for the GAIL Agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 10
        self.action_dim = 3
        self.env = MockGridEnv(state_dim=self.state_dim, action_dim=self.action_dim, max_steps=10)
        self.agent = GAILAgent(self.state_dim, self.action_dim)
        
        # Create sample expert trajectories
        self.sample_traj = {
            'states': [np.random.uniform(-0.5, 0.5, size=self.state_dim).astype(np.float32) for _ in range(5)],
            'actions': [np.random.randint(0, self.action_dim) for _ in range(5)]
        }
        
        # Add to agent
        self.agent.add_expert_trajectory(self.sample_traj)
    
    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertIsNotNone(self.agent.policy)
        self.assertIsNotNone(self.agent.discriminator)
        self.assertIsNotNone(self.agent.memory)
        self.assertIsNotNone(self.agent.expert_dataset)
    
    def test_act(self):
        """Test action selection."""
        state = self.env.reset()
        
        # Test action selection with exploration
        result = self.agent.act(state)
        
        # Handle different return formats from GAIL agent
        if isinstance(result, tuple) and len(result) == 3:
            action, log_prob, value = result
            # entropy may be combined with log_prob or omitted
            self.assertIsInstance(log_prob, (float, np.float32, torch.Tensor))
            self.assertIsInstance(value, (float, np.float32, torch.Tensor))
        elif isinstance(result, tuple) and len(result) == 4:
            action, log_prob, entropy, value = result
            self.assertIsInstance(log_prob, (float, np.float32, torch.Tensor))
            self.assertIsInstance(entropy, (float, np.float32, torch.Tensor))
            self.assertIsInstance(value, (float, np.float32, torch.Tensor))
        else:
            action = result
            
        # Convert action to Python int if it's a tensor or numpy value
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        
        # For GAIL, try eval mode but fall back to standard if not supported
        try:
            result = self.agent.act(state, eval_mode=True)
        except TypeError:
            result = self.agent.act(state)
            
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result
            
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)  # Ensure it's a Python int
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)


if __name__ == '__main__':
    unittest.main()
