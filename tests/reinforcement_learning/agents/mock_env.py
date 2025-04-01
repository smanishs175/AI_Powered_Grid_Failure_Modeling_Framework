"""
Mock environment for testing reinforcement learning agents.

This module provides a simple environment that can be used for testing RL agents,
with customizable state and action spaces.
"""

import numpy as np
import gym
from gym import spaces


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
        """
        Render the environment. Not implemented for the mock environment.
        
        Args:
            mode (str): Rendering mode
        """
        pass
    
    def close(self):
        """
        Close the environment.
        """
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
