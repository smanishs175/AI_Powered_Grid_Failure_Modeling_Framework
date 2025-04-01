"""
Unit tests for DQN Agent implementation.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import torch

from gfmf.reinforcement_learning.agents import DQNAgent
from tests.reinforcement_learning.agents.mock_env import MockGridEnv


class TestDQNAgent(unittest.TestCase):
    """Test cases for the DQN Agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 10
        self.action_dim = 3
        self.env = MockGridEnv(state_dim=self.state_dim, action_dim=self.action_dim, max_steps=10)
        self.agent = DQNAgent(self.state_dim, self.action_dim)
        
        # Create a temp directory for model saving/loading
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temp directory
        shutil.rmtree(self.temp_dir)
    
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
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        
        # Test action without exploration (eval mode)
        action = self.agent.act(state, eval_mode=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
    
    def test_step(self):
        """Test step function for storing experiences."""
        state = self.env.reset()
        action = self.agent.act(state)
        next_state, reward, done, _ = self.env.step(action)
        
        # Store the experience
        self.agent.step(state, action, reward, next_state, done)
        
        # Check that the experience was stored in memory
        self.assertEqual(len(self.agent.memory), 1)
    
    def test_learn(self):
        """Test the learning process."""
        # Fill the memory with some experiences
        state = self.env.reset()
        for _ in range(self.agent.config['batch_size'] * 2):
            action = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = self.env.reset()
        
        # Get initial network weights for comparison
        initial_weights = list(self.agent.qnetwork_local.parameters())[0].clone().detach()
        
        # Force a learning step
        self.agent._learn()
        
        # Get updated weights
        updated_weights = list(self.agent.qnetwork_local.parameters())[0].clone().detach()
        
        # Check that weights were updated
        self.assertFalse(torch.allclose(initial_weights, updated_weights))
    
    def test_save_load_policy(self):
        """Test saving and loading policy."""
        # Save the policy
        model_path = os.path.join(self.temp_dir, "dqn_test_model.pth")
        saved_path = self.agent.save_policy(model_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(saved_path))
        
        # Create a new agent
        new_agent = DQNAgent(self.state_dim, self.action_dim)
        
        # Get initial weights for comparison
        initial_weights = list(new_agent.qnetwork_local.parameters())[0].clone().detach()
        
        # Load the saved policy
        new_agent.load_policy(saved_path)
        
        # Get loaded weights
        loaded_weights = list(new_agent.qnetwork_local.parameters())[0].clone().detach()
        
        # Check that weights were loaded (should be different from initialization)
        self.assertFalse(torch.allclose(initial_weights, loaded_weights))
        
        # Check that loaded weights match original agent's weights
        original_weights = list(self.agent.qnetwork_local.parameters())[0].clone().detach()
        self.assertTrue(torch.allclose(loaded_weights, original_weights))
    
    def test_mini_training(self):
        """Test a minimal training run."""
        # Run training for a few episodes
        metrics = self.agent.train(self.env, num_episodes=2, max_steps=5)
        
        # Check that metrics were returned
        self.assertIsNotNone(metrics)
        self.assertIn('rewards', metrics)
        self.assertIn('avg_rewards', metrics)
    
    def test_evaluate(self):
        """Test agent evaluation."""
        # Evaluate the agent for a few episodes
        avg_reward = self.agent.evaluate(self.env, num_episodes=2)
        
        # Check that a reward was returned
        self.assertIsInstance(avg_reward, float)


if __name__ == '__main__':
    unittest.main()
