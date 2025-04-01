"""
Unit tests for PPO Agent implementation.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import torch

from gfmf.reinforcement_learning.agents import PPOAgent
from tests.reinforcement_learning.agents.mock_env import MockGridEnv


class TestPPOAgent(unittest.TestCase):
    """Test cases for the PPO Agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 10
        self.action_dim = 3
        self.env = MockGridEnv(state_dim=self.state_dim, action_dim=self.action_dim, max_steps=10)
        self.agent = PPOAgent(self.state_dim, self.action_dim)
        
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
        self.assertIsNotNone(self.agent.network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_act(self):
        """Test action selection."""
        state = self.env.reset()
        
        # Test action selection with exploration
        action, log_prob, entropy, value = self.agent.act(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(entropy, float)
        self.assertIsInstance(value, float)
        
        # Test action without exploration (eval mode)
        action = self.agent.act(state, eval_mode=True)[0]
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
    
    def test_step(self):
        """Test step function for storing experiences."""
        state = self.env.reset()
        action, log_prob, entropy, value = self.agent.act(state)
        next_state, reward, done, _ = self.env.step(action)
        
        # Store the experience
        self.agent.step(state, action, reward, next_state, done, log_prob, value)
        
        # Check that the experience was stored in memory
        self.assertEqual(len(self.agent.memory), 1)
    
    def test_update(self):
        """Test the policy update process."""
        # Fill the memory with experiences
        for _ in range(self.agent.config['batch_size']):
            state = self.env.reset()
            action, log_prob, entropy, value = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.step(state, action, reward, next_state, done, log_prob, value)
        
        # Get initial network weights for comparison
        initial_weights = list(self.agent.network.parameters())[0].clone().detach()
        
        # Update the policy
        self.agent.update()
        
        # Get updated weights
        updated_weights = list(self.agent.network.parameters())[0].clone().detach()
        
        # Check that weights were updated
        self.assertFalse(torch.allclose(initial_weights, updated_weights))
    
    def test_save_load_policy(self):
        """Test saving and loading policy."""
        # Save the policy
        model_path = os.path.join(self.temp_dir, "ppo_test_model.pth")
        saved_path = self.agent.save_policy(model_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(saved_path))
        
        # Create a new agent
        new_agent = PPOAgent(self.state_dim, self.action_dim)
        
        # Get initial weights for comparison
        initial_weights = list(new_agent.network.parameters())[0].clone().detach()
        
        # Load the saved policy
        new_agent.load_policy(saved_path)
        
        # Get loaded weights
        loaded_weights = list(new_agent.network.parameters())[0].clone().detach()
        
        # Check that weights were loaded (should be different from initialization)
        self.assertFalse(torch.allclose(initial_weights, loaded_weights, atol=1e-6))
        
        # Check that loaded weights match original agent's weights
        original_weights = list(self.agent.network.parameters())[0].clone().detach()
        self.assertTrue(torch.allclose(loaded_weights, original_weights, atol=1e-6))
    
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
