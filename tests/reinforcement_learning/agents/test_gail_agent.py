"""
Unit tests for GAIL Agent implementation.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import torch
import pickle

from gfmf.reinforcement_learning.agents import GAILAgent
from tests.reinforcement_learning.agents.mock_env import MockGridEnv


class TestGAILAgent(unittest.TestCase):
    """Test cases for the GAIL Agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 10
        self.action_dim = 3
        self.env = MockGridEnv(state_dim=self.state_dim, action_dim=self.action_dim, max_steps=10)
        self.agent = GAILAgent(self.state_dim, self.action_dim)
        
        # Create a temp directory for model saving/loading
        self.temp_dir = tempfile.mkdtemp()
        
        # Create and store some sample expert trajectories
        self.expert_traj_path = os.path.join(self.temp_dir, "expert_trajectories.pkl")
        self._create_sample_expert_trajectories()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temp directory
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_expert_trajectories(self):
        """Create and save sample expert trajectories for testing."""
        # Generate a simple expert trajectory
        expert_trajectories = []
        
        # Create 5 simple trajectories
        for _ in range(5):
            states = []
            actions = []
            
            # Each trajectory has 10 steps
            for _ in range(10):
                state = np.random.uniform(-0.5, 0.5, size=self.state_dim).astype(np.float32)
                action = np.random.randint(0, self.action_dim)
                
                states.append(state)
                actions.append(action)
            
            expert_trajectories.append({
                'states': states,
                'actions': actions
            })
        
        # Save trajectories to file
        with open(self.expert_traj_path, 'wb') as f:
            pickle.dump(expert_trajectories, f)
    
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
        
        # Test action with exploration
        action, log_prob, entropy, value = self.agent.act(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(entropy, float)
        self.assertIsInstance(value, float)
        
        # Test action without exploration (eval mode)
        action, _, _, _ = self.agent.act(state, eval_mode=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
    
    def test_step(self):
        """Test step function for storing experiences."""
        state = self.env.reset()
        action, log_prob, _, value = self.agent.act(state)
        next_state, reward, done, _ = self.env.step(action)
        
        # Store the experience
        self.agent.step(state, action, reward, next_state, done, log_prob, value)
        
        # Check that the experience was stored in memory
        self.assertEqual(len(self.agent.memory), 1)
    
    def test_compute_gail_reward(self):
        """Test GAIL reward computation."""
        state = self.env.reset()
        action = 0
        
        # Compute GAIL reward
        reward = self.agent._compute_gail_reward(state, action)
        
        # Check that a reward was returned
        self.assertIsInstance(reward, float)
    
    def test_load_expert_trajectories(self):
        """Test loading expert trajectories."""
        # Load the trajectories
        success = self.agent.load_expert_trajectories(self.expert_traj_path)
        
        # Check that loading was successful
        self.assertTrue(success)
        
        # Check that trajectories were loaded
        self.assertEqual(len(self.agent.expert_dataset), 5)
    
    def test_add_expert_trajectory(self):
        """Test adding an expert trajectory."""
        # Create a trajectory
        trajectory = {
            'states': [np.random.uniform(-0.5, 0.5, size=self.state_dim).astype(np.float32) for _ in range(5)],
            'actions': [np.random.randint(0, self.action_dim) for _ in range(5)]
        }
        
        # Initial trajectory count
        initial_count = len(self.agent.expert_dataset)
        
        # Add the trajectory
        self.agent.add_expert_trajectory(trajectory)
        
        # Check that trajectory was added
        self.assertEqual(len(self.agent.expert_dataset), initial_count + 1)
    
    def test_save_expert_trajectories(self):
        """Test saving expert trajectories."""
        # Add a trajectory
        trajectory = {
            'states': [np.random.uniform(-0.5, 0.5, size=self.state_dim).astype(np.float32) for _ in range(5)],
            'actions': [np.random.randint(0, self.action_dim) for _ in range(5)]
        }
        self.agent.add_expert_trajectory(trajectory)
        
        # Save trajectories
        save_path = os.path.join(self.temp_dir, "saved_trajectories.pkl")
        success = self.agent.save_expert_trajectories(save_path)
        
        # Check that saving was successful
        self.assertTrue(success)
        self.assertTrue(os.path.exists(save_path))
    
    def test_generate_demonstration(self):
        """Test generating a demonstration."""
        # Generate a demonstration
        traj = self.agent.generate_demonstration(self.env, max_steps=5)
        
        # Check the trajectory structure
        self.assertIn('states', traj)
        self.assertIn('actions', traj)
        self.assertEqual(len(traj['states']), len(traj['actions']))
        self.assertGreater(len(traj['states']), 0)
    
    def test_update_discriminator(self):
        """Test discriminator update."""
        # Load expert trajectories
        self.agent.load_expert_trajectories(self.expert_traj_path)
        
        # Fill memory with experiences
        for _ in range(10):
            state = self.env.reset()
            action, log_prob, _, value = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.step(state, action, reward, next_state, done, log_prob, value)
        
        # Get initial discriminator weights
        initial_weights = list(self.agent.discriminator.parameters())[0].clone().detach()
        
        # Update discriminator
        self.agent._update_discriminator()
        
        # Get updated weights
        updated_weights = list(self.agent.discriminator.parameters())[0].clone().detach()
        
        # Check that weights were updated
        self.assertFalse(torch.allclose(initial_weights, updated_weights, atol=1e-6))
    
    def test_save_load_policy(self):
        """Test saving and loading policy."""
        # Save the policy
        model_path = os.path.join(self.temp_dir, "gail_test_model.pth")
        saved_path = self.agent.save_policy(model_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(saved_path))
        
        # Create a new agent
        new_agent = GAILAgent(self.state_dim, self.action_dim)
        
        # Get initial weights for comparison
        initial_policy_weights = list(new_agent.policy.parameters())[0].clone().detach()
        
        # Load the saved policy
        new_agent.load_policy(saved_path)
        
        # Get loaded weights
        loaded_policy_weights = list(new_agent.policy.parameters())[0].clone().detach()
        
        # Check that weights were loaded (should be different from initialization)
        self.assertFalse(torch.allclose(initial_policy_weights, loaded_policy_weights, atol=1e-6))
        
        # Check that loaded weights match original agent's weights
        original_policy_weights = list(self.agent.policy.parameters())[0].clone().detach()
        self.assertTrue(torch.allclose(loaded_policy_weights, original_policy_weights, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
