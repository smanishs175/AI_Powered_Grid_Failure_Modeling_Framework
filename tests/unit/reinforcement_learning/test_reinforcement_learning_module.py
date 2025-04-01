"""
Unit tests for the Reinforcement Learning Module (Module 5)
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import gymnasium as gym
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gfmf.reinforcement_learning.reinforcement_learning_module import ReinforcementLearningModule
from gfmf.reinforcement_learning.environments.grid_env import GridEnv


class TestReinforcementLearningModule(unittest.TestCase):
    """Test cases for the ReinforcementLearningModule class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.rl_module = ReinforcementLearningModule()
        
        # Create mock vulnerability scores
        self.vulnerability_scores = pd.DataFrame({
            'component_id': range(1, 11),
            'vulnerability_score': np.random.uniform(0, 1, 10)
        })
        
        # Create mock scenario data
        self.scenario_data = [
            {
                'scenario_id': 'baseline_1',
                'type': 'baseline',
                'components': [{'id': i, 'status': 'operational'} for i in range(1, 11)]
            },
            {
                'scenario_id': 'failure_1',
                'type': 'component_failure',
                'failed_components': [1, 3, 5]
            },
            {
                'scenario_id': 'weather_1',
                'type': 'weather',
                'weather_conditions': {
                    'extreme_temperature': True,
                    'high_wind': True,
                    'heavy_precipitation': False
                }
            }
        ]
        
        # Mock environment configuration
        self.env_config = {
            'grid_size': 10,
            'vulnerability_data': self.vulnerability_scores,
            'scenario_data': self.scenario_data,
            'action_space_type': 'discrete',
            'max_steps': 100,
            'reward_weights': {
                'reliability': 0.5,
                'cost': 0.3,
                'resilience': 0.2
            }
        }

    def test_initialization(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.rl_module)
        self.assertIsNotNone(self.rl_module.config)
        
    def test_create_environment(self):
        """Test environment creation."""
        env = self.rl_module.create_environment(self.env_config)
        
        self.assertIsInstance(env, gym.Env)
        
    def test_train_agent(self):
        """Test agent training functionality."""
        # Create a mock environment
        mock_env = MagicMock(spec=GridEnv)
        mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        mock_env.action_space = gym.spaces.Discrete(3)
        
        # Mock the SACAgent class
        with patch('gfmf.reinforcement_learning.agents.sac_agent.SACAgent') as MockSACAgent:
            # Configure the mock agent
            mock_agent = MagicMock()
            mock_agent.train.return_value = {'rewards': [0.1, 0.2, 0.3]}
            MockSACAgent.return_value = mock_agent
            
            # Run the training
            with patch.object(
                self.rl_module, 'create_environment', 
                return_value=mock_env
            ):
                training_results = self.rl_module.train_agent(
                    mock_env, agent_type='sac', train_timesteps=1000
                )
                
                self.assertIsInstance(training_results, dict)
                mock_agent.train.assert_called_once()
        
    def test_evaluate_agent(self):
        """Test agent evaluation functionality."""
        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.evaluate.return_value = {
            'mean_reward': 0.75,
            'std_reward': 0.1,
            'success_rate': 0.8,
            'episodes': 10
        }
        
        # Create a mock environment
        mock_env = MagicMock(spec=GridEnv)
        
        # Test the evaluation
        evaluation_results = self.rl_module.evaluate_agent(
            mock_agent, mock_env, num_episodes=10
        )
        
        self.assertIsInstance(evaluation_results, dict)
        self.assertIn('mean_reward', evaluation_results)
        self.assertIn('success_rate', evaluation_results)
        
    def test_select_best_agent(self):
        """Test best agent selection."""
        # Mock agent performances
        agent_performances = {
            'sac': {'mean_reward': 0.8, 'success_rate': 0.85},
            'ppo': {'mean_reward': 0.7, 'success_rate': 0.8},
            'dqn': {'mean_reward': 0.9, 'success_rate': 0.9}
        }
        
        best_agent = self.rl_module.select_best_agent(agent_performances)
        
        self.assertEqual(best_agent, 'dqn')
        
    def test_generate_hardening_policy(self):
        """Test hardening policy generation."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.get_policy.return_value = lambda x: np.array([1, 0, 2])
        
        # Create mock vulnerability scores
        vulnerable_components = pd.DataFrame({
            'component_id': range(1, 11),
            'vulnerability_score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            'type': ['line', 'transformer', 'bus', 'line', 'transformer', 
                    'bus', 'line', 'transformer', 'bus', 'line']
        })
        
        hardening_policy = self.rl_module.generate_hardening_policy(
            mock_agent, vulnerable_components
        )
        
        self.assertIsInstance(hardening_policy, dict)
        self.assertIn('policy_id', hardening_policy)
        self.assertIn('actions', hardening_policy)
        self.assertIsInstance(hardening_policy['actions'], list)
        self.assertEqual(len(hardening_policy['actions']), len(vulnerable_components))


class TestGridEnv(unittest.TestCase):
    """Test cases for the GridEnv class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock data for environment
        self.grid_topology = pd.DataFrame({
            'component_id': range(1, 11),
            'type': ['line', 'transformer', 'bus', 'line', 'transformer', 
                    'bus', 'line', 'transformer', 'bus', 'line'],
            'capacity': np.random.uniform(50, 200, 10),
            'age': np.random.uniform(1, 20, 10),
            'criticality': np.random.uniform(0.1, 1.0, 10)
        })
        
        self.vulnerability_scores = pd.DataFrame({
            'component_id': range(1, 11),
            'vulnerability_score': np.random.uniform(0, 1, 10)
        })
        
        self.env_config = {
            'grid_size': 10,
            'vulnerability_data': self.vulnerability_scores,
            'scenario_data': [
                {
                    'scenario_id': 'baseline_1',
                    'type': 'baseline',
                    'components': [{'id': i, 'status': 'operational'} for i in range(1, 11)]
                }
            ],
            'action_space_type': 'discrete',
            'max_steps': 100,
            'reward_weights': {
                'reliability': 0.5,
                'cost': 0.3,
                'resilience': 0.2
            }
        }
        
        # Initialize the environment
        self.env = GridEnv(self.env_config)

    def test_initialization(self):
        """Test that environment initializes correctly."""
        self.assertIsNotNone(self.env)
        self.assertIsInstance(self.env.action_space, gym.spaces.Space)
        self.assertIsInstance(self.env.observation_space, gym.spaces.Space)
        
    def test_reset(self):
        """Test environment reset functionality."""
        observation, info = self.env.reset()
        
        self.assertIsNotNone(observation)
        # Check that observation has the right shape
        self.assertEqual(len(observation), self.env.observation_space.shape[0])
        
    def test_step(self):
        """Test environment step functionality."""
        # Reset first
        self.env.reset()
        
        # Take a step
        action = self.env.action_space.sample()
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertIsNotNone(observation)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
    def test_calculate_reward(self):
        """Test reward calculation."""
        # Reset first
        self.env.reset()
        
        # Calculate reward for a sample action
        action = self.env.action_space.sample()
        reward = self.env._calculate_reward(action)
        
        self.assertIsInstance(reward, float)


if __name__ == '__main__':
    unittest.main()
