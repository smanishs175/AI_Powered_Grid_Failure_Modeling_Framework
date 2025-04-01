"""
Grid Environment for Reinforcement Learning

This module provides a grid environment simulation for training reinforcement learning
agents to develop grid hardening policies.
"""

import os
import gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from gym import spaces
import logging

logger = logging.getLogger(__name__)

class GridEnv(gym.Env):
    """
    Grid Environment for simulating grid component failures and hardening actions.
    
    This environment allows reinforcement learning agents to learn optimal hardening
    policies by simulating:
    1. Grid topology and component relationships
    2. Environmental effects on components
    3. Component failures and cascading effects
    4. Hardening actions and their impacts
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the grid environment.
        
        Args:
            grid_topology: DataFrame with grid topology information
            vulnerability_scores: DataFrame with component vulnerability scores
            failure_predictions: DataFrame with component failure predictions
            scenarios: List of scenario dictionaries
            config: Environment configuration
        """
        super(GridEnv, self).__init__()
        
        # Store config
        self.config = config or {}
        
        # Extract data from config if provided in the new format
        self.grid_topology = None
        self.vulnerability_scores = None
        self.failure_predictions = None
        self.scenarios = []
        
        # Check if config is in the new format (from run_reinforcement_learning_module)
        if 'vulnerability_data' in self.config:
            self.vulnerability_scores = self.config['vulnerability_data']
            
        if 'scenario_data' in self.config:
            self.scenarios = self.config.get('scenario_data', [])
        
        # Set default values and ensure consistent data structures
        grid_size = self.config.get('grid_size', 10)
        
        # Create synthetic grid topology if not provided
        if self.grid_topology is None:
            self.grid_topology = pd.DataFrame({
                'component_id': range(grid_size),
                'type': ['transformer'] * (grid_size // 2) + ['line'] * (grid_size - grid_size // 2),
                'capacity': np.random.uniform(50, 100, grid_size),
                'age': np.random.uniform(1, 20, grid_size),
                'criticality': np.random.uniform(0.1, 1.0, grid_size)
            })
        
        # If vulnerability_scores is provided but doesn't have expected columns, adapt it
        if self.vulnerability_scores is not None and not isinstance(self.vulnerability_scores, pd.DataFrame):
            logger.warning("vulnerability_scores is not a DataFrame, converting")
            try:
                self.vulnerability_scores = pd.DataFrame(self.vulnerability_scores)
            except:
                self.vulnerability_scores = None
                
        if self.vulnerability_scores is not None:
            # Check if we need to add a component_id column
            if 'component_id' not in self.vulnerability_scores.columns:
                if 'id' in self.vulnerability_scores.columns:
                    self.vulnerability_scores['component_id'] = self.vulnerability_scores['id']  
            
            # Make sure vulnerability_score column exists
            if 'vulnerability_score' not in self.vulnerability_scores.columns:
                # Find best alternative column
                for col in ['vulnerability', 'risk_score', 'risk', 'score']:
                    if col in self.vulnerability_scores.columns:
                        self.vulnerability_scores['vulnerability_score'] = self.vulnerability_scores[col]
                        break
                else:
                    # If no suitable column found, generate random values
                    self.vulnerability_scores['vulnerability_score'] = np.random.uniform(0, 1, len(self.vulnerability_scores))
        
        # Generate synthetic vulnerability scores if still None
        if self.vulnerability_scores is None:
            component_ids = range(grid_size)
            self.vulnerability_scores = pd.DataFrame({
                'component_id': component_ids,
                'vulnerability_score': np.random.uniform(0, 1, grid_size)
            })
            
        # Generate synthetic failure predictions if None
        if self.failure_predictions is None:
            component_ids = self.vulnerability_scores['component_id'].values
            self.failure_predictions = pd.DataFrame({
                'component_id': component_ids,
                'failure_probability': np.random.uniform(0, 1, len(component_ids)),
                'predicted_failure': np.random.choice([0, 1], len(component_ids), p=[0.8, 0.2])
            })
        
        # Extract components
        self.components = self.grid_topology['component_id'].unique()
        self.num_components = len(self.components)
        
        # Define action and observation spaces
        # Action space: For each component, choose a hardening action (0: no action, 1: harden)
        self.action_space = spaces.Discrete(2**self.num_components)
        
        # Observation space: Grid state features
        # For each component: [vulnerability, failure_prob, age, criticality, is_failed, is_hardened]
        obs_dim = self.num_components * 6
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        logger.info(f"GridEnv initialized with {self.num_components} components")
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        # Initialize component states
        self.component_states = {
            'failed': np.zeros(self.num_components, dtype=bool),
            'hardened': np.zeros(self.num_components, dtype=bool),
            'remaining_capacity': np.ones(self.num_components)
        }
        
        # Select a random scenario or use baseline
        if self.scenarios is not None and len(self.scenarios) > 0:
            # Handle different data types for scenarios
            if isinstance(self.scenarios, pd.DataFrame):
                # If scenarios is a DataFrame, select a random row
                random_idx = np.random.randint(0, len(self.scenarios))
                scenario_row = self.scenarios.iloc[random_idx]
                self.current_scenario = {
                    'type': scenario_row.get('scenario_type', 'unknown'),
                    'intensity': float(scenario_row.get('cascading_impact', 0.0)) / 100.0,
                    'id': scenario_row.get('scenario_id', f'scenario_{random_idx}')
                }
            elif isinstance(self.scenarios, list) and len(self.scenarios) > 0:
                # If scenarios is a list, choose a random element
                self.current_scenario = np.random.choice(self.scenarios)
            else:
                # Default scenario
                self.current_scenario = {'type': 'baseline', 'intensity': 0.0}
        else:
            self.current_scenario = {'type': 'baseline', 'intensity': 0.0}
        
        # Reset time step
        self.current_step = 0
        self.max_steps = self.config.get('max_steps', 24)
        
        # Get initial observation
        observation = self._get_observation()
        
        # Return observation and empty info dict for compatibility with gymnasium
        return observation, {}
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: Integer representing the action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Decode action (binary representation of which components to harden)
        hardening_actions = self._decode_action(action)
        
        # Apply hardening actions
        for i, harden in enumerate(hardening_actions):
            if harden:
                self.component_states['hardened'][i] = True
        
        # Calculate cost of hardening
        hardening_cost = np.sum(hardening_actions) * self.config.get('hardening_cost', 0.1)
        
        # Simulate failures based on current scenario and component states
        new_failures = self._simulate_failures()
        
        # Update states based on failures
        self.component_states['failed'] = np.logical_or(
            self.component_states['failed'], new_failures
        )
        
        # Calculate reward
        # Negative reward for failed components, weighted by criticality
        criticality = self.grid_topology['criticality'].values
        failure_penalty = -np.sum(new_failures * criticality) * 10
        
        # Total reward is failure penalty minus hardening cost
        reward = failure_penalty - hardening_cost
        
        # Increment time step
        self.current_step += 1
        
        # Check if episode is done
        terminated = (self.current_step >= self.max_steps)
        
        # Truncated is False by default (not using time limit)
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'failures': np.sum(new_failures),
            'total_failed': np.sum(self.component_states['failed']),
            'hardened': np.sum(self.component_states['hardened']),
            'hardening_cost': hardening_cost,
            'failure_penalty': failure_penalty,
            'scenario': self.current_scenario['type']
        }
        
        return observation, reward, terminated, truncated, info
    
    def _decode_action(self, action):
        """
        Decode action integer into binary representation.
        
        Args:
            action: Integer action
            
        Returns:
            Binary array representing which components to harden
        """
        # Convert integer to binary representation
        binary = format(action, f'0{self.num_components}b')
        
        # Convert to array of 0s and 1s
        hardening_actions = np.array([int(b) for b in binary], dtype=int)
        
        return hardening_actions
    
    def _simulate_failures(self):
        """
        Simulate component failures based on current scenario and states.
        
        Returns:
            Binary array of new failures
        """
        # Get baseline failure probabilities
        failure_probs = self.failure_predictions['failure_probability'].values
        
        # Adjust probabilities based on scenario
        scenario_intensity = self.current_scenario.get('intensity', 0.0)
        adjusted_probs = failure_probs * (1 + scenario_intensity)
        
        # Reduce failure probability for hardened components
        hardening_effectiveness = self.config.get('hardening_effectiveness', 0.75)
        adjusted_probs = adjusted_probs * (1 - self.component_states['hardened'] * hardening_effectiveness)
        
        # Ensure probabilities are within [0, 1]
        adjusted_probs = np.clip(adjusted_probs, 0, 1)
        
        # Sample failures
        new_failures = np.random.random(self.num_components) < adjusted_probs
        
        # Don't fail already failed components
        new_failures = np.logical_and(new_failures, ~self.component_states['failed'])
        
        return new_failures
    
    def _get_observation(self):
        """
        Get current observation state.
        
        Returns:
            Observation array
        """
        # Component features
        vulnerability = self.vulnerability_scores['vulnerability_score'].values
        failure_prob = self.failure_predictions['failure_probability'].values
        age = self.grid_topology['age'].values / 20.0  # Normalize
        criticality = self.grid_topology['criticality'].values
        is_failed = self.component_states['failed'].astype(float)
        is_hardened = self.component_states['hardened'].astype(float)
        
        # Combine features for each component
        observation = np.concatenate([
            vulnerability, failure_prob, age, criticality, is_failed, is_hardened
        ])
        
        return observation
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            status = []
            for i in range(self.num_components):
                comp_id = self.components[i]
                state = "FAILED" if self.component_states['failed'][i] else "OK"
                hardened = " (Hardened)" if self.component_states['hardened'][i] else ""
                status.append(f"Component {comp_id}: {state}{hardened}")
            
            print(f"\nStep {self.current_step}/{self.max_steps}")
            print(f"Scenario: {self.current_scenario['type']}")
            print("\n".join(status))
            print(f"Failed: {np.sum(self.component_states['failed'])}/{self.num_components}")
            print(f"Hardened: {np.sum(self.component_states['hardened'])}/{self.num_components}")
            
    def close(self):
        """
        Clean up environment resources.
        """
        pass
