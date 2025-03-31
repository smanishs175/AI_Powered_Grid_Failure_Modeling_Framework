"""
Grid Environment for Reinforcement Learning.

This module implements a custom OpenAI Gym environment for grid management,
simulating grid operations under various conditions and failures.
"""

import os
import logging
import numpy as np
import gym
from gym import spaces
import networkx as nx
import pandas as pd
import random
from collections import defaultdict

class GridEnvironment(gym.Env):
    """
    Custom OpenAI Gym environment for power grid management.
    
    This environment simulates the operation of a power grid under various
    conditions, including component failures, weather events, and load dynamics.
    It integrates with scenario data from the Scenario Generation Module.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, scenarios=None, max_steps=100, reward_weights=None):
        """
        Initialize the grid environment.
        
        Args:
            scenarios (dict): Dictionary of scenario data from Module 4.
            max_steps (int): Maximum number of steps per episode.
            reward_weights (dict): Weights for different reward components.
        """
        super(GridEnvironment, self).__init__()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Grid Environment")
        
        # Store parameters
        self.scenarios = scenarios or {}
        self.max_steps = max_steps
        self.reward_weights = reward_weights or {
            'stability': 1.0,
            'outage': -1.0,
            'load_shedding': -0.5,
            'action': -0.1,
            'recovery': 0.5,
            'preventive': 0.2
        }
        
        # Process scenario data
        self._process_scenario_data()
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Initialize environment state
        self.reset()
        
        self.logger.info("Grid Environment initialized")
    
    def _process_scenario_data(self):
        """
        Process the scenario data from Module 4.
        
        This function extracts and formats data from the scenario dictionaries
        for use in the environment.
        """
        self.logger.info("Processing scenario data")
        
        # Extract component data
        self.components = []
        self.component_types = set()
        self.component_by_id = {}
        
        # Default scenario types if not provided
        default_scenario_types = ['normal', 'extreme', 'compound']
        self.scenario_types = [s_type for s_type in default_scenario_types 
                              if s_type in self.scenarios and self.scenarios[s_type]]
        
        if not self.scenario_types:
            self.logger.warning("No valid scenarios found, creating synthetic scenarios")
            self._create_synthetic_scenarios()
            self.scenario_types = ['synthetic']
            
        # Extract components from first scenario type
        first_type = self.scenario_types[0]
        if self.scenarios[first_type]:
            first_scenario = self.scenarios[first_type][0]
            if 'grid_components' in first_scenario:
                self.components = first_scenario['grid_components'].copy()
                for component in self.components:
                    self.component_types.add(component.get('type', 'unknown'))
                    self.component_by_id[component.get('id')] = component
        
        # If no components found, create synthetic ones
        if not self.components:
            self.logger.warning("No component data found, creating synthetic components")
            self._create_synthetic_components()
            
        # Extract cascade models if available
        self.cascade_models = self.scenarios.get('cascade_models', {})
        
        # Create grid network
        self._create_grid_network()
        
        self.logger.info(f"Processed data for {len(self.components)} components "
                        f"of {len(self.component_types)} types")
        self.logger.info(f"Available scenario types: {', '.join(self.scenario_types)}")
    
    def _create_synthetic_components(self, num_components=50):
        """
        Create synthetic grid components for testing.
        
        Args:
            num_components (int): Number of components to create.
        """
        self.logger.info(f"Creating {num_components} synthetic components")
        
        # Component types and their properties
        types = ['generator', 'transformer', 'substation', 'transmission_line', 'distribution_line']
        
        for i in range(num_components):
            comp_type = random.choice(types)
            
            component = {
                'id': f"comp_{i}",
                'type': comp_type,
                'capacity': random.uniform(50, 200) if comp_type != 'substation' else 0,
                'load': random.uniform(20, 100) if comp_type != 'generator' else 0,
                'age': random.randint(0, 30),
                'failure_rate': random.uniform(0.01, 0.1),
                'connections': []
            }
            
            # Add connections to existing components (ensure connected graph)
            if i > 0:
                num_connections = random.randint(1, min(3, i))
                connected_ids = random.sample([c['id'] for c in self.components], num_connections)
                
                for conn_id in connected_ids:
                    component['connections'].append(conn_id)
                    # Add the reverse connection
                    for existing in self.components:
                        if existing['id'] == conn_id:
                            if 'connections' not in existing:
                                existing['connections'] = []
                            existing['connections'].append(component['id'])
            
            self.components.append(component)
            self.component_types.add(comp_type)
            self.component_by_id[component['id']] = component
        
        self.logger.info(f"Created {len(self.components)} synthetic components "
                        f"of {len(self.component_types)} types")
    
    def _create_synthetic_scenarios(self, num_scenarios=20):
        """
        Create synthetic scenarios for testing.
        
        Args:
            num_scenarios (int): Number of scenarios to create.
        """
        self.logger.info(f"Creating {num_scenarios} synthetic scenarios")
        
        # Ensure we have components
        if not self.components:
            self._create_synthetic_components()
            
        # Create scenarios with random initial failures
        self.scenarios['synthetic'] = []
        
        for i in range(num_scenarios):
            # Select random components to fail initially
            num_failures = random.randint(1, max(1, len(self.components) // 10))
            failed_components = random.sample(self.components, num_failures)
            
            # Generate random weather conditions
            weather = {
                'temperature': random.uniform(-10, 35),
                'wind_speed': random.uniform(0, 40),
                'precipitation': random.uniform(0, 100)
            }
            
            # Create scenario
            scenario = {
                'id': f"synthetic_{i}",
                'description': f"Synthetic scenario {i}",
                'type': 'synthetic',
                'initial_failures': [c['id'] for c in failed_components],
                'weather_conditions': weather,
                'grid_components': self.components,
                'cascade_progression': [
                    {
                        'step': 0,
                        'newly_failed': [c['id'] for c in failed_components],
                        'all_failed': [c['id'] for c in failed_components]
                    }
                ]
            }
            
            self.scenarios['synthetic'].append(scenario)
            
        self.logger.info(f"Created {len(self.scenarios['synthetic'])} synthetic scenarios")
    
    def _create_grid_network(self):
        """
        Create a NetworkX graph representation of the grid.
        """
        self.logger.info("Creating grid network")
        
        self.grid_network = nx.DiGraph()
        
        # Add nodes for each component
        for component in self.components:
            self.grid_network.add_node(
                component['id'],
                **{k: v for k, v in component.items() if k != 'connections'}
            )
        
        # Add edges for connections
        for component in self.components:
            if 'connections' in component:
                for conn_id in component['connections']:
                    # Add directed edge
                    self.grid_network.add_edge(
                        component['id'],
                        conn_id,
                        weight=1.0
                    )
        
        self.logger.info(f"Created grid network with {self.grid_network.number_of_nodes()} nodes "
                        f"and {self.grid_network.number_of_edges()} edges")
    
    def _define_spaces(self):
        """
        Define action and observation spaces.
        """
        self.logger.info("Defining action and observation spaces")
        
        # Action space: 0=do nothing, 1=adjust generation, 2=reroute power
        self.action_space = spaces.Discrete(3)
        
        # State space components
        num_components = len(self.components)
        
        # Line status: 0=failed, 1=operational
        line_status_dim = num_components
        
        # Weather conditions: temperature, wind_speed, precipitation
        weather_dim = 3
        
        # Load demand for each component
        load_dim = num_components
        
        # Combine dimensions
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(line_status_dim + weather_dim + load_dim,),
            dtype=np.float32
        )
        
        self.logger.info(f"Observation space shape: {self.observation_space.shape}")
        self.logger.info(f"Action space: {self.action_space}")
    
    def reset(self):
        """
        Reset the environment to a new episode.
        
        Returns:
            numpy.ndarray: Initial observation.
        """
        self.logger.info("Resetting environment")
        
        # Reset step counter
        self.current_step = 0
        
        # Select a random scenario type and scenario
        scenario_type = random.choice(self.scenario_types)
        self.current_scenario = random.choice(self.scenarios[scenario_type])
        
        self.logger.info(f"Selected scenario {self.current_scenario['id']} of type {scenario_type}")
        
        # Initialize grid state
        self.failed_components = set(self.current_scenario.get('initial_failures', []))
        self.operational_components = set(c['id'] for c in self.components) - self.failed_components
        
        # Initialize weather conditions
        self.weather_conditions = self.current_scenario.get('weather_conditions', {
            'temperature': 20.0,
            'wind_speed': 5.0,
            'precipitation': 0.0
        })
        
        # Initialize load state
        self.current_loads = {}
        for component in self.components:
            # Default to component's base load or a random value
            self.current_loads[component['id']] = component.get('load', random.uniform(20, 100))
        
        # Track episode statistics
        self.episode_stats = {
            'total_outages': len(self.failed_components),
            'load_shed': 0.0,
            'actions_taken': 0,
            'recovered_components': 0
        }
        
        # Get initial observation
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current observation from the environment.
        
        Returns:
            numpy.ndarray: Current observation.
        """
        # Component status vector (1=operational, 0=failed)
        status_vector = np.zeros(len(self.components))
        
        for i, component in enumerate(self.components):
            status_vector[i] = 1.0 if component['id'] not in self.failed_components else 0.0
        
        # Weather conditions vector
        weather_vector = np.array([
            self.weather_conditions.get('temperature', 20.0),
            self.weather_conditions.get('wind_speed', 5.0),
            self.weather_conditions.get('precipitation', 0.0)
        ])
        
        # Normalize weather values to reasonable ranges
        weather_vector[0] = (weather_vector[0] + 10) / 45  # Temp from -10 to 35
        weather_vector[1] = weather_vector[1] / 40         # Wind from 0 to 40
        weather_vector[2] = weather_vector[2] / 100        # Precipitation from 0 to 100
        
        # Load vector
        load_vector = np.zeros(len(self.components))
        
        for i, component in enumerate(self.components):
            # Normalize load by component capacity or a default value
            capacity = component.get('capacity', 100.0)
            if capacity > 0:
                load_vector[i] = self.current_loads.get(component['id'], 0.0) / capacity
            else:
                load_vector[i] = self.current_loads.get(component['id'], 0.0) / 100.0
        
        # Combine vectors into observation
        observation = np.concatenate([status_vector, weather_vector, load_vector])
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take (0=do nothing, 1=adjust generation, 2=reroute power).
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.logger.debug(f"Step {self.current_step}, Action: {action}")
        
        # Increment step counter
        self.current_step += 1
        
        # Execute action
        action_result = self._execute_action(action)
        
        # Update grid state
        self._update_grid_state()
        
        # Calculate reward
        reward = self._calculate_reward(action, action_result)
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps or 
                len(self.operational_components) == 0)
        
        # Get observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'step': self.current_step,
            'action': action,
            'action_result': action_result,
            'num_failed': len(self.failed_components),
            'num_operational': len(self.operational_components),
            'weather': self.weather_conditions,
            'episode_stats': self.episode_stats
        }
        
        return observation, reward, done, info
    
    def _execute_action(self, action):
        """
        Execute the selected action.
        
        Args:
            action (int): Action to take.
            
        Returns:
            dict: Results of the action.
        """
        result = {
            'success': False,
            'prevented_failures': [],
            'recovered_components': [],
            'load_adjustment': 0.0
        }
        
        if action == 0:
            # Do nothing
            result['success'] = True
            return result
            
        elif action == 1:
            # Adjust generation
            self.episode_stats['actions_taken'] += 1
            
            # Find operational generators
            generators = [c for c in self.components 
                         if c['type'] == 'generator' and c['id'] not in self.failed_components]
            
            if not generators:
                # No operational generators
                return result
            
            # Adjust load distribution
            total_load = sum(self.current_loads.values())
            total_capacity = sum(c.get('capacity', 100.0) for c in generators)
            
            # Can't exceed capacity
            if total_load > total_capacity:
                load_adjustment = total_capacity - total_load
                for comp_id in self.current_loads:
                    self.current_loads[comp_id] *= (total_capacity / total_load)
                
                # Record load shed
                load_shed = total_load - total_capacity
                self.episode_stats['load_shed'] += load_shed
                result['load_adjustment'] = -load_shed
            else:
                # Even distribution across generators
                for gen in generators:
                    # Assign target load proportionally to capacity
                    capacity = gen.get('capacity', 100.0)
                    target_load = (capacity / total_capacity) * total_load
                    self.current_loads[gen['id']] = target_load
            
            result['success'] = True
            
        elif action == 2:
            # Reroute power
            self.episode_stats['actions_taken'] += 1
            
            # Identify vulnerable but still operational components
            vulnerable = []
            for component in self.components:
                if component['id'] in self.operational_components:
                    failure_rate = component.get('failure_rate', 0.05)
                    if failure_rate > 0.2:  # Consider high failure rate as vulnerable
                        vulnerable.append(component['id'])
            
            if not vulnerable:
                # No vulnerable components to protect
                return result
            
            # Try to reroute power around vulnerable components
            prevented = []
            for comp_id in vulnerable:
                # Check if this component would fail in the next step
                if self._would_component_fail(comp_id):
                    # Try to prevent failure by reducing load
                    if self._reduce_component_load(comp_id):
                        prevented.append(comp_id)
            
            result['prevented_failures'] = prevented
            result['success'] = len(prevented) > 0
            
        return result
    
    def _would_component_fail(self, component_id):
        """
        Check if a component would fail in the next step without intervention.
        
        Args:
            component_id (str): ID of the component to check.
            
        Returns:
            bool: True if component would fail, False otherwise.
        """
        component = self.component_by_id.get(component_id)
        if not component:
            return False
            
        # Base failure probability
        failure_rate = component.get('failure_rate', 0.05)
        
        # Adjust based on current load
        capacity = component.get('capacity', 100.0)
        current_load = self.current_loads.get(component_id, 0.0)
        
        if capacity > 0:
            load_ratio = current_load / capacity
            if load_ratio > 0.9:
                failure_rate *= 3.0
            elif load_ratio > 0.7:
                failure_rate *= 1.5
        
        # Adjust based on weather
        temp = self.weather_conditions.get('temperature', 20.0)
        wind = self.weather_conditions.get('wind_speed', 5.0)
        precip = self.weather_conditions.get('precipitation', 0.0)
        
        # Temperature effects (high or low temps increase failure rate)
        if temp > 30 or temp < 0:
            failure_rate *= 1.5
            
        # Wind effects (high wind increases failure rate for lines)
        if wind > 25 and component.get('type') in ['transmission_line', 'distribution_line']:
            failure_rate *= 2.0
            
        # Precipitation effects (high precipitation increases failure rate)
        if precip > 50:
            failure_rate *= 1.3
        
        # Check against threshold
        return random.random() < failure_rate
    
    def _reduce_component_load(self, component_id):
        """
        Reduce load on a component to prevent failure.
        
        Args:
            component_id (str): ID of the component to protect.
            
        Returns:
            bool: True if load was successfully reduced, False otherwise.
        """
        component = self.component_by_id.get(component_id)
        if not component:
            return False
            
        # Current load and capacity
        current_load = self.current_loads.get(component_id, 0.0)
        capacity = component.get('capacity', 100.0)
        
        if current_load <= 0 or capacity <= 0:
            return False
            
        # Reduce load to 50% of capacity
        target_load = capacity * 0.5
        if current_load > target_load:
            # Record load shed
            load_shed = current_load - target_load
            self.episode_stats['load_shed'] += load_shed
            
            # Reduce load
            self.current_loads[component_id] = target_load
            return True
            
        return False
    
    def _update_grid_state(self):
        """
        Update the grid state, including component failures and weather changes.
        """
        # Get current cascade step if available
        cascade_step = None
        if 'cascade_progression' in self.current_scenario:
            progression = self.current_scenario['cascade_progression']
            if self.current_step < len(progression):
                cascade_step = progression[self.current_step]
        
        # If we have cascade data, use it
        newly_failed = []
        if cascade_step is not None:
            newly_failed = cascade_step.get('newly_failed', [])
            self.failed_components = set(cascade_step.get('all_failed', self.failed_components))
            self.operational_components = set(c['id'] for c in self.components) - self.failed_components
        else:
            # Simulate failures based on current state
            for component_id in list(self.operational_components):
                if self._would_component_fail(component_id):
                    self.failed_components.add(component_id)
                    self.operational_components.remove(component_id)
                    newly_failed.append(component_id)
        
        # Update weather (small random changes if not from scenario)
        if 'weather_progression' in self.current_scenario:
            if self.current_step < len(self.current_scenario['weather_progression']):
                self.weather_conditions = self.current_scenario['weather_progression'][self.current_step]
        else:
            # Small random changes
            self.weather_conditions['temperature'] += random.uniform(-1.0, 1.0)
            self.weather_conditions['wind_speed'] += random.uniform(-2.0, 2.0)
            if self.weather_conditions['wind_speed'] < 0:
                self.weather_conditions['wind_speed'] = 0.0
            self.weather_conditions['precipitation'] += random.uniform(-5.0, 5.0)
            if self.weather_conditions['precipitation'] < 0:
                self.weather_conditions['precipitation'] = 0.0
        
        # Update component loads based on failures
        if newly_failed:
            self._redistribute_loads(newly_failed)
            
        # Update episode statistics
        self.episode_stats['total_outages'] += len(newly_failed)
    
    def _redistribute_loads(self, failed_components):
        """
        Redistribute loads after component failures.
        
        Args:
            failed_components (list): List of newly failed component IDs.
        """
        # Calculate total lost load
        lost_load = sum(self.current_loads.get(comp_id, 0) for comp_id in failed_components)
        
        if lost_load <= 0:
            return
        
        # Find components that can take additional load
        available_components = []
        for component in self.components:
            if component['id'] in self.operational_components:
                capacity = component.get('capacity', 0)
                current_load = self.current_loads.get(component['id'], 0)
                if capacity > current_load:
                    available_components.append({
                        'id': component['id'],
                        'available_capacity': capacity - current_load
                    })
        
        # Calculate total available capacity
        total_available_capacity = sum(c['available_capacity'] for c in available_components)
        
        if total_available_capacity <= 0:
            # No capacity available, must shed load
            self.episode_stats['load_shed'] += lost_load
            return
        
        # Redistribute load proportionally to available capacity
        if total_available_capacity >= lost_load:
            # Can redistribute all lost load
            for component in available_components:
                additional_load = (component['available_capacity'] / total_available_capacity) * lost_load
                self.current_loads[component['id']] += additional_load
        else:
            # Can only redistribute some load, rest must be shed
            for component in available_components:
                additional_load = (component['available_capacity'] / total_available_capacity) * total_available_capacity
                self.current_loads[component['id']] += additional_load
            
            # Record load shed
            load_shed = lost_load - total_available_capacity
            self.episode_stats['load_shed'] += load_shed
    
    def _calculate_reward(self, action, action_result):
        """
        Calculate reward based on current state and action.
        
        Args:
            action (int): Action taken.
            action_result (dict): Results of the action.
            
        Returns:
            float: Reward value.
        """
        reward = 0.0
        
        # Stability reward: +1 for each operational component
        num_operational = len(self.operational_components)
        stability_reward = num_operational / len(self.components)
        reward += self.reward_weights.get('stability', 1.0) * stability_reward
        
        # Outage penalty: -1 for each failed component
        num_failed = len(self.failed_components)
        outage_penalty = num_failed / len(self.components)
        reward += self.reward_weights.get('outage', -1.0) * outage_penalty
        
        # Load shedding penalty
        if 'load_shed' in self.episode_stats and self.episode_stats['load_shed'] > 0:
            # Calculate total possible load
            total_possible_load = sum(c.get('load', 0) for c in self.components)
            if total_possible_load > 0:
                load_shed_ratio = self.episode_stats['load_shed'] / total_possible_load
                reward += self.reward_weights.get('load_shedding', -0.5) * load_shed_ratio
        
        # Action cost for non-idle actions
        if action > 0:
            reward += self.reward_weights.get('action', -0.1)
        
        # Recovery reward
        if 'recovered_components' in action_result and action_result['recovered_components']:
            recovery_reward = len(action_result['recovered_components']) / len(self.components)
            reward += self.reward_weights.get('recovery', 0.5) * recovery_reward
            
        # Preventive action reward
        if 'prevented_failures' in action_result and action_result['prevented_failures']:
            preventive_reward = len(action_result['prevented_failures']) / len(self.components)
            reward += self.reward_weights.get('preventive', 0.2) * preventive_reward
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the current state of the environment.
        
        Args:
            mode (str): Rendering mode ('human' or 'rgb_array').
            
        Returns:
            Union[None, numpy.ndarray]: If mode is 'rgb_array', returns image array.
        """
        if mode == 'human':
            print(f"\n=== Grid Environment State (Step {self.current_step}) ===")
            print(f"Operational Components: {len(self.operational_components)}/{len(self.components)}")
            print(f"Failed Components: {len(self.failed_components)}/{len(self.components)}")
            print(f"Weather: Temp={self.weather_conditions.get('temperature', 20.0):.1f}°C, "
                 f"Wind={self.weather_conditions.get('wind_speed', 5.0):.1f} mph, "
                 f"Precip={self.weather_conditions.get('precipitation', 0.0):.1f} mm")
            
            print("\nEpisode Statistics:")
            for key, value in self.episode_stats.items():
                print(f"  {key}: {value}")
                
            return None
            
        elif mode == 'rgb_array':
            # Implement visualization code to render as RGB array
            # This would typically use matplotlib to create a visualization
            # and convert it to an RGB array
            # For now, return a placeholder
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self):
        """
        Clean up resources.
        """
        pass
