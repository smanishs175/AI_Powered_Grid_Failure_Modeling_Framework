#!/usr/bin/env python
"""
Extreme Event Scenario Generator

This module generates scenarios under extreme weather events.
"""

import numpy as np
import pandas as pd
from .base_scenario_generator import BaseScenarioGenerator

class ExtremeEventScenarioGenerator(BaseScenarioGenerator):
    """
    Generator for extreme weather event scenarios.
    
    This class generates scenarios that represent failure patterns
    under different types of extreme weather events.
    """
    
    def __init__(self, config=None):
        """
        Initialize the extreme event scenario generator.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        super().__init__(config)
        
        # Define extreme condition ranges
        self.extreme_conditions = {
            'high_temperature': {
                'temperature': (35, 45),
                'humidity': (60, 95),
                'wind_speed': (0, 10),
                'precipitation': (0, 2),
                'pressure': (980, 1010),
                'is_extreme_temperature': True,
                'heat_wave_day': True
            },
            'low_temperature': {
                'temperature': (-30, -5),
                'humidity': (40, 70),
                'wind_speed': (5, 15),
                'precipitation': (0, 5),
                'pressure': (990, 1030),
                'is_extreme_temperature': True,
                'cold_snap_day': True
            },
            'high_wind': {
                'temperature': (5, 30),
                'humidity': (30, 80),
                'wind_speed': (35, 70),
                'precipitation': (0, 10),
                'pressure': (960, 990),
                'is_extreme_wind': True,
                'storm_day': True
            },
            'precipitation': {
                'temperature': (5, 25),
                'humidity': (80, 100),
                'wind_speed': (10, 40),
                'precipitation': (30, 150),
                'pressure': (960, 990),
                'is_extreme_precipitation': True,
                'storm_day': True
            }
        }
        
        # Override with config values if provided
        if 'extreme_conditions' in config:
            for event_type, conditions in config['extreme_conditions'].items():
                if event_type in self.extreme_conditions:
                    self.extreme_conditions[event_type].update(conditions)
        
        # Set failure multipliers for extreme events
        self.failure_multipliers = config.get('failure_multipliers', {
            'high_temperature': 3.0,
            'low_temperature': 2.5,
            'high_wind': 4.0,
            'precipitation': 3.5
        })
    
    def generate_scenarios(self, input_data, event_types=None, count_per_type=20):
        """
        Generate extreme event scenarios.
        
        Args:
            input_data (dict): Dictionary containing input data.
            event_types (list, optional): List of extreme event types to generate. If None,
                                          uses all available types.
            count_per_type (int): Number of scenarios to generate per event type.
            
        Returns:
            dict: Dictionary mapping event types to lists of generated scenarios.
        """
        self.logger.info(f"Generating extreme event scenarios")
        
        # Extract required data
        components = input_data.get('components', pd.DataFrame())
        failure_probabilities = input_data.get('failure_probabilities', {})
        extreme_event_models = input_data.get('extreme_event_models', {})
        
        # Check that we have the necessary data
        if components.empty:
            self.logger.warning("No component data provided, cannot generate scenarios")
            return {}
        
        # Determine event types to generate
        if event_types is None or not event_types:
            event_types = list(self.extreme_conditions.keys())
        else:
            # Filter to valid event types
            event_types = [et for et in event_types if et in self.extreme_conditions]
        
        # Generate scenarios for each event type
        scenarios_by_type = {}
        
        for event_type in event_types:
            self.logger.info(f"Generating {count_per_type} scenarios for event type: {event_type}")
            
            # Generate scenarios for this event type
            event_scenarios = []
            
            for i in range(count_per_type):
                scenario_id = self.generate_scenario_id(prefix=f"extreme_{event_type}")
                
                # Generate weather conditions for this event type
                weather_conditions = self._generate_extreme_weather_conditions(event_type)
                
                # Apply extreme event impact models if available
                adjusted_probabilities = self._apply_extreme_event_impacts(
                    event_type,
                    failure_probabilities, 
                    components,
                    extreme_event_models
                )
                
                # Select components to fail
                # Extreme events generally cause more failures
                failure_count = np.random.randint(
                    max(1, int(components.shape[0] * 0.03)),
                    max(2, int(components.shape[0] * 0.1))
                )
                
                failed_components = self._select_components_to_fail(
                    components, 
                    adjusted_probabilities, 
                    count=failure_count,
                    max_percentage=0.2  # Allow higher percentage for extreme events
                )
                
                # Generate failure times - clustered more tightly for extreme events
                failure_times = self._generate_failure_times(
                    len(failed_components),
                    time_window=12.0  # Shorter time window for extreme events
                )
                
                # Determine failure causes
                failure_causes = self._determine_failure_causes(
                    failed_components, 
                    components, 
                    scenario_type=event_type
                )
                
                # Create component failures dictionary
                component_failures = {}
                for j, comp_id in enumerate(failed_components):
                    component_failures[comp_id] = {
                        'failed': True,
                        'failure_time': failure_times[j],
                        'failure_cause': failure_causes[comp_id]
                    }
                
                # Create scenario
                scenario = {
                    'scenario_id': scenario_id,
                    'weather_conditions': weather_conditions,
                    'component_failures': component_failures,
                    'metadata': self._create_scenario_metadata('extreme_event_model', event_type)
                }
                
                event_scenarios.append(scenario)
            
            scenarios_by_type[event_type] = event_scenarios
            self.logger.info(f"Generated {len(event_scenarios)} scenarios for {event_type} events")
        
        return scenarios_by_type
    
    def _generate_extreme_weather_conditions(self, event_type):
        """
        Generate weather conditions for an extreme event.
        
        Args:
            event_type (str): Type of extreme event.
            
        Returns:
            dict: Dictionary of weather conditions.
        """
        if event_type not in self.extreme_conditions:
            self.logger.warning(f"Unknown event type {event_type}, using high_temperature as default")
            event_type = 'high_temperature'
        
        # Get condition ranges for this event type
        conditions = self.extreme_conditions[event_type]
        
        # Generate random values within the specified ranges
        weather_conditions = {}
        for key, value in conditions.items():
            if isinstance(value, tuple) and len(value) == 2:
                # It's a range, generate a random value
                weather_conditions[key] = np.random.uniform(value[0], value[1])
            else:
                # It's a fixed value, use as is
                weather_conditions[key] = value
        
        # Initialize all boolean flags to False by default
        for flag in ['is_extreme_temperature', 'is_extreme_wind', 'is_extreme_precipitation',
                    'heat_wave_day', 'cold_snap_day', 'storm_day']:
            if flag not in weather_conditions:
                weather_conditions[flag] = False
        
        return weather_conditions
    
    def _apply_extreme_event_impacts(self, event_type, failure_probabilities, components, extreme_event_models):
        """
        Apply extreme event impact models to adjust failure probabilities.
        
        Args:
            event_type (str): Type of extreme event.
            failure_probabilities (dict): Base failure probabilities by component ID.
            components (DataFrame): DataFrame containing component information.
            extreme_event_models (dict): Extreme event impact models from Module 3.
            
        Returns:
            dict: Adjusted failure probabilities.
        """
        # Start with a copy of the original probabilities
        adjusted_probabilities = failure_probabilities.copy()
        
        # Get the multiplier for this event type
        multiplier = self.failure_multipliers.get(event_type, 2.0)
        
        # Apply the impact models if available
        if extreme_event_models and event_type in extreme_event_models:
            impact_model = extreme_event_models[event_type]
            
            # Apply sophisticated model from Module 3 (placeholder)
            # In a real implementation, this would use the actual impact model
            self.logger.info(f"Applying {event_type} impact model to adjust failure probabilities")
            
            # For now, just apply a basic adjustment based on vulnerability
            if not components.empty and 'vulnerability_score' in components.columns:
                for _, component in components.iterrows():
                    comp_id = component['component_id']
                    base_prob = adjusted_probabilities.get(comp_id, 0.01)
                    
                    # Apply vulnerability-based adjustment
                    vulnerability = component.get('vulnerability_score', 0.5)
                    
                    # Higher vulnerability means higher impact from extreme events
                    impact_factor = 1.0 + (vulnerability * (multiplier - 1.0))
                    
                    # Apply the adjustment
                    adjusted_probabilities[comp_id] = min(0.95, base_prob * impact_factor)
        else:
            # No specific impact model, apply general multiplier
            self.logger.info(f"No specific impact model for {event_type}, applying general multiplier")
            for comp_id, prob in adjusted_probabilities.items():
                adjusted_probabilities[comp_id] = min(0.95, prob * multiplier)
        
        return adjusted_probabilities
