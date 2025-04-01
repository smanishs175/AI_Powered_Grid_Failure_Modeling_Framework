#!/usr/bin/env python
"""
Compound Scenario Generator

This module generates scenarios under compound extreme weather events.
"""

import numpy as np
import pandas as pd
from .base_scenario_generator import BaseScenarioGenerator

class CompoundScenarioGenerator(BaseScenarioGenerator):
    """
    Generator for compound extreme event scenarios.
    
    This class generates scenarios that represent failure patterns
    under combinations of multiple extreme weather events.
    """
    
    def __init__(self, config=None):
        """
        Initialize the compound scenario generator.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        super().__init__(config)
        
        # Set default compound event types if not provided in config
        self.compound_types = config.get('compound_types', [
            ('high_temperature', 'high_wind'),
            ('high_wind', 'precipitation'),
            ('low_temperature', 'high_wind'),
            ('high_temperature', 'precipitation')
        ])
        
        # Additional severity multiplier for compound events
        self.compound_multiplier = config.get('compound_multiplier', 1.5)
    
    def generate_scenarios(self, input_data, extreme_scenarios=None, count=30):
        """
        Generate compound extreme event scenarios.
        
        Args:
            input_data (dict): Dictionary containing input data.
            extreme_scenarios (dict): Dictionary mapping event types to lists of scenarios.
            count (int): Number of compound scenarios to generate.
            
        Returns:
            list: List of generated compound scenario dictionaries.
        """
        self.logger.info(f"Generating {count} compound extreme event scenarios")
        
        # Extract required data
        components = input_data.get('components', pd.DataFrame())
        failure_probabilities = input_data.get('failure_probabilities', {})
        extreme_event_models = input_data.get('extreme_event_models', {})
        
        # Check that we have the necessary data
        if components.empty:
            self.logger.warning("No component data provided, cannot generate scenarios")
            return []
        
        # If no extreme scenarios provided, we can't create compound scenarios
        if not extreme_scenarios:
            self.logger.warning("No extreme scenarios provided, cannot generate compound scenarios")
            return []
        
        # Generate compound scenarios
        compound_scenarios = []
        
        # Distribute scenarios among available compound types
        scenarios_per_type = max(1, count // len(self.compound_types))
        
        for compound_type in self.compound_types:
            # Skip if any of the component types don't have scenarios
            if any(et not in extreme_scenarios for et in compound_type):
                continue
                
            self.logger.info(f"Generating scenarios for compound type: {'+'.join(compound_type)}")
            
            for i in range(scenarios_per_type):
                scenario_id = self.generate_scenario_id(prefix=f"compound_{'+'.join(compound_type)}")
                
                # Generate weather conditions by combining extreme events
                weather_conditions = self._combine_weather_conditions(
                    compound_type,
                    extreme_scenarios
                )
                
                # Apply compound event impact models
                adjusted_probabilities = self._apply_compound_impacts(
                    compound_type,
                    failure_probabilities, 
                    components,
                    extreme_event_models
                )
                
                # Select components to fail
                # Compound events generally cause even more failures
                failure_count = np.random.randint(
                    max(2, int(components.shape[0] * 0.05)),
                    max(3, int(components.shape[0] * 0.15))
                )
                
                failed_components = self._select_components_to_fail(
                    components, 
                    adjusted_probabilities, 
                    count=failure_count,
                    max_percentage=0.25  # Allow even higher percentage for compound events
                )
                
                # Generate failure times - clustered very tightly for compound events
                failure_times = self._generate_failure_times(
                    len(failed_components),
                    time_window=8.0  # Even shorter time window for compound events
                )
                
                # Determine failure causes
                failure_causes = self._determine_failure_causes(
                    failed_components, 
                    components, 
                    scenario_type='compound'
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
                    'metadata': self._create_scenario_metadata('compound_event_model', '+'.join(compound_type))
                }
                
                compound_scenarios.append(scenario)
        
        self.logger.info(f"Generated {len(compound_scenarios)} compound scenarios")
        return compound_scenarios
    
    def _combine_weather_conditions(self, compound_type, extreme_scenarios):
        """
        Combine weather conditions from multiple extreme events.
        
        Args:
            compound_type (tuple): Tuple of extreme event types to combine.
            extreme_scenarios (dict): Dictionary mapping event types to lists of scenarios.
            
        Returns:
            dict: Combined weather conditions.
        """
        # Get sample scenarios of each type
        sample_scenarios = []
        for event_type in compound_type:
            if event_type in extreme_scenarios and extreme_scenarios[event_type]:
                # Randomly select a scenario of this type
                sample_scenarios.append(np.random.choice(extreme_scenarios[event_type]))
        
        if not sample_scenarios:
            self.logger.warning("No sample scenarios found for compound type")
            # Return default severe weather conditions
            return {
                'temperature': 40.0,
                'wind_speed': 60.0,
                'precipitation': 100.0,
                'humidity': 90.0,
                'pressure': 960.0,
                'is_extreme_temperature': True,
                'is_extreme_wind': True,
                'is_extreme_precipitation': True,
                'heat_wave_day': True,
                'storm_day': True
            }
        
        # Combine weather conditions, taking the more extreme value for each
        combined_conditions = {}
        
        # Define which condition is "more extreme" for each parameter
        extreme_direction = {
            'temperature': 'max' if 'high_temperature' in compound_type else 'min',
            'wind_speed': 'max',
            'precipitation': 'max',
            'humidity': 'max' if 'precipitation' in compound_type else 'min',
            'pressure': 'min'  # Lower pressure is generally more extreme
        }
        
        # Combine numeric parameters
        for param, direction in extreme_direction.items():
            values = [s['weather_conditions'].get(param, 0) for s in sample_scenarios]
            if values:
                if direction == 'max':
                    combined_conditions[param] = max(values)
                else:
                    combined_conditions[param] = min(values)
        
        # Combine boolean flags (OR operation)
        for flag in ['is_extreme_temperature', 'is_extreme_wind', 'is_extreme_precipitation',
                     'heat_wave_day', 'cold_snap_day', 'storm_day']:
            combined_conditions[flag] = any(s['weather_conditions'].get(flag, False) 
                                           for s in sample_scenarios)
        
        return combined_conditions
    
    def _apply_compound_impacts(self, compound_type, failure_probabilities, components, extreme_event_models):
        """
        Apply compound event impact models to adjust failure probabilities.
        
        Args:
            compound_type (tuple): Tuple of extreme event types in this compound event.
            failure_probabilities (dict): Base failure probabilities by component ID.
            components (DataFrame): DataFrame containing component information.
            extreme_event_models (dict): Extreme event impact models from Module 3.
            
        Returns:
            dict: Adjusted failure probabilities.
        """
        # Start with a copy of the original probabilities
        adjusted_probabilities = failure_probabilities.copy()
        
        # Apply each individual impact model
        for event_type in compound_type:
            if extreme_event_models and event_type in extreme_event_models:
                # Apply impact model for this event type
                impact_model = extreme_event_models[event_type]
                
                # Placeholder for actual model application
                self.logger.info(f"Applying {event_type} impact model for compound scenario")
                
                # Basic adjustment based on vulnerability (placeholder)
                if not components.empty and 'vulnerability_score' in components.columns:
                    for _, component in components.iterrows():
                        comp_id = component['component_id']
                        base_prob = adjusted_probabilities.get(comp_id, 0.01)
                        
                        # Apply vulnerability-based adjustment
                        vulnerability = component.get('vulnerability_score', 0.5)
                        
                        # Adjust based on event type and vulnerability
                        if event_type == 'high_temperature':
                            # Example: transformers more vulnerable to high temperature
                            if 'type' in component.columns and component['type'].lower() == 'transformer':
                                multiplier = 3.0
                            else:
                                multiplier = 2.0
                        elif event_type == 'high_wind':
                            # Example: lines more vulnerable to high wind
                            if 'type' in component.columns and component['type'].lower() == 'line':
                                multiplier = 3.5
                            else:
                                multiplier = 2.0
                        else:
                            multiplier = 2.0
                        
                        # Apply the adjustment
                        impact_factor = 1.0 + (vulnerability * (multiplier - 1.0))
                        adjusted_probabilities[comp_id] = min(0.95, base_prob * impact_factor)
        
        # Apply additional compound effect (synergistic effects are worse than individual events)
        for comp_id, prob in adjusted_probabilities.items():
            adjusted_probabilities[comp_id] = min(0.99, prob * self.compound_multiplier)
        
        return adjusted_probabilities
