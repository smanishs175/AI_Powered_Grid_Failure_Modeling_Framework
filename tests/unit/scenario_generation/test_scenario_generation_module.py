"""
Unit tests for the Scenario Generation Module (Module 4)
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gfmf.scenario_generation.scenario_generation_module import ScenarioGenerationModule


class TestScenarioGenerationModule(unittest.TestCase):
    """Test cases for the ScenarioGenerationModule class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sg_module = ScenarioGenerationModule()
        
        # Create mock vulnerability scores
        self.vulnerability_scores = pd.DataFrame({
            'component_id': range(1, 11),
            'type': ['line', 'transformer', 'bus', 'line', 'transformer', 
                     'bus', 'line', 'transformer', 'bus', 'line'],
            'vulnerability_score': np.random.uniform(0, 1, 10)
        })
        
        # Create mock failure predictions
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        self.failure_predictions = pd.DataFrame({
            'date': np.repeat(dates, 5),
            'component_id': np.tile(range(1, 6), 10),
            'failure_probability': np.random.uniform(0, 1, 50)
        })
        
        # Create mock weather data
        self.weather_data = pd.DataFrame({
            'date': dates,
            'temperature': np.random.uniform(-10, 40, 10),
            'precipitation': np.random.uniform(0, 50, 10),
            'humidity': np.random.uniform(10, 100, 10),
            'wind_speed': np.random.uniform(0, 30, 10),
            'extreme_weather': np.random.choice([0, 1], 10, p=[0.8, 0.2])
        })
        
        # Create mock grid data
        self.grid_data = pd.DataFrame({
            'component_id': range(1, 11),
            'type': ['line', 'transformer', 'bus', 'line', 'transformer', 
                     'bus', 'line', 'transformer', 'bus', 'line'],
            'capacity': np.random.uniform(50, 200, 10),
            'age': np.random.uniform(1, 20, 10),
            'criticality': np.random.uniform(0.1, 1.0, 10),
            'from': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'to': [1, 2, 3, 4, 5, 6, 7, 8, 9, None]
        })

    def test_initialization(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.sg_module)
        self.assertIsNotNone(self.sg_module.config)

    def test_generate_baseline_scenario(self):
        """Test baseline scenario generation."""
        baseline = self.sg_module.generate_baseline_scenario(
            self.grid_data, self.vulnerability_scores
        )
        
        self.assertIsInstance(baseline, dict)
        self.assertIn('scenario_id', baseline)
        self.assertIn('type', baseline)
        self.assertEqual(baseline['type'], 'baseline')
        self.assertIn('components', baseline)

    def test_generate_component_failure_scenarios(self):
        """Test component failure scenario generation."""
        failure_scenarios = self.sg_module.generate_component_failure_scenarios(
            self.grid_data, self.vulnerability_scores, self.failure_predictions, num_scenarios=3
        )
        
        self.assertIsInstance(failure_scenarios, list)
        if len(failure_scenarios) > 0:
            self.assertLessEqual(len(failure_scenarios), 3)
            
            # Check structure of first scenario
            first_scenario = failure_scenarios[0]
            self.assertIn('scenario_id', first_scenario)
            self.assertIn('type', first_scenario)
            self.assertEqual(first_scenario['type'], 'component_failure')
            self.assertIn('failed_components', first_scenario)

    def test_generate_weather_scenarios(self):
        """Test weather scenario generation."""
        weather_scenarios = self.sg_module.generate_weather_scenarios(
            self.grid_data, self.vulnerability_scores, self.weather_data, num_scenarios=3
        )
        
        self.assertIsInstance(weather_scenarios, list)
        if len(weather_scenarios) > 0:
            self.assertLessEqual(len(weather_scenarios), 3)
            
            # Check structure of first scenario
            first_scenario = weather_scenarios[0]
            self.assertIn('scenario_id', first_scenario)
            self.assertIn('type', first_scenario)
            self.assertEqual(first_scenario['type'], 'weather')
            self.assertIn('weather_conditions', first_scenario)

    def test_generate_compound_scenarios(self):
        """Test compound scenario generation."""
        # Generate component failure and weather scenarios first
        failure_scenarios = self.sg_module.generate_component_failure_scenarios(
            self.grid_data, self.vulnerability_scores, self.failure_predictions, num_scenarios=2
        )
        
        weather_scenarios = self.sg_module.generate_weather_scenarios(
            self.grid_data, self.vulnerability_scores, self.weather_data, num_scenarios=2
        )
        
        compound_scenarios = self.sg_module.generate_compound_scenarios(
            failure_scenarios, weather_scenarios, num_scenarios=2
        )
        
        self.assertIsInstance(compound_scenarios, list)
        if len(compound_scenarios) > 0:
            self.assertLessEqual(len(compound_scenarios), 2)
            
            # Check structure of first scenario
            first_scenario = compound_scenarios[0]
            self.assertIn('scenario_id', first_scenario)
            self.assertIn('type', first_scenario)
            self.assertEqual(first_scenario['type'], 'compound')
            self.assertIn('failed_components', first_scenario)
            self.assertIn('weather_conditions', first_scenario)

    def test_generate_scenarios(self):
        """Test the main scenario generation method."""
        scenarios = self.sg_module.generate_scenarios(
            self.grid_data, self.vulnerability_scores, 
            self.failure_predictions, self.weather_data
        )
        
        self.assertIsInstance(scenarios, dict)
        self.assertIn('baseline', scenarios)
        self.assertIn('component_failure', scenarios)
        self.assertIn('weather', scenarios)
        self.assertIn('compound', scenarios)
        
        # Check that each scenario type is a list
        self.assertIsInstance(scenarios['baseline'], list)
        self.assertIsInstance(scenarios['component_failure'], list)
        self.assertIsInstance(scenarios['weather'], list)
        self.assertIsInstance(scenarios['compound'], list)


if __name__ == '__main__':
    unittest.main()
