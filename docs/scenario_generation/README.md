# Scenario Generation Module

## Overview
The Scenario Generation Module (Module 4) is a critical component of the AI-Powered Grid Failure Modeling Framework. It builds upon outputs from previous modules to generate a diverse set of realistic grid failure scenarios. These scenarios can be used for grid vulnerability analysis, planning, and resilience testing.

## Features
- Generation of multiple scenario types:
  - Normal operating condition scenarios
  - Extreme event scenarios
  - Compound extreme event scenarios
- Cascading failure modeling
- Scenario validation against historical data
- Configurable scenario parameters
- Integration with failure prediction, time-series forecasting, and extreme event modeling modules

## Module Structure
```
gfmf/scenario_generation/
│
├── config/                  # Configuration files
│   └── default_config.yaml  # Default configuration parameters
│
├── models/                  # Core scenario generation models
│   ├── base_scenario_generator.py     # Base class with common functionality
│   ├── normal_scenario_generator.py   # For normal operating conditions
│   ├── extreme_scenario_generator.py  # For extreme weather events
│   ├── compound_scenario_generator.py # For compound extreme events
│   ├── cascading_failure_model.py     # Failure propagation simulation
│   └── scenario_validator.py          # Ensures scenario validity
│
├── utils/                   # Utility functions
│   ├── data_loader.py       # Load data from previous modules
│   └── model_utils.py       # Common utilities for all components
│
└── scenario_generation_module.py  # Main module entry point
```

## How It Works
1. The module loads data from the Failure Prediction Module, including:
   - Component failure probabilities
   - Time-series forecasts
   - Extreme event impact models
   - Environmental correlation models

2. It generates multiple types of scenarios:
   - Normal scenarios: Based on typical operating conditions
   - Extreme event scenarios: For specific extreme weather events (high temperature, high wind, etc.)
   - Compound scenarios: Combining multiple extreme events

3. For each scenario, a cascading failure model simulates how failures might propagate through the grid network.

4. The generated scenarios are validated for realism, diversity, and consistency with physical laws.

5. The final output includes a set of validated scenarios ready for further analysis.

## Usage
```python
from gfmf.scenario_generation import ScenarioGenerationModule

# Initialize the module with default or custom configuration
scenario_module = ScenarioGenerationModule(config_path='path/to/config.yaml')

# Generate scenarios
scenarios = scenario_module.generate_scenarios()

# Access specific scenario types
normal_scenarios = scenarios['normal']
extreme_scenarios = scenarios['extreme']
compound_scenarios = scenarios['compound']

# Access validation metrics
validation_metrics = scenarios['validation_metrics']
```

## Configuration
The module's behavior can be customized through a YAML configuration file. Key parameters include:

- `data_paths`: Paths to input data files
- `scenario_counts`: Number of scenarios to generate for each type
- `failure_thresholds`: Thresholds for determining component failures
- `cascade_parameters`: Parameters for the cascading failure model
- `validation_thresholds`: Thresholds for scenario validation

See `default_config.yaml` for a complete list of configurable parameters.

## Outputs
The module produces:

1. **Scenario Data**: JSON files containing detailed scenario information
2. **Cascading Failure Graphs**: Network representations of failure propagation
3. **Validation Metrics**: Scores for realism, diversity, and consistency
4. **Summary Statistics**: Distribution of scenario types and severity levels

## Integration
This module is designed to integrate with other components of the Grid Failure Modeling Framework:

- **Input**: Uses outputs from the Failure Prediction Module
- **Output**: Provides scenarios for the Vulnerability Analysis Module

## Example
See `experiments/scenario_generation_test.py` for a complete example of using the Scenario Generation Module.
