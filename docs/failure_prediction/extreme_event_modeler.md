# Extreme Event Modeler

## Overview

The Extreme Event Modeler analyzes the impact of extreme environmental conditions on power grid component failures. It identifies extreme weather events based on configurable thresholds, quantifies their effects on failure rates, and provides insights into how different environmental extremes affect grid reliability.

## Features

- **Extreme event identification**: Automatically detects extreme environmental conditions using percentile-based thresholds
- **Impact quantification**: Measures the increase in failure rates during extreme conditions
- **Compound event analysis**: Assesses the combined effects of multiple extreme conditions occurring simultaneously
- **Statistical testing**: Determines if failure rate changes during events are statistically significant
- **Component-specific risk modeling**: Estimates component-level failure probabilities during extreme events
- **Visualization**: Creates visual representations of event impacts and risks

## Methods

### Event Detection & Analysis

- **`identify_extreme_events`**: Identifies extreme environmental conditions based on configurable thresholds
- **`analyze_event_impact`**: Quantifies the impact of extreme events on component failures
- **`calculate_compound_event_impact`**: Analyzes the effects of multiple extreme events occurring together

### Risk Modeling

- **`predict_event_failure_probability`**: Estimates component failure probabilities during specific extreme events
- **`visualize_event_impact`**: Creates visualizations of event impacts on failure rates
- **`save_results`**: Saves analysis results, thresholds, and visualizations to disk

### Data Management

- **`load_data`**: Loads environmental and failure data for analysis
- **`_map_var_to_event_type`**: Maps environmental variables to event types
- **`_load_config`**: Loads configuration settings for event analysis

## Usage Example

```python
from gfmf.failure_prediction.extreme_event_modeler import ExtremeEventModeler

# Initialize the modeler
modeler = ExtremeEventModeler(config_path='config/custom_config.yaml')

# Load data
env_df, failure_df = modeler.load_data(
    environmental_data_path='data/processed/weather_data.csv',
    failure_data_path='data/processed/failure_history.csv',
    date_column='date',
    location_column='location_id',
    component_column='component_id'
)

# Identify extreme events
extreme_events = modeler.identify_extreme_events(
    env_df=env_df,
    window_size=1,
    min_duration=1
)

# Analyze impact of extreme events on failures
impact_stats = modeler.analyze_event_impact(
    extreme_events=extreme_events,
    failure_df=failure_df,
    event_window=3,  # days
    baseline_window=30  # days
)

# Calculate the impact of compound events
compound_stats = modeler.calculate_compound_event_impact(
    extreme_events=extreme_events,
    failure_df=failure_df,
    event_window=3  # days
)

# Get component-specific failure probabilities
component_df = load_component_data('data/processed/components.csv')
event_failure_probs = modeler.predict_event_failure_probability(
    component_df=component_df,
    event_type='high_temperature',
    severity=1.5  # 1.5x standard deviations above threshold
)

# Visualize event impacts
impact_fig = modeler.visualize_event_impact(
    save_path='outputs/extreme_event_impact.png'
)

# Save all results
saved_files = modeler.save_results(
    save_dir='outputs/extreme_events',
    prefix='grid_extreme_events'
)
```

## Configuration Options

The Extreme Event Modeler can be configured through the `config/default_config.yaml` file:

```yaml
extreme_events:
  event_types: ["high_temperature", "low_temperature", "high_wind", "precipitation"]
  threshold_percentiles:
    high_temperature: 95
    low_temperature: 5
    high_wind: 95
    precipitation: 95
  min_duration: 1
  event_window: 3
  baseline_window: 30
```

## Output

The Extreme Event Modeler produces:

1. **Event identification results**: Details of identified extreme events
2. **Impact statistics**: Quantification of effects on failure rates
3. **Statistical significance**: P-values and significance indicators
4. **Component failure probabilities**: Risk estimates during extreme conditions
5. **Visualizations**: Impact charts and risk maps
6. **Analysis reports**: JSON and CSV files with detailed results

## Dependencies

- Pandas and NumPy: For data manipulation
- SciPy: For statistical tests
- Matplotlib and Seaborn: For visualization
