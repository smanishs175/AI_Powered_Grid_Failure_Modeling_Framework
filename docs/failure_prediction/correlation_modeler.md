# Correlation Modeler

## Overview

The Correlation Modeler identifies and quantifies relationships between environmental factors and component failures in power grids. It uses statistical and machine learning techniques to discover which environmental conditions are most strongly associated with failures, helping grid operators understand the key drivers of grid vulnerability.

## Features

- **Correlation analysis**: Supports multiple correlation methods (Pearson, Spearman, mutual information)
- **Predictive modeling**: Builds linear and non-linear models to predict failures from environmental factors
- **Feature importance**: Identifies the most influential environmental factors
- **Time lag analysis**: Evaluates the impact of lagged environmental conditions on failures
- **Visualization**: Creates heatmaps and importance plots to represent relationships
- **Model comparison**: Evaluates and compares different modeling approaches

## Methods

### Data Processing & Analysis

- **`prepare_data`**: Prepares and merges data for correlation analysis with time lags
- **`analyze_correlations`**: Computes different types of correlations between environmental factors and failures
- **`get_key_environmental_factors`**: Identifies the most influential environmental factors

### Modeling & Prediction

- **`train_models`**: Trains linear and non-linear models to predict failures
- **`predict_failures`**: Generates failure predictions based on environmental data
- **`visualize_key_factors`**: Creates visualizations of factor importance
- **`save_results`**: Saves analysis results, correlation matrices, and visualizations

## Usage Example

```python
from gfmf.failure_prediction.correlation_modeler import CorrelationModeler

# Initialize the modeler
modeler = CorrelationModeler(config_path='config/custom_config.yaml')

# Load and prepare data
environmental_df = pd.read_csv('data/processed/weather_data.csv')
failure_df = pd.read_csv('data/processed/failure_history.csv')
component_df = pd.read_csv('data/processed/components.csv')

# Prepare data for correlation analysis
merged_data = modeler.prepare_data(
    environmental_df=environmental_df,
    failure_df=failure_df,
    component_df=component_df,
    date_column='date',
    location_column='location_id',
    component_column='component_id',
    aggregation_period='D',  # Daily aggregation
    lag_periods=[0, 1, 2, 3, 7]  # Include data from previous days
)

# Analyze correlations
correlation_results = modeler.analyze_correlations(
    data=merged_data,
    target_column='failure_count',
    correlation_types=['pearson', 'spearman', 'mutual_info'],
    visualization_path='outputs/correlation_heatmap.png'
)

# Train predictive models
models = modeler.train_models(
    data=merged_data,
    target_column='failure_count',
    test_size=0.2,
    model_types=['linear', 'nonlinear']
)

# Get key environmental factors
key_factors = modeler.get_key_environmental_factors(
    top_n=10,
    model_type='nonlinear'  # or 'linear'
)

# Visualize key factors
factor_fig = modeler.visualize_key_factors(
    top_n=10,
    save_path='outputs/key_environmental_factors.png'
)

# Make predictions with the model
new_env_data = pd.read_csv('data/processed/new_weather_data.csv')
predictions = modeler.predict_failures(
    environmental_data=new_env_data,
    model_type='nonlinear',
    include_confidence=True
)

# Save all results
saved_files = modeler.save_results(
    save_dir='outputs/correlation_analysis',
    prefix='grid_correlations'
)
```

## Configuration Options

The Correlation Modeler can be configured through the `config/default_config.yaml` file:

```yaml
correlation_models:
  model_types: ["linear", "nonlinear"]
  environmental_factors: ["temperature", "wind_speed", "precipitation"]
  correlation_methods: ["pearson", "spearman", "mutual_info"]
  lag_periods: [0, 1, 2, 3, 7]
  top_factors: 10
```

## Output

The Correlation Modeler produces:

1. **Correlation matrices**: Tables showing relationships between variables
2. **Feature importance rankings**: Lists of most influential environmental factors
3. **Predictive models**: Trained models for failure prediction
4. **Performance metrics**: Model accuracy and reliability measures
5. **Visualizations**: Correlation heatmaps and feature importance charts
6. **Analysis reports**: CSV and JSON files with detailed results

## Dependencies

- Scikit-learn: For predictive modeling and feature selection
- Pandas and NumPy: For data manipulation
- SciPy: For statistical analysis
- Matplotlib and Seaborn: For visualization
