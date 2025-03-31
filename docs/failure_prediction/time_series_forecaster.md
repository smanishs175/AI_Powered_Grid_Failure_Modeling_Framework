# Time Series Forecaster

## Overview

The Time Series Forecaster is a specialized component of the Failure Prediction Module that analyzes temporal patterns in grid component failures and generates forecasts of future failure events. It supports multiple time series modeling approaches, including deep learning models (LSTM) and statistical methods (Prophet).

## Features

- **Time series data processing**: Handles data cleaning, resampling, and sequence creation
- **Multiple modeling approaches**: LSTM-based deep learning and Prophet statistical forecasting
- **Multi-step forecasting**: Predicts failures across different time horizons
- **Seasonal decomposition**: Identifies daily, weekly, and yearly patterns in failure data
- **Uncertainty quantification**: Provides confidence intervals for forecasts
- **Component-specific forecasts**: Generates predictions for different component types

## Methods

### Data Handling

- **`load_data`**: Loads time series data from various sources with date parsing
- **`create_sequences`**: Transforms time series data into sequences for LSTM training
- **`split_data`**: Divides data into training and validation sets

### Model Building & Forecasting

- **`build_lstm_model`**: Creates and compiles LSTM architectures for time series prediction
- **`build_prophet_model`**: Sets up Facebook Prophet models with seasonality components
- **`train`**: Fits models to historical data with validation
- **`forecast`**: Generates multi-step ahead predictions with confidence intervals
- **`evaluate`**: Calculates forecast accuracy metrics (MAE, RMSE, MAPE)

### Visualization & Analysis

- **`plot_forecast`**: Visualizes forecasts with prediction intervals
- **`plot_components`**: Decomposes and visualizes trend, seasonality, and residual components
- **`plot_historical_vs_predicted`**: Compares model predictions with historical data

## Usage Example

```python
from gfmf.failure_prediction.time_series_forecaster import TimeSeriesForecaster

# Initialize forecaster
forecaster = TimeSeriesForecaster(config_path='config/custom_config.yaml')

# Load historical failure data
data = forecaster.load_data(
    data_path='data/processed/failure_history.csv',
    date_column='date',
    target_column='failure_count',
    feature_columns=['temperature', 'wind_speed'],
    group_by='component_type'
)

# For LSTM approach
if forecaster.model_type == 'lstm':
    # Create sequences for LSTM training
    X, y = forecaster.create_sequences(
        data['transformer'], 
        sequence_length=14, 
        target_column='failure_count'
    )
    
    # Build and train LSTM model
    forecaster.build_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        lstm_units=[64, 32]
    )
    forecaster.train(X, y, validation_split=0.2, epochs=100)
    
    # Generate forecast
    forecast = forecaster.forecast(
        input_data=latest_data,
        forecast_horizon=30,
        return_confidence_intervals=True
    )

# For Prophet approach
else:
    # Build and train Prophet model
    forecaster.build_prophet_model(
        data['transformer'],
        seasonality_mode='multiplicative',
        include_holidays=True
    )
    
    # Generate forecast
    forecast = forecaster.forecast(
        forecast_horizon=30,
        return_components=True
    )

# Evaluate forecast accuracy
metrics = forecaster.evaluate(actual_values, forecast_values)
print(f"RMSE: {metrics['rmse']}, MAPE: {metrics['mape']}%")

# Visualize forecast
forecaster.plot_forecast(
    forecast_data=forecast,
    actual_data=historical_data,
    save_path='outputs/forecast_plot.png'
)

# Save model
forecaster.save('models/time_series_forecaster/transformer_model')
```

## Configuration Options

The Time Series Forecaster can be configured through the `config/default_config.yaml` file:

```yaml
time_series:
  model_type: "lstm"  # Options: lstm, prophet
  sequence_length: 14
  lstm_units: [64, 32]
  learning_rate: 0.001
  epochs: 100
  batch_size: 32
  forecast_horizon: 30
  seasonality_modes: ["daily", "weekly", "yearly"]
```

## Output

The Time Series Forecaster produces:

1. **Failure forecasts**: Predicted failure counts over the specified time horizon
2. **Confidence intervals**: Upper and lower bounds for predictions
3. **Seasonal components**: Decomposition of time series into trend, seasonality, and residuals
4. **Performance metrics**: Forecast accuracy measures
5. **Visualization plots**: Forecast plots, component plots, and error analysis
6. **Saved models**: Model files for future use

## Dependencies

- TensorFlow/Keras: For LSTM models
- Prophet (optional): For statistical forecasting
- Pandas: For time series data manipulation
- NumPy: For numerical operations
- Matplotlib and Seaborn: For visualization
