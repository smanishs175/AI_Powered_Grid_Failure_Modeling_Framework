"""
Failure Prediction Module

This module provides functionalities for predicting component failures in power grids
using machine learning techniques. It integrates with the output from the previous modules
(Component Property Extraction and Vulnerability Analysis) to predict failures based on
component properties, environmental conditions, and vulnerability assessments.

Components:
- Neural Predictor: Predicts component failure probabilities using neural networks
- Time Series Forecaster: Forecasts failure trends over time
- Extreme Event Modeler: Models the impact of extreme events on failure rates
- Correlation Modeler: Models correlations between environmental factors and failures

Example usage:
    from gfmf.failure_prediction.neural_predictor import NeuralPredictor
    
    # Initialize the predictor
    predictor = NeuralPredictor(config_path='path/to/config.yaml')
    
    # Load data
    predictor.load_data(module1_data_path, module2_data_path)
    
    # Train the model
    predictor.train()
    
    # Make predictions
    predictions = predictor.predict(new_data)
"""

# Import core components
from gfmf.failure_prediction.neural_predictor import NeuralPredictor
from gfmf.failure_prediction.time_series_forecaster import TimeSeriesForecaster
from gfmf.failure_prediction.extreme_event_modeler import ExtremeEventModeler
from gfmf.failure_prediction.correlation_modeler import CorrelationModeler

# Define version
__version__ = '0.1.0'
