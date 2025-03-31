#!/usr/bin/env python
"""
Failure Prediction Module - Integration Test Script

This script tests all components of the Failure Prediction Module to ensure they 
work properly with a small synthetic dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('failure_prediction_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("FailurePredictionTest")

# Add the root directory to the path
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

# Import the different components
from gfmf.failure_prediction.neural_predictor import NeuralPredictor
from gfmf.failure_prediction.time_series_forecaster import TimeSeriesForecaster
from gfmf.failure_prediction.extreme_event_modeler import ExtremeEventModeler
from gfmf.failure_prediction.correlation_modeler import CorrelationModeler

# Create output directory
output_dir = os.path.join(root_dir, 'outputs', 'failure_prediction_test')
os.makedirs(output_dir, exist_ok=True)

def load_test_data():
    """
    Load and prepare synthetic test data.
    """
    logger.info("Loading synthetic test data...")
    
    # Choose a synthetic data directory
    data_dir = os.path.join(root_dir, 'data', 'synthetic', 'synthetic_20250328_144932')
    
    # Load grid component data
    components_path = os.path.join(data_dir, 'synthetic_grid.csv')
    components_df = pd.read_csv(components_path)
    
    # Load weather data
    weather_path = os.path.join(data_dir, 'synthetic_weather.csv')
    weather_df = pd.read_csv(weather_path)
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    # Load outage data
    outages_path = os.path.join(data_dir, 'synthetic_outages.csv')
    outages_df = pd.read_csv(outages_path)
    outages_df['start_time'] = pd.to_datetime(outages_df['start_time'])
    outages_df['end_time'] = pd.to_datetime(outages_df['end_time'])
    
    # Create daily failure counts (aggregated)
    failure_df = pd.DataFrame({
        'date': pd.to_datetime(outages_df['start_time'].dt.date).unique()
    })
    
    # Count failures per day
    daily_counts = outages_df.groupby(outages_df['start_time'].dt.date).size()
    daily_counts.index = pd.to_datetime(daily_counts.index)
    failure_df = pd.DataFrame(daily_counts).reset_index()
    failure_df.columns = ['date', 'failure_count']
    
    # Create a vulnerability dataframe (using the vulnerability score from components_df)
    vulnerability_df = components_df[['component_id', 'vulnerability']].copy()
    vulnerability_df.rename(columns={'vulnerability': 'vulnerability_score'}, inplace=True)
    
    # Create stations mapping (for associating weather with grid components)
    # For simplicity, we'll just divide components among available weather stations
    unique_stations = weather_df['station_id'].unique()
    station_mapping = {}
    
    for i, component_id in enumerate(components_df['component_id']):
        station_idx = i % len(unique_stations)
        station_mapping[component_id] = unique_stations[station_idx]
    
    station_df = pd.DataFrame(list(station_mapping.items()), columns=['component_id', 'station_id'])
    
    logger.info(f"Loaded {len(components_df)} components, {len(weather_df)} weather records, {len(outages_df)} outages")
    
    return {
        'components': components_df,
        'weather': weather_df,
        'outages': outages_df,
        'failures': failure_df,
        'vulnerability': vulnerability_df,
        'stations': station_df
    }

def prepare_data_for_neural_predictor(data_dict):
    """
    Prepare data for the Neural Predictor component.
    """
    logger.info("Preparing data for Neural Predictor...")
    
    components_df = data_dict['components']
    weather_df = data_dict['weather']
    vulnerability_df = data_dict['vulnerability']
    outages_df = data_dict['outages']
    stations_df = data_dict['stations']
    
    # Create failure history
    failure_history = pd.DataFrame()
    failure_history['component_id'] = outages_df['component_id']
    failure_history['failure_date'] = outages_df['start_time'].dt.date
    failure_history['failure'] = 1  # 1 indicates failure
    
    # Convert date to datetime
    failure_history['failure_date'] = pd.to_datetime(failure_history['failure_date'])
    
    # Add non-failure records for balance
    # Get all component IDs
    all_component_ids = components_df['component_id'].unique()
    
    # Get all dates in the data range
    min_date = failure_history['failure_date'].min()
    max_date = failure_history['failure_date'].max()
    all_dates = pd.date_range(min_date, max_date, freq='D')
    
    # Create all possible component-date combinations
    non_failures = []
    
    # For simplicity, just add a sample of non-failures
    sample_size = min(len(failure_history) * 3, len(all_component_ids) * len(all_dates))
    
    import random
    random.seed(42)
    
    for _ in range(sample_size):
        component_id = random.choice(all_component_ids)
        date = random.choice(all_dates)
        
        # Check if this component failed on this date
        if not ((failure_history['component_id'] == component_id) & 
                (failure_history['failure_date'] == date)).any():
            non_failures.append({
                'component_id': component_id,
                'failure_date': date,
                'failure': 0  # 0 indicates no failure
            })
    
    # Add non-failures to the failure history
    non_failure_df = pd.DataFrame(non_failures)
    failure_history = pd.concat([failure_history, non_failure_df], ignore_index=True)
    
    # Prepare environmental data
    # We'll take the daily average for each station
    weather_daily = weather_df.copy()
    weather_daily['date'] = weather_daily['timestamp'].dt.date
    weather_daily = weather_daily.groupby(['station_id', 'date']).agg({
        'temperature': 'mean',
        'precipitation': 'sum',
        'wind_speed': 'mean',
        'humidity': 'mean',
        'pressure': 'mean',
        'is_extreme_temperature': 'max',
        'is_extreme_wind': 'max',
        'is_extreme_precipitation': 'max',
        'heat_wave_day': 'max',
        'cold_snap_day': 'max',
        'storm_day': 'max'
    }).reset_index()
    
    # Convert date to datetime
    weather_daily['date'] = pd.to_datetime(weather_daily['date'])
    
    # Add location_id column (required by Neural Predictor)
    weather_daily['location_id'] = weather_daily['station_id']
    
    return {
        'components': components_df,
        'environmental': weather_daily,
        'vulnerability': vulnerability_df,
        'failures': failure_history,
        'stations': stations_df
    }

def prepare_data_for_time_series(data_dict):
    """
    Prepare data for the Time Series Forecaster component.
    """
    logger.info("Preparing data for Time Series Forecaster...")
    
    failure_df = data_dict['failures'].copy()
    weather_df = data_dict['weather'].copy()
    
    # Aggregate weather data daily
    weather_daily = weather_df.copy()
    weather_daily['date'] = pd.to_datetime(weather_daily['timestamp'].dt.date)
    weather_daily = weather_daily.groupby('date').agg({
        'temperature': 'mean',
        'precipitation': 'sum',
        'wind_speed': 'mean',
        'humidity': 'mean',
        'is_extreme_temperature': 'max',
        'is_extreme_wind': 'max',
        'is_extreme_precipitation': 'max'
    }).reset_index()
    
    # Ensure failure_df date is in datetime format
    failure_df['date'] = pd.to_datetime(failure_df['date'])
    
    # Merge with failure data
    merged_df = pd.merge(
        failure_df,
        weather_daily,
        on='date',
        how='right'
    )
    
    # Fill missing failure counts with 0
    merged_df['failure_count'] = merged_df['failure_count'].fillna(0)
    
    # Sort by date
    merged_df.sort_values('date', inplace=True)
    
    return merged_df

def test_neural_predictor(data_dict):
    """
    Test the Neural Predictor component.
    """
    logger.info("\n\n==== Testing Neural Predictor ====")
    
    # Initialize the predictor
    predictor = NeuralPredictor()
    
    # Process and merge data
    merged_data = predictor._process_and_merge_data(
        component_df=data_dict['components'],
        vulnerability_df=data_dict['vulnerability'],
        environmental_df=data_dict['environmental'],
        failure_df=data_dict['failures']
    )
    
    logger.info(f"Merged data shape: {merged_data.shape}")
    
    # Check if 'failure' column exists in the merged data
    try:
        # Split data
        X = merged_data.drop('failure', axis=1)
        y = merged_data['failure']
    except KeyError:
        # If 'failure' column doesn't exist, add it
        logger.warning("'failure' column not found in merged data, adding it")
        merged_data['failure'] = 0
        # We'll consider a component having at least one failure in history
        failed_components = data_dict['failures']['component_id'].unique()
        merged_data.loc[merged_data['component_id'].isin(failed_components), 'failure'] = 1
        
        # Now split the data
        X = merged_data.drop('failure', axis=1)
        y = merged_data['failure']
    
    # Keep only numeric columns for simplicity
    X = X.select_dtypes(include=np.number)
    
    # Test-train split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    # Preprocess data
    X_train_processed = predictor.preprocess_data(X_train, fit=True)
    X_test_processed = predictor.preprocess_data(X_test, fit=False)
    
    # Build model with reduced complexity for the test
    predictor.build_model(input_dim=X_train_processed.shape[1], hidden_layers=[8, 4])
    
    # Train model with minimal epochs
    history = predictor.train(
        X_train_processed, 
        y_train, 
        validation_split=0.2, 
        epochs=10,  # Reduced for testing
        batch_size=8
    )
    
    # Make predictions
    predictions = predictor.predict(X_test_processed)
    
    # Evaluate
    metrics = predictor.evaluate(X_test_processed, y_test, save_plots=True, plots_dir=output_dir)
    
    logger.info(f"Neural Predictor metrics: {metrics}")
    
    # Save model
    predictor.save(os.path.join(output_dir, 'neural_predictor_model'))
    
    return {
        'predictor': predictor,
        'metrics': metrics,
        'predictions': predictions,
        'test_data': (X_test, y_test)
    }

def test_time_series_forecaster(data_dict):
    """
    Test the Time Series Forecaster component.
    """
    logger.info("\n\n==== Testing Time Series Forecaster ====")
    
    # Initialize the forecaster
    forecaster = TimeSeriesForecaster()
    
    # Prepare time series data
    time_series_data = prepare_data_for_time_series(data_dict)
    
    # Set date as index
    ts_data = time_series_data.set_index('date')
    
    # Create sequences for LSTM
    sequence_length = 7  # Use 7 days of history to predict the next day
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(ts_data) - sequence_length):
        sequences.append(ts_data.iloc[i:i+sequence_length].values)
        targets.append(ts_data.iloc[i+sequence_length]['failure_count'])
    
    if not sequences:
        logger.warning("Not enough data for time series forecasting!")
        return None
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)
    
    # Test-train split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    logger.info(f"Time Series Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    # Build a simple LSTM model
    forecaster.build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=[8]  # Simplified for testing
    )
    
    # Train with minimal epochs
    history = forecaster.train(
        X_train, 
        y_train, 
        validation_split=0.2,
        epochs=10,  # Reduced for testing
        batch_size=8
    )
    
    # Make predictions
    predictions = forecaster.predict(X_test)
    
    # Evaluate
    metrics = forecaster.evaluate(y_test, predictions)
    
    logger.info(f"Time Series Forecaster metrics: {metrics}")
    
    # Generate a simple forecast plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Failure Count Forecast')
    plt.xlabel('Test Sample')
    plt.ylabel('Failure Count')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'time_series_forecast.png'))
    plt.close()
    
    # Save model
    forecaster.save(os.path.join(output_dir, 'time_series_model'))
    
    return {
        'forecaster': forecaster,
        'metrics': metrics,
        'predictions': predictions,
        'test_data': (X_test, y_test)
    }

def test_extreme_event_modeler(data_dict):
    """
    Test the Extreme Event Modeler component.
    """
    logger.info("\n\n==== Testing Extreme Event Modeler ====")
    
    # Initialize the modeler
    modeler = ExtremeEventModeler()
    
    # Prepare weather and failure data
    weather_df = data_dict['weather'].copy()
    
    # Create failure data with component_id and date
    failure_df = pd.DataFrame()
    failure_df['component_id'] = data_dict['outages']['component_id']
    failure_df['date'] = pd.to_datetime(data_dict['outages']['start_time'])
    
    # First load the data to initialize date_column and location_column
    env_df, fail_df = modeler.load_data(
        environmental_data_path=None,  # We'll pass the DataFrames directly
        failure_data_path=None,
        date_column='timestamp',
        location_column='station_id',
        component_column='component_id'
    )
    
    # Since we're not actually loading from files, manually set the DataFrames
    modeler.date_column = 'timestamp'
    modeler.location_column = 'station_id'
    modeler.component_column = 'component_id'
    
    # Identify extreme events
    extreme_events = modeler.identify_extreme_events(
        env_df=weather_df,
        window_size=1,
        min_duration=1
    )
    
    logger.info(f"Identified extreme events: {list(extreme_events.keys())}")
    
    # Analyze event impact
    impact_stats = modeler.analyze_event_impact(
        extreme_events=extreme_events,
        failure_df=failure_df,
        event_window=3,
        baseline_window=30
    )
    
    logger.info(f"Impact statistics: {list(impact_stats.keys())}")
    
    # Calculate compound event impact
    compound_stats = modeler.calculate_compound_event_impact(
        extreme_events=extreme_events,
        failure_df=failure_df,
        event_window=3
    )
    
    if compound_stats:
        logger.info(f"Compound event statistics: Rate ratio = {compound_stats['rate_ratio']:.2f}")
    
    # Predict event failure probability
    event_failure_probs = modeler.predict_event_failure_probability(
        component_df=data_dict['components'],
        event_type=list(impact_stats.keys())[0] if impact_stats else "high_temperature",
        severity=1.5
    )
    
    logger.info(f"Event failure probabilities: {event_failure_probs.shape}")
    
    # Visualize impact
    try:
        fig = modeler.visualize_event_impact(
            save_path=os.path.join(output_dir, 'extreme_event_impact.png')
        )
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not create event impact visualization: {e}")
    
    # Save results
    saved_files = modeler.save_results(
        save_dir=output_dir,
        prefix='test_extreme_events'
    )
    
    logger.info(f"Saved extreme event files: {list(saved_files.keys())}")
    
    return {
        'modeler': modeler,
        'extreme_events': extreme_events,
        'impact_stats': impact_stats,
        'compound_stats': compound_stats,
        'failure_probs': event_failure_probs
    }

def test_correlation_modeler(data_dict):
    """
    Test the Correlation Modeler component.
    """
    logger.info("\n\n==== Testing Correlation Modeler ====")
    
    # Initialize the modeler
    modeler = CorrelationModeler()
    
    # Prepare time series data (reuse function)
    time_series_data = prepare_data_for_time_series(data_dict)
    
    # Analyze correlations
    correlation_results = modeler.analyze_correlations(
        data=time_series_data,
        target_column='failure_count',
        correlation_types=['pearson', 'spearman'],
        visualization_path=os.path.join(output_dir, 'correlation_heatmap.png')
    )
    
    logger.info(f"Correlation types: {list(correlation_results.keys())}")
    
    # Train models
    models = modeler.train_models(
        data=time_series_data,
        target_column='failure_count',
        test_size=0.3,
        model_types=['linear', 'nonlinear']
    )
    
    logger.info(f"Trained model types: {list(models.keys())}")
    
    # Get key environmental factors
    key_factors = modeler.get_key_environmental_factors(
        top_n=5,
        model_type='nonlinear'
    )
    
    logger.info(f"Key environmental factors: {key_factors['feature'].tolist()}")
    
    # Visualize key factors
    try:
        fig = modeler.visualize_key_factors(
            top_n=5,
            save_path=os.path.join(output_dir, 'key_environmental_factors.png')
        )
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not create key factors visualization: {e}")
    
    # Make predictions
    predictions = modeler.predict_failures(
        environmental_data=time_series_data,
        model_type='nonlinear',
        include_confidence=True
    )
    
    logger.info(f"Correlation model predictions: {predictions.shape}")
    
    # Save results
    saved_files = modeler.save_results(
        save_dir=output_dir,
        prefix='test_correlations'
    )
    
    logger.info(f"Saved correlation files: {list(saved_files.keys())}")
    
    return {
        'modeler': modeler,
        'correlation_results': correlation_results,
        'models': models,
        'key_factors': key_factors,
        'predictions': predictions
    }

def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting Failure Prediction Module Integration Test")
    
    # Load test data
    data_dict = load_test_data()
    
    # Prepare data for the Neural Predictor
    np_data = prepare_data_for_neural_predictor(data_dict)
    
    # Test each component
    try:
        neural_results = test_neural_predictor(np_data)
        logger.info("Neural Predictor test completed successfully!")
    except Exception as e:
        logger.error(f"Neural Predictor test failed: {e}", exc_info=True)
    
    try:
        timeseries_results = test_time_series_forecaster(data_dict)
        logger.info("Time Series Forecaster test completed successfully!")
    except Exception as e:
        logger.error(f"Time Series Forecaster test failed: {e}", exc_info=True)
    
    try:
        extreme_results = test_extreme_event_modeler(data_dict)
        logger.info("Extreme Event Modeler test completed successfully!")
    except Exception as e:
        logger.error(f"Extreme Event Modeler test failed: {e}", exc_info=True)
    
    try:
        correlation_results = test_correlation_modeler(data_dict)
        logger.info("Correlation Modeler test completed successfully!")
    except Exception as e:
        logger.error(f"Correlation Modeler test failed: {e}", exc_info=True)
    
    logger.info("Failure Prediction Module Integration Test completed!")
    logger.info(f"Results and outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
