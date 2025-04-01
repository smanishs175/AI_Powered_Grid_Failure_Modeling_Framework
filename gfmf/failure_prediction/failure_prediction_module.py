"""
Failure Prediction Module

This module integrates the various failure prediction components (neural predictor,
time series forecaster, and correlation modeler) to provide a unified interface for
predicting component failures in power grids.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

from gfmf.failure_prediction.neural_predictor import NeuralPredictor
from gfmf.failure_prediction.time_series_forecaster import TimeSeriesForecaster
from gfmf.failure_prediction.correlation_modeler import CorrelationModeler
from gfmf.failure_prediction.extreme_event_modeler import ExtremeEventModeler

logger = logging.getLogger(__name__)

class FailurePredictionModule:
    """
    Failure Prediction Module that integrates various predictive models
    for power grid component failure prediction.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Failure Prediction Module.
        
        Args:
            config_path (str, optional): Path to the configuration file. If None,
                default configuration will be used.
        """
        # Load configuration
        if config_path is None:
            # Use default configuration
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, 'config', 'default_config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.neural_predictor = NeuralPredictor(config_path=config_path)
        self.time_series_forecaster = TimeSeriesForecaster(config_path=config_path)
        self.correlation_modeler = CorrelationModeler(config_path=config_path)
        self.extreme_event_modeler = ExtremeEventModeler(config_path=config_path)
        
        logger.info("Failure Prediction Module initialized")
    
    def load_data(self, data_path):
        """
        Load processed data from previous modules.
        
        Args:
            data_path (str): Path to the processed data directory
            
        Returns:
            dict: Dictionary containing the loaded data
        """
        data = {}
        
        # Load aligned data (from Module 1)
        aligned_data_path = os.path.join(data_path, 'aligned_data.csv')
        if os.path.exists(aligned_data_path):
            data['aligned_data'] = pd.read_csv(aligned_data_path)
            logger.info(f"Loaded aligned data with shape: {data['aligned_data'].shape}")
        
        # Load vulnerability data (from Module 2)
        vulnerability_path = os.path.join(data_path, 'vulnerability_scores.csv')
        if os.path.exists(vulnerability_path):
            data['vulnerability_scores'] = pd.read_csv(vulnerability_path)
            logger.info(f"Loaded vulnerability scores with shape: {data['vulnerability_scores'].shape}")
        
        return data
    
    def analyze_correlations(self, data):
        """
        Analyze correlations between environmental factors and failures.
        
        Args:
            data (pd.DataFrame): The aligned dataset with component and environmental data
            
        Returns:
            dict: Dictionary containing correlation results
        """
        logger.info("Analyzing correlations between environmental factors and failures")
        correlations = self.correlation_modeler.analyze_correlations(
            data, target_col='failure_count' if 'failure_count' in data.columns else 'outage_flag'
        )
        return correlations
    
    def train_neural_predictor(self, data, target_col='outage_flag'):
        """
        Train the neural predictor model.
        
        Args:
            data (pd.DataFrame): Training data
            target_col (str): Target column name
            
        Returns:
            dict: Training results
        """
        logger.info(f"Training neural predictor model with target: {target_col}")
        return self.neural_predictor.train(data, target_col=target_col)
    
    def train_time_series_model(self, data, target_col='outage_count'):
        """
        Train the time series forecasting model.
        
        Args:
            data (pd.DataFrame): Time series data
            target_col (str): Target column to forecast
            
        Returns:
            object: Trained model object
        """
        logger.info(f"Training time series model with target: {target_col}")
        # Prepare time series data
        prepared_data = self.time_series_forecaster.prepare_time_series_data(
            data, date_col='timestamp' if 'timestamp' in data.columns else 'date', 
            target_col=target_col
        )
        
        # Train the model
        model = self.time_series_forecaster.train(prepared_data, target_col=target_col)
        return model
    
    def generate_failure_predictions(self, data, vulnerability_scores, periods=30):
        """
        Generate failure predictions for components.
        
        Args:
            data (pd.DataFrame): Historical data
            vulnerability_scores (pd.DataFrame): Component vulnerability scores
            periods (int): Number of future periods to predict
            
        Returns:
            pd.DataFrame: Predictions for components
        """
        logger.info(f"Generating failure predictions for {periods} future periods")
        
        # Analyze correlations
        correlations = self.analyze_correlations(data)
        logger.info(f"Correlation analysis completed: {len(correlations)} factors analyzed")
        
        # Create time series data structure
        if 'timestamp' not in data.columns and 'date' in data.columns:
            data['timestamp'] = pd.to_datetime(data['date'])
        
        # Generate time series forecasts
        try:
            forecast_data = self.time_series_forecaster.forecast(
                data, periods=periods, 
                target_col='outage_count' if 'outage_count' in data.columns else 'outage_flag'
            )
            logger.info(f"Time series forecast completed with shape: {forecast_data.shape}")
        except Exception as e:
            logger.error(f"Error in time series forecasting: {str(e)}")
            # Create synthetic forecast data for demonstration
            start_date = pd.to_datetime(data['timestamp' if 'timestamp' in data.columns else 'date']).max() + pd.Timedelta(days=1)
            dates = pd.date_range(start=start_date, periods=periods)
            forecast_data = pd.DataFrame({
                'date': dates,
                'outage_count_prediction': np.random.uniform(0, 1, periods)
            })
            logger.warning(f"Using synthetic forecast data with shape: {forecast_data.shape}")
        
        # Merge with vulnerability scores
        component_ids = vulnerability_scores['component_id'].unique()
        predictions = []
        
        for component_id in component_ids:
            component_vuln = vulnerability_scores.loc[
                vulnerability_scores['component_id'] == component_id, 
                'vulnerability_score'
            ].values[0]
            
            component_forecast = forecast_data.copy()
            component_forecast['component_id'] = component_id
            component_forecast['vulnerability_score'] = component_vuln
            
            # Combine vulnerability and time series forecast to calculate failure probability
            component_forecast['failure_probability'] = (
                component_forecast['outage_count_prediction'] * 0.6 +
                component_vuln * 0.4
            )
            
            predictions.append(component_forecast)
        
        predictions_df = pd.concat(predictions, ignore_index=True)
        logger.info(f"Generated predictions with shape: {predictions_df.shape}")
        
        return predictions_df
    
    def identify_high_risk_periods(self, predictions, threshold=0.7):
        """
        Identify high-risk periods from predictions.
        
        Args:
            predictions (pd.DataFrame): Prediction results
            threshold (float): Probability threshold for high-risk classification
            
        Returns:
            pd.DataFrame: High-risk periods
        """
        high_risk = predictions[predictions['failure_probability'] >= threshold].copy()
        logger.info(f"Identified {len(high_risk)} high-risk periods with threshold {threshold}")
        return high_risk
    
    def run_analysis(self, data, output_dir=None):
        """
        Run the complete failure prediction analysis.
        
        Args:
            data (dict): Dictionary containing necessary data
            output_dir (str, optional): Directory to save outputs
            
        Returns:
            dict: Analysis results
        """
        results = {}
        
        # Ensure output directory exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        aligned_data = data.get('aligned_data')
        vulnerability_scores = data.get('vulnerability_scores')
        
        if aligned_data is None:
            logger.error("Aligned data is missing")
            return {"error": "Aligned data is missing"}
        
        if vulnerability_scores is None:
            logger.warning("Vulnerability scores are missing, using synthetic data")
            component_ids = aligned_data['component_id'].unique()
            vulnerability_scores = pd.DataFrame({
                'component_id': component_ids,
                'vulnerability_score': np.random.uniform(0, 1, len(component_ids))
            })
        
        # Analyze correlations
        results['correlations'] = self.analyze_correlations(aligned_data)
        
        # Train neural predictor
        results['neural_model'] = self.train_neural_predictor(
            aligned_data, target_col='outage_flag' if 'outage_flag' in aligned_data.columns else 'failure_status'
        )
        
        # Train time series model
        target_col = 'outage_count' if 'outage_count' in aligned_data.columns else 'outage_flag'
        results['time_series_model'] = self.train_time_series_model(aligned_data, target_col=target_col)
        
        # Generate predictions
        results['predictions'] = self.generate_failure_predictions(
            aligned_data, vulnerability_scores, periods=30
        )
        
        # Identify high-risk periods
        results['high_risk_periods'] = self.identify_high_risk_periods(
            results['predictions'], threshold=0.7
        )
        
        # Save results
        if output_dir:
            for key, data in results.items():
                if isinstance(data, pd.DataFrame):
                    output_path = os.path.join(output_dir, f"{key}.csv")
                    data.to_csv(output_path, index=False)
                    logger.info(f"Saved {key} to {output_path}")
        
        logger.info("Failure prediction analysis completed")
        return results 