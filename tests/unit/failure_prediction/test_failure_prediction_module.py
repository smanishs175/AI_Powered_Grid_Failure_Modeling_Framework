"""
Unit tests for the Failure Prediction Module (Module 3)
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gfmf.failure_prediction.failure_prediction_module import FailurePredictionModule
from gfmf.failure_prediction.correlation_modeler import CorrelationModeler
from gfmf.failure_prediction.time_series_forecaster import TimeSeriesForecaster


class TestFailurePredictionModule(unittest.TestCase):
    """Test cases for the FailurePredictionModule class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.fp_module = FailurePredictionModule()
        
        # Create mock data
        # Aligned dataset with necessary features
        self.aligned_data = pd.DataFrame({
            'component_id': list(range(1, 11)) * 10,  # 10 components, 10 time periods
            'date': sorted(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                    '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'] * 10),
            'outage_flag': np.random.choice([0, 1], size=100, p=[0.8, 0.2]),
            'failure_count': np.random.choice([0, 1, 2], size=100, p=[0.8, 0.15, 0.05]),
            'age': np.random.uniform(1, 20, 100),
            'temperature': np.random.uniform(-10, 40, 100),
            'precipitation': np.random.uniform(0, 50, 100),
            'humidity': np.random.uniform(10, 100, 100),
            'wind_speed': np.random.uniform(0, 30, 100),
            'extreme_weather': np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        })
        
        # Vulnerability scores
        self.vulnerability_scores = pd.DataFrame({
            'component_id': list(range(1, 11)),
            'vulnerability_score': np.random.uniform(0, 1, 10)
        })
        
        # Time series data with timestamps
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.time_series_data = pd.DataFrame({
            'date': dates,
            'component_id': np.repeat(list(range(1, 11)), 10),
            'outage_count': np.random.choice([0, 1, 2], size=100, p=[0.8, 0.15, 0.05]),
            'temperature': np.random.uniform(-10, 40, 100),
            'precipitation': np.random.uniform(0, 50, 100),
            'wind_speed': np.random.uniform(0, 30, 100)
        })
        self.time_series_data['timestamp'] = pd.to_datetime(self.time_series_data['date'])

    def test_initialization(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.fp_module)
        self.assertIsNotNone(self.fp_module.config)
        self.assertIsInstance(self.fp_module.correlation_modeler, CorrelationModeler)
        self.assertIsInstance(self.fp_module.time_series_forecaster, TimeSeriesForecaster)

    def test_analyze_correlations(self):
        """Test correlation analysis."""
        correlations = self.fp_module.analyze_correlations(self.aligned_data)
        
        self.assertIsInstance(correlations, dict)
        # Check if key components of correlations are present
        if isinstance(correlations, dict) and 'feature_correlations' in correlations:
            self.assertIn('feature_correlations', correlations)
            self.assertIsInstance(correlations['feature_correlations'], dict)
        
        # Alternative structure some implementations might use
        elif isinstance(correlations, dict) and 'pearson' in correlations:
            self.assertIn('pearson', correlations)

    def test_train_time_series_model(self):
        """Test time series model training."""
        # Mock train method
        with patch.object(TimeSeriesForecaster, 'train', return_value=None) as mock_train:
            self.fp_module.train_time_series_model(
                self.time_series_data, target_col='outage_count'
            )
            mock_train.assert_called_once()

    def test_generate_failure_predictions(self):
        """Test failure prediction generation."""
        # Mock the internal method calls
        with patch.object(
            self.fp_module, 'analyze_correlations', 
            return_value={'feature_correlations': {'temperature': 0.3, 'age': 0.7}}
        ):
            with patch.object(
                self.fp_module.time_series_forecaster, 'forecast',
                return_value=pd.DataFrame({
                    'date': pd.date_range(start='2023-02-01', periods=10, freq='D'),
                    'outage_count_prediction': np.random.uniform(0, 1, 10)
                })
            ):
                predictions = self.fp_module.generate_failure_predictions(
                    self.aligned_data, self.vulnerability_scores, periods=10
                )
                
                self.assertIsInstance(predictions, pd.DataFrame)
                # Check for prediction columns
                prediction_cols = [col for col in predictions.columns if 'prediction' in col.lower()]
                self.assertTrue(len(prediction_cols) > 0)

    def test_identify_high_risk_periods(self):
        """Test identification of high-risk periods."""
        # Create predictions with a known structure
        predictions = pd.DataFrame({
            'date': pd.date_range(start='2023-02-01', periods=10, freq='D'),
            'component_id': np.repeat([1, 2], 5),
            'failure_probability': [0.1, 0.8, 0.3, 0.9, 0.4, 0.2, 0.7, 0.5, 0.6, 0.3]
        })
        
        high_risk_periods = self.fp_module.identify_high_risk_periods(
            predictions, threshold=0.7
        )
        
        self.assertIsInstance(high_risk_periods, pd.DataFrame)
        # Should have filtered to only high-risk periods
        self.assertTrue(all(high_risk_periods['failure_probability'] >= 0.7))


class TestCorrelationModeler(unittest.TestCase):
    """Test cases for the CorrelationModeler class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.correlation_modeler = CorrelationModeler()
        
        # Create mock data
        self.test_data = pd.DataFrame({
            'failure_count': [0, 1, 0, 2, 0, 1, 0, 0],
            'age': [5, 15, 8, 20, 10, 18, 7, 12],
            'temperature': [25, 32, 28, 35, 22, 30, 26, 29],
            'precipitation': [0, 15, 5, 20, 2, 10, 0, 8],
            'wind_speed': [10, 25, 15, 30, 12, 22, 8, 18]
        })

    def test_analyze_correlations(self):
        """Test correlation analysis functionality."""
        correlations = self.correlation_modeler.analyze_correlations(
            self.test_data, target_col='failure_count'
        )
        
        self.assertIsInstance(correlations, dict)
        # Check if proper structure is returned
        self.assertIn('pearson', correlations)
        self.assertIn('feature_importance', correlations)
        
    def test_get_top_correlations(self):
        """Test extraction of top correlations."""
        correlations = self.correlation_modeler.analyze_correlations(
            self.test_data, target_col='failure_count'
        )
        
        top_corr = self.correlation_modeler.get_top_correlations(correlations, n=3)
        
        self.assertIsInstance(top_corr, dict)
        self.assertLessEqual(len(top_corr), 3)


class TestTimeSeriesForecaster(unittest.TestCase):
    """Test cases for the TimeSeriesForecaster class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.forecaster = TimeSeriesForecaster()
        
        # Create mock time series data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.time_series_data = pd.DataFrame({
            'timestamp': dates,
            'component_id': np.repeat(list(range(1, 11)), 10),
            'outage_count': np.random.choice([0, 1, 2], size=100, p=[0.8, 0.15, 0.05]),
            'temperature': np.random.uniform(-10, 40, 100),
            'precipitation': np.random.uniform(0, 50, 100),
            'wind_speed': np.random.uniform(0, 30, 100)
        })

    def test_prepare_time_series_data(self):
        """Test time series data preparation."""
        prepared_data = self.forecaster.prepare_time_series_data(
            self.time_series_data, date_col='timestamp', target_col='outage_count'
        )
        
        self.assertIsInstance(prepared_data, pd.DataFrame)
        # Check that timestamp is set as index
        self.assertEqual(prepared_data.index.name, 'timestamp')
        
    def test_train_model(self):
        """Test model training functionality."""
        # This test will be implementation-specific
        # We'll mock the training to avoid actual computation
        with patch.object(
            self.forecaster, '_train_arima_model', 
            return_value=MagicMock()
        ) as mock_train:
            self.forecaster.train(
                self.time_series_data, 
                date_col='timestamp',
                target_col='outage_count',
                model_type='arima'
            )
            mock_train.assert_called_once()

    def test_forecast_future_values(self):
        """Test forecasting functionality."""
        # Mock the model
        self.forecaster.model = MagicMock()
        self.forecaster.model.forecast = MagicMock(return_value=np.array([0.5, 0.7, 0.3]))
        self.forecaster.model_type = 'arima'
        self.forecaster.trained = True
        
        # Create future dates
        future_dates = pd.date_range(start='2023-04-10', periods=3, freq='D')
        
        with patch.object(
            self.forecaster, '_forecast_arima',
            return_value=pd.Series([0.5, 0.7, 0.3], index=future_dates)
        ):
            forecast_result = self.forecaster.forecast(periods=3, future_dates=future_dates)
            
            self.assertIsInstance(forecast_result, pd.DataFrame)
            self.assertEqual(len(forecast_result), 3)


if __name__ == '__main__':
    unittest.main()
