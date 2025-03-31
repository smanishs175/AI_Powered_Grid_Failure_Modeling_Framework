"""
Time Series Forecaster Module

This module provides functionality for forecasting component failures over time using
time series analysis techniques such as LSTM and Prophet.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import logging
import pickle
import json
from typing import Dict, Any, List, Tuple, Union, Optional
from datetime import datetime, timedelta

# For conditional Prophet import
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
# Import utilities
from gfmf.failure_prediction.utils.model_utils import load_config, save_model
from gfmf.failure_prediction.utils.evaluation_utils import evaluate_time_series_model
from gfmf.failure_prediction.utils.visualization import plot_time_series_forecast_grid

# Configure logger
logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """
    Time series forecaster for predicting future component failures.
    
    This class provides functionality for training time series models
    to predict component failures over time using techniques such as LSTM and Prophet.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the time series forecaster.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.model_type = self.config['time_series'].get('model_type', 'lstm')
        self.sequence_length = self.config['time_series'].get('sequence_length', 14)
        self.forecast_horizon = self.config['time_series'].get('forecast_horizon', 30)
        self.model_path = None
        self.scaler = None
        self.feature_cols = None
        self.date_col = None
        self.target_col = None
        self.grouped_data = None
        
        logger.info(f"Initialized {self.__class__.__name__} with model type: {self.model_type}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file or use default.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path and os.path.exists(config_path):
            return load_config(config_path)
        
        # Default config path
        default_config_path = os.path.join(
            os.path.dirname(__file__), 
            'config', 
            'default_config.yaml'
        )
        
        if os.path.exists(default_config_path):
            return load_config(default_config_path)
        
        # Fallback default configuration
        logger.warning("No configuration file found. Using fallback default configuration.")
        return {
            'paths': {
                'module1_data': "data/processed/",
                'module2_data': "data/vulnerability_analysis/",
                'output_data': "data/failure_prediction/",
                'logs': "logs/failure_prediction/"
            },
            'time_series': {
                'model_type': 'lstm',
                'sequence_length': 14,
                'lstm_units': [64, 32],
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'forecast_horizon': 30
            }
        }
    
    def load_data(
        self,
        data_path: str,
        date_column: str = 'date',
        target_column: str = 'failure_count',
        feature_columns: Optional[List[str]] = None,
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load and prepare time series data.
        
        Args:
            data_path: Path to the data file
            date_column: Name of the date column
            target_column: Name of the target column
            feature_columns: List of feature columns to use (optional)
            group_by: Column to group by for aggregation (optional)
            
        Returns:
            Processed time series DataFrame
        """
        # Load data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            logger.error(f"Unsupported file format: {data_path}")
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded data: {df.shape}")
        
        # Ensure date column is in datetime format
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
        else:
            logger.error(f"Date column '{date_column}' not found in data")
            raise ValueError(f"Date column '{date_column}' not found in data")
        
        # Check target column
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Group data if specified
        if group_by and group_by in df.columns:
            # Aggregate by group and date
            grouped_df = df.groupby([group_by, date_column])[target_column].sum().reset_index()
            logger.info(f"Grouped data by {group_by}: {grouped_df[group_by].nunique()} groups")
            
            # Store as list of DataFrames, one per group
            self.grouped_data = []
            for group, group_df in grouped_df.groupby(group_by):
                self.grouped_data.append({
                    'group': group,
                    'data': group_df.sort_values(date_column)
                })
            
            # For simplicity, return the full grouped DataFrame
            df = grouped_df
        
        # Store column names
        self.date_col = date_column
        self.target_col = target_column
        self.feature_cols = feature_columns if feature_columns else []
        
        logger.info(f"Prepared time series data: {df.shape}")
        return df
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input time series data
            seq_length: Sequence length
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build an LSTM model for time series forecasting.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled LSTM model
        """
        # Get model parameters
        lstm_units = self.config['time_series'].get('lstm_units', [64, 32])
        learning_rate = self.config['time_series'].get('learning_rate', 0.001)
        
        # Build model
        model = keras.Sequential()
        
        # Add LSTM layers
        model.add(keras.layers.LSTM(
            lstm_units[0],
            input_shape=input_shape,
            return_sequences=len(lstm_units) > 1
        ))
        
        # Add additional LSTM layers
        for i, units in enumerate(lstm_units[1:]):
            return_sequences = i < len(lstm_units) - 2
            model.add(keras.layers.LSTM(units, return_sequences=return_sequences))
        
        # Add dense output layer
        model.add(keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        logger.info(f"Built LSTM model with {len(lstm_units)} layers")
        return model
    
    def _build_prophet_model(self) -> Any:
        """
        Build a Prophet model for time series forecasting.
        
        Returns:
            Prophet model
        """
        if not PROPHET_AVAILABLE:
            logger.error("Prophet is not installed. Please install it with 'pip install prophet'")
            raise ImportError("Prophet is not installed")
            
        # Create and return Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        logger.info("Built Prophet model")
        return model
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2):
        """
        Train the time series forecasting model.
        
        Args:
            df: Time series DataFrame
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history or trained model
        """
        # Check model type
        if self.model_type == 'lstm':
            return self._train_lstm(df, validation_split)
        elif self.model_type == 'prophet':
            return self._train_prophet(df)
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _train_lstm(self, df: pd.DataFrame, validation_split: float = 0.2):
        """
        Train an LSTM model for time series forecasting.
        
        Args:
            df: Time series DataFrame
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history
        """
        # Prepare data
        if len(self.feature_cols) > 0:
            # Use specified features
            data = df[self.feature_cols + [self.target_col]].values
        else:
            # Use only target column
            data = df[self.target_col].values.reshape(-1, 1)
        
        # Scale data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, self.sequence_length)
        
        # For LSTM with multiple features, the targets should only be the target column
        if len(self.feature_cols) > 0:
            y = y[:, -1]  # Last column is the target
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        n_features = X.shape[2]
        self.model = self._build_lstm_model((self.sequence_length, n_features))
        
        # Train model
        epochs = self.config['time_series'].get('epochs', 100)
        batch_size = self.config['time_series'].get('batch_size', 32)
        
        # Add early stopping
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"LSTM model training completed: {len(history.epoch)} epochs")
        return history
    
    def _train_prophet(self, df: pd.DataFrame):
        """
        Train a Prophet model for time series forecasting.
        
        Args:
            df: Time series DataFrame
            
        Returns:
            Trained Prophet model
        """
        # Create Prophet model
        self.model = self._build_prophet_model()
        
        # Prepare data for Prophet
        prophet_df = df[[self.date_col, self.target_col]].copy()
        prophet_df.columns = ['ds', 'y']  # Prophet requires these column names
        
        # Fit model
        self.model.fit(prophet_df)
        
        logger.info("Prophet model training completed")
        return self.model
    
    def forecast(
        self, 
        df: pd.DataFrame = None,
        periods: int = None, 
        future_dates: Optional[pd.DatetimeIndex] = None,
        return_confidence_intervals: bool = True
    ) -> pd.DataFrame:
        """
        Generate time series forecasts.
        
        Args:
            df: Input DataFrame (only needed if not using previously loaded data)
            periods: Number of periods to forecast
            future_dates: Specific dates to forecast for
            return_confidence_intervals: Whether to return confidence intervals
            
        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet")
        
        # Set periods if not provided
        if periods is None:
            periods = self.forecast_horizon
        
        # Generate forecasts based on model type
        if self.model_type == 'lstm':
            return self._forecast_lstm(df, periods, return_confidence_intervals)
        elif self.model_type == 'prophet':
            return self._forecast_prophet(periods, future_dates, return_confidence_intervals)
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _forecast_lstm(self, df: pd.DataFrame, periods: int, return_confidence_intervals: bool = True) -> pd.DataFrame:
        """
        Generate forecasts using LSTM model.
        
        Args:
            df: Input DataFrame
            periods: Number of periods to forecast
            return_confidence_intervals: Whether to return confidence intervals
            
        Returns:
            DataFrame with forecasts
        """
        # If df not provided, use the last loaded data
        if df is None:
            logger.error("DataFrame is required for LSTM forecasting")
            raise ValueError("DataFrame is required for LSTM forecasting")
        
        # Prepare data
        if len(self.feature_cols) > 0:
            # Use specified features
            data = df[self.feature_cols + [self.target_col]].values
        else:
            # Use only target column
            data = df[self.target_col].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.transform(data)
        
        # Get the last sequence
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Generate forecasts
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            # Make prediction
            pred = self.model.predict(current_sequence)[0][0]
            
            # Append prediction to forecasts
            forecasts.append(pred)
            
            # Update sequence for next prediction
            if len(self.feature_cols) > 0:
                # For multivariate forecasting, we only update the target column
                # and keep other features constant (using last known values)
                next_step = current_sequence[0, -1, :].copy()
                next_step[-1] = pred  # Update target column
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = next_step
            else:
                # For univariate forecasting
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred
        
        # Create forecast dates
        last_date = pd.to_datetime(df[self.date_col].iloc[-1])
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'  # Assuming daily frequency
        )
        
        # Inverse transform forecasts
        if len(self.feature_cols) > 0:
            # For multivariate forecasting, we need to create dummy values for other features
            dummy_features = np.tile(scaled_data[-1, :-1], (periods, 1))  # Use last known values
            scaled_forecasts = np.column_stack((dummy_features, forecasts))
            inv_forecasts = self.scaler.inverse_transform(scaled_forecasts)[:, -1]
        else:
            # For univariate forecasting
            inv_forecasts = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': inv_forecasts
        })
        
        # Add confidence intervals if requested
        if return_confidence_intervals:
            # Simple approach: use fixed percentage for confidence intervals
            forecast_df['lower_bound'] = forecast_df['forecast'] * 0.8
            forecast_df['upper_bound'] = forecast_df['forecast'] * 1.2
        
        logger.info(f"Generated {periods} LSTM forecasts")
        return forecast_df
    
    def _forecast_prophet(
        self, 
        periods: int, 
        future_dates: Optional[pd.DatetimeIndex] = None,
        return_confidence_intervals: bool = True
    ) -> pd.DataFrame:
        """
        Generate forecasts using Prophet model.
        
        Args:
            periods: Number of periods to forecast
            future_dates: Specific dates to forecast for
            return_confidence_intervals: Whether to return confidence intervals
            
        Returns:
            DataFrame with forecasts
        """
        # Create future DataFrame
        if future_dates is not None:
            future = pd.DataFrame({'ds': future_dates})
        else:
            future = self.model.make_future_dataframe(periods=periods)
        
        # Make forecast
        forecast = self.model.predict(future)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast['ds'],
            'forecast': forecast['yhat']
        })
        
        # Add confidence intervals if requested
        if return_confidence_intervals:
            forecast_df['lower_bound'] = forecast['yhat_lower']
            forecast_df['upper_bound'] = forecast['yhat_upper']
        
        logger.info(f"Generated {len(forecast_df)} Prophet forecasts")
        return forecast_df
    
    def evaluate(
        self, 
        test_df: pd.DataFrame, 
        plot: bool = True,
        save_plot: bool = False,
        plot_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_df: Test DataFrame
            plot: Whether to create evaluation plot
            save_plot: Whether to save the plot
            plot_path: Path to save the plot
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating time series model performance")
        
        # Prepare test data
        if len(self.feature_cols) > 0:
            # Use specified features
            test_data = test_df[self.feature_cols + [self.target_col]].values
        else:
            # Use only target column
            test_data = test_df[self.target_col].values.reshape(-1, 1)
        
        # Scale test data
        scaled_test_data = self.scaler.transform(test_data)
        
        # Create sequences
        X_test, y_test = self._create_sequences(scaled_test_data, self.sequence_length)
        
        # For LSTM with multiple features, the targets should only be the target column
        if len(self.feature_cols) > 0:
            y_test = y_test[:, -1]  # Last column is the target
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred = y_pred.flatten()
        
        # Inverse transform predictions and actual values for proper metrics
        if len(self.feature_cols) > 0:
            # Extract test points for inverse transform
            test_points = []
            for i in range(len(y_test)):
                test_points.append(scaled_test_data[i + self.sequence_length])
            test_points = np.array(test_points)
            
            # Replace target column with predictions
            test_points_pred = test_points.copy()
            test_points_pred[:, -1] = y_pred
            
            # Inverse transform
            inv_y_test = self.scaler.inverse_transform(test_points)[:, -1]
            inv_y_pred = self.scaler.inverse_transform(test_points_pred)[:, -1]
        else:
            # Inverse transform univariate data
            inv_y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            inv_y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        metrics = evaluate_time_series_model(inv_y_test, inv_y_pred)
        
        # Log metrics
        logger.info(f"Evaluation metrics: rmse={metrics['rmse']:.4f}, "
                    f"mae={metrics['mae']:.4f}, r2={metrics['r2']:.4f}")
        
        # Create and save plot if requested
        if plot:
            # Create date indices
            if self.date_col in test_df.columns:
                dates = test_df[self.date_col].iloc[self.sequence_length:].values
            else:
                dates = np.arange(len(inv_y_test))
            
            # Create plot
            fig = plot_time_series_forecast_grid(
                inv_y_test, 
                inv_y_pred, 
                dates=dates,
                title='Time Series Forecast Evaluation'
            )
            
            if save_plot:
                if plot_path is None:
                    plot_dir = os.path.join(self.config['paths']['output_data'], 'plots')
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_path = os.path.join(plot_dir, 'time_series_evaluation.png')
                
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved evaluation plot to {plot_path}")
            
            plt.close(fig)
        
        return metrics
    
    def save(self, save_dir: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            save_dir: Directory to save the model
            model_name: Name for the saved model
            
        Returns:
            Path to the saved model
        """
        # Check if model is trained
        if self.model is None:
            logger.error("No trained model to save")
            raise ValueError("No trained model to save")
        
        # Set save directory if not provided
        if save_dir is None:
            save_dir = os.path.join(self.config['paths']['output_data'], 'models')
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"time_series_{self.model_type}_{timestamp}"
        
        # Save model based on type
        if self.model_type == 'lstm':
            # Save Keras model
            model_path = os.path.join(save_dir, model_name)
            self.model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(save_dir, f"{model_name}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
        elif self.model_type == 'prophet':
            # Save Prophet model
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            logger.error(f"Unsupported model type for saving: {self.model_type}")
            raise ValueError(f"Unsupported model type for saving: {self.model_type}")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_columns': self.feature_cols,
            'date_column': self.date_col,
            'target_column': self.target_col,
            'sequence_length': self.sequence_length,
            'config': self.config['time_series'],
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save model path
        self.model_path = model_path
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load(self, model_path: str, metadata_path: Optional[str] = None) -> Any:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            metadata_path: Path to the saved metadata
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        # Determine model type from path if not set
        if model_path.endswith('.h5') or os.path.isdir(model_path):
            self.model_type = 'lstm'
        elif model_path.endswith('.pkl'):
            self.model_type = 'prophet'
        
        # Load model based on type
        if self.model_type == 'lstm':
            # Load Keras model
            self.model = keras.models.load_model(model_path)
            
            # Try to find matching scaler
            scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.warning("No scaler found. Predictions may be incorrect.")
                
        elif self.model_type == 'prophet':
            # Load Prophet model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            logger.error(f"Unsupported model type for loading: {self.model_type}")
            raise ValueError(f"Unsupported model type for loading: {self.model_type}")
        
        # Load metadata if available
        if metadata_path is None:
            # Try to find matching metadata
            potential_metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
            if os.path.exists(potential_metadata_path):
                metadata_path = potential_metadata_path
        
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load metadata
            self.model_type = metadata.get('model_type', self.model_type)
            self.feature_cols = metadata.get('feature_columns', None)
            self.date_col = metadata.get('date_column', None)
            self.target_col = metadata.get('target_column', None)
            self.sequence_length = metadata.get('sequence_length', self.sequence_length)
            
            logger.info(f"Loaded model metadata from {metadata_path}")
        
        # Save model path
        self.model_path = model_path
        
        logger.info(f"Successfully loaded {self.model_type} model from {model_path}")
        return self.model
