"""
Neural Predictor Module

This module provides functionality for predicting component failures using neural networks.
It takes component properties, environmental conditions, and vulnerability scores as input
and predicts failure probabilities for each component.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import yaml
import json
from typing import Dict, Any, List, Tuple, Union, Optional
from datetime import datetime

# Import utilities
from gfmf.failure_prediction.utils.model_utils import (
    load_config, save_model, create_model_registry_entry
)
from gfmf.failure_prediction.utils.evaluation_utils import (
    evaluate_classification_model, plot_confusion_matrix, 
    plot_roc_curve, plot_precision_recall_curve
)

# Configure logger
logger = logging.getLogger(__name__)


class NeuralPredictor:
    """
    Neural network-based component failure predictor.
    
    This class provides functionality for training neural network models
    to predict component failure probabilities based on their properties
    and environmental conditions.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the neural predictor.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.preprocessing_pipeline = None
        self.feature_columns = None
        self.model_type = self.config['neural_predictor'].get('model_type', 'deep_neural_network')
        self.model_path = None
        self.training_history = None
        
        # Set up TensorFlow session
        tf.random.set_seed(42)  # For reproducibility
        
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
            'neural_predictor': {
                'model_type': 'deep_neural_network',
                'hidden_layers': [128, 64, 32],
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'test_size': 0.2,
                'early_stopping_patience': 10,
                'dropout_rate': 0.2
            }
        }
    
    def load_data(
        self,
        module1_data_path: str = None,
        module2_data_path: str = None,
        component_data_file: str = 'component_properties.csv',
        vulnerability_data_file: str = 'vulnerability_scores.csv',
        environmental_data_file: str = 'environmental_data.csv',
        failure_data_file: str = 'historical_failures.csv'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data from previous modules.
        
        Args:
            module1_data_path: Path to Module 1 output data
            module2_data_path: Path to Module 2 output data
            component_data_file: Filename for component properties data
            vulnerability_data_file: Filename for vulnerability scores data
            environmental_data_file: Filename for environmental data
            failure_data_file: Filename for historical failure data
            
        Returns:
            Tuple of (features_df, target_df)
        """
        # Get paths from config if not provided
        if module1_data_path is None:
            module1_data_path = self.config['paths']['module1_data']
        if module2_data_path is None:
            module2_data_path = self.config['paths']['module2_data']
        
        # Load component properties
        component_path = os.path.join(module1_data_path, component_data_file)
        if os.path.exists(component_path):
            comp_df = pd.read_csv(component_path)
            logger.info(f"Loaded component data: {comp_df.shape}")
        else:
            logger.error(f"Component data file not found: {component_path}")
            raise FileNotFoundError(f"Component data file not found: {component_path}")
        
        # Load vulnerability scores
        vuln_path = os.path.join(module2_data_path, vulnerability_data_file)
        if os.path.exists(vuln_path):
            vuln_df = pd.read_csv(vuln_path)
            logger.info(f"Loaded vulnerability data: {vuln_df.shape}")
        else:
            logger.warning(f"Vulnerability data file not found: {vuln_path}")
            vuln_df = None
        
        # Load environmental data
        env_path = os.path.join(module1_data_path, environmental_data_file)
        if os.path.exists(env_path):
            env_df = pd.read_csv(env_path)
            logger.info(f"Loaded environmental data: {env_df.shape}")
        else:
            logger.warning(f"Environmental data file not found: {env_path}")
            env_df = None
        
        # Load historical failure data
        fail_path = os.path.join(module1_data_path, failure_data_file)
        if os.path.exists(fail_path):
            fail_df = pd.read_csv(fail_path)
            logger.info(f"Loaded historical failure data: {fail_df.shape}")
        else:
            logger.error(f"Historical failure data file not found: {fail_path}")
            raise FileNotFoundError(f"Historical failure data file not found: {fail_path}")
        
        # Process and merge data
        merged_df = self._process_and_merge_data(comp_df, vuln_df, env_df, fail_df)
        
        # Separate features and target
        target_column = 'failure_indicator'
        if target_column not in merged_df.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        features_df = merged_df.drop(columns=[target_column])
        target_df = merged_df[target_column]
        
        # Store feature columns for future use
        self.feature_columns = features_df.columns.tolist()
        
        logger.info(f"Prepared features ({features_df.shape}) and target ({target_df.shape})")
        return features_df, target_df
        
    def _process_and_merge_data(
        self,
        component_df: pd.DataFrame,
        vulnerability_df: Optional[pd.DataFrame] = None,
        environmental_df: Optional[pd.DataFrame] = None,
        failure_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Process and merge data from different sources.
        
        Args:
            component_df: DataFrame with component properties
            vulnerability_df: DataFrame with vulnerability scores
            environmental_df: DataFrame with environmental data
            failure_df: DataFrame with historical failure data
            
        Returns:
            Merged DataFrame with features and target
        """
        logger.info("Processing and merging input data")
        
        # Ensure component_id is present
        if 'component_id' not in component_df.columns:
            logger.error("component_id column not found in component data")
            raise ValueError("component_id column not found in component data")
        
        # Base DataFrame
        merged_df = component_df.copy()
        
        # Merge vulnerability data if available
        if vulnerability_df is not None and 'component_id' in vulnerability_df.columns:
            # Select only numeric columns and component_id
            numeric_cols = vulnerability_df.select_dtypes(include=[np.number]).columns.tolist()
            vuln_cols = ['component_id'] + numeric_cols
            
            if len(vuln_cols) > 1:  # At least one numeric column besides component_id
                vuln_subset = vulnerability_df[vuln_cols]
                merged_df = pd.merge(
                    merged_df, 
                    vuln_subset,
                    on='component_id',
                    how='left'
                )
                logger.info(f"Merged vulnerability data: {len(numeric_cols)} features")
            else:
                logger.warning("No numeric vulnerability features found")
        
        # Merge environmental data if available
        if environmental_df is not None:
            # Ensure it has datetime and location information
            if 'date' in environmental_df.columns and 'location_id' in environmental_df.columns:
                # Create a recent snapshot of environmental data
                env_snapshot = environmental_df.sort_values('date').groupby('location_id').last().reset_index()
                
                # If component data has location_id, merge on that
                if 'location_id' in merged_df.columns:
                    merged_df = pd.merge(
                        merged_df,
                        env_snapshot.drop(columns=['date']),
                        on='location_id',
                        how='left'
                    )
                    logger.info("Merged environmental data based on location_id")
                else:
                    logger.warning("Cannot merge environmental data: location_id not found in component data")
            else:
                logger.warning("Environmental data missing required columns (date, location_id)")
        
        # Process failure data
        if failure_df is not None:
            if 'component_id' in failure_df.columns and 'failure_date' in failure_df.columns:
                # Create binary indicator for components that have failed
                failure_indicator = failure_df.groupby('component_id').size().reset_index()
                failure_indicator.columns = ['component_id', 'failure_count']
                failure_indicator['failure_indicator'] = 1
                
                # Merge with main dataset
                merged_df = pd.merge(
                    merged_df,
                    failure_indicator[['component_id', 'failure_indicator']],
                    on='component_id',
                    how='left'
                )
                
                # Fill missing values (components that never failed)
                merged_df['failure_indicator'] = merged_df['failure_indicator'].fillna(0)
                logger.info("Added failure indicator based on historical failures")
            else:
                logger.error("Failure data missing required columns (component_id, failure_date)")
                raise ValueError("Failure data missing required columns (component_id, failure_date)")
        else:
            logger.error("Failure data is required but not provided")
            raise ValueError("Failure data is required but not provided")
        
        # Handle missing values
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].mean())
        
        # Drop non-numeric columns except component_id
        non_numeric_cols = merged_df.select_dtypes(exclude=[np.number]).columns
        cols_to_drop = [col for col in non_numeric_cols if col != 'component_id']
        merged_df = merged_df.drop(columns=cols_to_drop)
        
        logger.info(f"Final merged data shape: {merged_df.shape}")
        return merged_df
    
    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build the neural network model.
        
        Args:
            input_dim: Input dimension (number of features)
            
        Returns:
            Built model
        """
        # Get model configuration
        hidden_layers = self.config['neural_predictor'].get('hidden_layers', [128, 64, 32])
        learning_rate = self.config['neural_predictor'].get('learning_rate', 0.001)
        dropout_rate = self.config['neural_predictor'].get('dropout_rate', 0.2)
        
        # Create model
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in hidden_layers:
            model.add(keras.layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            ))
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        logger.info(f"Built neural network model: {hidden_layers} hidden layers")
        return model
    
    def preprocess_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None, fit: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Preprocess data for model training or prediction.
        
        Args:
            X: Features DataFrame
            y: Target Series (optional)
            fit: Whether to fit the preprocessing pipeline
            
        Returns:
            Preprocessed X (and y if provided)
        """
        if self.preprocessing_pipeline is None and fit:
            # Create preprocessing pipeline for numeric features
            numeric_features = X.select_dtypes(include=[np.number]).columns
            
            # Create column transformer with StandardScaler for numeric features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features)
                ],
                remainder='drop'
            )
            
            # Create preprocessing pipeline
            self.preprocessing_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor)
            ])
        
        # Apply preprocessing
        if self.preprocessing_pipeline is not None:
            if fit:
                X_processed = self.preprocessing_pipeline.fit_transform(X)
            else:
                X_processed = self.preprocessing_pipeline.transform(X)
        else:
            logger.warning("No preprocessing pipeline available. Using raw features.")
            X_processed = X.values
        
        # Return preprocessed data
        if y is not None:
            return X_processed, np.array(y)
        else:
            return X_processed
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = None, early_stopping: bool = True) -> Dict[str, List[float]]:
        """
        Train the model on the provided data.
        
        Args:
            X: Features DataFrame
            y: Target Series
            validation_split: Fraction of data to use for validation
            early_stopping: Whether to use early stopping
            
        Returns:
            Training history
        """
        logger.info("Training neural predictor model")
        
        # Set validation split if not provided
        if validation_split is None:
            validation_split = self.config['neural_predictor'].get('test_size', 0.2)
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Preprocess data
        X_train_processed, y_train = self.preprocess_data(X_train, y_train, fit=True)
        X_val_processed, y_val = self.preprocess_data(X_val, y_val, fit=False)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(X_train_processed.shape[1])
        
        # Set up callbacks
        callbacks = []
        
        # Add early stopping if requested
        if early_stopping:
            early_stopping_patience = self.config['neural_predictor'].get('early_stopping_patience', 10)
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ))
        
        # Train model
        epochs = self.config['neural_predictor'].get('epochs', 100)
        batch_size = self.config['neural_predictor'].get('batch_size', 32)
        
        history = self.model.fit(
            X_train_processed, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_processed, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history
        self.training_history = history.history
        
        logger.info(f"Model training completed: {len(history.epoch)} epochs")
        return history.history
        
    def predict(self, X: pd.DataFrame, return_proba: bool = True) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features DataFrame
            return_proba: Whether to return probabilities or binary predictions
            
        Returns:
            Predictions
        """
        logger.info(f"Making predictions on {len(X)} samples")
        
        # Check if model is trained
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet")
        
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Return probabilities or binary predictions
        if return_proba:
            return predictions.flatten()
        else:
            return (predictions >= 0.5).astype(int).flatten()
            
    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5, 
               save_plots: bool = False, plots_dir: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            threshold: Classification threshold
            save_plots: Whether to save evaluation plots
            plots_dir: Directory to save plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Preprocess data
        X_processed, y_true = self.preprocess_data(X, y)
        
        # Get predictions
        y_prob = self.model.predict(X_processed).flatten()
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        metrics = evaluate_classification_model(y_true, y_pred, y_prob, threshold)
        
        # Log metrics
        logger.info(f"Evaluation metrics: accuracy={metrics['accuracy']:.4f}, "
                    f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                    f"f1={metrics['f1']:.4f}, roc_auc={metrics.get('roc_auc', 0):.4f}")
        
        # Create plots
        if save_plots:
            if plots_dir is None:
                plots_dir = os.path.join(self.config['paths']['output_data'], 'plots')
            
            os.makedirs(plots_dir, exist_ok=True)
            
            # Confusion matrix
            cm_fig = plot_confusion_matrix(y_true, y_pred)
            cm_fig.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300)
            
            # ROC curve
            roc_fig = plot_roc_curve(y_true, y_prob)
            roc_fig.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300)
            
            # Precision-recall curve
            pr_fig = plot_precision_recall_curve(y_true, y_prob)
            pr_fig.savefig(os.path.join(plots_dir, 'precision_recall_curve.png'), dpi=300)
            
            logger.info(f"Saved evaluation plots to {plots_dir}")
        
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
            model_name = f"neural_predictor_{timestamp}"
        
        # Save model
        model_path = os.path.join(save_dir, model_name)
        self.model.save(model_path)
        
        # Save preprocessing pipeline
        if self.preprocessing_pipeline is not None:
            pipeline_path = os.path.join(save_dir, f"{model_name}_pipeline.pkl")
            with open(pipeline_path, 'wb') as f:
                pickle.dump(self.preprocessing_pipeline, f)
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'config': self.config['neural_predictor'],
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Add to model registry if configured
        registry_path = os.path.join(save_dir, 'model_registry.json')
        
        # Create registry entry
        metrics = {}
        if self.training_history is not None:
            # Get final metrics from training history
            for metric, values in self.training_history.items():
                if not metric.startswith('val_'):  # Only training metrics
                    metrics[metric] = float(values[-1])  # Last value
        
        create_model_registry_entry(
            model_path=model_path,
            model_type=self.model_type,
            performance_metrics=metrics,
            features_used=self.feature_columns,
            training_params=self.config['neural_predictor'],
            registry_path=registry_path
        )
        
        # Save model path
        self.model_path = model_path
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load(self, model_path: str, pipeline_path: Optional[str] = None) -> keras.Model:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            pipeline_path: Path to the saved preprocessing pipeline
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load model
        self.model = keras.models.load_model(model_path)
        self.model_path = model_path
        
        # Load preprocessing pipeline if provided
        if pipeline_path is None:
            # Try to find matching pipeline
            base_path = os.path.splitext(model_path)[0]
            potential_pipeline_path = f"{base_path}_pipeline.pkl"
            if os.path.exists(potential_pipeline_path):
                pipeline_path = potential_pipeline_path
        
        if pipeline_path is not None and os.path.exists(pipeline_path):
            with open(pipeline_path, 'rb') as f:
                self.preprocessing_pipeline = pickle.load(f)
            logger.info(f"Loaded preprocessing pipeline from {pipeline_path}")
        
        # Check for metadata file
        metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load metadata
            self.model_type = metadata.get('model_type', self.model_type)
            self.feature_columns = metadata.get('feature_columns', None)
            
            logger.info(f"Loaded model metadata from {metadata_path}")
        
        logger.info(f"Successfully loaded model from {model_path}")
        return self.model
