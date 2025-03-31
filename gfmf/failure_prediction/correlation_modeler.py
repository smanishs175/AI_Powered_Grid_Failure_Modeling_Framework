"""
Correlation Modeler Module

This module provides functionality for analyzing correlations between environmental factors
and component failures in power grids. It identifies key relationships, quantifies their
strength, and helps understand the environmental conditions that contribute to failures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, List, Tuple, Union, Optional
import logging
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import mutual_info_regression

# Import utilities
from gfmf.failure_prediction.utils.model_utils import load_config
from gfmf.failure_prediction.utils.visualization import plot_correlation_heatmap

# Configure logger
logger = logging.getLogger(__name__)


class CorrelationModeler:
    """
    Correlation modeler for analyzing relationships between environmental factors and failures.
    
    This class provides functionality for identifying and quantifying correlations
    between environmental conditions and component failures using various statistical
    and machine learning techniques.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the correlation modeler.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.model_types = self.config['correlation_models'].get('model_types', ['linear', 'nonlinear'])
        self.env_factors = self.config['correlation_models'].get('environmental_factors', 
                                                                ['temperature', 'wind_speed', 'precipitation'])
        
        self.correlation_results = {}  # Dictionary to store correlation results
        self.feature_importance = {}  # Dictionary to store feature importance results
        self.models = {}  # Dictionary to store trained models
        
        logger.info(f"Initialized {self.__class__.__name__} with model types: {self.model_types}")
    
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
            'correlation_models': {
                'model_types': ['linear', 'nonlinear'],
                'environmental_factors': ['temperature', 'wind_speed', 'precipitation']
            }
        }
    
    def prepare_data(
        self,
        environmental_df: pd.DataFrame,
        failure_df: pd.DataFrame,
        component_df: Optional[pd.DataFrame] = None,
        date_column: str = 'date',
        location_column: Optional[str] = 'location_id',
        component_column: str = 'component_id',
        aggregation_period: str = 'D',  # 'D' for daily, 'W' for weekly, 'M' for monthly
        lag_periods: List[int] = [0, 1, 2, 3, 7]
    ) -> pd.DataFrame:
        """
        Prepare and merge data for correlation analysis.
        
        Args:
            environmental_df: DataFrame with environmental data
            failure_df: DataFrame with failure data
            component_df: DataFrame with component properties (optional)
            date_column: Name of the date column
            location_column: Name of the location column (optional)
            component_column: Name of the component column in failure data
            aggregation_period: Time period for aggregating data
            lag_periods: List of lag periods to include for environmental variables
            
        Returns:
            Merged DataFrame for correlation analysis
        """
        logger.info("Preparing data for correlation analysis")
        
        # Ensure date columns are in datetime format
        environmental_df[date_column] = pd.to_datetime(environmental_df[date_column])
        failure_df[date_column] = pd.to_datetime(failure_df[date_column])
        
        # Convert to time series with date index
        env_ts = environmental_df.set_index(date_column)
        
        # Aggregate failures by date
        failure_counts = failure_df.groupby(failure_df[date_column].dt.to_period(aggregation_period)).size()
        failure_ts = failure_counts.reset_index()
        failure_ts.columns = [date_column, 'failure_count']
        failure_ts[date_column] = failure_ts[date_column].dt.to_timestamp()
        failure_ts = failure_ts.set_index(date_column)
        
        # Resample environmental data to match aggregation period
        env_ts = env_ts.resample(aggregation_period).mean()
        
        # Create lagged features
        env_features = []
        
        # Identify numeric environmental columns
        env_cols = env_ts.select_dtypes(include=[np.number]).columns
        
        # Create lagged versions of each environmental variable
        for lag in lag_periods:
            if lag == 0:
                # Current values (no lag)
                lag_df = env_ts[env_cols].copy()
                lag_df.columns = [f"{col}" for col in env_cols]
                env_features.append(lag_df)
            else:
                # Lagged values
                lag_df = env_ts[env_cols].shift(lag)
                lag_df.columns = [f"{col}_lag{lag}" for col in env_cols]
                env_features.append(lag_df)
        
        # Combine all environmental features
        env_features_df = pd.concat(env_features, axis=1)
        
        # Merge with failure data
        merged_df = pd.merge(
            failure_ts,
            env_features_df,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Add component information if provided
        if component_df is not None and component_column in component_df.columns:
            # Count components by type
            if 'component_type' in component_df.columns:
                type_counts = component_df.groupby('component_type').size()
                
                # Merge with failure data to get failure rates by component type
                failure_component = failure_df.merge(
                    component_df[[component_column, 'component_type']],
                    on=component_column,
                    how='left'
                )
                
                # Calculate failure rate by component type and date
                type_failure_rates = failure_component.groupby(
                    [failure_component[date_column].dt.to_period(aggregation_period), 'component_type']
                ).size().unstack(fill_value=0)
                
                # Convert to rate by dividing by count of each component type
                for component_type in type_failure_rates.columns:
                    if component_type in type_counts:
                        type_failure_rates[component_type] = type_failure_rates[component_type] / type_counts[component_type]
                
                # Reset index and convert period to timestamp
                type_failure_rates = type_failure_rates.reset_index()
                type_failure_rates[date_column] = type_failure_rates[date_column].dt.to_timestamp()
                type_failure_rates = type_failure_rates.set_index(date_column)
                
                # Rename columns to indicate they are failure rates
                type_failure_rates.columns = [f"{col}_failure_rate" for col in type_failure_rates.columns]
                
                # Merge with main dataframe
                merged_df = pd.merge(
                    merged_df,
                    type_failure_rates,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
        
        # Drop rows with missing values
        merged_df = merged_df.dropna()
        
        # Store column types for later use
        self.feature_cols = env_features_df.columns.tolist()
        self.target_col = 'failure_count'
        
        logger.info(f"Prepared data for correlation analysis: {merged_df.shape}")
        return merged_df
    
    def analyze_correlations(
        self,
        data: pd.DataFrame,
        target_column: str = 'failure_count',
        correlation_types: List[str] = ['pearson', 'spearman'],
        visualization_path: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze correlations between environmental factors and failures.
        
        Args:
            data: DataFrame with merged data
            target_column: Name of the target column
            correlation_types: Types of correlation to calculate
            visualization_path: Path to save correlation heatmap visualization
            
        Returns:
            Dictionary with correlation results by type
        """
        logger.info("Analyzing correlations between environmental factors and failures")
        
        # Check if target column exists
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Calculate different types of correlations
        correlation_results = {}
        
        for corr_type in correlation_types:
            if corr_type == 'pearson':
                # Pearson correlation (linear relationships)
                corr_matrix = numeric_data.corr(method='pearson')
                corr_with_target = corr_matrix[target_column].sort_values(ascending=False)
                
            elif corr_type == 'spearman':
                # Spearman correlation (monotonic relationships)
                corr_matrix = numeric_data.corr(method='spearman')
                corr_with_target = corr_matrix[target_column].sort_values(ascending=False)
                
            elif corr_type == 'mutual_info':
                # Mutual information (non-linear relationships)
                # Normalize data for MI calculation
                X = numeric_data.drop(columns=[target_column])
                y = numeric_data[target_column]
                
                # Calculate mutual information
                mi_scores = mutual_info_regression(X, y)
                mi_scores = pd.Series(mi_scores, index=X.columns)
                corr_with_target = mi_scores.sort_values(ascending=False)
                
                # Create a dummy correlation matrix with MI scores
                corr_matrix = pd.DataFrame(index=numeric_data.columns, columns=numeric_data.columns)
                corr_matrix.loc[:, target_column] = 0
                corr_matrix.loc[target_column, :] = 0
                for col in X.columns:
                    corr_matrix.loc[col, target_column] = mi_scores[col]
                    corr_matrix.loc[target_column, col] = mi_scores[col]
                corr_matrix.loc[target_column, target_column] = 1.0
            
            else:
                logger.warning(f"Unsupported correlation type: {corr_type}")
                continue
            
            # Store results
            correlation_results[corr_type] = {
                'matrix': corr_matrix,
                'with_target': corr_with_target
            }
            
            # Log top correlations
            top_n = min(5, len(corr_with_target) - 1)  # Exclude self-correlation
            logger.info(f"Top {top_n} {corr_type} correlations with {target_column}:")
            for idx, (feature, corr) in enumerate(corr_with_target.iloc[1:top_n+1].items()):
                logger.info(f"  {idx+1}. {feature}: {corr:.4f}")
        
        # Visualize correlation heatmap
        if visualization_path is not None and 'pearson' in correlation_results:
            # Extract correlation matrix
            pearson_corr = correlation_results['pearson']['matrix']
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
            heatmap = sns.heatmap(
                pearson_corr,
                mask=mask,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                linewidths=.5,
                annot=False,
                cbar_kws={'shrink': .8}
            )
            plt.title('Correlation Heatmap (Pearson)')
            plt.tight_layout()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
            
            # Save heatmap
            plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved correlation heatmap to {visualization_path}")
        
        # Store results
        self.correlation_results = correlation_results
        
        return correlation_results
    
    def train_models(
        self,
        data: pd.DataFrame,
        target_column: str = 'failure_count',
        test_size: float = 0.2,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train models to predict failures based on environmental factors.
        
        Args:
            data: DataFrame with merged data
            target_column: Name of the target column
            test_size: Fraction of data to use for testing
            model_types: Types of models to train
            
        Returns:
            Dictionary with trained models and evaluation metrics
        """
        logger.info("Training correlation models")
        
        # Check if target column exists
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Use configured model types if not provided
        if model_types is None:
            model_types = self.model_types
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Prepare data
        X = numeric_data.drop(columns=[target_column])
        y = numeric_data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {}
        
        for model_type in model_types:
            if model_type == 'linear':
                # Linear regression
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Get feature importances (coefficients for linear model)
                feature_importances = pd.Series(
                    np.abs(model.coef_),
                    index=X.columns
                ).sort_values(ascending=False)
                
            elif model_type == 'nonlinear':
                # Random forest regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Get feature importances
                feature_importances = pd.Series(
                    model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
                
            else:
                logger.warning(f"Unsupported model type: {model_type}")
                continue
            
            # Store model and results
            models[model_type] = {
                'model': model,
                'scaler': scaler,
                'feature_importances': feature_importances,
                'metrics': {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse
                }
            }
            
            # Log results
            logger.info(f"{model_type.capitalize()} model results:")
            logger.info(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
            logger.info(f"  Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
            
            # Log top feature importances
            top_n = min(5, len(feature_importances))
            logger.info(f"  Top {top_n} features:")
            for idx, (feature, importance) in enumerate(feature_importances.iloc[:top_n].items()):
                logger.info(f"    {idx+1}. {feature}: {importance:.4f}")
        
        # Store models and feature importances
        self.models = models
        self.feature_importance = {
            model_type: results['feature_importances'] 
            for model_type, results in models.items()
        }
        
        return models
    
    def predict_failures(
        self,
        environmental_data: pd.DataFrame,
        model_type: str = 'nonlinear',
        include_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Predict failures based on environmental data.
        
        Args:
            environmental_data: DataFrame with environmental data
            model_type: Type of model to use for prediction
            include_confidence: Whether to include confidence intervals
            
        Returns:
            DataFrame with failure predictions
        """
        logger.info(f"Predicting failures using {model_type} model")
        
        # Check if model is trained
        if not self.models or model_type not in self.models:
            logger.error(f"Model {model_type} not trained")
            raise ValueError(f"Model {model_type} not trained")
        
        # Extract model and scaler
        model = self.models[model_type]['model']
        scaler = self.models[model_type]['scaler']
        
        # Prepare data
        X = environmental_data.select_dtypes(include=[np.number])
        
        # Remove any columns not used in training
        trained_features = set(scaler.feature_names_in_)
        X = X[[col for col in X.columns if col in trained_features]]
        
        # Check if any features are missing
        missing_features = trained_features - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}")
            # Could add feature imputation here if needed
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Create output DataFrame
        result_df = pd.DataFrame()
        result_df['predicted_failures'] = np.maximum(0, predictions)  # Ensure non-negative
        
        # Add confidence intervals for Random Forest
        if include_confidence and model_type == 'nonlinear' and isinstance(model, RandomForestRegressor):
            # Get individual tree predictions
            tree_preds = np.array([tree.predict(X_scaled) for tree in model.estimators_])
            
            # Calculate confidence intervals
            lower_bound = np.percentile(tree_preds, 2.5, axis=0)
            upper_bound = np.percentile(tree_preds, 97.5, axis=0)
            
            result_df['lower_bound'] = np.maximum(0, lower_bound)  # Ensure non-negative
            result_df['upper_bound'] = upper_bound
        
        return result_df
    
    def get_key_environmental_factors(
        self,
        top_n: int = 10,
        model_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get key environmental factors that influence failures.
        
        Args:
            top_n: Number of top factors to return
            model_type: Type of model to use (if None, aggregate results from all models)
            
        Returns:
            DataFrame with key environmental factors
        """
        logger.info("Getting key environmental factors")
        
        # Check if feature importance is available
        if not self.feature_importance:
            logger.error("Feature importance not available")
            raise ValueError("Feature importance not available")
        
        if model_type is not None:
            # Use specified model type
            if model_type not in self.feature_importance:
                logger.error(f"Model type {model_type} not found")
                raise ValueError(f"Model type {model_type} not found")
            
            # Get feature importance for specified model
            importance = self.feature_importance[model_type].copy()
            
            # Normalize importance
            importance = importance / importance.sum()
            
            # Get top N features
            top_features = importance.nlargest(top_n)
            
            # Create DataFrame
            result_df = pd.DataFrame({
                'feature': top_features.index,
                'importance': top_features.values,
                'model_type': model_type
            })
            
        else:
            # Aggregate results from all models
            all_features = []
            
            for model_type, importance in self.feature_importance.items():
                # Normalize importance
                norm_importance = importance / importance.sum()
                
                # Get top N features
                top_features = norm_importance.nlargest(top_n)
                
                # Add to list
                for feature, imp in top_features.items():
                    all_features.append({
                        'feature': feature,
                        'importance': imp,
                        'model_type': model_type
                    })
            
            # Create DataFrame
            result_df = pd.DataFrame(all_features)
        
        return result_df
    
    def visualize_key_factors(
        self,
        top_n: int = 10,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Visualize key environmental factors.
        
        Args:
            top_n: Number of top factors to visualize
            save_path: Path to save the visualization
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Visualizing top {top_n} environmental factors")
        
        # Get key factors
        key_factors_df = self.get_key_environmental_factors(top_n=top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot by model type
        model_types = key_factors_df['model_type'].unique()
        
        for i, model_type in enumerate(model_types):
            model_data = key_factors_df[key_factors_df['model_type'] == model_type]
            
            # Sort by importance
            model_data = model_data.sort_values('importance', ascending=True)
            
            # Plot horizontal bar chart
            ax.barh(
                y=np.arange(len(model_data)) + i * 0.4,
                width=model_data['importance'],
                height=0.3,
                label=model_type.capitalize()
            )
            
            # Add feature names
            for j, (_, row) in enumerate(model_data.iterrows()):
                ax.text(
                    0.01,
                    j + i * 0.4,
                    row['feature'],
                    ha='left',
                    va='center',
                    color='white' if row['importance'] > 0.2 else 'black',
                    fontsize=9
                )
        
        # Customize plot
        ax.set_yticks([])
        ax.set_xlabel('Relative Importance')
        ax.set_title(f'Top {top_n} Environmental Factors Influencing Failures')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # Add model-specific R² values if available
        if self.models:
            r2_text = "Model Performance (R²):\n"
            for model_type, results in self.models.items():
                r2 = results['metrics']['test_r2']
                r2_text += f"{model_type.capitalize()}: {r2:.4f}\n"
            
            ax.text(
                0.98,
                0.02,
                r2_text,
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved factor visualization to {save_path}")
        
        return fig
    
    def save_results(
        self,
        save_dir: Optional[str] = None,
        prefix: str = 'correlation_analysis'
    ) -> Dict[str, str]:
        """
        Save analysis results to files.
        
        Args:
            save_dir: Directory to save results
            prefix: Prefix for output filenames
            
        Returns:
            Dictionary with paths to saved files
        """
        # Set save directory if not provided
        if save_dir is None:
            save_dir = os.path.join(self.config['paths']['output_data'], 'correlation_analysis')
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save correlation results
        if self.correlation_results:
            for corr_type, results in self.correlation_results.items():
                # Save correlation with target
                corr_path = os.path.join(save_dir, f"{prefix}_{corr_type}_correlations.csv")
                results['with_target'].to_csv(corr_path)
                saved_files[f"{corr_type}_correlations"] = corr_path
                
                # Save correlation matrix
                matrix_path = os.path.join(save_dir, f"{prefix}_{corr_type}_matrix.csv")
                results['matrix'].to_csv(matrix_path)
                saved_files[f"{corr_type}_matrix"] = matrix_path
                
                # Save correlation heatmap
                heatmap_path = os.path.join(save_dir, f"{prefix}_{corr_type}_heatmap.png")
                fig = plt.figure(figsize=(12, 10))
                mask = np.triu(np.ones_like(results['matrix'], dtype=bool))
                sns.heatmap(
                    results['matrix'],
                    mask=mask,
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1,
                    center=0,
                    square=True,
                    linewidths=.5,
                    annot=False,
                    cbar_kws={'shrink': .8}
                )
                plt.title(f'{corr_type.capitalize()} Correlation Heatmap')
                plt.tight_layout()
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files[f"{corr_type}_heatmap"] = heatmap_path
        
        # Save feature importance results
        if self.feature_importance:
            # Save for each model type
            for model_type, importance in self.feature_importance.items():
                importance_path = os.path.join(save_dir, f"{prefix}_{model_type}_importance.csv")
                importance.to_csv(importance_path, header=['importance'])
                saved_files[f"{model_type}_importance"] = importance_path
            
            # Save visualization
            viz_path = os.path.join(save_dir, f"{prefix}_key_factors.png")
            fig = self.visualize_key_factors(save_path=viz_path)
            plt.close(fig)
            saved_files['key_factors_viz'] = viz_path
        
        # Save model metrics
        if self.models:
            metrics = {}
            for model_type, results in self.models.items():
                metrics[model_type] = results['metrics']
            
            metrics_path = os.path.join(save_dir, f"{prefix}_model_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            saved_files['model_metrics'] = metrics_path
        
        logger.info(f"Saved correlation analysis results to {save_dir}")
        return saved_files
