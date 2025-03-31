"""
Extreme Event Modeler Module

This module provides functionality for modeling the impact of extreme environmental events
on component failures in power grids. It identifies extreme events, quantifies their effects,
and predicts failure probabilities during extreme conditions.
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

# Import utilities
from gfmf.failure_prediction.utils.model_utils import load_config
from gfmf.failure_prediction.utils.visualization import plot_extreme_event_impact

# Configure logger
logger = logging.getLogger(__name__)


class ExtremeEventModeler:
    """
    Extreme event modeler for analyzing the impact of extreme environmental conditions.
    
    This class provides functionality for identifying extreme environmental events,
    analyzing their impact on component failures, and predicting failure probabilities
    during extreme conditions.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the extreme event modeler.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.event_types = self.config['extreme_events'].get('event_types', 
                                                           ['high_temperature', 'low_temperature', 
                                                            'high_wind', 'precipitation'])
        self.threshold_percentiles = self.config['extreme_events'].get('threshold_percentiles', {
            'high_temperature': 95,
            'low_temperature': 5,
            'high_wind': 95,
            'precipitation': 95
        })
        
        self.extreme_events = {}  # Dictionary to store identified extreme events
        self.impact_statistics = {}  # Dictionary to store impact statistics
        self.event_thresholds = {}  # Dictionary to store thresholds for extreme events
        
        logger.info(f"Initialized {self.__class__.__name__} with events: {self.event_types}")
    
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
            'extreme_events': {
                'event_types': ['high_temperature', 'low_temperature', 'high_wind', 'precipitation'],
                'threshold_percentiles': {
                    'high_temperature': 95,
                    'low_temperature': 5,
                    'high_wind': 95,
                    'precipitation': 95
                }
            }
        }
    
    def _map_var_to_event_type(self, variable_name: str) -> str:
        """
        Map variable name to event type.
        
        Args:
            variable_name: Name of the environmental variable
            
        Returns:
            Corresponding event type
        """
        # Mapping for common variable names
        mapping = {
            'temperature': ['temperature', 'temp', 'air_temperature'],
            'wind_speed': ['wind_speed', 'wind', 'windspeed'],
            'precipitation': ['precipitation', 'precip', 'rainfall'],
            'humidity': ['humidity', 'relative_humidity'],
            'pressure': ['pressure', 'air_pressure', 'barometric_pressure']
        }
        
        # Find matching event type
        for event_type, keywords in mapping.items():
            if any(keyword.lower() in variable_name.lower() for keyword in keywords):
                return event_type
        
        # If no match found, return the original name
        return variable_name
    
    def load_data(
        self,
        environmental_data_path: str,
        failure_data_path: str,
        date_column: str = 'date',
        location_column: Optional[str] = 'location_id',
        component_column: str = 'component_id'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load environmental and failure data.
        
        Args:
            environmental_data_path: Path to environmental data file
            failure_data_path: Path to failure data file
            date_column: Name of the date column
            location_column: Name of the location column (optional)
            component_column: Name of the component column in failure data
            
        Returns:
            Tuple of (environmental_df, failure_df)
        """
        # Load environmental data
        if environmental_data_path.endswith('.csv'):
            env_df = pd.read_csv(environmental_data_path)
        elif environmental_data_path.endswith('.parquet'):
            env_df = pd.read_parquet(environmental_data_path)
        else:
            logger.error(f"Unsupported file format: {environmental_data_path}")
            raise ValueError(f"Unsupported file format: {environmental_data_path}")
        
        # Load failure data
        if failure_data_path.endswith('.csv'):
            fail_df = pd.read_csv(failure_data_path)
        elif failure_data_path.endswith('.parquet'):
            fail_df = pd.read_parquet(failure_data_path)
        else:
            logger.error(f"Unsupported file format: {failure_data_path}")
            raise ValueError(f"Unsupported file format: {failure_data_path}")
        
        # Ensure date columns are in datetime format
        if date_column in env_df.columns:
            env_df[date_column] = pd.to_datetime(env_df[date_column])
        else:
            logger.error(f"Date column '{date_column}' not found in environmental data")
            raise ValueError(f"Date column '{date_column}' not found in environmental data")
        
        if date_column in fail_df.columns:
            fail_df[date_column] = pd.to_datetime(fail_df[date_column])
        else:
            logger.error(f"Date column '{date_column}' not found in failure data")
            raise ValueError(f"Date column '{date_column}' not found in failure data")
        
        # Check component column
        if component_column not in fail_df.columns:
            logger.error(f"Component column '{component_column}' not found in failure data")
            raise ValueError(f"Component column '{component_column}' not found in failure data")
        
        # Store column names
        self.date_column = date_column
        self.location_column = location_column
        self.component_column = component_column
        
        logger.info(f"Loaded environmental data: {env_df.shape}")
        logger.info(f"Loaded failure data: {fail_df.shape}")
        
        return env_df, fail_df
    
    def identify_extreme_events(
        self,
        env_df: pd.DataFrame,
        custom_thresholds: Optional[Dict[str, float]] = None,
        window_size: int = 1,
        min_duration: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        Identify extreme environmental events.
        
        Args:
            env_df: Environmental data DataFrame
            custom_thresholds: Custom thresholds for extreme events
            window_size: Window size for filtering events (days)
            min_duration: Minimum duration for an event to be considered extreme (days)
            
        Returns:
            Dictionary of DataFrames with extreme events by type
        """
        logger.info("Identifying extreme environmental events")
        
        # Get environmental variables
        env_vars = env_df.select_dtypes(include=[np.number]).columns.tolist()
        env_vars = [var for var in env_vars if var not in [self.date_column, self.location_column]]
        
        # Map variables to event types
        var_to_event = {var: self._map_var_to_event_type(var) for var in env_vars}
        
        # Calculate thresholds for each variable
        self.event_thresholds = {}
        for var in env_vars:
            event_type = var_to_event[var]
            
            # Use custom threshold if provided
            if custom_thresholds and event_type in custom_thresholds:
                threshold = custom_thresholds[event_type]
                logger.info(f"Using custom threshold for {event_type}: {threshold}")
                
                # Store threshold directly
                self.event_thresholds[var] = threshold
            else:
                # Use percentile threshold based on event type
                percentile = None
                
                if 'temperature' in event_type:
                    if 'high' in event_type:
                        percentile = self.threshold_percentiles.get('high_temperature', 95)
                    elif 'low' in event_type:
                        percentile = self.threshold_percentiles.get('low_temperature', 5)
                elif 'wind' in event_type:
                    percentile = self.threshold_percentiles.get('high_wind', 95)
                elif 'precipitation' in event_type or 'rain' in event_type:
                    percentile = self.threshold_percentiles.get('precipitation', 95)
                else:
                    # Default to high percentile for other variables
                    percentile = 95
                
                # Calculate threshold
                if percentile <= 50:  # Low percentile (lower tail)
                    threshold = np.percentile(env_df[var].dropna(), percentile)
                    self.event_thresholds[var] = {'threshold': threshold, 'direction': 'below', 'percentile': percentile}
                else:  # High percentile (upper tail)
                    threshold = np.percentile(env_df[var].dropna(), percentile)
                    self.event_thresholds[var] = {'threshold': threshold, 'direction': 'above', 'percentile': percentile}
                
                logger.info(f"Calculated threshold for {var} ({event_type}): {threshold}")
        
        # Identify extreme events
        extreme_events = {}
        for var in env_vars:
            # Skip non-numeric columns
            if not np.issubdtype(env_df[var].dtype, np.number):
                continue
                
            event_type = var_to_event[var]
            threshold_info = self.event_thresholds[var]
            
            # Handle different threshold types
            if isinstance(threshold_info, dict):
                threshold = threshold_info['threshold']
                direction = threshold_info['direction']
            else:
                threshold = threshold_info
                direction = 'above'  # Default
            
            # Create mask for extreme values
            if direction == 'above':
                mask = env_df[var] >= threshold
            else:
                mask = env_df[var] <= threshold
            
            # Apply mask to identify extreme events
            extreme_df = env_df[mask].copy()
            
            # Skip if no extreme events found
            if len(extreme_df) == 0:
                logger.info(f"No extreme events found for {var}")
                continue
            
            # Add event type information
            extreme_df['event_type'] = event_type
            extreme_df['event_var'] = var
            extreme_df['threshold'] = threshold
            extreme_df['direction'] = direction
            
            # Calculate deviation from threshold
            if direction == 'above':
                extreme_df['deviation'] = extreme_df[var] - threshold
            else:
                extreme_df['deviation'] = threshold - extreme_df[var]
            
            # Normalize deviation
            std_dev = env_df[var].std()
            if std_dev > 0:
                extreme_df['normalized_deviation'] = extreme_df['deviation'] / std_dev
            else:
                extreme_df['normalized_deviation'] = extreme_df['deviation']
            
            # Store extreme events
            extreme_events[event_type] = extreme_df
            
            logger.info(f"Identified {len(extreme_df)} extreme events for {event_type}")
        
        # Store extreme events
        self.extreme_events = extreme_events
        
        return extreme_events
    
    def analyze_event_impact(
        self,
        extreme_events: Dict[str, pd.DataFrame],
        failure_df: pd.DataFrame,
        event_window: int = 3,  # days
        baseline_window: int = 30,  # days
        min_events: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the impact of extreme events on component failures.
        
        Args:
            extreme_events: Dictionary of DataFrames with extreme events by type
            failure_df: DataFrame with component failure data
            event_window: Window size after extreme event to look for failures (days)
            baseline_window: Window size for baseline failure rate calculation (days)
            min_events: Minimum number of events required for reliable statistics
            
        Returns:
            Dictionary with impact statistics by event type
        """
        logger.info("Analyzing impact of extreme events on failures")
        
        impact_statistics = {}
        
        # Ensure failure data has datetime format
        failure_df[self.date_column] = pd.to_datetime(failure_df[self.date_column])
        
        # Calculate baseline failure rate (failures per day)
        total_days = (failure_df[self.date_column].max() - failure_df[self.date_column].min()).days + 1
        total_failures = len(failure_df)
        baseline_rate = total_failures / max(1, total_days)
        
        logger.info(f"Baseline failure rate: {baseline_rate:.4f} failures per day")
        
        # Analyze impact by event type
        for event_type, event_df in extreme_events.items():
            # Skip if too few events
            if len(event_df) < min_events:
                logger.warning(f"Too few events for {event_type} ({len(event_df)} < {min_events})")
                continue
            
            # Extract event dates
            event_dates = event_df[self.date_column].unique()
            
            # Count failures during and after extreme events
            event_failures = []
            
            for event_date in event_dates:
                # Define window after event
                window_end = event_date + pd.Timedelta(days=event_window)
                
                # Count failures in window
                window_failures = failure_df[
                    (failure_df[self.date_column] >= event_date) & 
                    (failure_df[self.date_column] <= window_end)
                ]
                
                # Store failures for this event
                if len(window_failures) > 0:
                    event_failures.append({
                        'event_date': event_date,
                        'failures': len(window_failures),
                        'failure_rate': len(window_failures) / event_window,
                        'component_ids': window_failures[self.component_column].tolist()
                    })
            
            # Skip if no failures during events
            if not event_failures:
                logger.warning(f"No failures during {event_type} events")
                continue
            
            # Calculate statistics
            failures_df = pd.DataFrame(event_failures)
            mean_rate = failures_df['failure_rate'].mean()
            rate_ratio = mean_rate / baseline_rate
            
            # Perform statistical test if enough data
            p_value = None
            if len(failures_df) >= 10:
                # Get baseline daily failure counts for comparison
                # Group failures by date and count
                baseline_daily = failure_df.groupby(failure_df[self.date_column].dt.date).size()
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(
                    failures_df['failures'].values, 
                    baseline_daily.values,
                    equal_var=False  # Welch's t-test for unequal variances
                )
            
            # Store statistics
            impact_statistics[event_type] = {
                'event_count': len(event_dates),
                'affected_failure_count': failures_df['failures'].sum(),
                'mean_failure_rate': mean_rate,
                'baseline_failure_rate': baseline_rate,
                'rate_ratio': rate_ratio,
                'significant': p_value is not None and p_value < 0.05,
                'p_value': p_value,
                'failure_events': event_failures
            }
            
            logger.info(f"Impact of {event_type}: rate ratio = {rate_ratio:.2f}, significant = {impact_statistics[event_type]['significant']}")
        
        # Store impact statistics
        self.impact_statistics = impact_statistics
        
        return impact_statistics
    
    def calculate_compound_event_impact(
        self,
        extreme_events: Dict[str, pd.DataFrame],
        failure_df: pd.DataFrame,
        event_window: int = 3  # days
    ) -> Dict[str, Any]:
        """
        Calculate the impact of compound extreme events (multiple event types occurring together).
        
        Args:
            extreme_events: Dictionary of DataFrames with extreme events by type
            failure_df: DataFrame with component failure data
            event_window: Window size after extreme event to look for failures (days)
            
        Returns:
            Dictionary with compound event impact statistics
        """
        logger.info("Analyzing impact of compound extreme events")
        
        # Extract all event dates by type
        event_dates_by_type = {}
        for event_type, event_df in extreme_events.items():
            event_dates_by_type[event_type] = set(event_df[self.date_column].dt.date)
        
        # Find dates with multiple event types (compound events)
        all_dates = set()
        for dates in event_dates_by_type.values():
            all_dates.update(dates)
        
        compound_dates = {}
        for date in all_dates:
            event_types = [et for et, dates in event_dates_by_type.items() if date in dates]
            if len(event_types) > 1:
                compound_dates[date] = event_types
        
        # Skip if no compound events
        if not compound_dates:
            logger.warning("No compound events found")
            return None
        
        logger.info(f"Found {len(compound_dates)} compound events")
        
        # Analyze impact of compound events
        compound_failures = []
        
        for event_date, event_types in compound_dates.items():
            # Convert to datetime
            dt_date = pd.Timestamp(event_date)
            
            # Define window after event
            window_end = dt_date + pd.Timedelta(days=event_window)
            
            # Count failures in window
            window_failures = failure_df[
                (failure_df[self.date_column] >= dt_date) & 
                (failure_df[self.date_column] <= window_end)
            ]
            
            # Store failures for this compound event
            if len(window_failures) > 0:
                compound_failures.append({
                    'event_date': dt_date,
                    'event_types': event_types,
                    'failures': len(window_failures),
                    'failure_rate': len(window_failures) / event_window,
                    'component_ids': window_failures[self.component_column].tolist()
                })
        
        # Skip if no failures during compound events
        if not compound_failures:
            logger.warning("No failures during compound events")
            return None
        
        # Calculate statistics
        failures_df = pd.DataFrame(compound_failures)
        
        # Calculate baseline failure rate (failures per day)
        total_days = (failure_df[self.date_column].max() - failure_df[self.date_column].min()).days + 1
        total_failures = len(failure_df)
        baseline_rate = total_failures / max(1, total_days)
        
        mean_rate = failures_df['failure_rate'].mean()
        rate_ratio = mean_rate / baseline_rate
        
        # Perform statistical test if enough data
        p_value = None
        if len(failures_df) >= 10:
            # Get baseline daily failure counts for comparison
            # Group failures by date and count
            baseline_daily = failure_df.groupby(failure_df[self.date_column].dt.date).size()
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                failures_df['failures'].values, 
                baseline_daily.values,
                equal_var=False  # Welch's t-test for unequal variances
            )
        
        # Create compound event statistics
        compound_stats = {
            'event_count': len(compound_dates),
            'affected_failure_count': failures_df['failures'].sum(),
            'mean_failure_rate': mean_rate,
            'baseline_failure_rate': baseline_rate,
            'rate_ratio': rate_ratio,
            'significant': p_value is not None and p_value < 0.05,
            'p_value': p_value,
            'failure_events': compound_failures
        }
        
        logger.info(f"Impact of compound events: rate ratio = {rate_ratio:.2f}, significant = {compound_stats['significant']}")
        
        # Add to impact statistics
        self.impact_statistics['compound'] = compound_stats
        
        return compound_stats
    
    def predict_event_failure_probability(
        self,
        component_df: pd.DataFrame,
        event_type: str,
        severity: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Predict component failure probabilities during extreme events.
        
        Args:
            component_df: DataFrame with component information
            event_type: Type of extreme event
            severity: Severity of the event (normalized deviation from threshold)
            
        Returns:
            DataFrame with failure probabilities by component
        """
        logger.info(f"Predicting failure probabilities during {event_type} event")
        
        # Check if impact statistics are available
        if not self.impact_statistics or event_type not in self.impact_statistics:
            logger.error(f"No impact statistics available for {event_type}")
            raise ValueError(f"No impact statistics available for {event_type}")
        
        # Get impact statistics for this event type
        stats = self.impact_statistics[event_type]
        
        # Base rate ratio from analysis
        base_ratio = stats['rate_ratio']
        
        # Adjust for severity if provided
        if severity is not None:
            # Simple linear scaling of impact with severity
            # Severity 1.0 = base ratio, higher severity = higher ratio
            adjusted_ratio = base_ratio * severity
        else:
            adjusted_ratio = base_ratio
        
        # Calculate component-specific failure probabilities
        # Start with baseline probability (can be enhanced with component-specific factors)
        baseline_probability = 0.01  # Default baseline
        
        # Create output DataFrame
        result_df = component_df.copy()
        result_df['baseline_failure_prob'] = baseline_probability
        result_df['event_failure_prob'] = baseline_probability * adjusted_ratio
        result_df['rate_ratio'] = adjusted_ratio
        result_df['event_type'] = event_type
        
        # Ensure probabilities are within [0, 1]
        result_df['event_failure_prob'] = np.clip(result_df['event_failure_prob'], 0, 1)
        
        return result_df
    
    def visualize_event_impact(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Visualize the impact of extreme events on failure rates.
        
        Args:
            save_path: Path to save the visualization
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        logger.info("Visualizing extreme event impact")
        
        # Check if impact statistics are available
        if not self.impact_statistics:
            logger.error("No impact statistics available for visualization")
            raise ValueError("No impact statistics available for visualization")
        
        # Create impact data for visualization
        event_impacts = {}
        for event_type, stats in self.impact_statistics.items():
            event_impacts[event_type] = {
                'statistics': stats
            }
        
        # Create visualization
        fig = plot_extreme_event_impact(event_impacts, figsize=figsize)
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved impact visualization to {save_path}")
        
        return fig
    
    def save_results(
        self,
        save_dir: Optional[str] = None,
        prefix: str = 'extreme_events'
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
            save_dir = os.path.join(self.config['paths']['output_data'], 'extreme_events')
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save thresholds
        thresholds_path = os.path.join(save_dir, f"{prefix}_thresholds.json")
        with open(thresholds_path, 'w') as f:
            json.dump(self.event_thresholds, f, indent=2, default=str)
        saved_files['thresholds'] = thresholds_path
        
        # Save impact statistics
        impact_path = os.path.join(save_dir, f"{prefix}_impact_statistics.json")
        
        # Convert complex objects to serializable format
        serializable_impact = {}
        for event_type, stats in self.impact_statistics.items():
            serializable_stats = {}
            for k, v in stats.items():
                if k == 'failure_events':
                    # Convert event date to string
                    serializable_events = []
                    for event in v:
                        serializable_event = event.copy()
                        serializable_event['event_date'] = str(event['event_date'])
                        serializable_events.append(serializable_event)
                    serializable_stats[k] = serializable_events
                else:
                    serializable_stats[k] = v
            serializable_impact[event_type] = serializable_stats
        
        with open(impact_path, 'w') as f:
            json.dump(serializable_impact, f, indent=2)
        saved_files['impact_statistics'] = impact_path
        
        # Save extreme events (if available)
        if self.extreme_events:
            for event_type, event_df in self.extreme_events.items():
                events_path = os.path.join(save_dir, f"{prefix}_{event_type}_events.csv")
                event_df.to_csv(events_path, index=False)
                saved_files[f"{event_type}_events"] = events_path
        
        # Create and save visualization
        viz_path = os.path.join(save_dir, f"{prefix}_impact_visualization.png")
        try:
            fig = self.visualize_event_impact()
            fig.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files['visualization'] = viz_path
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
        
        logger.info(f"Saved extreme event analysis results to {save_dir}")
        return saved_files
