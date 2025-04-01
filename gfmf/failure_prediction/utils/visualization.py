"""
Visualization utilities for the Failure Prediction Module.
This module provides utilities for visualizing prediction results and model outputs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Dict, Any, List, Tuple, Union, Optional
import logging

# Set up logger
logger = logging.getLogger(__name__)


def setup_plot_style(style: str = 'seaborn-v0_8-whitegrid'):
    """
    Set up matplotlib plot style.
    
    Args:
        style: Matplotlib style name
    """
    try:
        plt.style.use(style)
    except Exception as e:
        logger.warning(f"Failed to set plot style {style}: {e}. Using default style.")


def save_figure(
    fig: plt.Figure,
    save_path: str,
    dpi: int = 300,
    format_type: str = 'png'
) -> str:
    """
    Save a matplotlib figure to disk.
    
    Args:
        fig: Matplotlib figure object
        save_path: Path where the figure will be saved
        dpi: DPI resolution for the saved figure
        format_type: File format (png, pdf, svg, etc.)
        
    Returns:
        Path to the saved figure
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Add file extension if not present
    if not save_path.endswith(f'.{format_type}'):
        save_path = f"{save_path}.{format_type}"
    
    try:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format=format_type)
        logger.info(f"Saved figure to {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save figure to {save_path}: {e}")
        raise


def plot_failure_probability_heatmap(
    failure_probabilities: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'YlOrRd',
    title: str = 'Component Failure Probability Heatmap'
) -> plt.Figure:
    """
    Plot a heatmap of failure probabilities for grid components.
    
    Args:
        failure_probabilities: DataFrame with failure probabilities
        figsize: Figure size as (width, height)
        cmap: Colormap for the heatmap
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Pivot data for heatmap if needed
    if 'component_id' in failure_probabilities.columns and 'failure_probability' in failure_probabilities.columns:
        if 'component_type' in failure_probabilities.columns:
            # Pivot by component type
            pivot_data = failure_probabilities.pivot_table(
                index='component_type',
                values='failure_probability',
                aggfunc=['mean', 'max', 'count']
            )
            
            # Flatten column multi-index
            pivot_data.columns = [f"{col[0]}_{col[1]}" for col in pivot_data.columns]
            
        else:
            # Sort by failure probability
            pivot_data = failure_probabilities.sort_values(
                by='failure_probability', 
                ascending=False
            ).set_index('component_id')['failure_probability']
    else:
        pivot_data = failure_probabilities
    
    # Create figure
    plt.figure(figsize=figsize)
    
    if isinstance(pivot_data, pd.Series):
        # Plot top N components as a bar chart
        top_n = min(20, len(pivot_data))
        top_data = pivot_data.head(top_n)
        
        plt.barh(range(len(top_data)), top_data.values)
        plt.yticks(range(len(top_data)), top_data.index)
        plt.xlabel('Failure Probability')
        plt.title(f"Top {top_n} Components by Failure Probability")
        plt.colorbar = lambda: None  # Dummy colorbar function
        
    elif isinstance(pivot_data, pd.DataFrame):
        # Create heatmap
        sns.heatmap(
            pivot_data,
            cmap=cmap,
            annot=True,
            fmt='.2f',
            linewidths=0.5,
            cbar=True
        )
        plt.title(title)
        plt.tight_layout()
    
    return plt.gcf()


def plot_time_series_forecast_grid(
    forecasts: Dict[str, Dict[str, Union[List, np.ndarray]]],
    figsize: Tuple[int, int] = (15, 10),
    max_cols: int = 2,
    title: str = 'Time Series Forecasts by Component Type'
) -> plt.Figure:
    """
    Plot a grid of time series forecasts.
    
    Args:
        forecasts: Dictionary of forecast data by component type
        figsize: Figure size as (width, height)
        max_cols: Maximum number of columns in the grid
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Calculate grid dimensions
    n_plots = len(forecasts)
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=16)
    
    # Plot each component type
    for i, (comp_type, forecast_data) in enumerate(forecasts.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Extract forecast data
        dates = forecast_data.get('dates', np.arange(len(forecast_data['predictions'])))
        predictions = forecast_data['predictions']
        lower_bound = forecast_data.get('lower_bound', None)
        upper_bound = forecast_data.get('upper_bound', None)
        
        # Plot predictions
        ax.plot(dates, predictions, 'r-', label='Forecast')
        
        # Plot confidence intervals if available
        if lower_bound is not None and upper_bound is not None:
            ax.fill_between(dates, lower_bound, upper_bound, color='r', alpha=0.2, label='95% CI')
        
        # Customize plot
        ax.set_title(f"{comp_type}")
        ax.set_xlabel('Time (days ahead)')
        ax.set_ylabel('Predicted Failures')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    
    return fig


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu_r',
    annot: bool = True,
    title: str = 'Correlation Heatmap'
) -> plt.Figure:
    """
    Plot a correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix as DataFrame
        figsize: Figure size as (width, height)
        cmap: Colormap for the heatmap
        annot: Whether to annotate cells with correlation values
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create mask for upper triangle to show only lower triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot,
        fmt='.2f',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()


def plot_extreme_event_impact(
    event_impacts: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 6),
    title: str = 'Extreme Event Impact on Failure Rates'
) -> plt.Figure:
    """
    Plot the impact of extreme events on failure rates.
    
    Args:
        event_impacts: Dictionary of extreme event impact data
        figsize: Figure size as (width, height)
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Extract event types and rate ratios
    event_types = []
    rate_ratios = []
    
    for event_type, impact in event_impacts.items():
        if event_type != 'compound' and impact is not None:
            if 'statistics' in impact and 'rate_ratio' in impact['statistics']:
                event_types.append(event_type)
                rate_ratios.append(impact['statistics']['rate_ratio'])
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot horizontal bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(event_types)))
    bars = plt.barh(event_types, rate_ratios, color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.1,
            bar.get_y() + bar.get_height()/2,
            f'{width:.2f}x',
            va='center'
        )
    
    # Add baseline
    plt.axvline(x=1.0, color='red', linestyle='--', label='Baseline (Normal Rate)')
    
    # Customize plot
    plt.xlabel('Failure Rate Ratio (Event vs. Normal)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()


def plot_feature_importance_by_model(
    feature_importance_dict: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (12, 8),
    n_features: int = 10,
    title: str = 'Feature Importance by Model Type'
) -> plt.Figure:
    """
    Plot feature importance across different model types.
    
    Args:
        feature_importance_dict: Dictionary of feature importances by model type
        figsize: Figure size as (width, height)
        n_features: Number of top features to plot
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Get number of models and features
    n_models = len(feature_importance_dict)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_models, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # If only one model, convert axes to array for consistent indexing
    if n_models == 1:
        axes = np.array([axes])
    
    # Plot feature importance for each model
    for i, (model_name, importance) in enumerate(feature_importance_dict.items()):
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Take top n features
        top_features = sorted_features[:n_features]
        
        # Extract names and values
        names = [x[0] for x in top_features]
        values = [x[1] for x in top_features]
        
        # Plot
        axes[i].barh(np.arange(len(names)), values, color=plt.cm.tab10(i))
        axes[i].set_yticks(np.arange(len(names)))
        axes[i].set_yticklabels(names)
        axes[i].set_title(f"Model: {model_name}")
        axes[i].grid(True, alpha=0.3)
    
    # Set common labels
    fig.text(0.5, 0.01, 'Importance', ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    
    return fig


def create_failure_risk_map(
    failure_probabilities: pd.DataFrame,
    location_data: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Grid Component Failure Risk Map',
    background_map: bool = False
) -> plt.Figure:
    """
    Create a map visualization of component failure risks.
    
    Args:
        failure_probabilities: DataFrame with failure probabilities
        location_data: DataFrame with component location data
        figsize: Figure size as (width, height)
        title: Title for the plot
        background_map: Whether to include a background map
        
    Returns:
        Matplotlib figure object
    """
    # Merge failure probabilities with location data
    if 'component_id' in failure_probabilities.columns and 'component_id' in location_data.columns:
        data = pd.merge(
            failure_probabilities,
            location_data,
            on='component_id',
            how='inner'
        )
    else:
        data = failure_probabilities
    
    # Check if required columns exist
    required_cols = ['location_x', 'location_y', 'failure_probability']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    scatter = plt.scatter(
        data['location_x'],
        data['location_y'],
        c=data['failure_probability'],
        cmap='YlOrRd',
        alpha=0.7,
        s=50 + data['failure_probability'] * 100,  # Size based on failure probability
        edgecolors='k',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Failure Probability')
    
    # Customize plot
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_temporal_failure_patterns(
    temporal_data: pd.DataFrame,
    time_column: str,
    value_column: str,
    groupby_column: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = 'Temporal Failure Patterns'
) -> plt.Figure:
    """
    Plot temporal patterns in failure data.
    
    Args:
        temporal_data: DataFrame with temporal failure data
        time_column: Name of the column containing time information
        value_column: Name of the column containing failure or probability values
        groupby_column: Optional column to group data by (e.g., component_type)
        figsize: Figure size as (width, height)
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Check if required columns exist
    required_cols = [time_column, value_column]
    if groupby_column:
        required_cols.append(groupby_column)
    
    missing_cols = [col for col in required_cols if col not in temporal_data.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot data
    if groupby_column:
        # Group data and plot each group
        grouped_data = temporal_data.groupby(groupby_column)
        for name, group in grouped_data:
            plt.plot(
                group[time_column],
                group[value_column],
                marker='o',
                linestyle='-',
                label=name
            )
        plt.legend()
    else:
        # Plot single line
        plt.plot(
            temporal_data[time_column],
            temporal_data[value_column],
            marker='o',
            linestyle='-',
            color='blue'
        )
    
    # Customize plot
    plt.xlabel(time_column)
    plt.ylabel(value_column)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()
