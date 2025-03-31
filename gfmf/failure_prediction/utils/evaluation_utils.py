"""
Evaluation utilities for the Failure Prediction Module.
This module provides utilities for evaluating machine learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score, precision_recall_curve,
    roc_curve, auc
)

# Set up logger
logger = logging.getLogger(__name__)


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a binary classification model.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities (optional)
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Ensure predictions are binary
    if y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Calculate ROC AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Failed to calculate ROC AUC: {e}")
            metrics['roc_auc'] = np.nan
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    return metrics


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a regression model.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Additional metrics
    metrics['explained_variance'] = np.var(y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0
    metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return metrics


def evaluate_time_series_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_index: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate a time series forecasting model.
    
    Args:
        y_true: Ground truth time series values
        y_pred: Predicted time series values
        time_index: Time indices for the time series (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Basic regression metrics
    metrics = evaluate_regression_model(y_true, y_pred)
    
    # Time series specific metrics
    # Mean Absolute Scaled Error (MASE)
    if len(y_true) > 1:
        # Seasonal differencing (default to 1 period if no seasonality)
        d = np.abs(np.diff(y_true, n=1))
        scale = np.mean(d) if np.mean(d) > 0 else 1
        metrics['mase'] = mean_absolute_error(y_true, y_pred) / scale
    else:
        metrics['mase'] = np.nan
    
    # Directional accuracy (correct prediction of up/down movement)
    if len(y_true) > 1 and len(y_pred) > 1:
        y_true_dir = np.sign(np.diff(y_true))
        y_pred_dir = np.sign(np.diff(y_pred))
        dir_accuracy = np.mean((y_true_dir == y_pred_dir).astype(int))
        metrics['directional_accuracy'] = dir_accuracy
    else:
        metrics['directional_accuracy'] = np.nan
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    normalize: bool = False,
    title: str = 'Confusion Matrix'
) -> plt.Figure:
    """
    Plot a confusion matrix for classification results.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class labels (optional)
        figsize: Figure size as (width, height)
        cmap: Colormap for the plot
        normalize: Whether to normalize the confusion matrix
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        cbar=True,
        square=True,
        xticklabels=labels if labels else ['Negative', 'Positive'],
        yticklabels=labels if labels else ['Negative', 'Positive']
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'Receiver Operating Characteristic (ROC) Curve'
) -> plt.Figure:
    """
    Plot a ROC curve for binary classification.
    
    Args:
        y_true: Ground truth binary labels
        y_score: Predicted probabilities or scores
        figsize: Figure size as (width, height)
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'Precision-Recall Curve'
) -> plt.Figure:
    """
    Plot a precision-recall curve for binary classification.
    
    Args:
        y_true: Ground truth binary labels
        y_score: Predicted probabilities or scores
        figsize: Figure size as (width, height)
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    # Create figure
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.axhline(y=np.mean(y_true), color='navy', lw=2, linestyle='--', label=f'Baseline (y_mean = {np.mean(y_true):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_time_series_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[np.ndarray] = None,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = 'Time Series Forecast',
    train_size: Optional[int] = None
) -> plt.Figure:
    """
    Plot time series forecasting results.
    
    Args:
        y_true: Ground truth time series values
        y_pred: Predicted time series values
        dates: Dates for the time series (optional)
        lower_bound: Lower confidence bound (optional)
        upper_bound: Upper confidence bound (optional)
        figsize: Figure size as (width, height)
        title: Title for the plot
        train_size: Size of training set to mark train/test split (optional)
        
    Returns:
        Matplotlib figure object
    """
    # Create x-axis values
    if dates is None:
        x = np.arange(len(y_true))
    else:
        x = dates
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot training and testing data if train_size is provided
    if train_size is not None and train_size < len(y_true):
        plt.plot(x[:train_size], y_true[:train_size], 'b-', label='Training data')
        plt.plot(x[train_size:], y_true[train_size:], 'g-', label='Testing data')
    else:
        plt.plot(x, y_true, 'b-', label='Actual')
    
    # Plot predictions
    plt.plot(x, y_pred, 'r--', label='Predicted')
    
    # Plot confidence intervals if provided
    if lower_bound is not None and upper_bound is not None:
        plt.fill_between(x, lower_bound, upper_bound, color='r', alpha=0.2, label='95% Confidence Interval')
    
    # Customize plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def calculate_feature_importance(
    model: Any,
    feature_names: List[str],
    model_type: str = 'tree'
) -> Dict[str, float]:
    """
    Calculate feature importance from a model.
    
    Args:
        model: Trained model object
        feature_names: List of feature names
        model_type: Type of model ('tree', 'linear', 'permutation')
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    importance_dict = {}
    
    try:
        # For tree-based models (Random Forest, XGBoost, etc.)
        if model_type == 'tree' and hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            
        # For linear models (Linear/Logistic Regression, etc.)
        elif model_type == 'linear' and hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
            
        # For neural networks or other models without direct feature importance
        else:
            logger.warning(f"Cannot extract feature importance for model type: {model_type}")
            return {name: 0.0 for name in feature_names}
        
        # Normalize importance scores
        total = np.sum(importance_scores)
        if total > 0:
            importance_scores = importance_scores / total
        
        # Create dictionary mapping feature names to importance scores
        for name, score in zip(feature_names, importance_scores):
            importance_dict[name] = float(score)
            
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return {name: 0.0 for name in feature_names}
    
    return importance_dict


def plot_feature_importance(
    importance_dict: Dict[str, float],
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Feature Importance',
    top_n: Optional[int] = None,
    color: str = 'skyblue'
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        figsize: Figure size as (width, height)
        title: Title for the plot
        top_n: Number of top features to plot (optional)
        color: Bar color
        
    Returns:
        Matplotlib figure object
    """
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Take top N features if specified
    if top_n is not None:
        sorted_features = sorted_features[:top_n]
    
    # Extract names and values
    names = [x[0] for x in sorted_features]
    values = [x[1] for x in sorted_features]
    
    # Create figure
    plt.figure(figsize=figsize)
    plt.barh(range(len(names)), values, color=color)
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()
