"""
Model utilities for the Failure Prediction Module.
This module provides utilities for loading, saving, and managing machine learning models.
"""

import os
import pickle
import json
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple, Optional
from tensorflow import keras

# Set up logger
logger = logging.getLogger(__name__)


def load_model(model_path: str, model_type: str = 'tensorflow') -> Any:
    """
    Load a machine learning model from disk.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model to load (tensorflow, sklearn, custom)
        
    Returns:
        Loaded model object
    """
    logger.info(f"Loading {model_type} model from {model_path}")
    
    try:
        if model_type == 'tensorflow':
            # Load TensorFlow/Keras model
            model = keras.models.load_model(model_path)
        elif model_type == 'sklearn' or model_type == 'custom':
            # Load scikit-learn or custom model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def save_model(model: Any, save_path: str, model_type: str = 'tensorflow', 
              metadata: Dict[str, Any] = None) -> str:
    """
    Save a machine learning model to disk.
    
    Args:
        model: Model object to save
        save_path: Path where the model will be saved
        model_type: Type of model to save (tensorflow, sklearn, custom)
        metadata: Optional metadata to save alongside the model
        
    Returns:
        Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    logger.info(f"Saving {model_type} model to {save_path}")
    
    try:
        if model_type == 'tensorflow':
            # Save TensorFlow/Keras model
            model.save(save_path)
        elif model_type == 'sklearn' or model_type == 'custom':
            # Save scikit-learn or custom model
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = f"{os.path.splitext(save_path)[0]}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully saved model to {save_path}")
        return save_path
    
    except Exception as e:
        logger.error(f"Failed to save model to {save_path}: {e}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_prediction_results(results: pd.DataFrame, save_path: str, 
                           format_type: str = 'pickle') -> str:
    """
    Save prediction results to disk.
    
    Args:
        results: DataFrame containing prediction results
        save_path: Path where results will be saved
        format_type: Format to save results in (pickle, csv, json)
        
    Returns:
        Path to the saved results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    logger.info(f"Saving prediction results to {save_path}")
    
    try:
        if format_type == 'pickle':
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
        elif format_type == 'csv':
            results.to_csv(save_path, index=False)
        elif format_type == 'json':
            results.to_json(save_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        logger.info(f"Successfully saved prediction results to {save_path}")
        return save_path
    
    except Exception as e:
        logger.error(f"Failed to save prediction results to {save_path}: {e}")
        raise


def create_model_registry_entry(model_path: str, model_type: str, 
                               performance_metrics: Dict[str, float],
                               features_used: List[str], 
                               training_params: Dict[str, Any],
                               registry_path: str) -> Dict[str, Any]:
    """
    Create an entry in the model registry for model versioning and tracking.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model
        performance_metrics: Dictionary of performance metrics
        features_used: List of features used in the model
        training_params: Dictionary of training parameters
        registry_path: Path to the model registry file
        
    Returns:
        The created registry entry
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    # Load existing registry or create new one
    try:
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"models": []}
    except Exception:
        registry = {"models": []}
    
    # Create new entry
    timestamp = pd.Timestamp.now().isoformat()
    entry = {
        "model_id": f"{model_type}_{timestamp}",
        "model_path": model_path,
        "model_type": model_type,
        "created_at": timestamp,
        "performance_metrics": performance_metrics,
        "features_used": features_used,
        "training_params": training_params
    }
    
    # Add to registry
    registry["models"].append(entry)
    
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Added model to registry: {model_path}")
    return entry
