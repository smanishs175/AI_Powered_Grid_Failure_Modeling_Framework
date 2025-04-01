# Neural Predictor

## Overview

The Neural Predictor is a core component of the Failure Prediction Module that uses deep learning to predict the probability of component failures in power grids. It combines multiple data sources including component properties, vulnerability assessments, environmental conditions, and historical failure data to make accurate predictions.

## Features

- **Multi-source data integration**: Combines data from various sources to create a comprehensive feature set
- **Flexible neural architecture**: Configurable network depth, width, and regularization
- **Preprocessing pipeline**: Handles missing values, scaling, and feature engineering
- **Model evaluation**: Computes and visualizes key performance metrics
- **Model persistence**: Saves and loads trained models with their preprocessing pipelines

## Methods

### Data Processing

- **`_process_and_merge_data`**: Merges component properties, vulnerability scores, environmental data, and failure history into a single dataset
- **`preprocess_data`**: Handles missing values, scales numerical features, and encodes categorical variables

### Model Building & Training

- **`build_model`**: Constructs a neural network architecture based on configuration parameters
- **`train`**: Trains the neural network with early stopping and validation
- **`predict`**: Generates failure probabilities for new data
- **`evaluate`**: Assesses model performance using metrics like accuracy, AUC, precision, and recall

### Model Management

- **`save`**: Persists the trained model, preprocessing pipeline, and metadata
- **`load`**: Retrieves a previously saved model and its associated files

## Usage Example

```python
from gfmf.failure_prediction.neural_predictor import NeuralPredictor

# Initialize the predictor
predictor = NeuralPredictor(config_path='config/custom_config.yaml')

# Load data
component_df = predictor.load_component_data('data/processed/components.csv')
vulnerability_df = predictor.load_vulnerability_data('data/vulnerability_analysis/scores.csv')
environmental_df = predictor.load_environmental_data('data/processed/weather.csv')
failure_df = predictor.load_failure_data('data/processed/failures.csv')

# Process and merge data
merged_data = predictor._process_and_merge_data(
    component_df=component_df,
    vulnerability_df=vulnerability_df,
    environmental_df=environmental_df,
    failure_df=failure_df
)

# Train-test split
X_train, X_test, y_train, y_test = predictor.train_test_split(
    merged_data, 
    target_column='failure',
    test_size=0.2
)

# Preprocess data
X_train_processed = predictor.preprocess_data(X_train, fit=True)
X_test_processed = predictor.preprocess_data(X_test, fit=False)

# Build and train model
predictor.build_model(input_dim=X_train_processed.shape[1])
predictor.train(X_train_processed, y_train, validation_split=0.2)

# Evaluate model
metrics = predictor.evaluate(X_test_processed, y_test, save_plots=True)
print("AUC:", metrics['auc'])

# Save model
predictor.save('models/failure_prediction/neural_predictor')

# Later, load the model
predictor.load('models/failure_prediction/neural_predictor')

# Make new predictions
new_data = predictor.preprocess_data(new_component_data)
failure_probs = predictor.predict(new_data)
```

## Configuration Options

The Neural Predictor can be configured through the `config/default_config.yaml` file:

```yaml
neural_predictor:
  model_type: "deep_neural_network"  
  hidden_layers: [128, 64, 32]
  learning_rate: 0.001
  epochs: 200
  batch_size: 32
  test_size: 0.2
  early_stopping_patience: 10
  dropout_rate: 0.2
```

## Output

The Neural Predictor produces:

1. **Failure probabilities**: Component-level probabilities of failure
2. **Performance metrics**: Accuracy, precision, recall, F1-score, and AUC
3. **Evaluation plots**: ROC curves, precision-recall curves, and confusion matrices
4. **Saved models**: Model files, preprocessing pipelines, and metadata

## Dependencies

- TensorFlow/Keras: Deep learning framework
- Scikit-learn: Data preprocessing and model evaluation
- Pandas: Data manipulation
- NumPy: Numerical operations
- Matplotlib: Visualization
