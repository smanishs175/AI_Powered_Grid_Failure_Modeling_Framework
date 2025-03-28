# Grid Failure Modeling Framework (GFMF)
## Software Requirements Specification

### 1. Introduction

#### 1.1 Purpose
This document specifies the requirements for an AI-powered Grid Failure Modeling Framework (GFMF) that predicts, analyzes, and helps mitigate power grid failures under various environmental conditions and threats. This specification is designed to provide precise implementation guidance for AI coding assistants.

#### 1.2 Scope
The GFMF will integrate public datasets (or generate synthetic data when necessary) to create a comprehensive framework for grid vulnerability analysis, failure prediction, and policy optimization using AI techniques, particularly reinforcement learning.

#### 1.3 System Context
This framework addresses the critical need to model and predict grid failures resulting from extreme weather events, component vulnerabilities, and cascading failures. It will help grid operators optimize response strategies through intelligent policy learning.

### 2. System Architecture

#### 2.1 High-Level Architecture
The system follows a modular architecture with the following layers:
- **Data Layer**: Data acquisition, preprocessing, and synthetic generation
- **Analysis Layer**: Component vulnerability profiling and environmental threat modeling
- **Prediction Layer**: Failure prediction and scenario generation
- **Learning Layer**: Reinforcement learning for policy optimization
- **Application Layer**: Visualization and decision support

#### 2.2 Module Overview

1. **Data Management Module**
   - Data loading and integration
   - Data preprocessing and feature engineering
   - Synthetic data generation

2. **Vulnerability Analysis Module**
   - Component vulnerability profiling
   - Environmental threat modeling
   - Correlation analysis

3. **Failure Prediction Module**
   - Neural network-based failure probability prediction
   - Extreme event modeling
   - Time-series forecasting

4. **Scenario Generation Module**
   - Conditional GAN for realistic scenario generation
   - Cascading failure modeling
   - Extreme event simulation

5. **Reinforcement Learning Module**
   - Custom OpenAI Gym environment
   - Multiple RL agent implementations
   - Policy optimization

6. **Visualization and Reporting Module**
   - Grid topology visualization
   - Performance metrics visualization
   - Decision support dashboards

### 3. Detailed Requirements

#### 3.1 Data Management

##### 3.1.1 Data Loading
- **Requirement**: The system shall load grid topology, weather, and outage data from standardized file formats.
- **Implementation Details**:
  - File Formats:
    - Grid Topology: JSON
    - Weather Data: CSV with columns [date, temperature, wind_speed, precipitation, ...]
    - Outage Data: CSV with columns [date, component_id, outage, duration, ...]

##### 3.1.2 Data Preprocessing
- **Requirement**: The system shall preprocess raw data into formats suitable for ML models.
- **Implementation Details**:
  - Datetime Parsing: Convert all timestamps to consistent datetime format
  - Normalization: Apply StandardScaler to numerical features
  - Feature Engineering: 
    - Daily/monthly aggregations for weather data
    - Component-level failure statistics
    - Time-based features (month, season, etc.)

##### 3.1.3 Synthetic Data Generation
- **Requirement**: The system shall generate realistic synthetic data for testing when real data is unavailable.
- **Implementation Details**:
  - Grid Topology Generation: 
    - Configurable number of nodes and lines
    - Realistic node properties (type, capacity, location)
    - Line properties with physical characteristics
  - Weather Data Generation:
    - Seasonal temperature patterns
    - Wind speed patterns
    - Extreme event insertions
  - Outage Generation:
    - Correlated with weather conditions
    - Component-specific vulnerability factors
    - Realistic duration distributions

#### 3.2 Vulnerability Analysis

##### 3.2.1 Component Vulnerability Profiling
- **Requirement**: The system shall model vulnerability profiles for grid components.
- **Implementation Details**:
  - Features to Consider:
    - Component type (line, transformer, generator, etc.)
    - Material properties
    - Age and maintenance history
    - Environmental exposure
  - Model Type: Neural network classifier
  - Output: Vulnerability score (0-1) for each component

##### 3.2.2 Environmental Threat Modeling
- **Requirement**: The system shall model the impact of environmental conditions on grid components.
- **Implementation Details**:
  - Weather Factors:
    - Temperature extremes (high/low)
    - Wind events (speed thresholds)
    - Precipitation (flood risk, ice loading)
    - Lightning
  - Correlation Analysis:
    - Weather condition × component type × failure rate
    - Extreme event statistics

#### 3.3 Failure Prediction

##### 3.3.1 Neural Network Predictor
- **Requirement**: The system shall implement a neural network to predict component failure probabilities.
- **Implementation Details**:
  - Architecture:
    - Input: Component features + weather conditions
    - Hidden layers: 2 dense layers (64, 32 units)
    - Output: Failure probability
  - Training:
    - Loss: Binary cross-entropy
    - Optimizer: Adam with learning rate 0.001
    - Validation: 20% holdout, AUC metric

##### 3.3.2 Correlation Modeling
- **Requirement**: The system shall model correlations between environmental conditions and outages.
- **Implementation Details**:
  - Features: 
    - Temperature (daily min/max/avg)
    - Wind speed (daily max/avg)
    - Precipitation (daily sum)
  - Target: Outage events (binary)
  - Model: Feedforward neural network with 2 hidden layers

#### 3.4 Scenario Generation

##### 3.4.1 Conditional GAN Implementation
- **Requirement**: The system shall implement a Conditional GAN for scenario generation.
- **Implementation Details**:
  - Architecture:
    - Generator: 3 dense layers (128, 256, 128 units)
    - Discriminator: 2 dense layers (128, 64 units)
    - Condition: Weather forecast vector
    - Output: Binary line status vector
  - Training:
    - Adversarial loss with gradient penalty
    - 1000 epochs with batch size 64
    - Adam optimizer with learning rate 0.0002

##### 3.4.2 Extreme Event Generation
- **Requirement**: The system shall generate realistic extreme event scenarios.
- **Implementation Details**:
  - Event Types:
    - Heat waves (3-5 days, temperature > 35°C)
    - Cold snaps (2-4 days, temperature < -10°C)
    - High wind events (wind speed > 30 mph)
    - Flooding (precipitation > 50mm in 24h)
    - Compound events (combinations of above)

#### 3.5 Reinforcement Learning

##### 3.5.1 OpenAI Gym Environment
- **Requirement**: The system shall implement a custom OpenAI Gym environment for grid management.
- **Implementation Details**:
  - State Space:
    - Line status vector (binary)
    - Weather conditions vector
    - Load demand vector
  - Action Space:
    - Discrete actions: [0: do nothing, 1: adjust generation, 2: reroute power]
  - Reward Function:
    - +1 for stable operation
    - -1 for outages or load shedding
    - -0.1 for unnecessary actions

##### 3.5.2 RL Algorithms
- **Requirement**: The system shall implement multiple RL algorithms and compare their performance.
- **Implementation Details**:
  - Deep Q-Network (DQN):
    - Double DQN with prioritized experience replay
    - Network: 3 fully connected layers (128, 64, 32)
    - Learning rate: 0.001, discount factor: 0.99
  
  - Proximal Policy Optimization (PPO):
    - Clip parameter: 0.2
    - Value function coefficient: 0.5
    - Entropy coefficient: 0.01
  
  - Soft Actor-Critic (SAC):
    - Temperature parameter: auto-adjusted
    - Twin Q-networks
    - Target update rate: 0.005
  
  - Twin Delayed DDPG (TD3):
    - Policy noise: 0.2
    - Noise clip: 0.5
    - Policy update frequency: 2
  
  - Generative Adversarial Imitation Learning (GAIL):
    - Discriminator network: 2 layers (128, 64)
    - Expert demonstrations: 100 episodes

##### 3.5.3 Policy Optimization
- **Requirement**: The system shall optimize policies for grid management under various conditions.
- **Implementation Details**:
  - Training Process:
    - Train each agent for 100,000 steps
    - Test on 50 generated scenarios
    - Metric: Average episodic reward
  
  - Model Saving:
    - Save best policy after every 10,000 steps
    - Format: PyTorch/TensorFlow model files

#### 3.6 Visualization and Reporting

##### 3.6.1 Grid Visualization
- **Requirement**: The system shall visualize grid topology and component status.
- **Implementation Details**:
  - Library: NetworkX with Matplotlib
  - Node Color Coding:
    - Green: Operational
    - Yellow: At-risk (vulnerability > 0.7)
    - Red: Failed
  - Line Width: Proportional to capacity

##### 3.6.2 Performance Visualization
- **Requirement**: The system shall visualize model performance and RL agent behavior.
- **Implementation Details**:
  - Learning Curves:
    - X-axis: Episodes/Steps
    - Y-axis: Episodic reward
    - One line per algorithm
  
  - Prediction Accuracy:
    - ROC curves for failure prediction
    - Confusion matrices
  
  - Component Heatmaps:
    - Color based on failure probability

### 4. Data Requirements

#### 4.1 Required Data Types

##### 4.1.1 Grid Infrastructure Data
- **Format**: JSON or CSV
- **Content**:
  - Nodes:
    - ID, type, location
    - Capacity, voltage
    - Age, material
  
  - Lines:
    - ID, from_node, to_node
    - Capacity, length
    - Age, material

##### 4.1.2 Weather Data
- **Format**: CSV
- **Content**:
  - Timestamp/date (hourly or daily)
  - Temperature (°C)
  - Wind speed (mph)
  - Precipitation (mm)
  - Optional: humidity, pressure, etc.

##### 4.1.3 Outage Data
- **Format**: CSV
- **Content**:
  - Date and time
  - Component ID
  - Outage flag (0/1)
  - Duration (hours)
  - Optional: cause code

#### 4.2 Public Datasets
- **Weather Data**: NOAA climate data (https://www.ncdc.noaa.gov/cdo-web/)
- **Grid Data**: EIA data (https://www.eia.gov/electricity/data.php)
- **Outage Data**: DOE Form OE-417 reports

### 5. Technical Implementation

#### 5.1 Development Environment

##### 5.1.1 Programming Language and Libraries
- **Language**: Python 3.8+
- **Core Libraries**:
  - Data Processing: pandas, numpy
  - Machine Learning: scikit-learn, tensorflow, pytorch
  - Reinforcement Learning: gym, stable-baselines3
  - Visualization: matplotlib, seaborn, networkx
  - Utilities: tqdm, pyyaml, joblib

##### 5.1.2 Configuration Management
- **Configuration Format**: YAML
- **Default Config Structure**:

```yaml
data:
  paths:
    grid_topology: "./data_collection_by_hollis/grid_topology.json"
    weather_data: "./data_collection_by_hollis/weather_data.csv"
    outage_data: "./data_collection_by_hollis/outage_data.csv"
  preprocessing:
    missing_strategy: "interpolate"
    feature_engineering: true

models:
  vulnerability:
    hidden_layers: [64, 32]
    learning_rate: 0.001
    epochs: 100
  failure:
    hidden_layers: [128, 64]
    learning_rate: 0.001
    epochs: 100
  gan:
    generator_layers: [128, 256, 128]
    discriminator_layers: [128, 64]
    learning_rate: 0.0002
    epochs: 1000

rl:
  environment:
    max_steps: 100
    reward_weights:
      stability: 1.0
      outage: -1.0
      action: -0.1
  agents:
    dqn:
      learning_rate: 0.001
      gamma: 0.99
      buffer_size: 100000
    ppo:
      learning_rate: 0.0003
      clip_range: 0.2
      n_steps: 2048
    sac:
      learning_rate: 0.0003
      buffer_size: 100000
      ent_coef: "auto"
    td3:
      learning_rate: 0.0003
      buffer_size: 100000
      policy_delay: 2
    gail:
      learning_rate: 0.0003
      expert_episodes: 100
visualization:
  grid:
    node_size: 300
    font_size: 8
  performance:
    figure_size: [10, 6]
    dpi: 100
```





#### 5.2 Implementation Approach

##### 5.2.1 Modularity Guidelines
- Implement each module as a separate Python package
- Use object-oriented design with clear interfaces
- Follow Python style guidelines (PEP 8)
- Provide comprehensive docstrings for all functions and classes

##### 5.2.2 Testing Strategy
- Unit tests for core functionality
- Integration tests for module interactions
- Synthetic data-based validation
- Performance benchmarks for computational components

### 6. Use Cases and Workflows

#### 6.1 Data Preprocessing Workflow
1. Load raw data (grid, weather, outages)
2. Clean and normalize data
3. Generate features (daily/monthly aggregates)
4. Align temporal data to common timestamps
5. Create modeling dataset
6. Save processed data

#### 6.2 Failure Prediction Workflow
1. Load processed data
2. Split into training/validation sets
3. Train neural network predictor
4. Evaluate on validation set
5. Calculate feature importance
6. Generate vulnerability heatmap

#### 6.3 Reinforcement Learning Workflow
1. Initialize environment with processed data
2. Train multiple RL agents (DQN, PPO, SAC, TD3, GAIL)
3. Evaluate agent performance on test scenarios
4. Compare metrics across agents
5. Save best-performing policy
6. Visualize learning curves and policy behavior

### 7. Implementation Timeline

#### 7.1 Phase 1: Data Management (Week 1-2)
- Implement data loading
- Implement preprocessing pipeline
- Implement synthetic data generation

#### 7.2 Phase 2: Modeling (Week 3-4)
- Implement component vulnerability profiling
- Implement environmental threat modeling
- Implement failure prediction models

#### 7.3 Phase 3: Scenario Generation (Week 5-6)
- Implement Conditional GAN
- Implement extreme event generator
- Validate scenario realism

#### 7.4 Phase 4: Reinforcement Learning (Week 7-9)
- Implement OpenAI Gym environment
- Implement DQN, PPO, SAC, TD3, and GAIL agents
- Train and evaluate policies

#### 7.5 Phase 5: Visualization and Integration (Week 10-12)
- Implement visualization components
- Create integrated pipeline
- Document full system and create examples

### 8. Integration and Evaluation

#### 8.1 Integration Points
- Data preprocessing → Vulnerability modeling
- Vulnerability modeling → Failure prediction
- Failure prediction → Scenario generation
- Scenario generation → Reinforcement learning
- All components → Visualization

#### 8.2 Evaluation Metrics
- Failure Prediction: Accuracy, Precision, Recall, AUC
- Scenario Generation: Realism score, Statistical similarity to historical data
- Reinforcement Learning: Average reward, Outage frequency, Load shedding events

### 9. Appendix

#### 9.1 Glossary
- GAN: Generative Adversarial Network
- DQN: Deep Q-Network
- PPO: Proximal Policy Optimization
- SAC: Soft Actor-Critic
- TD3: Twin Delayed DDPG
- GAIL: Generative Adversarial Imitation Learning

#### 9.2 Example Dataset Format

##### Grid Topology Example
```json
{
  "nodes": {
    "n1": {
      "id": "n1", 
      "type": "generator", 
      "capacity": 100, 
      "voltage": 230, 
      "age": 5,
      "material": "copper"
    },
    "n2": {
      "id": "n2", 
      "type": "load", 
      "demand": 80, 
      "priority": 1
    }
  },
  "lines": {
    "l1": {
      "id": "l1", 
      "from": "n1", 
      "to": "n2", 
      "capacity": 150, 
      "length": 20, 
      "voltage": 230, 
      "age": 8,
      "material": "aluminum"
    }
  }
}
```

#### 9.3 Development Workflow

##### 9.3.1 Git Branching Strategy
This project follows a feature branch workflow for all development activities:

1. **Main Branch**: The `main` branch contains the stable, production-ready code.
2. **Feature Branches**: All new features and changes must be developed in dedicated feature branches.
   - Branch naming convention: `feature/feature-name`, `bugfix/issue-description`, `data/update-description`
   - Current active branches:
     - `data-optimization`: Data collection optimization and reduction

**Branch Workflow**:
1. Create a new branch from `main` for each feature or change
2. Develop and test changes in the feature branch
3. Submit a pull request for code review
4. Merge back to `main` after approval

##### 9.3.2 Data Collection Structure
The framework uses two primary data collection directories:

1. **data_collection_by_hollis**: 175MB dataset containing:
   - `correlated_outage`: Power outage data organized by year (2014-2023)
   - Documentation and data links

2. **data_collection_by_manish**: 926MB dataset containing:
   - `NOAA_Daily_Summaries_Reduced`: 540MB optimized subset of weather data
   - `Outage Data`: 199MB of power outage records
   - `RTS_Data`: 166MB of grid system test case data
   - Other smaller datasets including IEEE test cases and environmental data

**Data Optimization**:
- Original NOAA dataset of 134GB was reduced to 540MB using random sampling of US weather stations
- Approximately 551 representative weather stations were selected from the original ~130,000 stations
- The reduced dataset preserves geographical diversity while making processing feasible

**Note**: When working with the data, use the reduced datasets to ensure processing remains manageable. For more extensive analysis requiring the full dataset, consider processing in smaller batches or utilizing high-performance computing resources at CHPC.
