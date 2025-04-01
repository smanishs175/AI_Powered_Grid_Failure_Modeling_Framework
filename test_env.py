import numpy as np
import pandas as pd
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import required components
from gfmf.reinforcement_learning.environments.grid_env import GridEnv
from gfmf.reinforcement_learning.agents.sac_agent import SACAgent

# Create synthetic data
def create_test_data():
    # Create grid topology data
    grid_size = 5
    grid_topology = pd.DataFrame({
        'component_id': range(grid_size),
        'type': ['transformer'] * (grid_size // 2) + ['line'] * (grid_size - grid_size // 2),
        'capacity': np.random.uniform(50, 100, grid_size),
        'age': np.random.uniform(1, 20, grid_size),
        'criticality': np.random.uniform(0.1, 1.0, grid_size)
    })
    
    # Create vulnerability scores
    vulnerability_scores = pd.DataFrame({
        'component_id': range(grid_size),
        'vulnerability_score': np.random.uniform(0, 1, grid_size),
        'environmental_vulnerability': np.random.uniform(0, 1, grid_size)
    })
    
    # Create scenario data
    scenario_impacts = pd.DataFrame({
        'scenario_id': ['baseline', 'heat_wave', 'storm'],
        'scenario_type': ['baseline', 'extreme_weather', 'extreme_weather'],
        'operational_percentage': [100, 80, 70],
        'outage_impact': [0, 20, 30],
        'cascading_impact': [0, 30, 45]
    })
    
    return grid_topology, vulnerability_scores, scenario_impacts

if __name__ == "__main__":
    print("Testing Grid Environment and RL Agents")
    
    # Create test data
    grid_topology, vulnerability_scores, scenario_impacts = create_test_data()
    
    # Configure environment
    env_config = {
        'grid_size': len(vulnerability_scores),
        'vulnerability_data': vulnerability_scores,
        'scenario_data': scenario_impacts,
        'max_steps': 10
    }
    
    # Create environment
    print("Creating environment...")
    env = GridEnv(env_config)
    
    # Test environment reset
    print("Testing environment reset...")
    state, _ = env.reset()
    print(f"State shape: {state.shape}")
    
    # Test environment step
    print("Testing environment step...")
    action = 0  # Do nothing action
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}, Done: {terminated or truncated}")
    print(f"Info: {info}")
    
    # Create SAC agent
    print("Creating SAC agent...")
    sac_config = {'learning_rate': 3e-4, 'batch_size': 8}
    sac_agent = SACAgent(env, sac_config)
    
    # Test agent training (very short)
    print("Testing SAC agent training...")
    results = sac_agent.train(total_timesteps=20, eval_freq=10, n_eval_episodes=2)
    print(f"Training results: {results['best_reward']}")
    
    print("Test completed successfully!") 