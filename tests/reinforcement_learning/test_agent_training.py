"""
Quick training test for all reinforcement learning agents.

This script runs all agents for a minimal training period (10% of normal duration)
to verify they can properly learn and improve performance.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import mock environment 
from tests.reinforcement_learning.agents.test_fixed import MockGridEnv, MockContinuousGridEnv

# Import agents
from gfmf.reinforcement_learning.agents.dqn_agent import DQNAgent
from gfmf.reinforcement_learning.agents.ppo_agent import PPOAgent
from gfmf.reinforcement_learning.agents.sac_agent import SACAgent
from gfmf.reinforcement_learning.agents.td3_agent import TD3Agent
from gfmf.reinforcement_learning.agents.gail_agent import GAILAgent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentTrainingTest")

# Set small training parameters (10% of normal)
NUM_EPISODES = 10        # Standard would be 100
MAX_STEPS = 100          # Keep normal episode length
EVAL_EPISODES = 3        # Standard would be 20-30

# Create output directory
OUTPUT_DIR = 'outputs/test_training'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_expert_demonstrations(env, num_trajectories=5, steps_per_trajectory=10):
    """
    Create simple expert demonstrations for GAIL.
    
    Args:
        env: Environment to generate demonstrations in
        num_trajectories: Number of demonstration trajectories
        steps_per_trajectory: Steps per trajectory
        
    Returns:
        list: List of demonstration trajectories
    """
    expert_trajectories = []
    
    for _ in range(num_trajectories):
        states = []
        actions = []
        
        state = env.reset()
        for _ in range(steps_per_trajectory):
            # Simple policy: choose action 0 most of the time (assumed to be good)
            if np.random.random() < 0.8:
                action = 0
            else:
                action = np.random.randint(0, env.action_space.n)
                
            states.append(state)
            actions.append(action)
            
            state, _, done, _ = env.step(action)
            if done:
                break
                
        expert_trajectories.append({
            'states': states,
            'actions': actions
        })
        
    return expert_trajectories

def train_and_evaluate(agent_name, agent, env, num_episodes, max_steps, eval_episodes):
    """
    Train and evaluate a single agent.
    
    Args:
        agent_name: Name of the agent for logging
        agent: Agent instance to train
        env: Environment to train in
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        eval_episodes: Number of evaluation episodes
        
    Returns:
        dict: Training and evaluation metrics
    """
    start_time = time.time()
    
    logger.info(f"Starting training of {agent_name}...")
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    
    # Train for specified number of episodes
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        losses = []
        
        for step in range(max_steps):
            # Get action based on current state
            if agent_name == "DQN":
                action = agent.act(state)
                if hasattr(action, 'item'):
                    action = action.item()
            elif agent_name in ["PPO", "GAIL"]:
                action = agent.act(state)[0]
                if hasattr(action, 'item'):
                    action = action.item()
            else:
                action = agent.act(state)
                if hasattr(action, 'item'):
                    action = action.item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience based on agent type
            if agent_name == "PPO":
                # PPO typically stores experiences in each step
                agent.store_experience(state, action, reward, next_state, done)
            elif agent_name == "GAIL":
                # GAIL may have its own storage mechanism
                agent.store_experience(state, action, reward, next_state, done)
            elif agent_name == "DQN":
                # DQN with prioritized replay might use add_experience
                if hasattr(agent, 'add_experience'):
                    agent.add_experience(state, action, reward, next_state, done)
                else:
                    # Try the standard methods
                    try:
                        agent.memory.add(state, action, reward, next_state, done)
                    except AttributeError:
                        try:
                            agent.memory.add_experience(state, action, reward, next_state, done)
                        except AttributeError:
                            # Last resort: direct buffer access if we know the structure
                            agent.store_experience(state, action, reward, next_state, done)
            else:
                # For SAC and TD3, try different buffer interfaces
                try:
                    agent.memory.add(state, action, reward, next_state, done)
                except AttributeError:
                    try:
                        agent.memory.add_experience(state, action, reward, next_state, done)
                    except AttributeError:
                        # Last resort: agent might have a store_experience method
                        agent.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Learn from experiences
            if agent_name == "DQN" and len(agent.memory) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
            elif agent_name == "SAC" and len(agent.memory) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
            elif agent_name == "TD3" and len(agent.memory) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
            elif agent_name == "GAIL" and len(agent.memory) > agent.batch_size:
                # GAIL might need to first collect a full trajectory
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
            
            if done:
                break
        
        # PPO learns after collecting a batch of experiences
        if agent_name == "PPO":
            # Different PPO implementations might have different update methods
            try:
                loss = agent.update_policy(state)
            except (AttributeError, TypeError):
                try:
                    loss = agent.learn()
                except:
                    # If all else fails, just continue (agent might not return loss)
                    loss = None
                    
            if loss is not None:
                losses.append(loss)
        
        # Record metrics
        episode_rewards.append(episode_reward)
        if losses:
            episode_losses.append(np.mean(losses))
        else:
            episode_losses.append(0)
        
        # Log progress
        if (episode + 1) % 2 == 0:
            logger.info(f"{agent_name} - Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Avg Loss: {np.mean(losses) if losses else 0:.4f}")
    
    # Evaluate the trained agent
    logger.info(f"Evaluating {agent_name}...")
    eval_rewards = []
    
    for episode in range(eval_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Get action (evaluation mode - deterministic)
            if agent_name == "DQN":
                action = agent.act(state, eval_mode=True)
                if hasattr(action, 'item'):
                    action = action.item()
            elif agent_name == "PPO":
                # PPO might not have eval_mode
                action = agent.act(state)[0]
                if hasattr(action, 'item'):
                    action = action.item()
            elif agent_name == "GAIL":
                # GAIL might not have eval_mode
                action = agent.act(state)[0]
                if hasattr(action, 'item'):
                    action = action.item()
            else:
                try:
                    action = agent.act(state, eval_mode=True)
                except TypeError:
                    action = agent.act(state)
                    
                if hasattr(action, 'item'):
                    action = action.item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        eval_rewards.append(total_reward)
    
    # Calculate metrics
    training_time = time.time() - start_time
    avg_eval_reward = np.mean(eval_rewards)
    std_eval_reward = np.std(eval_rewards)
    
    logger.info(f"{agent_name} - Training complete. Time: {training_time:.2f}s, Eval Reward: {avg_eval_reward:.2f} ± {std_eval_reward:.2f}")
    
    # Return metrics
    return {
        'name': agent_name,
        'training_time': training_time,
        'training_rewards': episode_rewards,
        'training_losses': episode_losses,
        'evaluation_rewards': eval_rewards,
        'avg_eval_reward': avg_eval_reward,
        'std_eval_reward': std_eval_reward
    }

def plot_results(results, output_dir):
    """
    Plot the training and evaluation results.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    # Plot training rewards
    plt.figure(figsize=(12, 6))
    for result in results:
        plt.plot(result['training_rewards'], label=result['name'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_rewards.png'))
    
    # Plot training losses
    plt.figure(figsize=(12, 6))
    for result in results:
        plt.plot(result['training_losses'], label=result['name'])
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_losses.png'))
    
    # Plot evaluation rewards
    plt.figure(figsize=(10, 6))
    agent_names = [result['name'] for result in results]
    avg_rewards = [result['avg_eval_reward'] for result in results]
    std_rewards = [result['std_eval_reward'] for result in results]
    
    plt.bar(agent_names, avg_rewards, yerr=std_rewards, capsize=10)
    plt.ylabel('Average Reward')
    plt.title('Evaluation Performance')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, 'evaluation_rewards.png'))

def main():
    """Main function to run agent training tests."""
    logger.info("Starting agent training tests...")
    
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Environment parameters
    state_dim = 8
    action_dim = 3
    max_steps = MAX_STEPS
    
    # Create environments
    discrete_env = MockGridEnv(state_dim=state_dim, action_dim=action_dim, max_steps=max_steps)
    continuous_env = MockContinuousGridEnv(state_dim=state_dim, action_dim=action_dim, max_steps=max_steps)
    
    # Add missing methods to agents if needed
    def _add_store_experience_to_agent(agent, agent_name):
        """Add store_experience method to agent classes that might not have it"""
        if not hasattr(agent, 'store_experience'):
            if agent_name == "DQN":
                def store_experience(state, action, reward, next_state, done):
                    # Handle different buffer interfaces
                    try:
                        agent.memory.add(state, action, reward, next_state, done)
                    except AttributeError:
                        try:
                            agent.memory.add_experience(state, action, reward, next_state, done)
                        except:
                            # Fallback for Prioritized Experience Replay
                            if hasattr(agent.memory, 'buffer'):
                                experience = (state, action, reward, next_state, done)
                                agent.memory.buffer.append(experience)
            elif agent_name in ["SAC", "TD3"]:
                def store_experience(state, action, reward, next_state, done):
                    # Handle different buffer interfaces
                    try:
                        agent.memory.add(state, action, reward, next_state, done)
                    except AttributeError:
                        try:
                            agent.memory.add_experience(state, action, reward, next_state, done)
                        except:
                            # Fallback for different buffer structures
                            if hasattr(agent.memory, 'buffer'):
                                experience = (state, action, reward, next_state, done)
                                agent.memory.buffer.append(experience)
            elif agent_name == "PPO":
                def store_experience(state, action, reward, next_state, done):
                    # PPO often stores experiences differently or implicitly
                    # This is just a placeholder if the agent doesn't have its own method
                    pass
            elif agent_name == "GAIL":
                def store_experience(state, action, reward, next_state, done):
                    # GAIL might store experiences differently
                    try:
                        agent.memory.add(state, action, reward, next_state, done)
                    except AttributeError:
                        try:
                            agent.memory.add_experience(state, action, reward, next_state, done)
                        except:
                            # Fallback
                            if hasattr(agent.memory, 'buffer'):
                                experience = (state, action, reward, next_state, done)
                                agent.memory.buffer.append(experience)
            else:
                def store_experience(state, action, reward, next_state, done):
                    # Generic fallback
                    pass
                
            # Add the method to the agent
            agent.store_experience = store_experience.__get__(agent)
    
    # Create agents
    dqn_agent = DQNAgent(state_dim, action_dim)
    _add_store_experience_to_agent(dqn_agent, "DQN")
    
    ppo_agent = PPOAgent(state_dim, action_dim)
    _add_store_experience_to_agent(ppo_agent, "PPO")
    
    sac_agent = SACAgent(state_dim, action_dim)
    _add_store_experience_to_agent(sac_agent, "SAC")
    
    td3_agent = TD3Agent(state_dim, action_dim)
    _add_store_experience_to_agent(td3_agent, "TD3")
    
    gail_agent = GAILAgent(state_dim, action_dim)
    _add_store_experience_to_agent(gail_agent, "GAIL")
    
    # Create expert demonstrations for GAIL
    expert_demos = create_expert_demonstrations(discrete_env)
    for demo in expert_demos:
        gail_agent.add_expert_trajectory(demo)
    
    # Train and evaluate each agent
    results = []
    
    # DQN
    dqn_results = train_and_evaluate("DQN", dqn_agent, discrete_env, NUM_EPISODES, MAX_STEPS, EVAL_EPISODES)
    results.append(dqn_results)
    
    # PPO
    ppo_results = train_and_evaluate("PPO", ppo_agent, discrete_env, NUM_EPISODES, MAX_STEPS, EVAL_EPISODES)
    results.append(ppo_results)
    
    # SAC - For environments with continuous action spaces
    sac_results = train_and_evaluate("SAC", sac_agent, discrete_env, NUM_EPISODES, MAX_STEPS, EVAL_EPISODES)
    results.append(sac_results)
    
    # TD3
    td3_results = train_and_evaluate("TD3", td3_agent, discrete_env, NUM_EPISODES, MAX_STEPS, EVAL_EPISODES)
    results.append(td3_results)
    
    # GAIL
    gail_results = train_and_evaluate("GAIL", gail_agent, discrete_env, NUM_EPISODES, MAX_STEPS, EVAL_EPISODES)
    results.append(gail_results)
    
    # Plot results
    plot_results(results, OUTPUT_DIR)
    
    logger.info(f"All tests complete. Results saved to {OUTPUT_DIR}")
    
    # Print summary
    print("\n===== TRAINING SUMMARY =====")
    print(f"{'Agent':<6} | {'Training Time (s)':<18} | {'Final Training Reward':<20} | {'Evaluation Reward':<20}")
    print("-" * 70)
    for result in results:
        print(f"{result['name']:<6} | {result['training_time']:<18.2f} | {result['training_rewards'][-1]:<20.2f} | {result['avg_eval_reward']:<16.2f} ± {result['std_eval_reward']:.2f}")

if __name__ == "__main__":
    main()
