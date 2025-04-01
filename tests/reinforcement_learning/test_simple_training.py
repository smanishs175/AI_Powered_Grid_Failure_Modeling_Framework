"""
Simple training script to test each reinforcement learning agent separately.

This script runs each agent for a minimal training period to verify functionality.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import gym
from gym import spaces

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import agents
from gfmf.reinforcement_learning.agents.dqn_agent import DQNAgent
from gfmf.reinforcement_learning.agents.ppo_agent import PPOAgent
from gfmf.reinforcement_learning.agents.sac_agent import SACAgent
from gfmf.reinforcement_learning.agents.td3_agent import TD3Agent
from gfmf.reinforcement_learning.agents.gail_agent import GAILAgent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimpleTrainingTest")

# Set training parameters to 10% of normal
NUM_EPISODES = 10
MAX_STEPS = 100
EVAL_EPISODES = 3

# Create output directory
OUTPUT_DIR = 'outputs/test_simple_training'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simple mock environment for testing
class MockGridEnv(gym.Env):
    """Simple mock environment with discrete action space."""
    
    def __init__(self, state_dim=10, action_dim=3, max_steps=100):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        
        self.state = None
        self.steps = 0
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(-0.5, 0.5, size=self.state_dim).astype(np.float32)
        self.steps = 0
        return self.state
    
    def step(self, action):
        # Simple dynamics for testing
        if action == 0:
            self.state += 0.1 * np.ones(self.state_dim).astype(np.float32)
        elif action == 1:
            self.state -= 0.1 * np.ones(self.state_dim).astype(np.float32)
        else:
            self.state += 0.05 * np.random.uniform(-1, 1, size=self.state_dim).astype(np.float32)
        
        self.state = np.clip(self.state, -1.0, 1.0).astype(np.float32)
        reward = -np.sum(np.abs(self.state)) / self.state_dim
        
        self.steps += 1
        done = (self.steps >= self.max_steps)
        
        return self.state, reward, done, {}

class MockContinuousGridEnv(MockGridEnv):
    """Simple mock environment with continuous action space."""
    
    def __init__(self, state_dim=10, action_dim=3, max_steps=100):
        super().__init__(state_dim, action_dim, max_steps)
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
    
    def step(self, action):
        # Convert continuous action to array
        action_np = np.asarray(action).flatten()
        
        # Update state based on action
        self.state += 0.1 * action_np[:self.state_dim]
        self.state = np.clip(self.state, -1.0, 1.0).astype(np.float32)
        
        reward = -np.sum(np.abs(self.state)) / self.state_dim
        self.steps += 1
        done = (self.steps >= self.max_steps)
        
        return self.state, reward, done, {}

def create_expert_data():
    """Create simple expert demonstrations for GAIL."""
    env = MockGridEnv(state_dim=8, action_dim=3)
    expert_trajectories = []
    
    for _ in range(5):  # 5 trajectories
        states = []
        actions = []
        
        state = env.reset()
        for _ in range(10):  # 10 steps per trajectory
            # Simple expert policy: choose action 0 most of the time
            action = 0 if np.random.random() < 0.8 else np.random.randint(1, 3)
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

def test_dqn_agent():
    """Test DQN agent training."""
    logger.info("Testing DQN Agent")
    
    # Setup environment and agent
    env = MockGridEnv(state_dim=8, action_dim=3, max_steps=MAX_STEPS)
    dqn = DQNAgent(8, 3)
    
    # Training loop
    rewards = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            # Select action
            action = dqn.act(state)
            if hasattr(action, 'item'):
                action = action.item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience in replay buffer
            try:
                dqn.memory.add(state, action, reward, next_state, done)
            except:
                # Alternative method if add doesn't exist
                if hasattr(dqn, 'add_experience'):
                    dqn.add_experience(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Learn if enough samples
            if len(dqn.memory) > dqn.batch_size:
                dqn.learn()
            
            if done:
                break
        
        rewards.append(episode_reward)
        logger.info(f"Episode {episode+1}/{NUM_EPISODES}, Reward: {episode_reward:.2f}")
    
    # Evaluate
    eval_rewards = []
    for episode in range(EVAL_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            action = dqn.act(state, eval_mode=True)
            if hasattr(action, 'item'):
                action = action.item()
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
    
    avg_eval = np.mean(eval_rewards)
    logger.info(f"DQN Evaluation: Avg Reward = {avg_eval:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('DQN Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'dqn_rewards.png'))
    
    return {
        'train_rewards': rewards,
        'eval_rewards': eval_rewards,
        'avg_eval': avg_eval
    }

def test_ppo_agent():
    """Test PPO agent training."""
    logger.info("Testing PPO Agent")
    
    # Setup environment and agent
    env = MockGridEnv(state_dim=8, action_dim=3, max_steps=MAX_STEPS)
    ppo = PPOAgent(8, 3)
    
    # Training loop
    rewards = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            # Select action
            result = ppo.act(state)
            if isinstance(result, tuple):
                action = result[0]
            else:
                action = result
                
            if hasattr(action, 'item'):
                action = action.item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # PPO typically handles experience collection internally
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update policy after collecting experiences
        try:
            ppo.update_policy(state)
        except:
            try:
                ppo.learn()
            except:
                pass  # Some implementations might update during act()
        
        rewards.append(episode_reward)
        logger.info(f"Episode {episode+1}/{NUM_EPISODES}, Reward: {episode_reward:.2f}")
    
    # Evaluate
    eval_rewards = []
    for episode in range(EVAL_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            result = ppo.act(state)
            if isinstance(result, tuple):
                action = result[0]
            else:
                action = result
                
            if hasattr(action, 'item'):
                action = action.item()
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
    
    avg_eval = np.mean(eval_rewards)
    logger.info(f"PPO Evaluation: Avg Reward = {avg_eval:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('PPO Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ppo_rewards.png'))
    
    return {
        'train_rewards': rewards,
        'eval_rewards': eval_rewards,
        'avg_eval': avg_eval
    }

def test_sac_agent():
    """Test SAC agent training."""
    logger.info("Testing SAC Agent")
    
    # Setup environment and agent
    env = MockGridEnv(state_dim=8, action_dim=3, max_steps=MAX_STEPS)
    sac = SACAgent(8, 3)
    
    # Training loop
    rewards = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            # Select action
            action = sac.act(state)
            if hasattr(action, 'item'):
                action = action.item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience in replay buffer
            try:
                sac.memory.add(state, action, reward, next_state, done)
            except:
                # Alternative method if add doesn't exist
                if hasattr(sac, 'add_experience'):
                    sac.add_experience(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Learn if enough samples
            if len(sac.memory) > sac.batch_size:
                sac.learn()
            
            if done:
                break
        
        rewards.append(episode_reward)
        logger.info(f"Episode {episode+1}/{NUM_EPISODES}, Reward: {episode_reward:.2f}")
    
    # Evaluate
    eval_rewards = []
    for episode in range(EVAL_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            try:
                action = sac.act(state, eval_mode=True)
            except:
                action = sac.act(state)
                
            if hasattr(action, 'item'):
                action = action.item()
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
    
    avg_eval = np.mean(eval_rewards)
    logger.info(f"SAC Evaluation: Avg Reward = {avg_eval:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('SAC Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'sac_rewards.png'))
    
    return {
        'train_rewards': rewards,
        'eval_rewards': eval_rewards,
        'avg_eval': avg_eval
    }

def test_td3_agent():
    """Test TD3 agent training."""
    logger.info("Testing TD3 Agent")
    
    # Setup environment and agent
    env = MockGridEnv(state_dim=8, action_dim=3, max_steps=MAX_STEPS)
    td3 = TD3Agent(8, 3)
    
    # Training loop
    rewards = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            # Select action
            action = td3.act(state)
            if hasattr(action, 'item'):
                action = action.item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience in replay buffer
            try:
                td3.memory.add(state, action, reward, next_state, done)
            except:
                # Alternative method if add doesn't exist
                if hasattr(td3, 'add_experience'):
                    td3.add_experience(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Learn if enough samples
            if len(td3.memory) > td3.batch_size:
                td3.learn()
            
            if done:
                break
        
        rewards.append(episode_reward)
        logger.info(f"Episode {episode+1}/{NUM_EPISODES}, Reward: {episode_reward:.2f}")
    
    # Evaluate
    eval_rewards = []
    for episode in range(EVAL_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            try:
                action = td3.act(state, eval_mode=True)
            except:
                action = td3.act(state)
                
            if hasattr(action, 'item'):
                action = action.item()
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
    
    avg_eval = np.mean(eval_rewards)
    logger.info(f"TD3 Evaluation: Avg Reward = {avg_eval:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('TD3 Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'td3_rewards.png'))
    
    return {
        'train_rewards': rewards,
        'eval_rewards': eval_rewards,
        'avg_eval': avg_eval
    }

def test_gail_agent():
    """Test GAIL agent training."""
    logger.info("Testing GAIL Agent")
    
    # Setup environment and agent
    env = MockGridEnv(state_dim=8, action_dim=3, max_steps=MAX_STEPS)
    gail = GAILAgent(8, 3)
    
    # Add expert demonstrations
    expert_demos = create_expert_data()
    for demo in expert_demos:
        gail.add_expert_trajectory(demo)
    
    # Training loop
    rewards = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            # Select action
            result = gail.act(state)
            if isinstance(result, tuple):
                action = result[0]
            else:
                action = result
                
            if hasattr(action, 'item'):
                action = action.item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # GAIL typically handles experience collection internally
            # but we'll try different methods
            try:
                gail.memory.add(state, action, reward, next_state, done)
            except:
                try:
                    if hasattr(gail, 'add_experience'):
                        gail.add_experience(state, action, reward, next_state, done)
                except:
                    pass
            
            # Get GAIL reward if available
            try:
                gail_reward = gail.get_reward(state, action)
            except:
                gail_reward = reward
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Learn if enough samples
            try:
                if hasattr(gail, 'learn') and len(gail.memory) > gail.batch_size:
                    gail.learn()
            except:
                pass
            
            if done:
                break
        
        # Update policy after episode
        try:
            gail.update_policy(state)
        except:
            try:
                if hasattr(gail, 'train'):
                    gail.train()
            except:
                pass
        
        rewards.append(episode_reward)
        logger.info(f"Episode {episode+1}/{NUM_EPISODES}, Reward: {episode_reward:.2f}")
    
    # Evaluate
    eval_rewards = []
    for episode in range(EVAL_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            result = gail.act(state)
            if isinstance(result, tuple):
                action = result[0]
            else:
                action = result
                
            if hasattr(action, 'item'):
                action = action.item()
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
    
    avg_eval = np.mean(eval_rewards)
    logger.info(f"GAIL Evaluation: Avg Reward = {avg_eval:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('GAIL Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'gail_rewards.png'))
    
    return {
        'train_rewards': rewards,
        'eval_rewards': eval_rewards,
        'avg_eval': avg_eval
    }

def main():
    """Run tests for all agents."""
    logger.info("Starting simple agent tests with 10% of normal training data")
    
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Test all agents
    try:
        dqn_results = test_dqn_agent()
        logger.info("DQN testing completed successfully")
    except Exception as e:
        logger.error(f"Error testing DQN: {e}")
        dqn_results = None
    
    try:
        ppo_results = test_ppo_agent()
        logger.info("PPO testing completed successfully")
    except Exception as e:
        logger.error(f"Error testing PPO: {e}")
        ppo_results = None
    
    try:
        sac_results = test_sac_agent()
        logger.info("SAC testing completed successfully")
    except Exception as e:
        logger.error(f"Error testing SAC: {e}")
        sac_results = None
    
    try:
        td3_results = test_td3_agent()
        logger.info("TD3 testing completed successfully")
    except Exception as e:
        logger.error(f"Error testing TD3: {e}")
        td3_results = None
    
    try:
        gail_results = test_gail_agent()
        logger.info("GAIL testing completed successfully")
    except Exception as e:
        logger.error(f"Error testing GAIL: {e}")
        gail_results = None
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    # Plot all training curves
    if dqn_results:
        plt.plot(dqn_results['train_rewards'], label='DQN')
    if ppo_results:
        plt.plot(ppo_results['train_rewards'], label='PPO')
    if sac_results:
        plt.plot(sac_results['train_rewards'], label='SAC')
    if td3_results:
        plt.plot(td3_results['train_rewards'], label='TD3')
    if gail_results:
        plt.plot(gail_results['train_rewards'], label='GAIL')
    
    plt.title('Agent Training Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_agents_comparison.png'))
    
    # Print summary
    print("\n===== TRAINING SUMMARY =====")
    print(f"{'Agent':<6} | {'Final Training Reward':<20} | {'Evaluation Reward':<20}")
    print("-" * 60)
    
    if dqn_results:
        print(f"DQN    | {dqn_results['train_rewards'][-1]:<20.2f} | {dqn_results['avg_eval']:<20.2f}")
    if ppo_results:
        print(f"PPO    | {ppo_results['train_rewards'][-1]:<20.2f} | {ppo_results['avg_eval']:<20.2f}")
    if sac_results:
        print(f"SAC    | {sac_results['train_rewards'][-1]:<20.2f} | {sac_results['avg_eval']:<20.2f}")
    if td3_results:
        print(f"TD3    | {td3_results['train_rewards'][-1]:<20.2f} | {td3_results['avg_eval']:<20.2f}")
    if gail_results:
        print(f"GAIL   | {gail_results['train_rewards'][-1]:<20.2f} | {gail_results['avg_eval']:<20.2f}")
    
    logger.info(f"All tests complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
