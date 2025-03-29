import numpy as np
import torch
import os
import sys
import logging
from datetime import datetime
from environment import make_env
from marl_agents.maddpg_algorithm import MADDPG
from marl_agents.replay_buffer import MAReplayBuffer  # Updated import path
import time

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'  # Simplified format for training output
    )
    return logging.getLogger(__name__)

def create_log_dir():
    """Create directory for logs"""
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', f'training_{current_time}')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def train(episodes=1000, save_interval=100, log_dir=None, debug=False):
    """
    Train MADDPG agents
    Args:
        episodes: Number of training episodes
        save_interval: Interval to save model checkpoints
        log_dir: Directory for saving logs and models
        debug: Enable debug logging
    """
    logger = setup_logger()
    
    # Print header similar to training_output.log
    logger.info("# Training Log for MADDPG Pursuit Environment")
    logger.info(f"# Date: {datetime.now().strftime('%B %d, %Y')}")
    logger.info("# Environment: Pursuit-v4")
    logger.info("# Number of Agents: 8")
    logger.info(f"# Training Episodes: {episodes}\n")
    
    logger.info("Initializing environment and agents...\n")
    
    env = make_env()
    maddpg = MADDPG(env)
    replay_buffer = MAReplayBuffer(capacity=100000)

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.agents}
        step_count = 0
        done = {agent: False for agent in env.agents}

        logger.info(f"[Episode {episode}]")
        
        while not all(done.values()):
            # Select actions
            actions = {agent: maddpg.select_action(obs[agent], i) 
                      for i, agent in enumerate(env.agents)}
            
            # Execute actions
            next_obs, rewards, done, _ = env.step(actions)
            
            # Store experience
            replay_buffer.push(obs, actions, rewards, next_obs, done)
            
            # Update observations and rewards
            obs = next_obs
            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]
            step_count += 1
            
            # Train if enough samples
            if len(replay_buffer) > maddpg.batch_size:
                maddpg.update(replay_buffer)

        # Log episode results
        for agent in env.agents:
            logger.info(f"{agent} Reward: {episode_rewards[agent]:.2f}")
        logger.info(f"Steps: {step_count}\n")

        # Save model checkpoint
        if episode % save_interval == 0:
            maddpg.save_model(f"models/maddpg_{episode}.pth")

    # Save final model
    maddpg.save_model("models/maddpg_final.pth")

if __name__ == "__main__":
    train()