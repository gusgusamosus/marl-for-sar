import numpy as np
import torch
from environment import make_env
from marl_agents.maddpg_algorithm import MADDPG
import time
from datetime import datetime
import os

def evaluate(model_path, num_episodes=10, render=False):
    """
    Evaluate trained MADDPG agents
    Args:
        model_path: Path to the trained model
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
    """
    # Initialize environment and load model
    env = make_env()
    maddpg = MADDPG(env)
    maddpg.load_model(model_path)

    # Create log file
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = open(f'logs/evaluation_{current_time}.log', 'w')
    log_file.write(f"Evaluation started at: {datetime.now()}\n")
    log_file.write("Episode,Total Reward,Average Agent Reward,Captures,Steps\n")

    # Evaluation loop
    eval_rewards = []
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = {agent: np.array(obs[agent], dtype=np.float32) for agent in env.agents}
        episode_reward = 0
        captures = 0
        steps = 0

        # Episode loop
        done = {agent: False for agent in env.agents}
        while not all(done.values()):
            if render:
                env.render()

            # Select actions (no exploration during evaluation)
            actions = {agent: maddpg.select_action(state[agent], agent_id=i, evaluate=True) 
                      for i, agent in enumerate(env.agents)}
            
            # Execute actions
            next_obs, rewards, done, _ = env.step(actions)
            next_state = {agent: np.array(next_obs[agent], dtype=np.float32) 
                         for agent in env.agents}

            # Update state and metrics
            state = next_state
            episode_reward += sum(rewards.values())
            steps += 1

        # Log episode results
        avg_reward = episode_reward / len(env.agents)
        eval_rewards.append(episode_reward)
        
        log_message = f"Episode {episode}/{num_episodes}\n"
        log_message += f"Total Reward: {episode_reward:.2f}\n"
        log_message += f"Average Agent Reward: {avg_reward:.2f}\n"
        log_message += f"Steps: {steps}\n"
        print(log_message)
        
        log_file.write(f"{episode},{episode_reward:.2f},{avg_reward:.2f},{captures},{steps}\n")

    # Write summary statistics
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    log_file.write(f"\nEvaluation Results:\n")
    log_file.write(f"Average Total Reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
    log_file.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/maddpg_final.pth',
                      help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment')
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render)