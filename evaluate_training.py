import numpy as np
from datetime import datetime
import os

def evaluate_model(env, maddpg, num_episodes=10, log_file="evaluation_output.log"):
    """Evaluate trained model and log results."""
    with open(log_file, 'w') as f:
        # Write header
        f.write(f"MADDPG Evaluation Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: models/maddpg.pth\n")
        f.write(f"Number of Episodes: {num_episodes}\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("Loading trained model from models/maddpg.pth...\n")
        f.write(f"Starting evaluation over {num_episodes} episodes...\n\n")
        
        # Track metrics
        total_rewards = []
        agent_rewards = []
        captures_list = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            captures = 0
            done = {agent: False for agent in env.agents}
            
            while not all(done.values()):
                actions = maddpg.select_actions(state)
                next_state, rewards, done, _ = env.step(actions)
                
                episode_reward += sum(rewards.values())
                episode_steps += 1
                if any(r > 0 for r in rewards.values()):  # Assuming positive reward means capture
                    captures += 1
                    
                state = next_state
            
            # Log episode results
            avg_agent_reward = episode_reward / len(env.agents)
            f.write(f"\nEpisode {episode + 1}/{num_episodes}\n")
            f.write(f"Total Reward: {episode_reward:.2f}\n")
            f.write(f"Average Agent Reward: {avg_agent_reward:.2f}\n")
            f.write(f"Captures: {captures}\n")
            f.write(f"Episode Duration: {episode_steps} steps\n")
            
            # Store metrics
            total_rewards.append(episode_reward)
            agent_rewards.append(avg_agent_reward)
            captures_list.append(captures)
            episode_lengths.append(episode_steps)
        
        # Write summary statistics
        f.write("\nEvaluation Results:\n")
        f.write("-" * 17 + "\n")
        f.write(f"Average Total Reward: {np.mean(total_rewards):.2f}\n")
        f.write(f"Average Agent Reward per Episode: {np.mean(agent_rewards):.2f}\n")
        f.write(f"Average Captures per Episode: {np.mean(captures_list):.1f}\n")
        f.write(f"Average Episode Duration: {np.mean(episode_lengths):.1f} steps\n")
        f.write(f"Standard Deviation of Rewards: {np.std(total_rewards):.2f}\n")
        f.write(f"Best Episode Reward: {max(total_rewards):.2f}\n")
        f.write(f"Worst Episode Reward: {min(total_rewards):.2f}\n")

if __name__ == "__main__":
    from environment import make_env
    from marl_agents.maddpg_algorithm import MADDPG
    
    env = make_env(render_mode=None)
    maddpg = MADDPG(env)
    evaluate_model(env, maddpg)