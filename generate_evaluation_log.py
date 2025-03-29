import os
from datetime import datetime
import numpy as np
import re

def parse_episode_data(lines, start_idx):
    """Parse data for a single episode from the log file."""
    data = {'rewards': [], 'steps': 0, 'captures': 0}
    i = start_idx
    
    while i < len(lines) and not lines[i].startswith('[Episode'):
        line = lines[i].strip()
        if line.startswith('Steps:'):
            data['steps'] = int(line.split(': ')[1])
        elif line.startswith('Agent') and 'Reward:' in line:
            reward = float(line.split(': ')[1])
            data['rewards'].append(reward)
        elif line.startswith('Captures:'):
            data['captures'] = int(line.split(': ')[1])
        i += 1
    
    return data, i - 1

def generate_evaluation_log(training_log="training_output.log"):
    """Generate evaluation log from training data"""
    
    # Read training log
    with open(training_log, 'r') as f:
        lines = f.readlines()
    
    # Extract metadata
    date_line = next(line for line in lines if "# Date:" in line)
    date = date_line.split(': ')[1].strip()
    
    # Parse episodes
    episodes_data = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('[Episode'):
            episode_data, last_idx = parse_episode_data(lines, i+1)
            episodes_data.append(episode_data)
            i = last_idx
        i += 1
    
    # Calculate statistics
    num_episodes = len(episodes_data)
    total_steps = sum(ep['steps'] for ep in episodes_data)
    avg_steps = total_steps / num_episodes
    
    # Calculate reward statistics 
    all_rewards = [reward for ep in episodes_data for reward in ep['rewards']]
    avg_reward = np.mean(all_rewards)
    min_reward = np.min(all_rewards)
    max_reward = np.max(all_rewards)
    std_reward = np.std(all_rewards)
    
    # Calculate improvement metrics
    first_10_avg = np.mean([np.mean(ep['rewards']) for ep in episodes_data[:10]])
    last_10_avg = np.mean([np.mean(ep['rewards']) for ep in episodes_data[-10:]])
    improvement = last_10_avg - first_10_avg
    
    # Calculate capture statistics
    total_captures = sum(ep['captures'] for ep in episodes_data)
    avg_captures = total_captures / num_episodes
    
    # Write evaluation report
    output_file = 'evaluation_output.log'
    
    with open(output_file, 'w') as f:
        f.write("MADDPG Training Evaluation Report\n")
        f.write("===============================\n\n")
        f.write(f"Training Date: {date}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%B %d, %Y')}\n\n")
        
        f.write("Episode Statistics\n")
        f.write("-----------------\n")
        f.write(f"Total Episodes: {num_episodes}\n")
        f.write(f"Total Environment Steps: {total_steps:,}\n")
        f.write(f"Average Steps per Episode: {avg_steps:.2f}\n\n")
        
        f.write("Reward Statistics\n")
        f.write("----------------\n")
        f.write(f"Average Agent Reward: {avg_reward:.2f}\n")
        f.write(f"Minimum Reward: {min_reward:.2f}\n")
        f.write(f"Maximum Reward: {max_reward:.2f}\n")
        f.write(f"Reward Standard Deviation: {std_reward:.2f}\n\n")
        
        f.write("Performance Metrics\n")
        f.write("-----------------\n")
        f.write(f"First 10 Episodes Average Reward: {first_10_avg:.2f}\n")
        f.write(f"Last 10 Episodes Average Reward: {last_10_avg:.2f}\n")
        f.write(f"Overall Improvement: {improvement:.2f}\n")
        f.write(f"Average Captures per Episode: {avg_captures:.2f}\n\n")
        
        f.write("Training Progress\n")
        f.write("----------------\n")
        f.write("Episode Ranges    | Avg Reward | Avg Captures\n")
        f.write("-" * 45 + "\n")
        
        # Show progress in chunks of episodes
        chunk_size = num_episodes // 5
        for i in range(0, num_episodes, chunk_size):
            chunk = episodes_data[i:i+chunk_size]
            chunk_avg_reward = np.mean([np.mean(ep['rewards']) for ep in chunk])
            chunk_avg_captures = np.mean([ep['captures'] for ep in chunk])
            f.write(f"Episodes {i+1}-{i+chunk_size:4d} | {chunk_avg_reward:10.2f} | {chunk_avg_captures:12.2f}\n")

    print(f"Evaluation report generated successfully: {output_file}")

if __name__ == "__main__":
    generate_evaluation_log()