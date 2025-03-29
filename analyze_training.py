import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Dict
import os

def parse_training_log(log_file: str) -> Tuple[List[float], List[float], List[int], List[int]]:
    """Parse the training log file and extract metrics."""
    episode_rewards = []
    avg_rewards = []
    captures = []
    buffer_sizes = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    current_episode_data = {}
    for line in lines:
        if "Total Episode Reward:" in line:
            reward = float(line.split(":")[1].strip())
            episode_rewards.append(reward)
        elif "Average Agent Reward:" in line:
            avg = float(line.split(":")[1].strip())
            avg_rewards.append(avg)
        elif "Captures:" in line:
            cap = int(line.split(":")[1].strip())
            captures.append(cap)
        elif "Buffer Size:" in line:
            size = int(line.split(":")[1].strip())
            buffer_sizes.append(size)
    
    return episode_rewards, avg_rewards, captures, buffer_sizes

def plot_metrics(metrics: Dict[str, List[float]], save_dir: str = "results"):
    """Plot training metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['episode_rewards'], label='Total Episode Reward')
    plt.plot(metrics['avg_rewards'], label='Average Agent Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/rewards.png")
    plt.close()
    
    # Plot captures
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['captures'], label='Captures per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Captures')
    plt.title('Capture Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/captures.png")
    plt.close()
    
    # Plot buffer size
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['buffer_sizes'], label='Replay Buffer Size')
    plt.xlabel('Episode')
    plt.ylabel('Buffer Size')
    plt.title('Replay Buffer Growth')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/buffer_size.png")
    plt.close()

def analyze_performance(metrics: Dict[str, List[float]], window: int = 100) -> None:
    """Analyze and print performance statistics."""
    # Calculate moving averages
    reward_ma = np.convolve(metrics['episode_rewards'], 
                           np.ones(window)/window, 
                           mode='valid')
    
    print("\nPerformance Analysis:")
    print("-" * 50)
    print(f"Total Episodes: {len(metrics['episode_rewards'])}")
    print(f"Final Average Reward (last {window} episodes): {reward_ma[-1]:.2f}")
    print(f"Peak Average Reward: {max(reward_ma):.2f}")
    print(f"Final Number of Captures: {metrics['captures'][-1]}")
    print(f"Final Buffer Size: {metrics['buffer_sizes'][-1]:,}")
    
    # Calculate improvement
    first_100_avg = np.mean(metrics['episode_rewards'][:100])
    last_100_avg = np.mean(metrics['episode_rewards'][-100:])
    improvement = ((last_100_avg - first_100_avg) / abs(first_100_avg)) * 100
    
    print(f"\nImprovement Analysis:")
    print(f"First 100 Episodes Average: {first_100_avg:.2f}")
    print(f"Last 100 Episodes Average: {last_100_avg:.2f}")
    print(f"Improvement: {improvement:.1f}%")

def main():
    log_file = "training_output.log"
    
    # Parse metrics
    episode_rewards, avg_rewards, captures, buffer_sizes = parse_training_log(log_file)
    
    # Organize metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards,
        'captures': captures,
        'buffer_sizes': buffer_sizes
    }
    
    # Generate plots
    plot_metrics(metrics)
    
    # Analyze performance
    analyze_performance(metrics)

if __name__ == "__main__":
    main()