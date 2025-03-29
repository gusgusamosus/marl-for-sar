import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def read_training_data(filename="training_output.log"):
    """Extract training metrics from log file."""
    episodes = []
    total_rewards = []
    avg_rewards = []
    captures = []
    agent_rewards = {i: [] for i in range(1, 9)}
    
    try:
        with open(filename, 'r') as f:
            current_episode_rewards = {}
            
            for line in f:
                line = line.strip()
                
                # Skip empty lines and header
                if not line or line.startswith('#'):
                    continue
                    
                # Extract episode number
                if line.startswith('[Episode'):
                    try:
                        episode_num = int(re.search(r'Episode (\d+)', line).group(1))
                        episodes.append(episode_num)
                    except (AttributeError, ValueError):
                        continue
                
                # Extract agent rewards
                elif line.startswith('Agent'):
                    try:
                        parts = line.split()
                        agent_id = int(parts[1])
                        reward = float(parts[3])
                        agent_rewards[agent_id].append(reward)
                    except (IndexError, ValueError):
                        continue
                
                # Extract total reward
                elif line.startswith('Total Episode Reward:'):
                    try:
                        reward = float(line.split(':')[1].strip())
                        total_rewards.append(reward)
                    except (IndexError, ValueError):
                        continue
                
                # Extract average reward
                elif line.startswith('Average Agent Reward:'):
                    try:
                        reward = float(line.split(':')[1].strip())
                        avg_rewards.append(reward)
                    except (IndexError, ValueError):
                        continue
                
                # Extract captures
                elif line.startswith('Captures:'):
                    try:
                        capture = int(line.split(':')[1].strip())
                        captures.append(capture)
                    except (IndexError, ValueError):
                        continue
        
        # Verify data consistency
        min_len = min(len(episodes), len(total_rewards), len(avg_rewards), len(captures))
        
        return {
            'episodes': episodes[:min_len],
            'total_rewards': total_rewards[:min_len],
            'avg_rewards': avg_rewards[:min_len],
            'captures': captures[:min_len],
            'agent_rewards': {k: v[:min_len] for k, v in agent_rewards.items()}
        }
        
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        return None
    except Exception as e:
        print(f"Error reading training data: {e}")
        return None

def read_eval_data(filename="evaluation_output.log"):
    """Extract evaluation metrics from log file."""
    eval_data = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Average Agent Reward:" in line:
                eval_data['avg_reward'] = float(line.split(':')[1].strip())
            elif "Average Captures per Episode:" in line:
                eval_data['avg_captures'] = float(line.split(':')[1].strip())
            elif "First 10 Episodes Average Reward:" in line:
                eval_data['first_10_avg'] = float(line.split(':')[1].strip())
            elif "Last 10 Episodes Average Reward:" in line:
                eval_data['last_10_avg'] = float(line.split(':')[1].strip())
            elif "Overall Improvement:" in line:
                eval_data['improvement'] = float(line.split(':')[1].strip())
    
    return eval_data

def create_performance_plots(train_data, eval_data):
    """Generate comprehensive performance plots."""
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Set plotting style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # 1. Training Rewards with Evaluation Reference
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['episodes'], train_data['total_rewards'], 
             label='Training Rewards', alpha=0.7)
    plt.axhline(y=eval_data['avg_reward'], color='r', linestyle='--', 
                label='Evaluation Average')
    plt.title('Training Rewards Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_rewards.png')
    plt.close()
    
    # 2. Agent-wise Performance
    plt.figure(figsize=(12, 6))
    for agent_id, rewards in train_data['agent_rewards'].items():
        plt.plot(train_data['episodes'], rewards, 
                 label=f'Agent {agent_id}', alpha=0.6)
    plt.title('Individual Agent Performance')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/agent_performance.png')
    plt.close()
    
    # 3. Captures Progress
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['episodes'], train_data['captures'], 
             label='Training Captures', color='green')
    plt.axhline(y=eval_data['avg_captures'], color='r', linestyle='--', 
                label='Evaluation Average')
    plt.title('Capture Performance')
    plt.xlabel('Episode')
    plt.ylabel('Number of Captures')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/captures.png')
    plt.close()
    
    # 4. Moving Average of Rewards
    window = 20
    moving_avg = np.convolve(train_data['avg_rewards'], 
                            np.ones(window)/window, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['episodes'][window-1:], moving_avg, 
             label=f'Moving Average (window={window})', color='blue')
    plt.title('Smoothed Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/smoothed_performance.png')
    plt.close()
    
    # 5. Training Progress Summary
    plt.figure(figsize=(12, 6))
    episodes = train_data['episodes']
    rewards = train_data['avg_rewards']
    
    plt.plot(episodes, rewards, label='Training', alpha=0.5)
    plt.scatter([episodes[0], episodes[-1]], 
                [eval_data['first_10_avg'], eval_data['last_10_avg']], 
                color='red', s=100, label='Evaluation Checkpoints')
    plt.annotate(f"Improvement: {eval_data['improvement']:.2f}", 
                xy=(episodes[-1], eval_data['last_10_avg']),
                xytext=(10, 10), textcoords='offset points')
    
    plt.title('Training Progress Summary')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_summary.png')
    plt.close()

def main():
    try:
        # Read data
        train_data = read_training_data()
        if train_data is None:
            raise ValueError("Failed to read training data")
            
        eval_data = read_eval_data()
        if eval_data is None:
            raise ValueError("Failed to read evaluation data")
        
        # Generate plots
        create_performance_plots(train_data, eval_data)
        
        print("Performance plots have been generated in the results/ directory:")
        print("- training_rewards.png")
        print("- agent_performance.png")
        print("- captures.png")
        print("- smoothed_performance.png")
        print("- training_summary.png")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find log files - {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()