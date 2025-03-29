import matplotlib.pyplot as plt
import numpy as np

def read_training_log(filename="training_output.log"):
    episodes = []
    rewards = []
    captures = []
    
    with open(filename, 'r') as f:
        for line in f:
            if "Episode" in line and "[" in line:
                episodes.append(int(line.strip('[]').split()[1]))
            elif "Total Episode Reward:" in line:
                rewards.append(float(line.split(':')[1].strip()))
            elif "Captures:" in line:
                captures.append(int(line.split(':')[1].strip()))
                
    return episodes, rewards, captures

def read_eval_log(filename="evaluation_output.log"):
    eval_data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Average Agent Reward:" in line:
                eval_data['avg_reward'] = float(line.split(':')[1].strip())
            elif "Average Captures per Episode:" in line:
                eval_data['avg_captures'] = float(line.split(':')[1].strip())
            elif "Total Episodes:" in line:
                eval_data['total_episodes'] = int(line.split(':')[1].strip())
    return eval_data

def make_plots():
    # Read data
    episodes, rewards, captures = read_training_log()
    eval_data = read_eval_log()
    
    # Create plots directory
    plt.style.use('seaborn')
    
    # Plot 1: Training Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, 'b-')
    plt.axhline(y=eval_data['avg_reward'], color='r', linestyle='--', label='Evaluation Average')
    plt.title('Training Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_rewards.png')
    plt.close()
    
    # Plot 2: Captures
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, captures, 'g-')
    plt.axhline(y=eval_data['avg_captures'], color='r', linestyle='--', label='Evaluation Average')
    plt.title('Captures per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Captures')
    plt.legend()
    plt.grid(True)
    plt.savefig('captures.png')
    plt.close()
    
    # Plot 3: Moving Average of Rewards
    window = 20
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(episodes[window-1:], moving_avg, 'r-')
    plt.title(f'Moving Average of Rewards (Window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig('reward_moving_avg.png')
    plt.close()

if __name__ == "__main__":
    make_plots()
    print("Plots generated: training_rewards.png, captures.png, reward_moving_avg.png")