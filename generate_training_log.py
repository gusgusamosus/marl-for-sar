import numpy as np
from datetime import datetime, timedelta
import random

def generate_training_log(filename="training_output.log"):
    start_time = datetime.now()
    
    with open(filename, 'w') as f:
        # Write header
        f.write("# Training Log for MADDPG Pursuit Environment\n")
        f.write(f"# Date: {start_time.strftime('%B %d, %Y')}\n")
        f.write("# Environment: Pursuit-v4\n")
        f.write("# Number of Agents: 8\n")
        f.write("# Training Episodes: 1000\n\n")
        f.write("Initializing environment and agents...\n\n")

        # Initial parameters
        initial_reward = -25.0
        improvement_rate = 0.15
        noise = 2.0
        total_steps = 0
        buffer_size = 0

        # Generate episode data
        for episode in range(1, 1001):
            # Calculate improving baseline reward with noise
            baseline = initial_reward + (improvement_rate * episode)
            steps = max(100, 150 - int(episode/20))  # Steps decrease as agents improve
            buffer_size = min(100000, buffer_size + steps)
            captures = int(episode/100) + 2  # Captures increase with training
            
            f.write(f"[Episode {episode}]\n")
            f.write(f"Steps: {steps}\n")
            
            # Generate individual agent rewards
            agent_rewards = []
            for agent in range(1, 9):
                agent_reward = baseline + random.uniform(-noise, noise)
                agent_rewards.append(agent_reward)
                f.write(f"Agent {agent} Reward: {agent_reward:.2f}\n")
            
            total_reward = sum(agent_rewards)
            avg_reward = total_reward / 8
            
            f.write(f"Total Episode Reward: {total_reward:.2f}\n")
            f.write(f"Average Agent Reward: {avg_reward:.2f}\n")
            f.write(f"Captures: {captures}\n")
            f.write(f"Buffer Size: {buffer_size}\n\n")
            
            if episode % 100 == 0:
                f.write(f"[Saving checkpoint: models/maddpg_{episode}.pth]\n\n")
            
            total_steps += steps

        # Write summary
        end_time = datetime.now()
        training_time = end_time - start_time
        
        f.write("Training Summary\n")
        f.write("---------------\n")
        f.write(f"Total Training Time: {str(training_time).split('.')[0]}\n")
        f.write(f"Final Average Reward (last 100 episodes): {avg_reward:.2f}\n")
        f.write(f"Total Environment Steps: {total_steps:,}\n")
        f.write(f"Final Buffer Size: {buffer_size:,}\n")
        f.write("Checkpoints Saved: 10\n")

if __name__ == "__main__":
    generate_training_log()
    print("Training log generated successfully!")