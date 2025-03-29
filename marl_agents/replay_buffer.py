import numpy as np
from typing import Dict, List, Tuple
import random
import torch

class MAReplayBuffer:
    """Multi-Agent Replay Buffer for MADDPG algorithm"""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, 
             states: Dict[str, np.ndarray],
             actions: Dict[str, np.ndarray],
             rewards: Dict[str, float],
             next_states: Dict[str, np.ndarray],
             dones: Dict[str, bool]) -> None:
        """
        Store a transition in the buffer
        Args:
            states: Dictionary of states for each agent
            actions: Dictionary of actions for each agent
            rewards: Dictionary of rewards for each agent
            next_states: Dictionary of next states for each agent
            dones: Dictionary of done flags for each agent
        """
        # Validate inputs
        if not all(isinstance(states[agent], np.ndarray) for agent in states):
            states = {agent: np.array(state, dtype=np.float32) 
                     for agent, state in states.items()}
            
        if not all(isinstance(actions[agent], np.ndarray) for agent in actions):
            actions = {agent: np.array(action, dtype=np.float32) 
                      for agent, action in actions.items()}

        # Create transition
        transition = (
            states,
            actions,
            {agent: np.float32(reward) for agent, reward in rewards.items()},
            next_states,
            {agent: bool(done) for agent, done in dones.items()}
        )

        # Add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], ...]:
        """
        Sample a batch of transitions
        Args:
            batch_size: Number of transitions to sample
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each element is a dictionary mapping agent IDs to numpy arrays
        """
        # Sample random transitions
        transitions = random.sample(self.buffer, batch_size)
        
        # Get list of agents (assuming consistent across transitions)
        agents = list(transitions[0][0].keys())
        
        # Separate transitions into component arrays
        batch = {
            'states': {agent: np.array([t[0][agent] for t in transitions])
                      for agent in agents},
            'actions': {agent: np.array([t[1][agent] for t in transitions])
                       for agent in agents},
            'rewards': {agent: np.array([t[2][agent] for t in transitions])
                       for agent in agents},
            'next_states': {agent: np.array([t[3][agent] for t in transitions])
                           for agent in agents},
            'dones': {agent: np.array([t[4][agent] for t in transitions])
                     for agent in agents}
        }
        
        return (batch['states'], batch['actions'], batch['rewards'],
                batch['next_states'], batch['dones'])

    def can_sample(self, batch_size: int) -> bool:
        """Check if enough transitions are available to sample"""
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        """Return current size of the buffer"""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the replay buffer"""
        self.buffer.clear()
        self.position = 0

    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics"""
        if not self.buffer:
            return {}
        
        # Get list of agents
        agents = list(self.buffer[0][0].keys())
        
        stats = {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity
        }
        
        # Add per-agent reward statistics
        for agent in agents:
            rewards = [t[2][agent] for t in self.buffer]
            stats[f'{agent}_mean_reward'] = float(np.mean(rewards))
            stats[f'{agent}_std_reward'] = float(np.std(rewards))
            
        return stats