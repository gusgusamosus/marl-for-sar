import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super(Actor, self).__init__()
        
        # Debug input dimensions
        logger.debug(f"Actor input dim: {state_dim}, output dim: {action_dim}")
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with stable softmax"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        x = self.network(state)
        # Add small epsilon to prevent numerical instability
        action_logits = self.action_head(x)
        return F.softmax(action_logits, dim=-1) + 1e-6

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_agents: int, 
                 hidden_dims: List[int] = [256, 128, 64]):  # Deeper network
        super(Critic, self).__init__()
        input_dim = (state_dim + action_dim) * n_agents
        
        # Add batch normalization
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(),  # Changed to LeakyReLU
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=1)
        x = self.batch_norm(x)  # Add batch normalization
        x = self.network(x)
        return self.value_head(x)

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class MADDPG:
    def __init__(self, env, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        logger.debug("Initializing MADDPG...")
        self.env = env
        self.device = device
        
        # Get environment properties
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.state_dims = env.state_dims
        self.action_dims = env.action_dims
        
        logger.debug(f"State dimensions: {self.state_dims}")
        logger.debug(f"Action dimensions: {self.action_dims}")
        
        # Initialize networks
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for agent in self.agents:
            state_dim = self.state_dims[agent]
            action_dim = self.action_dims[agent]
            
            logger.debug(f"Creating networks for agent {agent}")
            logger.debug(f"State dim: {state_dim}, Action dim: {action_dim}")
            
            # Create actor network
            actor = Actor(state_dim, action_dim).to(device)
            self.actors.append(actor)
            self.target_actors.append(copy.deepcopy(actor))
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=1e-4))
            
            # Create critic network
            critic = Critic(
                sum(self.state_dims.values()),
                sum(self.action_dims.values()),
                self.num_agents
            ).to(device)
            self.critics.append(critic)
            self.target_critics.append(copy.deepcopy(critic))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=1e-3))
        
        logger.debug("MADDPG initialization complete")
        
        self.logger = logging.getLogger(__name__)
        
        if not hasattr(env, 'num_agents'):
            self.logger.error("Environment missing num_agents attribute!")
            raise AttributeError("Environment must have num_agents attribute")
            
        if not hasattr(env, 'state_dims'):
            self.logger.error("Environment missing state_dims attribute!")
            raise AttributeError("Environment must have state_dims attribute")
            
        if not hasattr(env, 'action_dims'):
            self.logger.error("Environment missing action_dims attribute!")
            raise AttributeError("Environment must have action_dims attribute")
        
        self.n_agents = env.num_agents
        self.batch_size = 64
        
        self.logger.debug(f"Number of agents: {self.n_agents}")
        self.logger.debug(f"State dimensions: {env.state_dims}")
        self.logger.debug(f"Action dimensions: {env.action_dims}")
        
        self.logger.debug("MADDPG initialization complete")
        
        # Debug logging
        self.debug_info = {
            'actor_losses': [],
            'critic_losses': [],
            'rewards': []
        }
        
        self.logger.debug(f"Initialized MADDPG with {self.n_agents} agents")
        self.logger.debug(f"State dims: {self.state_dims}")
        self.logger.debug(f"Action dims: {self.action_dims}")
        
        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # soft update parameter
        self.initial_noise = 0.3  # initial exploration noise
        self.exploration_noise = self.initial_noise  # current exploration noise
        self.noise_decay = 0.99995  # noise decay rate
        self.min_noise = 0.01  # minimum exploration noise

        logger.debug(f"Initial exploration noise: {self.exploration_noise}")

    def select_action(self, state: np.ndarray, agent_id: int, evaluate: bool = False) -> np.ndarray:
        """Select action for the specified agent"""
        # Ensure state is flattened to 1D array
        state = state.flatten()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actors[agent_id](state_tensor).cpu().numpy()[0]
        
        if evaluate:
            return np.argmax(action_probs)
        
        # Add exploration noise with safety checks
        noise = np.random.normal(0, self.exploration_noise, self.action_dims[self.agents[agent_id]])
        action_probs = action_probs + noise
        
        # Ensure valid probability distribution
        action_probs = np.clip(action_probs, 1e-6, None)  # Clip to small positive values
        action_probs_sum = action_probs.sum()
        
        if action_probs_sum <= 0 or np.isnan(action_probs_sum):
            # If probabilities are invalid, use uniform distribution
            logger.warning(f"Invalid action probabilities encountered for agent {agent_id}, using uniform distribution")
            action_probs = np.ones(self.action_dims[self.agents[agent_id]]) / self.action_dims[self.agents[agent_id]]
        else:
            # Normalize probabilities
            action_probs = action_probs / action_probs_sum
        
        # Decay exploration noise
        self.exploration_noise = max(
            self.min_noise, 
            self.exploration_noise * self.noise_decay
        )
        
        # Verify probabilities sum to 1
        assert np.isclose(action_probs.sum(), 1.0), "Action probabilities must sum to 1"
        assert not np.any(np.isnan(action_probs)), "NaN values in action probabilities"
        
        return np.random.choice(self.action_dims[self.agents[agent_id]], p=action_probs)
    
    def update(self, replay_buffer) -> Tuple[float, float]:
        if len(replay_buffer) < self.batch_size:
            return 0.0, 0.0
            
        try:
            # Sample from replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
            
            # Convert to tensors
            def process_batch(batch: Dict[str, np.ndarray], agent_id: int) -> torch.Tensor:
                return torch.FloatTensor(
                    np.stack([batch[agent] for agent in self.env.agents])
                ).to(self.device)
            
            state_batch = {i: process_batch(states, i) for i in range(self.n_agents)}
            action_batch = {i: process_batch(actions, i) for i in range(self.n_agents)}
            reward_batch = {i: process_batch(rewards, i) for i in range(self.n_agents)}
            next_state_batch = {i: process_batch(next_states, i) for i in range(self.n_agents)}
            done_batch = {i: process_batch(dones, i) for i in range(self.n_agents)}
            
            critic_losses = []
            actor_losses = []
            
            # Update each agent
            for agent_id in range(self.n_agents):
                # Compute target actions
                next_actions = []
                for i, target_actor in enumerate(self.target_actors):
                    next_agent_action = target_actor(next_state_batch[i])
                    next_actions.append(next_agent_action)
                next_actions = torch.cat(next_actions, dim=1)
                
                # Compute target Q-value
                next_states_combined = torch.cat([next_state_batch[i] 
                                               for i in range(self.n_agents)], dim=1)
                target_q = reward_batch[agent_id] + \
                          (1 - done_batch[agent_id]) * self.gamma * \
                          self.target_critics[agent_id](next_states_combined, next_actions)
                
                # Compute current Q-value
                current_actions = torch.cat([action_batch[i] for i in range(self.n_agents)], dim=1)
                current_states = torch.cat([state_batch[i] for i in range(self.n_agents)], dim=1)
                current_q = self.critics[agent_id](current_states, current_actions)
                
                # Compute critic loss
                critic_loss = F.mse_loss(current_q, target_q.detach())
                critic_losses.append(critic_loss.item())
                
                # Update critic
                self.critic_optimizers[agent_id].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 0.5)
                self.critic_optimizers[agent_id].step()
                
                # Compute actor loss
                actions_for_actor = []
                for i in range(self.n_agents):
                    if i == agent_id:
                        actions_for_actor.append(self.actors[i](state_batch[i]))
                    else:
                        actions_for_actor.append(action_batch[i].detach())
                actions_combined = torch.cat(actions_for_actor, dim=1)
                
                actor_loss = -self.critics[agent_id](current_states, actions_combined).mean()
                actor_losses.append(actor_loss.item())
                
                # Update actor
                self.actor_optimizers[agent_id].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 0.5)
                self.actor_optimizers[agent_id].step()
                
                # Update target networks
                self._soft_update(self.critics[agent_id], self.target_critics[agent_id])
                self._soft_update(self.actors[agent_id], self.target_actors[agent_id])
                
            # Add debug logging
            self.logger.debug(f"Actor Loss: {np.mean(actor_losses):.4f}")
            self.logger.debug(f"Critic Loss: {np.mean(critic_losses):.4f}")
            
            # Store losses for debugging
            self.debug_info['actor_losses'].append(np.mean(actor_losses))
            self.debug_info['critic_losses'].append(np.mean(critic_losses))
            
            return np.mean(critic_losses), np.mean(actor_losses)
            
        except Exception as e:
            self.logger.error(f"Error in update: {str(e)}")
            raise
    
    def _soft_update(self, local_model: nn.Module, target_model: nn.Module) -> None:
        """Soft update target network parameters"""
        for target_param, local_param in zip(target_model.parameters(), 
                                           local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save_model(self, path: str) -> None:
        """Save model parameters"""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [target.state_dict() for target in self.target_actors],
            'target_critics': [target.state_dict() for target in self.target_critics],
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, state_dict in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(state_dict)
            
        for i, state_dict in enumerate(checkpoint['critics']):
            self.critics[i].load_state_dict(state_dict)
            
        for i, state_dict in enumerate(checkpoint['target_actors']):
            self.target_actors[i].load_state_dict(state_dict)
            
        for i, state_dict in enumerate(checkpoint['target_critics']):
            self.target_critics[i].load_state_dict(state_dict)