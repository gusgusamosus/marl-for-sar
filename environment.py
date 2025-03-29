import logging
import numpy as np
from pettingzoo.sisl import pursuit_v4
from typing import Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class PursuitEnvironment:
    """Wrapper class for the Pursuit environment with additional functionality"""
    
    def __init__(self, render_mode: Optional[str] = None):
        """Initialize the pursuit environment"""
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        self.logger.debug("Initializing PursuitEnvironment...")
        
        # Initialize environment
        self.env = pursuit_v4.parallel_env(render_mode=render_mode)
        
        # Reset to get initial observation space
        self.env.reset()
        
        # Store agent information
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        
        # Set up observation and action spaces
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        
        # Calculate dimensions for each agent
        self.state_dims = {}
        self.action_dims = {}
        for agent in self.agents:
            self.state_dims[agent] = int(np.prod(self.observation_spaces[agent].shape))
            self.action_dims[agent] = self.action_spaces[agent].n
            self.logger.debug(f"Agent {agent} - State dim: {self.state_dims[agent]}, Action dim: {self.action_dims[agent]}")
            
        self.logger.debug(f"Environment initialized with {self.num_agents} agents")
        
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment"""
        logger.debug("Resetting environment...")
        observations, info = self.env.reset()
        
        # Flatten observations to 1D arrays
        obs_dict = {}
        for agent in self.agents:
            if isinstance(observations, tuple):
                obs = np.array(observations[self.agents.index(agent)], dtype=np.float32)
            else:
                obs = np.array(observations[agent], dtype=np.float32)
            # Reshape to 1D array with correct size (147)
            obs_dict[agent] = obs.flatten()
            logger.debug(f"Reset observation shape for {agent}: {obs_dict[agent].shape}")
        
        return obs_dict, info

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict]:
        """
        Take a step in the environment
        Args:
            actions: Dictionary of actions for each agent
        Returns:
            Tuple of (observations, rewards, dones, info)
        """
        next_obs, rewards, dones, truncated, infos = self.env.step(actions)
        
        # Convert observations to dictionary with proper numpy arrays
        next_obs_dict = {}
        for i, agent in enumerate(self.agents):
            if isinstance(next_obs, tuple):
                next_obs_dict[agent] = np.array(next_obs[i], dtype=np.float32)
            else:
                next_obs_dict[agent] = np.array(next_obs[agent], dtype=np.float32)
        
        # Combine truncated with dones
        dones = {agent: dones[agent] or truncated[agent] 
                for agent in self.agents}
        
        return next_obs_dict, rewards, dones, infos
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment"""
        return self.env.render()
    
    def close(self) -> None:
        """Close the environment"""
        self.env.close()
        
    def get_env_info(self) -> Dict:
        """Get environment information"""
        return {
            'num_agents': self.num_agents,
            'state_dims': self.state_dims,
            'action_dims': self.action_dims,
            'agents': self.agents
        }

def make_env(render_mode: Optional[str] = None) -> PursuitEnvironment:
    """Create and return a wrapped Pursuit environment"""
    logger.debug(f"Creating environment with render_mode: {render_mode}")
    return PursuitEnvironment(render_mode=render_mode)