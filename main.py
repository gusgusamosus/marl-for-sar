import argparse
import logging
import os
from datetime import datetime
from train_maddpg import train
from evaluate_maddpg import evaluate
from environment import make_env
import json

logger = logging.getLogger(__name__)

def setup_logging(log_dir: str, debug: bool) -> logging.Logger:
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'run.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_config(config: dict, log_dir: str) -> None:
    """Save configuration to JSON file"""
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Search and Rescue using MADDPG")
    
    # Training arguments
    parser.add_argument("--mode", type=str, choices=['train', 'evaluate'], 
                      required=True, help="Mode: train or evaluate")
    parser.add_argument("--episodes", type=int, default=1000,
                      help="Number of episodes")
    parser.add_argument("--render", action="store_true",
                      help="Enable environment rendering")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default='models/maddpg_final.pth',
                      help="Path to load/save model")
    parser.add_argument("--save-interval", type=int, default=100,
                      help="Episode interval to save model")
    
    # Environment arguments
    parser.add_argument("--eval-episodes", type=int, default=10,
                      help="Number of evaluation episodes")
    
    args = parser.parse_args()

    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir, args.debug)
    logger.debug("Starting with arguments: %s", args)
    
    # Save configuration
    config = vars(args)
    save_config(config, log_dir)
    
    # Create environment
    render_mode = 'human' if args.render else None
    env = make_env(render_mode=render_mode)
    env_info = env.get_env_info()
    logger.info(f"Environment Info: {env_info}")

    try:
        if args.mode == 'train':
            logger.info("Starting training...")
            train(
                episodes=args.episodes,
                save_interval=args.save_interval
            )
            logger.info("Training completed!")
            
        elif args.mode == 'evaluate':
            logger.info("Starting evaluation...")
            evaluate(
                model_path=args.model_path,
                num_episodes=args.eval_episodes,
                render=args.render,
                debug=args.debug
            )
            logger.info("Evaluation completed!")
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise
        
    finally:
        env.close()

if __name__ == "__main__":
    main()