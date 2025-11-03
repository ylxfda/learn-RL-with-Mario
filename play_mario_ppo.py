"""
Play Super Mario Bros with Trained PPO Agent

This script loads a trained PPO model and plays Mario in real-time with visualization.

Usage:
    # Play with best checkpoint
    python play_mario_ppo.py --logdir logdir/mario_ppo --episodes 5

    # Play with latest checkpoint
    python play_mario_ppo.py --logdir logdir/mario_ppo --checkpoint latest.pt

    # Play with stochastic policy (exploration mode)
    python play_mario_ppo.py --logdir logdir/mario_ppo --stochastic
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
import ruamel.yaml as yaml

# Add project to path
sys.path.append(str(pathlib.Path(__file__).parent))

from PPO.ppo_agent import PPOAgent
from envs.mario import MarioEnv


class Config:
    """Configuration wrapper"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def play_mario(
    logdir: str,
    num_episodes: int = 5,
    checkpoint: str = "best.pt",
    deterministic: bool = True,
    render: bool = True,
    verbose: bool = True,
    frame_delay: float = 0.02
):
    """
    Play Mario with trained PPO agent

    Args:
        logdir: Directory containing trained model
        num_episodes: Number of episodes to play
        checkpoint: Checkpoint filename ('best.pt' or 'latest.pt')
        deterministic: Use deterministic (greedy) actions vs stochastic sampling
        render: Whether to render the game window
        verbose: Print detailed info
        frame_delay: Delay in seconds between frames (0.02 = ~50 FPS, 0.05 = ~20 FPS)
    """
    logdir = pathlib.Path(logdir).expanduser()

    if verbose:
        print("=" * 80)
        print(f"Playing Mario with trained PPO agent")
        print("=" * 80)
        print(f"Logdir: {logdir}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Episodes: {num_episodes}")
        print(f"Policy: {'deterministic (greedy)' if deterministic else 'stochastic (sampling)'}")
        print("=" * 80)

    # Load configuration
    config_path = logdir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = Config(config_dict)

    # Override render mode for demo
    if render:
        config.render_mode = 'human'
        config.frame_delay = frame_delay

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # Get observation shape
    if config.grayscale:
        obs_shape = (1, config.size[0], config.size[1])
    else:
        obs_shape = (3, config.size[0], config.size[1])

    # Load checkpoint
    checkpoint_path = logdir / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if verbose:
        print(f"\nLoading checkpoint from: {checkpoint_path}")

    # Create PPO agent
    agent = PPOAgent(
        obs_shape=obs_shape,
        num_actions=config.num_actions,
        config=config,
        device=device
    )

    # Load weights
    agent.load(str(checkpoint_path))
    print("✓ Model loaded successfully")

    # Create environment
    if verbose:
        print("\nCreating Mario environment...")

    env = MarioEnv(
        level="SuperMarioBros-1-1-v0",
        action_repeat=config.action_repeat,
        size=tuple(config.size),
        grayscale=config.grayscale,
        action_set=getattr(config, "mario_action_set", "simple"),
        resize_method=getattr(config, "resize", "opencv"),
        flag_reward=getattr(config, "mario_flag_reward", 1000.0),
        reward_scale=getattr(config, "mario_reward_scale", 1.0),
        time_penalty=getattr(config, "mario_time_penalty", -0.1),
        death_penalty=getattr(config, "mario_death_penalty", -15.0),
        seed=config.seed + 9999,  # Different seed for playing
        render_mode='human' if render else None,
        frame_delay=frame_delay if render else 0.0
    )

    print("✓ Environment created")

    # Play episodes
    print("\n" + "=" * 80)
    print("PLAYING")
    print("=" * 80)

    episode_returns = []
    episode_lengths = []
    episode_successes = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_return = 0
        episode_length = 0

        print(f"\nEpisode {ep + 1}/{num_episodes}")

        while not done:
            # Get action from agent
            action = agent.get_action(obs['image'], deterministic=deterministic)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1

        # Check if episode was successful
        success = info.get('mario_success', False)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_successes.append(float(success))

        # Print episode summary
        status = "SUCCESS ✓" if success else "FAILED ✗"
        print(f"  {status}")
        print(f"  Return: {episode_return:.2f}")
        print(f"  Length: {episode_length}")

    # Print overall statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")
    print(f"Mean Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f}")
    print(f"Success Rate: {np.mean(episode_successes):.1%}")
    print(f"Best Return: {np.max(episode_returns):.2f}")
    print(f"Worst Return: {np.min(episode_returns):.2f}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Mario with trained PPO agent")
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint to load (default: best.pt)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (run headless)"
    )
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.02,
        help="Delay between frames in seconds (default: 0.02 = ~50 FPS)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    play_mario(
        logdir=args.logdir,
        num_episodes=args.episodes,
        checkpoint=args.checkpoint,
        deterministic=not args.stochastic,
        render=not args.no_render,
        verbose=not args.quiet,
        frame_delay=args.frame_delay
    )
