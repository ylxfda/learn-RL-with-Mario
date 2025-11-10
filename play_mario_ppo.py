"""
Play Super Mario Bros with Trained PPO Agent

This script loads a trained PPO model and plays Mario in real-time with visualization.
Optionally save each episode as an animated GIF.

Usage:
    # Play with best checkpoint
    python play_mario_ppo.py --logdir logdir/mario_ppo --episodes 5

    # Play with latest checkpoint
    python play_mario_ppo.py --logdir logdir/mario_ppo --checkpoint latest.pt

    # Play with stochastic policy (exploration mode)
    python play_mario_ppo.py --logdir logdir/mario_ppo --stochastic

    # Save episodes as GIFs
    python play_mario_ppo.py --logdir logdir/mario_ppo --episodes 5 --save-gif
"""

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="moviepy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
from ruamel.yaml import YAML
from PIL import Image

# Add project to path
sys.path.append(str(pathlib.Path(__file__).parent))

from PPO.ppo_agent import PPOAgent
from envs.mario import MarioEnv


class Config:
    """Configuration wrapper"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def save_episode_gif(
    frames: list,
    save_path: pathlib.Path,
    duration: int = 50,
    loop: int = 0
):
    """
    Save a list of frames as an animated GIF

    Args:
        frames: List of numpy arrays (H, W, C) representing RGB frames
        save_path: Path where to save the GIF file
        duration: Duration of each frame in milliseconds (default: 50ms = 20 FPS)
        loop: Number of loops (0 = infinite loop, default: 0)
    """
    if not frames:
        print(f"Warning: No frames to save for {save_path}")
        return

    # Convert numpy arrays to PIL Images
    pil_frames = []
    for frame in frames:
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        # Convert to PIL Image
        pil_frame = Image.fromarray(frame)
        pil_frames.append(pil_frame)

    # Save as animated GIF
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
        optimize=False  # Set to True to reduce file size (but slower)
    )

    print(f"Saved GIF with {len(frames)} frames to: {save_path}")


def play_mario(
    logdir: str,
    num_episodes: int = 5,
    checkpoint: str = "best.pt",
    deterministic: bool = True,
    render: bool = True,
    verbose: bool = True,
    frame_delay: float = 0.02,
    save_gif: bool = False,
    gif_dir: str = None,
    gif_fps: int = 20,
    max_episode_steps: int = 2000
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
        save_gif: Save each episode as an animated GIF (default: False)
        gif_dir: Directory to save GIF files (default: logdir/gifs)
        gif_fps: Frames per second for GIF animation (default: 20)
        max_episode_steps: Maximum steps per episode before timeout (default: 2000)
    """
    logdir = pathlib.Path(logdir).expanduser()

    # Setup GIF saving directory
    if save_gif:
        if gif_dir is None:
            gif_dir = logdir / "gifs"
        else:
            gif_dir = pathlib.Path(gif_dir).expanduser()

        gif_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory for this run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        gif_run_dir = gif_dir / f"run_{timestamp}"
        gif_run_dir.mkdir(parents=True, exist_ok=True)
    else:
        gif_run_dir = None

    if verbose:
        print("=" * 80)
        print(f"Playing Mario with trained PPO agent")
        print("=" * 80)
        print(f"Logdir: {logdir}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Episodes: {num_episodes}")
        print(f"Policy: {'deterministic (greedy)' if deterministic else 'stochastic (sampling)'}")
        if save_gif:
            print(f"Saving GIFs: {gif_run_dir} ({gif_fps} FPS)")
        print("=" * 80)

    # Load configuration
    config_path = logdir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    yaml = YAML(typ='safe', pure=True)
    config_dict = yaml.load(config_path)

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

        # Collect frames for GIF
        episode_frames = [] if save_gif else None

        print(f"\nEpisode {ep + 1}/{num_episodes}")

        while not done and episode_length < max_episode_steps:
            # Get action from agent
            action = agent.get_action(obs['image'], deterministic=deterministic)

            # Step environment
            obs, reward, done, info = env.step(action)

            # Capture frame for GIF (use high-resolution rendered screen)
            if save_gif:
                try:
                    # Access the raw screen from gym-super-mario-bros
                    raw_screen = env._env.unwrapped.screen
                    if raw_screen is not None:
                        # raw_screen is already uint8 RGB format
                        episode_frames.append(raw_screen.copy())
                except (AttributeError, KeyError):
                    # Fallback to low-res observation if screen not available
                    if 'image' in obs:
                        frame = obs['image']
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                        episode_frames.append(frame.copy())

            episode_return += reward
            episode_length += 1

        # Check if episode was successful or timed out
        success = info.get('mario_success', False)
        timed_out = episode_length >= max_episode_steps

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_successes.append(float(success))

        # Save episode as GIF
        if save_gif and episode_frames:
            gif_filename = f"episode_{ep+1:03d}_reward_{episode_return:.0f}_steps_{episode_length}.gif"
            gif_path = gif_run_dir / gif_filename
            duration_ms = int(1000 / gif_fps)  # Convert FPS to milliseconds per frame
            save_episode_gif(episode_frames, gif_path, duration=duration_ms, loop=0)

        # Print episode summary
        if success:
            status = "SUCCESS ✓"
        elif timed_out:
            status = "TIMEOUT ⏱"
        else:
            status = "FAILED ✗"
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
        default=0.015,
        help="Delay between frames in seconds (default: 0.015 = ~67 FPS)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Save each episode as an animated GIF (default: False)"
    )
    parser.add_argument(
        "--gif-dir",
        type=str,
        default=None,
        help="Directory to save GIF files (default: logdir/gifs)"
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=20,
        help="Frames per second for GIF animation (default: 20)"
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2000,
        help="Maximum steps per episode before timeout (default: 2000)"
    )

    args = parser.parse_args()

    play_mario(
        logdir=args.logdir,
        num_episodes=args.episodes,
        checkpoint=args.checkpoint,
        deterministic=not args.stochastic,
        render=not args.no_render,
        verbose=not args.quiet,
        frame_delay=args.frame_delay,
        save_gif=args.save_gif,
        gif_dir=args.gif_dir,
        gif_fps=args.gif_fps,
        max_episode_steps=args.max_episode_steps
    )
