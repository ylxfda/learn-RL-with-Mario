"""
Play Super Mario Bros with Trained DreamerV3 Agent

This script loads a trained model and plays Mario in real-time with visualization.
Optionally save each episode as an animated GIF.

By default, uses stochastic action sampling for more varied behavior.
Use --action-mode for deterministic (mode) action selection.

Usage:
    # Basic usage (stochastic actions by default)
    python play_mario.py --logdir logdir/mario --episodes 5

    # Use deterministic actions
    python play_mario.py --logdir logdir/mario --episodes 5 --action-mode

    # Save episodes as GIFs
    python play_mario.py --logdir logdir/mario --episodes 5 --save-gif

    # Customize GIF settings
    python play_mario.py --logdir logdir/mario --episodes 5 --save-gif --gif-fps 30 --gif-dir ./my_gifs
"""

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="moviepy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

import argparse
import pathlib
import torch
import numpy as np
from ruamel.yaml import YAML
from PIL import Image
import time

from train_mario_dreamer import DreamerAgent
from envs.mario import make_mario_env
from dreamerv3.utils import tools


class DemoConfig:
    """Configuration wrapper for demo"""
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
    use_mode_for_z: bool = True,
    use_mode_for_action: bool = False,
    render: bool = True,
    verbose: bool = True,
    frame_delay: float = 0.02,
    use_best: bool = True,
    save_gif: bool = False,
    gif_dir: str = None,
    gif_fps: int = 20,
    max_episode_steps: int = 2000
):
    """
    Play Mario with trained agent

    Args:
        logdir: Directory containing trained model
        num_episodes: Number of episodes to play
        use_mode_for_z: Use mode (vs sample) for z (latent state)
        use_mode_for_action: Use mode (vs sample) for action (default: False = sample/stochastic)
        render: Whether to render the game window
        verbose: Print detailed info
        frame_delay: Delay in seconds between frames (0.02 = ~50 FPS, 0.05 = ~20 FPS)
        use_best: Use best.pt checkpoint (default: True, can use latest.pt instead)
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
        print("=" * 60)
        print(f"Playing Mario with trained agent from: {logdir}")
        print(f"Episodes: {num_episodes}")
        print(f"z strategy: {'mode (deterministic)' if use_mode_for_z else 'sample (stochastic)'}")
        print(f"action strategy: {'mode (deterministic)' if use_mode_for_action else 'sample (stochastic)'}")
        if save_gif:
            print(f"Saving GIFs: {gif_run_dir} ({gif_fps} FPS)")
        print("=" * 60)

    # Load configuration
    config_path = logdir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    yaml = YAML(typ='safe', pure=True)
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f)

    config = DemoConfig(config_dict)

    # Override render mode for demo
    if render:
        # Note: gym-super-mario-bros uses 'human' mode for rendering
        config.render_mode = 'human'
        config.frame_delay = frame_delay

    # Load checkpoint
    checkpoint_name = "best.pt" if use_best else "latest.pt"
    checkpoint_path = logdir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if verbose:
        print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Create environment
    if verbose:
        print("Creating Mario environment...")

    env = make_mario_env(config)

    # Get observation and action spaces
    obs_space = env.observation_space
    act_space = env.action_space

    # Set num_actions from action space (needed for WorldModel)
    config.num_actions = act_space.n

    # Create agent (we only need the policy, not training components)
    if verbose:
        print("Creating agent...")

    # Dummy logger and dataset for agent initialization
    class DummyLogger:
        def __init__(self):
            self.step = 0
        def scalar(self, *args, **kwargs): pass
        def write(self, *args, **kwargs): pass

    class DummyDataset:
        def __iter__(self):
            return self
        def __next__(self):
            # Return dummy batch (not used during play)
            return {
                'image': torch.zeros(1, 1, 64, 64, 3),
                'action': torch.zeros(1, 1, config.num_actions),
                'reward': torch.zeros(1, 1),
                'discount': torch.ones(1, 1),
                'is_first': torch.zeros(1, 1),
                'is_terminal': torch.zeros(1, 1)
            }

    dummy_logger = DummyLogger()
    dummy_dataset = DummyDataset()

    agent = DreamerAgent(
        obs_space,
        act_space,
        config,
        dummy_logger,
        dummy_dataset
    )

    # Load model weights
    agent._world_model.load_state_dict(checkpoint["world_model"])
    agent._actor_critic.load_state_dict(checkpoint["actor_critic"])

    if verbose:
        print("Model loaded successfully!")
        print("=" * 60)

    # Set to eval mode
    agent._world_model.eval()
    agent._actor_critic.eval()

    # Play episodes
    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        if verbose:
            print(f"\nEpisode {ep + 1}/{num_episodes}")
            print("-" * 60)

        obs = env.reset()
        done = False
        state = None  # Agent state (latent, action)
        total_reward = 0
        steps = 0

        # Collect frames for GIF
        episode_frames = [] if save_gif else None

        while not done and steps < max_episode_steps:
            # Prepare observation
            obs_dict = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            obs_dict = agent._world_model.preprocess(obs_dict)

            # Get agent state
            if state is None:
                latent = action = None
            else:
                latent, action = state

            # Encode observation
            embed = agent._world_model.encoder(obs_dict)

            # Update latent state
            with torch.no_grad():
                latent, _ = agent._world_model.dynamics.obs_step(
                    latent,
                    action,
                    embed,
                    obs_dict["is_first"],
                    sample=not use_mode_for_z  # True=sample, False=mode
                )

            # Get features
            feat = agent._world_model.dynamics.get_latent_state_feature(latent)

            # Select action
            with torch.no_grad():
                actor = agent._actor_critic.actor(feat)
                if use_mode_for_action:
                    action = actor.mode()
                else:
                    action = actor.sample()

            # Convert action to index
            action_index = action.argmax(dim=-1)
            action_np = tools.to_np(action_index)[0]

            # Step environment
            obs, reward, done, info = env.step(int(action_np))

            # Capture frame for GIF (use high-resolution rendered screen)
            if save_gif:
                # Get the high-resolution screen from the base environment
                # This is the actual NES screen resolution (240x256) before downsampling
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

            total_reward += reward
            steps += 1

            # Detach state for next step
            latent = {k: v.detach() for k, v in latent.items()}
            action = action.detach()
            state = (latent, action)

            if verbose and steps % 50 == 0:
                print(f"  Step {steps}: reward={reward:.2f}, total={total_reward:.2f}")

        # Check if episode was successful or timed out
        flag_reached = info.get('flag_get', False) if 'flag_get' in info else False
        timed_out = steps >= max_episode_steps

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

        # Save episode as GIF
        if save_gif and episode_frames:
            gif_filename = f"episode_{ep+1:03d}_reward_{total_reward:.0f}_steps_{steps}.gif"
            gif_path = gif_run_dir / gif_filename
            duration_ms = int(1000 / gif_fps)  # Convert FPS to milliseconds per frame
            save_episode_gif(episode_frames, gif_path, duration=duration_ms, loop=0)

        if verbose:
            print(f"Episode {ep + 1} finished:")
            if flag_reached:
                print("  Status: SUCCESS ✓ - FLAG REACHED!")
            elif timed_out:
                print("  Status: TIMEOUT ⏱")
            else:
                print("  Status: FAILED ✗")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Length: {steps} steps")

    env.close()

    # Print statistics
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Episodes played: {num_episodes}")
        print(f"Average return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
        print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Best return: {np.max(episode_returns):.2f}")
        print(f"Worst return: {np.min(episode_returns):.2f}")
        print("=" * 60)

    return {
        'returns': episode_returns,
        'lengths': episode_lengths,
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }


def main():
    parser = argparse.ArgumentParser(description="Play Mario with trained DreamerV3 agent")
    parser.add_argument(
        '--logdir',
        type=str,
        default='logdir/mario',
        help='Path to log directory containing trained model'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to play'
    )
    parser.add_argument(
        '--z-sample',
        action='store_true',
        help='Use sample (instead of mode) for latent state z'
    )
    parser.add_argument(
        '--action-mode',
        action='store_true',
        help='Use mode (instead of sample) for action selection'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering (headless mode)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--frame-delay',
        type=float,
        default=0.015,
        help='Delay between frames in seconds (default: 0.015 = ~67 FPS)'
    )
    parser.add_argument(
        '--latest',
        action='store_false',
        dest='best',
        help='Use latest.pt checkpoint instead of best.pt (default: use best.pt)'
    )
    parser.add_argument(
        '--save-gif',
        action='store_true',
        help='Save each episode as an animated GIF (default: False)'
    )
    parser.add_argument(
        '--gif-dir',
        type=str,
        default=None,
        help='Directory to save GIF files (default: logdir/gifs)'
    )
    parser.add_argument(
        '--gif-fps',
        type=int,
        default=20,
        help='Frames per second for GIF animation (default: 20)'
    )
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        default=2000,
        help='Maximum steps per episode before timeout (default: 2000)'
    )

    args = parser.parse_args()

    play_mario(
        logdir=args.logdir,
        num_episodes=args.episodes,
        use_mode_for_z=not args.z_sample,
        use_mode_for_action=args.action_mode,
        render=not args.no_render,
        verbose=not args.quiet,
        frame_delay=args.frame_delay,
        use_best=args.best,
        save_gif=args.save_gif,
        gif_dir=args.gif_dir,
        gif_fps=args.gif_fps,
        max_episode_steps=args.max_episode_steps
    )


if __name__ == "__main__":
    main()
