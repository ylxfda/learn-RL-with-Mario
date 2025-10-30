"""
Play Super Mario Bros with Trained DreamerV3 Agent

This script loads a trained model and plays Mario in real-time with visualization.

Usage:
    python play_mario.py --logdir logdir/mario --episodes 5
"""

import argparse
import pathlib
import torch
import numpy as np
import ruamel.yaml as yaml

from train_mario import DreamerAgent, make_mario_env
from dreamerv3.utils import tools


class DemoConfig:
    """Configuration wrapper for demo"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def play_mario(
    logdir: str,
    num_episodes: int = 5,
    use_mode_for_z: bool = True,
    use_mode_for_action: bool = True,
    render: bool = True,
    verbose: bool = True,
    frame_delay: float = 0.02
):
    """
    Play Mario with trained agent

    Args:
        logdir: Directory containing trained model
        num_episodes: Number of episodes to play
        use_mode_for_z: Use mode (vs sample) for z (latent state)
        use_mode_for_action: Use mode (vs sample) for action
        render: Whether to render the game window
        verbose: Print detailed info
        frame_delay: Delay in seconds between frames (0.02 = ~50 FPS, 0.05 = ~20 FPS)
    """
    logdir = pathlib.Path(logdir).expanduser()

    if verbose:
        print("=" * 60)
        print(f"Playing Mario with trained agent from: {logdir}")
        print(f"Episodes: {num_episodes}")
        print(f"z strategy: {'mode (deterministic)' if use_mode_for_z else 'sample (stochastic)'}")
        print(f"action strategy: {'mode (deterministic)' if use_mode_for_action else 'sample (stochastic)'}")
        print("=" * 60)

    # Load configuration
    config_path = logdir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = DemoConfig(config_dict)

    # Override render mode for demo
    if render:
        # Note: gym-super-mario-bros uses 'human' mode for rendering
        config.render_mode = 'human'
        config.frame_delay = frame_delay

    # Load checkpoint
    checkpoint_path = logdir / "latest.pt"
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

        while not done:
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

            total_reward += reward
            steps += 1

            # Detach state for next step
            latent = {k: v.detach() for k, v in latent.items()}
            action = action.detach()
            state = (latent, action)

            if verbose and steps % 50 == 0:
                print(f"  Step {steps}: reward={reward:.2f}, total={total_reward:.2f}")

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

        if verbose:
            print(f"Episode {ep + 1} finished:")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Length: {steps} steps")
            if 'flag_get' in info and info['flag_get']:
                print("  🎉 FLAG REACHED!")

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
        '--action-sample',
        action='store_true',
        help='Use sample (instead of mode) for action selection'
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
        help='Delay between frames in seconds (default: 0.02 = ~50 FPS)'
    )

    args = parser.parse_args()

    play_mario(
        logdir=args.logdir,
        num_episodes=args.episodes,
        use_mode_for_z=not args.z_sample,
        use_mode_for_action=not args.action_sample,
        render=not args.no_render,
        verbose=not args.quiet,
        frame_delay=args.frame_delay
    )


if __name__ == "__main__":
    main()
