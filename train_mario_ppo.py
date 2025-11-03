"""
Training Script for PPO on Super Mario Bros

This script trains a PPO agent to play Super Mario Bros stage 1-1.
The training process follows Algorithm 1 from the PPO paper:

1. Collect rollout using current policy π_{θ_old}
2. Compute advantages using GAE
3. Update policy and value function for K epochs
4. Repeat until convergence

Usage:
    # Train with default config
    python train_mario_ppo.py

    # Train with debug mode (faster, for testing)
    python train_mario_ppo.py --configs debug

    # Resume from checkpoint
    python train_mario_ppo.py --resume --logdir ./logdir/mario_ppo

Paper Reference: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
https://arxiv.org/abs/1707.06347
"""

import argparse
import pathlib
import time
import sys
from typing import Any

import numpy as np
import torch
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter

# Add project to path
sys.path.append(str(pathlib.Path(__file__).parent))

from PPO.ppo_agent import PPOAgent
from envs.vec_mario import make_vec_mario_env
from envs.mario import MarioEnv
from dreamerv3.utils import tools


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def log_videos_to_tensorboard(writer, videos, name, timestep, fps=16):
    """
    Log videos to TensorBoard

    Args:
        writer: TensorBoard SummaryWriter
        videos: List of video arrays, each of shape (T, H, W, C)
        name: Name for the video in TensorBoard
        timestep: Current timestep for logging
        fps: Frames per second for video playback
    """
    if not videos or len(videos) == 0:
        return

    # Stack videos: List[(T, H, W, C)] -> (B, T, H, W, C)
    # First, pad all videos to same length
    max_length = max(v.shape[0] for v in videos)

    padded_videos = []
    for video in videos:
        if video.shape[0] < max_length:
            # Pad with last frame repeated
            padding = np.repeat(
                video[-1:],
                max_length - video.shape[0],
                axis=0
            )
            video = np.concatenate([video, padding], axis=0)
        padded_videos.append(video)

    # Stack into (B, T, H, W, C)
    video_batch = np.stack(padded_videos, axis=0)
    B, T, H, W, C = video_batch.shape

    # Convert to uint8 if needed
    if np.issubdtype(video_batch.dtype, np.floating):
        video_batch = np.clip(255 * video_batch, 0, 255).astype(np.uint8)

    # Reshape for TensorBoard: (B, T, H, W, C) -> (1, T, C, H, B*W)
    # This displays videos side-by-side horizontally
    video_batch = video_batch.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))

    # Add to TensorBoard
    writer.add_video(name, video_batch, timestep, fps=fps)


def evaluate_agent(
    agent: PPOAgent,
    config: Any,
    num_episodes: int = 10,
    render: bool = False,
    record_video: bool = False,
    num_video_episodes: int = 3
) -> dict:
    """
    Evaluate agent performance

    Args:
        agent: PPO agent to evaluate
        config: Configuration object
        num_episodes: Number of episodes to run
        render: Whether to render episodes
        record_video: Whether to record video frames for TensorBoard
        num_video_episodes: Number of episodes to record (randomly selected)

    Returns:
        Dictionary with evaluation metrics:
        - mean_return: Average episode return
        - mean_length: Average episode length
        - success_rate: Fraction of episodes that reached flag
        - videos: List of recorded episode frames (if record_video=True)
                  Each video is np.array of shape (T, H, W, C)
    """
    # Create single environment for evaluation
    eval_env = MarioEnv(
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
        seed=config.seed + 10000,  # Different seed for eval
        render_mode='human' if render else None,
        frame_delay=0.02 if render else 0.0
    )

    returns = []
    lengths = []
    successes = []
    all_episodes_frames = []  # Store frames from all episodes

    # Randomly select episodes to record
    if record_video:
        episodes_to_record = set(np.random.choice(
            num_episodes,
            size=min(num_video_episodes, num_episodes),
            replace=False
        ))
    else:
        episodes_to_record = set()

    for ep in range(num_episodes):
        obs = eval_env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        episode_frames = []

        # Record this episode if selected
        should_record = ep in episodes_to_record

        if should_record:
            # Record initial frame
            episode_frames.append(obs['image'].copy())

        while not done:
            # Get action from agent (deterministic)
            action = agent.get_action(obs['image'], deterministic=True)
            action = int(action)  # Convert numpy array to int

            # Step environment
            obs, reward, done, info = eval_env.step(action)
            episode_return += reward
            episode_length += 1

            if should_record:
                # Record frame
                episode_frames.append(obs['image'].copy())

        # Check if episode was successful (reached flag)
        success = info.get('mario_success', False)

        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(float(success))

        if should_record and len(episode_frames) > 0:
            # Convert frames list to numpy array: (T, H, W, C)
            episode_video = np.array(episode_frames)
            all_episodes_frames.append(episode_video)

    eval_env.close()

    result = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'success_rate': np.mean(successes)
    }

    if record_video and len(all_episodes_frames) > 0:
        result['videos'] = all_episodes_frames

    return result


def main(config):
    """
    Main training loop

    Args:
        config: Configuration object
    """
    # Set random seeds
    set_seed(config.seed)

    # Create log directory
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"PPO Training: Super Mario Bros")
    print("=" * 80)
    print(f"Logdir: {logdir}")
    print(f"Device: {config.device}")
    print(f"Parallel Envs: {config.num_envs}")
    print(f"Total Timesteps: {config.total_timesteps:,}")
    print("=" * 80)

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=str(logdir))

    # Create vectorized environments
    print("\nCreating vectorized environments...")
    vec_env = make_vec_mario_env(config, num_envs=config.num_envs)
    print(f"✓ Created {config.num_envs} parallel environments")

    # Get observation and action space info
    obs_space = vec_env.observation_space
    act_space = vec_env.action_space

    # Observation shape: (C, H, W)
    if config.grayscale:
        obs_shape = (1, config.size[0], config.size[1])
    else:
        obs_shape = (3, config.size[0], config.size[1])

    num_actions = act_space.n
    config.num_actions = num_actions

    print(f"Observation shape: {obs_shape}")
    print(f"Action space: {num_actions} discrete actions")

    # Save configuration
    config_path = logdir / 'config.yaml'
    yaml = YAML()
    yaml.default_flow_style = False
    with open(config_path, 'w') as f:
        yaml.dump(vars(config), f)
    print(f"✓ Saved config to {config_path}")

    # Create PPO agent
    print("\nInitializing PPO agent...")
    agent = PPOAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        config=config,
        device=device
    )
    print("✓ Agent initialized")

    # Load checkpoint if resuming
    latest_checkpoint_path = logdir / 'latest.pt'
    start_timestep = 0
    if latest_checkpoint_path.exists() and getattr(config, 'resume', False):
        print(f"\nLoading checkpoint from {latest_checkpoint_path}")
        agent.load(str(latest_checkpoint_path))
        # Note: We don't track timestep in checkpoint, so we start from 0
        # In production, you'd want to save/load this
        print("✓ Checkpoint loaded")

    # Initialize environment
    print("\nStarting training...")
    obs = vec_env.reset()
    episode_returns = np.zeros(config.num_envs)
    episode_lengths = np.zeros(config.num_envs)

    # Training statistics
    num_updates = 0
    best_eval_return = -float('inf')
    start_time = time.time()

    # Main training loop
    timestep = start_timestep
    while timestep < config.total_timesteps:
        # === Collect Rollout ===
        rollout_start = time.time()
        obs = agent.collect_rollout(vec_env, obs)
        rollout_time = time.time() - rollout_start

        # Update timestep counter
        timestep += config.num_steps * config.num_envs

        # === PPO Update ===
        update_start = time.time()
        metrics = agent.update()
        update_time = time.time() - update_start
        num_updates += 1

        # === Logging ===
        if timestep % config.log_freq < (config.num_steps * config.num_envs):
            elapsed_time = time.time() - start_time
            fps = timestep / elapsed_time

            print(f"\n[Timestep {timestep:,}/{config.total_timesteps:,}]")
            print(f"  Update: {num_updates}")
            print(f"  FPS: {fps:.0f}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Approx KL: {metrics['approx_kl']:.4f}")
            print(f"  Clip Fraction: {metrics['clipfrac']:.3f}")
            print(f"  Rollout Time: {rollout_time:.2f}s")
            print(f"  Update Time: {update_time:.2f}s")

            # Log to TensorBoard
            writer.add_scalar('train/policy_loss', metrics['policy_loss'], timestep)
            writer.add_scalar('train/value_loss', metrics['value_loss'], timestep)
            writer.add_scalar('train/entropy', metrics['entropy'], timestep)
            writer.add_scalar('train/approx_kl', metrics['approx_kl'], timestep)
            writer.add_scalar('train/clipfrac', metrics['clipfrac'], timestep)
            writer.add_scalar('train/fps', fps, timestep)

        # === Evaluation ===
        if timestep % config.eval_freq < (config.num_steps * config.num_envs):
            print("\n" + "=" * 80)
            print("EVALUATION")
            print("=" * 80)

            # Enable video recording (controlled by config)
            eval_metrics = evaluate_agent(
                agent,
                config,
                num_episodes=config.eval_episodes,
                render=False,
                record_video=getattr(config, 'record_eval_videos', True),
                num_video_episodes=getattr(config, 'num_video_episodes', 3)
            )

            print(f"Episodes: {config.eval_episodes}")
            print(f"Mean Return: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f}")
            print(f"Mean Length: {eval_metrics['mean_length']:.1f}")
            print(f"Success Rate: {eval_metrics['success_rate']:.2%}")

            # Log to TensorBoard
            writer.add_scalar('eval/mean_return', eval_metrics['mean_return'], timestep)
            writer.add_scalar('eval/success_rate', eval_metrics['success_rate'], timestep)
            writer.add_scalar('eval/mean_length', eval_metrics['mean_length'], timestep)

            # Log videos to TensorBoard
            if 'videos' in eval_metrics:
                print(f"Recording {len(eval_metrics['videos'])} evaluation episodes to TensorBoard...")
                log_videos_to_tensorboard(
                    writer,
                    eval_metrics['videos'],
                    'eval/episodes',
                    timestep,
                    fps=getattr(config, 'video_fps', 16)
                )

            # Save best checkpoint
            if eval_metrics['mean_return'] > best_eval_return:
                best_eval_return = eval_metrics['mean_return']
                best_path = logdir / 'best.pt'
                agent.save(str(best_path))
                print(f"✓ New best model saved (return: {best_eval_return:.2f})")

            print("=" * 80)

        # === Save Latest Checkpoint ===
        if timestep % config.save_freq < (config.num_steps * config.num_envs):
            agent.save(str(latest_checkpoint_path))
            print(f"✓ Latest checkpoint saved to {latest_checkpoint_path}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    final_metrics = evaluate_agent(
        agent,
        config,
        num_episodes=config.eval_episodes * 2,  # More episodes for final eval
        render=False,
        record_video=True,
        num_video_episodes=5  # Record more videos for final eval
    )

    print(f"Episodes: {config.eval_episodes * 2}")
    print(f"Mean Return: {final_metrics['mean_return']:.2f} ± {final_metrics['std_return']:.2f}")
    print(f"Mean Length: {final_metrics['mean_length']:.1f}")
    print(f"Success Rate: {final_metrics['success_rate']:.2%}")

    # Log final videos to TensorBoard
    if 'videos' in final_metrics:
        print(f"Recording {len(final_metrics['videos'])} final evaluation episodes to TensorBoard...")
        log_videos_to_tensorboard(
            writer,
            final_metrics['videos'],
            'eval/final_episodes',
            timestep,
            fps=getattr(config, 'video_fps', 16)
        )

    print("=" * 80)

    # Save final latest checkpoint
    agent.save(str(latest_checkpoint_path))
    print(f"\n✓ Training complete!")
    print(f"  Latest checkpoint: {latest_checkpoint_path}")
    print(f"  Best checkpoint: {logdir / 'best.pt'} (return: {best_eval_return:.2f})")

    # Cleanup
    vec_env.close()
    writer.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train PPO on Super Mario Bros")
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--logdir", type=str, default=None)
    args, remaining = parser.parse_known_args()

    # Load config
    config_file = pathlib.Path(__file__).parent / "configs" / "ppo_configs.yaml"
    yaml = YAML(typ='safe', pure=True)
    configs = yaml.load(config_file)

    # Merge configs (always start with defaults, then apply others)
    if 'defaults' not in configs:
        raise ValueError("'defaults' config not found in ppo_configs.yaml")

    config_dict = dict(configs['defaults'])  # Start with defaults

    # Apply requested configs on top (skip 'defaults' if already in args.configs)
    for name in args.configs:
        if name == 'defaults':
            continue  # Already loaded
        if name not in configs:
            raise ValueError(f"Config '{name}' not found in ppo_configs.yaml")
        # Simple merge (later configs override earlier ones)
        config_dict.update(configs[name])

    # Override with command-line arguments
    parser = argparse.ArgumentParser()
    for key, value in sorted(config_dict.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    config = parser.parse_args(remaining)

    # Override logdir if specified
    if args.logdir:
        config.logdir = args.logdir

    # Set resume flag
    config.resume = args.resume

    # Run training
    main(config)
