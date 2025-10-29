"""
Training Script for DreamerV3 on Super Mario Bros

This script trains a DreamerV3 agent to play Super Mario Bros stage 1-1.
The training process follows Algorithm 1 from the paper:

1. Collect experience using current policy
2. Train world model on replay buffer
3. Imagine trajectories using world model
4. Train actor-critic on imagined trajectories
5. Evaluate periodically and save checkpoints

Usage:
    # Train with default config
    python train_mario.py

    # Train with debug mode (faster, for testing)
    python train_mario.py --configs debug

    # Resume from checkpoint
    python train_mario.py --logdir ./logdir/mario

Paper Reference: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
https://arxiv.org/abs/2301.04104
"""

import argparse
import collections
import pathlib
import sys

import numpy as np
import torch
import ruamel.yaml as yaml

# Add project to path
sys.path.append(str(pathlib.Path(__file__).parent))

from dreamerv3.world_model import WorldModel
from dreamerv3.actor_critic import ActorCritic
from dreamerv3.utils import tools
from envs.mario import make_mario_env


class DreamerAgent:
    """
    DreamerV3 Agent

    Combines world model and actor-critic for model-based RL.
    The agent:
    1. Learns a world model from experience
    2. Uses the world model to imagine future trajectories
    3. Learns a policy and value function from imagined experience
    """

    def __init__(
        self,
        obs_space,
        act_space,
        config,
        logger,
        dataset
    ):
        """
        Initialize DreamerV3 agent

        Args:
            obs_space: Observation space
            act_space: Action space
            config: Configuration object
            logger: Logger for metrics
            dataset: Replay buffer dataset
        """
        self._config = config
        self._logger = logger
        self._dataset = dataset

        # Scheduling
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()

        # Tracking
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0

        # === World Model ===
        # Learns environment dynamics: (s_t, a_t) -> (s_{t+1}, r_t, done_t)
        self._world_model = WorldModel(
            obs_space, act_space, self._step, config
        )

        # === Actor-Critic ===
        # Learns policy and value function using imagination
        self._actor_critic = ActorCritic(config, self._world_model)

        # Move to device
        self._world_model = self._world_model.to(config.device)
        self._actor_critic = self._actor_critic.to(config.device)

        # Optionally compile with PyTorch 2.0
        if config.compile and hasattr(torch, 'compile'):
            print("Compiling models with torch.compile...")
            self._world_model = torch.compile(self._world_model)
            self._actor_critic = torch.compile(self._actor_critic)

    def __call__(self, obs, reset, state=None, training=True):
        """
        Agent forward pass (for data collection)

        Args:
            obs: Observations from environment
            reset: Episode reset flags
            state: Previous agent state (latent state + action)
            training: Whether in training mode

        Returns:
            Tuple of (policy_output, new_state)
            - policy_output: Dict with 'action' and 'logprob'
            - new_state: Tuple of (latent_state, action)
        """
        step = self._step

        # === Training Updates ===
        if training:
            # Pretrain on random data once
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )

            # Multiple gradient updates per environment step
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count

            # Periodic logging
            if self._should_log(step):
                # Log scalar metrics
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []

                # Log video predictions
                if self._config.video_pred_log:
                    openl = self._world_model.video_pred(next(self._dataset))
                    self._logger.video("train_openl", tools.to_np(openl))

                self._logger.write(fps=True)

        # === Policy Execution ===
        policy_output, state = self._policy(obs, state, training)

        # Update step counter
        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step

        return policy_output, state

    def _policy(self, obs, state, training):
        """
        Execute policy to select actions

        Process:
        1. Encode observation to embedding
        2. Update latent state (posterior inference)
        3. Extract features
        4. Sample action from policy

        Args:
            obs: Observations
            state: Previous (latent, action) tuple
            training: Whether in training mode

        Returns:
            Tuple of (policy_output, new_state)
        """
        # Unpack previous state
        if state is None:
            latent = action = None
        else:
            latent, action = state

        # Preprocess observations
        obs = self._world_model.preprocess(obs)

        # Encode observations
        embed = self._world_model.encoder(obs)

        # Update latent state with observation
        # This computes the posterior: p(z_t | h_t, o_t)
        latent, _ = self._world_model.dynamics.obs_step(
            latent, action, embed, obs["is_first"]
        )

        # Use mean for deterministic evaluation
        if self._config.eval_state_mean and not training:
            latent["stoch"] = latent["mean"]

        # Extract features for policy
        feat = self._world_model.dynamics.get_latent_state_feature(latent)

        # Select action
        if not training:
            # Deterministic (mode) during evaluation
            actor = self._actor_critic.actor(feat)
            action = actor.mode()
        else:
            # Stochastic (sample) during training
            actor = self._actor_critic.actor(feat)
            action = actor.sample()

        # Compute log probability (for off-policy learning)
        # OneHotDist.log_prob will automatically handle straight-through gradients
        logprob = actor.log_prob(action)

        # Detach for next step
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()

        # Package outputs
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)

        return policy_output, state

    def _train(self, data):
        """
        Single training iteration

        Process:
        1. Train world model on batch
        2. Imagine trajectories
        3. Train actor-critic on imagination

        Args:
            data: Batch from replay buffer
        """
        metrics = {}

        # === 1. Train World Model ===
        # Learn dynamics, reward, and continuation from real experience
        posterior, context, mets = self._world_model._train(data)
        metrics.update(mets)

        # Use posterior states as starting points for imagination
        start = posterior

        # Reward function for actor-critic (from world model)
        reward = lambda f, s, a: self._world_model.heads["reward"](
            self._world_model.dynamics.get_latent_state_feature(s)
        ).mode()

        # === 2. Train Actor-Critic ===
        # Imagine trajectories and learn policy + value function
        metrics.update(
            self._actor_critic._train(start, reward)[-1]
        )

        # Accumulate metrics
        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


class Logger:
    """
    Logger for Scalar Metrics and Videos

    Logs to:
    - TensorBoard
    - JSON lines file (metrics.jsonl)
    - Console output
    """

    def __init__(self, logdir, step):
        """
        Initialize logger

        Args:
            logdir: Log directory path
            step: Initial step count
        """
        from torch.utils.tensorboard import SummaryWriter
        import json
        import time

        self._logdir = pathlib.Path(logdir)
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        """Log scalar value"""
        self._scalars[name] = float(value)

    def video(self, name, value):
        """Log video"""
        self._videos[name] = np.array(value)

    def write(self, fps=False):
        """Write logs to disk and TensorBoard"""
        import json
        import time

        step = self.step
        scalars = list(self._scalars.items())

        # Compute FPS if requested
        if fps:
            scalars.append(("fps", self._compute_fps(step)))

        # Console output
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))

        # JSON lines
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")

        # TensorBoard scalars
        for name, value in scalars:
            self._writer.add_scalar(
                "scalars/" + name if "/" not in name else name,
                value,
                step
            )

        # TensorBoard videos
        for name, value in self._videos.items():
            # Convert to uint8 if float
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)

            # Reshape: (B, T, H, W, C) -> (1, T, C, H, B*W)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, fps=16)

        self._writer.flush()

        # Clear buffers
        self._scalars = {}
        self._videos = {}

    def _compute_fps(self, step):
        """Compute frames per second"""
        import time

        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0

        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step

        return steps / duration if duration > 0 else 0


def count_steps(folder):
    """
    Count total steps in saved episodes

    Args:
        folder: Directory containing episode files

    Returns:
        Total number of steps
    """
    return sum(
        int(str(n).split("-")[-1][:-4]) - 1
        for n in folder.glob("*.npz")
    )


def make_dataset(episodes, config):
    """
    Create batched dataset from episodes

    Args:
        episodes: Ordered dict of episodes
        config: Configuration

    Returns:
        Infinite iterator yielding batches
    """
    generator = tools.sample_episodes(
        episodes,
        config.batch_length
    )
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def simulate(
    agent,
    env,
    episodes_dict,
    logger,
    is_eval=False,
    episodes=0,
    steps=0
):
    """
    Simulate agent in single environment

    Runs agent in environment, collecting experience and logging metrics.

    Args:
        agent: Agent to execute
        env: Single environment instance
        episodes_dict: OrderedDict of saved episodes (for replay buffer)
        logger: Logger
        is_eval: Whether this is evaluation (vs training)
        episodes: Number of episodes to run (0 = until steps reached)
        steps: Number of steps to run (0 = until episodes reached)

    Returns:
        None
    """
    # Initialize
    step = episode = 0
    done = True
    length = 0
    obs = None
    agent_state = None

    # Current episode buffer (temporary storage for ongoing episode)
    current_episode = None

    while (steps and step < steps) or (episodes and episode < episodes):
        # Reset environment if needed
        if done:
            obs = env.reset()
            # Initialize new episode buffer (don't save initial obs yet)
            current_episode = {}
            length = 0

        # Add batch dimension for agent
        obs_batch = {
            k: np.expand_dims(obs[k], axis=0)
            for k in obs if "log_" not in k
        }
        done_batch = np.array([done])

        # Get action from agent (for current obs)
        action, agent_state = agent(obs_batch, done_batch, agent_state)

        # Convert one-hot action to index for environment
        action_tensor = action["action"]
        action_index = action_tensor.argmax(dim=-1)
        action_np = tools.to_np(action_index)
        env_action = int(action_np[0])

        # Convert action dict for cache (keep one-hot encoding)
        action_dict = {
            k: tools.to_np(action[k][0]) if hasattr(action[k][0], 'cpu') else np.array(action[k][0])
            for k in action
        }

        # Step environment to get next obs and reward
        next_obs, reward, done, info = env.step(env_action)

        # NEW FORMAT: Save (current_obs, action_taken_from_this_obs, reward_received)
        # This makes the data format: data[t] = (obs_t, action_t, reward_t)
        t = {k: tools.convert(v) for k, v in obs.items()}  # Current obs (before step)
        t.update(action_dict)  # Action taken in current obs
        t["reward"] = reward  # Reward received after executing action
        t["discount"] = info.get("discount", np.array(1 - float(done)))

        # Append to current episode buffer
        for key, val in t.items():
            if key not in current_episode:
                current_episode[key] = []
            current_episode[key].append(val)

        # Update state for next iteration
        obs = next_obs
        episode += int(done)
        length += 1
        step += 1

        # Log completed episode
        if done:
            # Convert lists to numpy arrays for saving
            episode_data = {
                k: np.array(v) for k, v in current_episode.items()
            }

            # Calculate episode metrics first
            # In new format, each element is a complete transition, so length is direct count
            ep_length = len(current_episode["reward"])

            # Generate unique episode ID (length will be added by save_episodes)
            import time
            episode_id = f"{int(time.time() * 1000)}-{episode}"

            # Save episode to disk
            save_dir = pathlib.Path(logger._logdir) / ("eval_eps" if is_eval else "train_eps")
            tools.save_episodes(save_dir, {episode_id: episode_data})
            ep_return = float(np.array(current_episode["reward"]).sum())

            # Log episode-specific metrics
            for key in list(current_episode.keys()):
                if "log_" in key:
                    logger.scalar(key, float(np.array(current_episode[key]).sum()))

            if not is_eval:
                # Training metrics
                logger.scalar("train_return", ep_return)
                logger.scalar("train_length", ep_length)
                # logger.scalar("train_episodes", 1)
                logger.write()

                # Add to episodes_dict for replay buffer
                episodes_dict[episode_id] = episode_data

                # Erase old episodes to maintain dataset size limit (from memory and disk)
                tools.erase_over_episodes(episodes_dict, config.dataset_size, save_dir)
            else:
                # Evaluation metrics
                logger.scalar("eval_return", ep_return)
                logger.scalar("eval_length", ep_length)

                # Add to episodes_dict for video prediction
                # Keep only the most recent eval episode
                episodes_dict.clear()
                episodes_dict[episode_id] = episode_data

            # Reset for next episode
            current_episode = None
            length = 0


def main(config):
    """
    Main training loop

    Args:
        config: Configuration object
    """
    # Set random seeds
    tools.set_seed_everywhere(config.seed)

    # Create directories
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "train_eps").mkdir(exist_ok=True)
    (logdir / "eval_eps").mkdir(exist_ok=True)

    # Adjust config for action repeat
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir:", logdir)

    # Count existing steps
    step = count_steps(logdir / "train_eps")
    logger = Logger(logdir, config.action_repeat * step)

    print("Create environment")

    # Create environment
    env = make_mario_env(config)
    obs_space = env.observation_space
    act_space = env.action_space
    config.num_actions = act_space.n

    print("Action space:", act_space, f"({config.num_actions} actions)")

    # Load replay buffer
    print("Load replay buffer")
    train_eps = tools.load_episodes(logdir / "train_eps", limit=config.dataset_size)
    eval_eps = tools.load_episodes(logdir / "eval_eps", limit=1)

    # Create dataset
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)

    # Create agent
    print("Create agent")
    agent = DreamerAgent(
        obs_space,
        act_space,
        config,
        logger,
        train_dataset
    )

    # Load checkpoint if exists
    if (logdir / "latest.pt").exists():
        print("Loading checkpoint")
        checkpoint = torch.load(logdir / "latest.pt")
        agent._world_model.load_state_dict(checkpoint["world_model"])
        agent._actor_critic.load_state_dict(checkpoint["actor_critic"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optimizers"])
        agent._should_pretrain._once = False

    # Initial data collection
    if step < config.prefill:
        print(f"Prefill dataset ({config.prefill - step} steps)")

        # Random policy
        def random_agent(o, d, s):
            batch_size = o["image"].shape[0]
            action = torch.randint(0, config.num_actions, (batch_size,))
            action_onehot = torch.nn.functional.one_hot(action, config.num_actions).float()
            return {"action": action_onehot, "logprob": torch.zeros(batch_size)}, None

        simulate(
            random_agent,
            env,
            train_eps,
            logger,
            steps=config.prefill - step
        )
        logger.step += (config.prefill - step) * config.action_repeat

    print("Start training")

    # Main training loop
    while agent._step < config.steps:
        # Evaluate (current model for eval_episode_num episodes)
        if config.eval_episode_num > 0:
            print("Evaluation")
            eval_env = make_mario_env(config)
            simulate(
                lambda o, d, s: agent(o, d, s, training=False),
                eval_env,
                eval_eps,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num
            )
            eval_env.close()

            # Log video prediction
            if config.video_pred_log and len(eval_eps) > 0:
                video_pred = agent._world_model.video_pred(next(eval_dataset))
                logger.video("eval_openl", tools.to_np(video_pred))

        # Train
        print(f"Training (step {agent._step}/{config.steps})")
        simulate(
            agent,
            env,
            train_eps,
            logger,
            steps=config.eval_every
        )

        # Save checkpoint
        print("Saving checkpoint")
        torch.save({
            "world_model": agent._world_model.state_dict(),
            "actor_critic": agent._actor_critic.state_dict(),
            "optimizers": tools.recursively_collect_optim_state_dict(agent)
        }, logdir / "latest.pt")

    print("Training complete!")
    env.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train DreamerV3 on Super Mario Bros")
    parser.add_argument("--configs", nargs="+", default=["defaults", "mario"])
    parser.add_argument("--logdir", type=str, default=None)
    args, remaining = parser.parse_known_args()

    # Load config
    configs = yaml.safe_load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )

    # Merge configs
    config_dict = {}
    for name in args.configs:
        if name not in configs:
            raise ValueError(f"Config '{name}' not found in configs.yaml")
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

    # Run training
    main(config)
