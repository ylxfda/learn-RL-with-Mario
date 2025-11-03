"""
Vectorized Super Mario Bros Environment for PPO

This module implements a SubprocVecEnv wrapper that runs multiple Mario
environments in parallel processes for efficient data collection with PPO.
PPO is an on-policy algorithm that benefits from collecting large batches
of experience from multiple environments simultaneously.

Based on OpenAI Baselines SubprocVecEnv implementation.
"""

import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
import gym


def worker(remote, parent_remote, env_fn):
    """
    Worker process function for SubprocVecEnv

    Each worker runs in a separate process and manages one environment.
    It receives commands from the main process via a pipe and sends back results.

    Args:
        remote: Child end of the pipe (used by worker)
        parent_remote: Parent end of the pipe (closed in worker)
        env_fn: Callable that creates an environment instance
    """
    parent_remote.close()
    env = env_fn()

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                # Execute action and return transition
                obs, reward, done, info = env.step(data)
                if done:
                    # Auto-reset on episode end
                    obs = env.reset()
                remote.send((obs, reward, done, info))

            elif cmd == 'reset':
                # Reset environment
                obs = env.reset()
                remote.send(obs)

            elif cmd == 'close':
                # Clean up and exit
                env.close()
                remote.close()
                break

            elif cmd == 'get_spaces':
                # Return observation and action spaces
                remote.send((env.observation_space, env.action_space))

            else:
                raise NotImplementedError(f"Command {cmd} not implemented")

        except EOFError:
            break


class SubprocVecEnv:
    """
    Vectorized Environment that runs multiple environments in parallel subprocesses.

    This is the standard approach for PPO to collect experience efficiently.
    Each environment runs in its own process, allowing true parallel execution.

    Interface follows the single environment API but with batched inputs/outputs:
    - step(actions) takes a batch of actions, returns batched observations, rewards, etc.
    - reset() returns batched initial observations

    Paper Reference:
    This vectorized environment is commonly used with PPO to collect the
    on-policy data needed for policy optimization. See Schulman et al. (2017)
    "Proximal Policy Optimization Algorithms", Section 3.2.

    Attributes:
        num_envs (int): Number of parallel environments
        observation_space: Observation space of a single environment
        action_space: Action space of a single environment
    """

    def __init__(self, env_fns: List[callable]):
        """
        Initialize vectorized environment

        Args:
            env_fns: List of callables that create environment instances
                     Each callable should return a configured environment

        Example:
            >>> env_fns = [lambda: MarioEnv(seed=i) for i in range(8)]
            >>> vec_env = SubprocVecEnv(env_fns)
        """
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        # Create pipe for each environment
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])

        # Start worker processes
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = mp.Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get spaces from first environment
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with given actions

        This is the main method used during PPO rollout collection.
        Each environment steps with its corresponding action in parallel.

        Args:
            actions: Array of actions, shape (num_envs,) for discrete actions
                     For Mario: integer action indices [0, num_actions-1]

        Returns:
            Tuple of (observations, rewards, dones, infos):
            - observations: Dict with batched observations
                - 'image': shape (num_envs, H, W, C)
                - 'is_first': shape (num_envs,)
                - 'is_terminal': shape (num_envs,)
            - rewards: Array of rewards, shape (num_envs,)
            - dones: Array of done flags, shape (num_envs,)
            - infos: List of info dicts, length num_envs

        Note:
            Environments automatically reset when done=True, so the returned
            observation is the first observation of the next episode.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: np.ndarray):
        """
        Send step commands to all workers (non-blocking)

        Args:
            actions: Array of actions, shape (num_envs,)
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self) -> Tuple[Dict, np.ndarray, np.ndarray, List[Dict]]:
        """
        Wait for step results from all workers

        Returns:
            Batched (observations, rewards, dones, infos)
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        obs_list, rewards, dones, infos = zip(*results)

        # Stack observations into batched format
        obs_batch = self._stack_obs(obs_list)

        return obs_batch, np.array(rewards), np.array(dones), infos

    def reset(self) -> Dict:
        """
        Reset all environments

        Returns:
            Batched initial observations with same structure as step()
            - 'image': shape (num_envs, H, W, C)
            - 'is_first': shape (num_envs,), all True
            - 'is_terminal': shape (num_envs,), all False
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        obs_list = [remote.recv() for remote in self.remotes]
        return self._stack_obs(obs_list)

    def close(self):
        """
        Close all worker processes and clean up resources
        """
        if self.closed:
            return

        if self.waiting:
            # Wait for pending operations
            for remote in self.remotes:
                remote.recv()

        # Send close command to all workers
        for remote in self.remotes:
            remote.send(('close', None))

        # Wait for processes to finish
        for process in self.processes:
            process.join()

        self.closed = True

    def _stack_obs(self, obs_list: List[Dict]) -> Dict:
        """
        Stack list of observations into batched dictionary

        Args:
            obs_list: List of observation dicts from each environment

        Returns:
            Dictionary with batched observations:
            - 'image': np.array of shape (num_envs, H, W, C)
            - 'is_first': np.array of shape (num_envs,)
            - 'is_terminal': np.array of shape (num_envs,)
        """
        # Stack image observations
        images = np.stack([obs['image'] for obs in obs_list], axis=0)

        # Stack scalar flags
        is_first = np.array([obs['is_first'] for obs in obs_list])
        is_terminal = np.array([obs['is_terminal'] for obs in obs_list])

        return {
            'image': images,
            'is_first': is_first,
            'is_terminal': is_terminal
        }

    def __len__(self):
        """Return number of environments"""
        return self.num_envs


def make_vec_mario_env(config: Any, num_envs: int = 32) -> SubprocVecEnv:
    """
    Create vectorized Mario environment for PPO

    Creates multiple parallel Mario environments with different random seeds
    to ensure diverse experience collection.

    Args:
        config: Configuration object with environment parameters
        num_envs: Number of parallel environments (default: 32)
                  PPO typically uses 8-128 parallel environments

    Returns:
        SubprocVecEnv with num_envs parallel Mario environments

    Example:
        >>> config = load_config('ppo_configs.yaml')
        >>> vec_env = make_vec_mario_env(config, num_envs=32)
        >>> obs = vec_env.reset()  # shape: (32, 64, 64, 3)
        >>> actions = np.random.randint(0, 7, size=32)
        >>> obs, rewards, dones, infos = vec_env.step(actions)

    Note:
        Each environment gets a unique seed based on config.seed + env_index
        to ensure reproducibility while maintaining diversity.
    """
    from envs.mario import MarioEnv

    def make_env(rank: int) -> callable:
        """
        Create environment factory function for worker process

        Args:
            rank: Environment index (0 to num_envs-1)

        Returns:
            Callable that creates a MarioEnv instance
        """
        def _init():
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
                seed=config.seed + rank if hasattr(config, 'seed') else rank,
                render_mode=None,  # No rendering in parallel envs
                frame_delay=0.0
            )
            return env
        return _init

    # Create list of environment factories
    env_fns = [make_env(i) for i in range(num_envs)]

    return SubprocVecEnv(env_fns)
