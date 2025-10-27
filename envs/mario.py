"""
Super Mario Bros Environment Wrapper for DreamerV3

A simplified environment wrapper for Super Mario Bros stage 1-1, tailored
for DreamerV3's observation and reward format. This wrapper:
- Processes visual observations
- Shapes rewards to encourage progress
- Handles episode termination
- Provides consistent interface for the agent

Based on gym_super_mario_bros environment.
"""

import gym
import numpy as np
from typing import Tuple, Dict, Optional

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class MarioEnv:
    """
    Super Mario Bros Environment Wrapper

    Wraps the gym_super_mario_bros environment to provide:
    1. Reward shaping based on progress (x-position change)
    2. Frame skipping (action repeat)
    3. Frame stacking with max pooling
    4. Observation resizing
    5. Episode termination handling

    The environment encourages the agent to move right and reach the flag.
    """

    # Metadata for gym compatibility
    metadata = {}

    def __init__(
        self,
        level: str = "SuperMarioBros-1-1-v0",
        action_repeat: int = 4,
        size: Tuple[int, int] = (64, 64),
        grayscale: bool = False,
        action_set: str = "simple",
        resize_method: str = "opencv",
        flag_reward: float = 1000.0,
        reward_scale: float = 1.0,
        time_penalty: float = -0.1,
        death_penalty: float = -15.0,
        seed: Optional[int] = None
    ):
        """
        Initialize Mario environment

        Args:
            level: Mario level name (default: "SuperMarioBros-1-1-v0")
            action_repeat: Number of times to repeat each action
            size: Output observation size (height, width)
            grayscale: Whether to convert to grayscale
            action_set: Action set to use ("simple" recommended)
            resize_method: Image resize method ("opencv" or "pillow")
            flag_reward: Bonus reward for completing the level
            reward_scale: Scale factor for distance-based rewards
            time_penalty: Penalty per unit time elapsed
            death_penalty: Penalty for losing a life
            seed: Random seed for reproducibility
        """
        assert size[0] == size[1], "Mario observations must be square"
        assert resize_method in ("opencv", "pillow"), resize_method

        # Store configuration
        self._level = level
        self._repeat = action_repeat
        self._size = size
        self._grayscale = grayscale
        self._flag_reward = flag_reward
        self._reward_scale = reward_scale
        self._time_penalty = time_penalty
        self._death_penalty = death_penalty
        self._random = np.random.RandomState(seed)
        self._seed = seed
        self._base_seed = seed if seed is not None else self._random.randint(0, 2**31)

        # Setup resize method
        self._resize_method = resize_method
        if self._resize_method == "opencv":
            import cv2
            self._cv2 = cv2
        else:
            from PIL import Image
            self._image = Image

        # Create base Mario environment
        base_env = gym_super_mario_bros.make(level)

        # Wrap with action space (use simple 7-action set)
        if action_set == "simple":
            actions = SIMPLE_MOVEMENT
        else:
            raise ValueError(f"Action set '{action_set}' not supported")

        self._env = JoypadSpace(base_env, actions)

        # Initialize frame buffer for max pooling
        # Stores last 2 frames to reduce flickering
        initial_obs = self._ensure_obs(self._env.reset())
        self._buffer = [
            np.zeros_like(initial_obs),
            np.zeros_like(initial_obs)
        ]
        self._reset_buffers(initial_obs)

        # Episode state
        self._done = True
        self._episode_flag = False  # Whether flag was reached
        self._prev_x_pos = 0.0
        self._prev_time = 400
        self._prev_lives = 2

        # For gym compatibility
        self.reward_range = (-np.inf, np.inf)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """
        Get observation space

        Returns:
            Dictionary space with 'image' key
        """
        img_shape = self._size + ((1,) if self._grayscale else (3,))
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, np.uint8)
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """
        Get action space

        Returns:
            Discrete action space (7 actions for SIMPLE_MOVEMENT)
        """
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action: any) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action in environment

        Process:
        1. Repeat action for self._repeat steps
        2. Accumulate rewards based on x-position progress
        3. Apply reward shaping (flag bonus, time penalty, death penalty)
        4. Return processed observation

        Args:
            action: Action to execute (int or one-hot array)

        Returns:
            Tuple of (observation, reward, done, info):
            - observation: Dict with 'image', 'is_first', 'is_terminal'
            - reward: Shaped reward
            - done: Whether episode ended
            - info: Additional information
        """
        # Convert one-hot to integer if needed
        if hasattr(action, "shape") and action.shape:
            action = int(np.argmax(action))

        total_reward = 0.0
        terminal = False
        obs = None
        step_info = {}

        # Repeat action
        for repeat_idx in range(self._repeat):
            obs, _, done, step_info = self._env.step(action)

            # === Reward Shaping: Progress-Based ===
            # Reward = change in x-position (encourage moving right)
            x_pos = float(step_info.get("x_pos", self._prev_x_pos))
            progress = x_pos - self._prev_x_pos
            if progress < 0.0:  # Don't penalize moving left
                progress = 0.0
            total_reward += self._reward_scale * progress
            self._prev_x_pos = x_pos

            # Store second-to-last frame for max pooling
            if repeat_idx == self._repeat - 2:
                self._buffer[1][:] = obs

            # Check for flag (level completion)
            if step_info.get("flag_get"):
                total_reward += self._flag_reward
                self._episode_flag = True

            # === Time Penalty ===
            # Penalize time spent (optional)
            if self._time_penalty:
                time_value = step_info.get("time")
                if time_value is not None and self._prev_time is not None:
                    elapsed = self._prev_time - float(time_value)
                    if elapsed > 0:
                        total_reward += self._time_penalty * elapsed
                    self._prev_time = float(time_value)

            # Track lives for death penalty
            life_value = step_info.get("life")
            if life_value is not None:
                self._prev_lives = int(life_value)

            # Check termination
            if done:
                terminal = True
                break

        # Update buffer with final frame
        if obs is not None:
            self._buffer[0][:] = obs

        self._done = terminal

        # === Death Penalty ===
        # Apply penalty if episode ended due to death (not flag)
        is_last = self._done
        if (
            is_last and
            self._death_penalty and
            not step_info.get("flag_get")
        ):
            total_reward += self._death_penalty

        # Discount factor for next step (0 if terminal, 1 otherwise)
        info = {"discount": np.array(0.0 if is_last else 1.0, dtype=np.float32)}

        return self._create_observation(
            total_reward,
            is_last=is_last,
            is_terminal=terminal,
            info=info
        )

    def reset(self) -> Dict:
        """
        Reset environment to initial state

        Returns:
            Initial observation dictionary
        """
        # Reset with seed
        if self._seed is not None:
            try:
                obs = self._env.reset(seed=int(self._base_seed))
            except TypeError:
                # Older gym versions
                self._env.seed(int(self._base_seed))
                obs = self._env.reset()
            self._base_seed = (self._base_seed + 1) % (2**31)
        else:
            obs = self._env.reset()

        # Reset buffers and state
        obs = self._reset_buffers(obs)
        self._prev_x_pos = float(getattr(self._env.unwrapped, "_x_position", 0.0))
        self._prev_time = 400
        self._prev_lives = 2

        # Create initial observation (is_first=True)
        transition = self._create_observation(0.0, is_first=True)
        obs_dict, _, _, _ = transition
        return obs_dict

    def close(self):
        """Close environment"""
        return self._env.close()

    def _reset_buffers(self, obs: np.ndarray) -> np.ndarray:
        """
        Reset frame buffers

        Args:
            obs: Initial observation

        Returns:
            Processed observation
        """
        obs = self._ensure_obs(obs)

        # Initialize buffers
        if not hasattr(self, "_buffer"):
            self._buffer = [np.zeros_like(obs), np.zeros_like(obs)]
        elif self._buffer[0].shape != obs.shape:
            self._buffer = [np.zeros_like(obs), np.zeros_like(obs)]

        self._buffer[0][:] = obs
        self._buffer[1].fill(0)

        # Reset episode state
        self._done = False
        self._episode_flag = False
        self._prev_time = 400
        self._prev_lives = 2

        return obs

    def _ensure_obs(self, obs: any) -> np.ndarray:
        """
        Extract observation from potential tuple

        Args:
            obs: Observation (possibly wrapped in tuple)

        Returns:
            Raw observation array
        """
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs

    def _create_observation(
        self,
        reward: float,
        is_first: bool = False,
        is_last: bool = False,
        is_terminal: bool = False,
        info: Optional[Dict] = None
    ) -> Tuple[Dict, float, bool, Dict]:
        """
        Create observation dictionary in DreamerV3 format

        Process:
        1. Apply max pooling over last 2 frames
        2. Resize to target size
        3. Convert to grayscale if needed
        4. Package with metadata flags

        Args:
            reward: Reward value
            is_first: Whether this is the first step of episode
            is_last: Whether this is the last step of episode
            is_terminal: Whether episode ended
            info: Additional information

        Returns:
            Tuple of (obs_dict, reward, is_last, info)
        """
        # Max pooling over last 2 frames (reduces flicker)
        np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
        image = self._buffer[0]

        # Resize to target size
        if image.shape[:2] != self._size:
            if self._resize_method == "opencv":
                image = self._cv2.resize(
                    image, self._size, interpolation=self._cv2.INTER_AREA
                )
            else:
                image = self._image.fromarray(image)
                image = image.resize(self._size, self._image.NEAREST)
                image = np.array(image)

        # Convert to grayscale if needed
        if self._grayscale:
            # Standard RGB to grayscale weights
            weights = [0.299, 0.587, 0.114]
            image = np.tensordot(image, weights, axes=(-1, 0)).astype(image.dtype)
            image = image[:, :, None]

        # Create observation dictionary
        obs = {
            "image": image,
            "is_terminal": is_terminal,
            "is_first": is_first
        }

        # Add info metadata
        info = info or {}
        if is_last:
            # Log episode success (reached flag)
            info["mario_success"] = bool(self._episode_flag)
            info["log_mario_success"] = np.array(
                float(self._episode_flag), dtype=np.float32
            )
            info["log_mario_episodes"] = np.array(1.0, dtype=np.float32)

        return obs, reward, is_last, info


def make_mario_env(config: any) -> MarioEnv:
    """
    Create Mario environment from configuration

    Args:
        config: Configuration object with environment parameters

    Returns:
        Configured Mario environment
    """
    return MarioEnv(
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
        seed=config.seed
    )
