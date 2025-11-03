"""
Rollout Buffer for PPO with GAE (Generalized Advantage Estimation)

This module implements on-policy experience storage and advantage computation
for Proximal Policy Optimization.

Key Differences from DreamerV3's Replay Buffer:
1. On-policy: Data is used once then discarded (no replay)
2. Fixed size: Stores exactly one rollout of experience
3. GAE computation: Computes advantages using λ-returns

Paper References:
- GAE: Schulman et al. (2016) "High-Dimensional Continuous Control Using
  Generalized Advantage Estimation"
- PPO: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"

Naming Conventions (from papers):
- γ (gamma): Discount factor for returns
- λ (lambda): GAE parameter for bias-variance tradeoff
- A^{GAE(γ,λ)}_t: Generalized advantage estimate
- δ_t: TD residual = r_t + γV(s_{t+1}) - V(s_t)
"""

import torch
import numpy as np
from typing import Tuple, Dict


class RolloutBuffer:
    """
    Storage for on-policy rollouts with GAE computation

    The buffer stores trajectories from parallel environments and computes
    advantage estimates using Generalized Advantage Estimation (GAE).

    Paper Reference: Schulman et al. (2016), Equation 16
    GAE formula:
        Â_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    The advantage estimates are used in the PPO policy gradient:
        L^{CLIP}(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

    Buffer Organization:
        - Fixed capacity: num_steps × num_envs
        - Filled sequentially during rollout
        - Reset after each training update

    Attributes:
        num_steps (int): Number of steps per rollout
        num_envs (int): Number of parallel environments
        obs_shape (tuple): Shape of observations (C, H, W)
        device (torch.device): Device for tensor operations
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, int, int],
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize rollout buffer

        Args:
            num_steps: Number of steps per rollout (typically 128-2048)
                      Trade-off: larger = more data but less on-policy
            num_envs: Number of parallel environments
            obs_shape: Shape of observations (channels, height, width)
                      For Mario: (3, 64, 64)
            device: Device for storing tensors ('cpu' or 'cuda')

        Buffer Size:
            Total transitions stored = num_steps × num_envs
            For PPO with 32 envs and 128 steps: 4096 transitions
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.device = device
        self.step = 0

        # Allocate storage buffers
        # All tensors have shape (num_steps, num_envs, ...)
        self.observations = torch.zeros(
            (num_steps, num_envs) + obs_shape,
            dtype=torch.uint8,
            device=device
        )
        self.actions = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.long,
            device=device
        )
        self.rewards = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.values = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.log_probs = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.dones = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=device
        )

        # These are computed after rollout is complete
        self.advantages = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.returns = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=device
        )

    def add(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        reward: np.ndarray,
        done: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ):
        """
        Add one step of experience from all parallel environments

        This method is called at each timestep during rollout collection.

        Args:
            obs: Observations from all environments
                 Shape: (num_envs, C, H, W)
                 Dtype: uint8, range [0, 255]
            action: Actions taken in all environments
                   Shape: (num_envs,)
                   Dtype: long, range [0, num_actions-1]
            reward: Rewards received in all environments
                   Shape: (num_envs,)
                   Dtype: float32
            done: Episode termination flags
                 Shape: (num_envs,)
                 Dtype: float32, values {0, 1}
            value: Value estimates V_ϕ(s_t) from critic
                  Shape: (num_envs,)
                  Dtype: float32
            log_prob: Log probabilities log π_θ(a_t|s_t) from actor
                     Shape: (num_envs,)
                     Dtype: float32
        """
        if self.step >= self.num_steps:
            raise RuntimeError(f"Buffer is full (capacity: {self.num_steps})")

        # Convert numpy arrays to tensors
        obs_tensor = torch.from_numpy(obs).to(self.device)
        reward_tensor = torch.from_numpy(reward).to(self.device)
        done_tensor = torch.from_numpy(done).to(self.device)

        # Store transition
        self.observations[self.step] = obs_tensor
        self.actions[self.step] = action
        self.rewards[self.step] = reward_tensor
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        self.dones[self.step] = done_tensor

        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Compute returns and advantages using GAE

        This method is called once after collecting a full rollout, before
        starting PPO training updates.

        Paper Reference: Schulman et al. (2016), Equation 16
        GAE computes advantages as exponentially-weighted average of TD residuals:
            Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
            where δ_t = r_t + γV(s_{t+1})(1-done_t) - V(s_t)

        Args:
            last_values: Value estimates V_ϕ(s_T) for states after rollout
                        Shape: (num_envs,)
                        Used to bootstrap returns at rollout boundary
            last_dones: Done flags for states after rollout
                       Shape: (num_envs,)
                       Used to zero out bootstrap if episode ended
            gamma: Discount factor γ (default: 0.99)
                  Higher γ = more far-sighted (considers distant rewards)
                  Lower γ = more myopic (considers immediate rewards)
            gae_lambda: GAE parameter λ (default: 0.95)
                       Higher λ = lower bias, higher variance (like MC)
                       Lower λ = higher bias, lower variance (like TD)
                       λ=1 recovers Monte Carlo returns
                       λ=0 recovers 1-step TD

        Effects:
            - Sets self.advantages: Advantage estimates Â_t
            - Sets self.returns: Target returns V_t^{targ} = Â_t + V(s_t)

        Implementation:
            Uses reverse iteration to compute advantages efficiently in O(T)
            instead of naive O(T^2) implementation.
        """
        if self.step != self.num_steps:
            raise RuntimeError(
                f"Buffer not full: {self.step}/{self.num_steps} steps filled"
            )

        # Convert last_dones to tensor
        last_dones_tensor = torch.from_numpy(last_dones).float().to(self.device)

        # Initialize for backward pass
        # gae: running sum of discounted advantages
        # next_values: V(s_{t+1}) for bootstrapping
        gae = 0
        next_values = last_values
        next_non_terminal = 1.0 - last_dones_tensor

        # Compute advantages backward through time
        # This is more efficient than forward computation
        for step in reversed(range(self.num_steps)):
            # TD residual: δ_t = r_t + γV(s_{t+1})(1-done_t) - V(s_t)
            # The (1-done_t) factor zeros out V(s_{t+1}) if episode ended
            delta = (
                self.rewards[step]
                + gamma * next_values * next_non_terminal
                - self.values[step]
            )

            # GAE: Â_t = δ_t + (γλ)Â_{t+1}
            # This implements the infinite sum via recursion
            gae = delta + gamma * gae_lambda * next_non_terminal * gae

            # Store advantage for this timestep
            self.advantages[step] = gae

            # Update for next iteration (moving backward)
            next_values = self.values[step]
            next_non_terminal = 1.0 - self.dones[step]

        # Compute returns as advantages + values
        # V_t^{targ} = Â_t + V(s_t)
        # This is the target for value function training
        self.returns = self.advantages + self.values

    def get_batches(
        self,
        batch_size: int,
        normalize_advantages: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Get all data as randomized mini-batches for training

        After computing advantages, this method prepares the data for
        PPO training by:
        1. Flattening (num_steps, num_envs) -> (num_steps * num_envs)
        2. Optionally normalizing advantages (recommended for stability)
        3. Yielding random mini-batches

        Paper Reference: Schulman et al. (2017), Algorithm 1
        PPO performs multiple epochs of mini-batch SGD on the rollout data.

        Args:
            batch_size: Size of mini-batches (typically 256-2048)
                       Smaller = more updates but noisier gradients
                       Larger = fewer updates but more stable
            normalize_advantages: Whether to normalize advantages (default: True)
                                 Normalization: Â = (Â - mean(Â)) / (std(Â) + 1e-8)
                                 This improves training stability

        Yields:
            Dictionary with mini-batch data:
            - 'observations': Tensor of shape (batch_size, C, H, W)
            - 'actions': Tensor of shape (batch_size,)
            - 'old_log_probs': Tensor of shape (batch_size,)
            - 'advantages': Tensor of shape (batch_size,)
            - 'returns': Tensor of shape (batch_size,)

        Example:
            >>> for batch in buffer.get_batches(batch_size=256):
            >>>     # Train actor and critic on this batch
            >>>     actor_loss = compute_actor_loss(batch)
            >>>     critic_loss = compute_critic_loss(batch)
        """
        if self.step != self.num_steps:
            raise RuntimeError(
                f"Buffer not full: {self.step}/{self.num_steps} steps filled"
            )

        # Flatten all buffers: (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
        num_samples = self.num_steps * self.num_envs

        obs_flat = self.observations.reshape(num_samples, *self.obs_shape)
        actions_flat = self.actions.reshape(num_samples)
        log_probs_flat = self.log_probs.reshape(num_samples)
        advantages_flat = self.advantages.reshape(num_samples)
        returns_flat = self.returns.reshape(num_samples)

        # Normalize advantages (recommended for training stability)
        if normalize_advantages:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (
                advantages_flat.std() + 1e-8
            )

        # Generate random permutation for mini-batch sampling
        indices = torch.randperm(num_samples, device=self.device)

        # Yield mini-batches
        start_idx = 0
        while start_idx < num_samples:
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            yield {
                'observations': obs_flat[batch_indices],
                'actions': actions_flat[batch_indices],
                'old_log_probs': log_probs_flat[batch_indices],
                'advantages': advantages_flat[batch_indices],
                'returns': returns_flat[batch_indices]
            }

            start_idx = end_idx

    def reset(self):
        """
        Reset buffer for next rollout

        Clears all stored data and resets step counter.
        Called after PPO training update is complete.
        """
        self.step = 0
        # Note: We don't need to zero out the buffers as they will be overwritten

    def __len__(self) -> int:
        """Return current number of transitions stored"""
        return self.step * self.num_envs
