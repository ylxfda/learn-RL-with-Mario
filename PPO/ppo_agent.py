"""
PPO Agent - Core Algorithm Implementation

This module implements the Proximal Policy Optimization algorithm from:
Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347

PPO is an on-policy actor-critic algorithm that uses a clipped surrogate
objective to prevent destructively large policy updates. It has become
one of the most popular RL algorithms due to its simplicity and effectiveness.

Algorithm Overview (Schulman et al. 2017, Algorithm 1):
1. Collect trajectories using current policy π_{θ_old}
2. Compute advantages Â_t using GAE
3. For K epochs:
    For each mini-batch:
        a. Update policy by maximizing: L^{CLIP}(θ)
        b. Update value function by minimizing: L^{VF}(ϕ)

Key Components:
- Clipped Surrogate Objective: Prevents destructively large policy updates
- Value Function Loss: MSE between predicted and target returns
- Entropy Bonus: Encourages exploration

Naming Conventions (from paper):
- θ: Policy parameters
- ϕ: Value function parameters
- π_θ: Policy (actor)
- V_ϕ: Value function (critic)
- ε: Clipping parameter (typically 0.2)
- γ: Discount factor
- λ: GAE parameter
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Any

from PPO.networks import Actor, Critic
from PPO.rollout_buffer import RolloutBuffer


class PPOAgent:
    """
    PPO Agent with Clipped Surrogate Objective

    This class implements the complete PPO algorithm including:
    - Policy updates with clipped objective
    - Value function updates
    - Entropy regularization
    - Gradient clipping

    Paper Reference: Schulman et al. (2017), Algorithm 1

    The agent maintains:
    - π_θ: Policy network (Actor)
    - V_ϕ: Value network (Critic)
    - Rollout buffer for on-policy data
    - Optimizers for both networks
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        num_actions: int,
        config: Any,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize PPO agent

        Args:
            obs_shape: Observation shape (C, H, W)
                      For Mario: (3, 64, 64)
            num_actions: Number of discrete actions
                        For Mario: 7 (simple controls)
            config: Configuration object with hyperparameters
            device: Device for computation ('cpu' or 'cuda')

        Required config parameters:
            - num_steps: Rollout length (e.g., 128)
            - num_envs: Number of parallel environments (e.g., 32)
            - learning_rate: Learning rate for both networks (e.g., 2.5e-4)
            - gamma: Discount factor (e.g., 0.99)
            - gae_lambda: GAE parameter (e.g., 0.95)
            - clip_coef: PPO clipping parameter ε (e.g., 0.2)
            - vf_coef: Value function loss coefficient (e.g., 0.5)
            - ent_coef: Entropy bonus coefficient (e.g., 0.01)
            - max_grad_norm: Gradient clipping threshold (e.g., 0.5)
            - update_epochs: Number of update epochs per rollout (e.g., 4)
            - batch_size: Mini-batch size (e.g., 256)
        """
        self.config = config
        self.device = device
        self.num_actions = num_actions

        # Extract hyperparameters from config
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_coef = config.clip_coef
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.max_grad_norm = config.max_grad_norm
        self.update_epochs = config.update_epochs
        self.batch_size = config.batch_size

        # Initialize networks
        self.actor = Actor(
            input_channels=obs_shape[0],
            num_actions=num_actions,
            feature_dim=getattr(config, 'feature_dim', 512)
        ).to(device)

        self.critic = Critic(
            input_channels=obs_shape[0],
            feature_dim=getattr(config, 'feature_dim', 512)
        ).to(device)

        # Initialize optimizers
        # Note: Original PPO paper uses separate optimizers for actor and critic
        # with the same learning rate
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )

        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_envs=config.num_envs,
            obs_shape=obs_shape,
            device=device
        )

        # Training statistics
        self.update_count = 0

    def collect_rollout(
        self,
        vec_env,
        obs: np.ndarray
    ) -> np.ndarray:
        """
        Collect one rollout of experience from parallel environments

        This implements the data collection phase of PPO (Algorithm 1, line 3).
        The agent interacts with the environment using the current policy π_{θ_old}
        and stores transitions in the rollout buffer.

        Args:
            vec_env: Vectorized environment (SubprocVecEnv)
            obs: Current observations from all environments
                 Shape: (num_envs, C, H, W)

        Returns:
            Final observations after rollout
            Shape: (num_envs, C, H, W)

        Side Effects:
            - Fills self.rollout_buffer with num_steps transitions
            - Computes advantages using GAE
        """
        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            for step in range(self.config.num_steps):
                # Convert observation to tensor
                # obs shape: (num_envs, H, W, C) -> (num_envs, C, H, W)
                obs_tensor = torch.from_numpy(obs['image']).permute(0, 3, 1, 2).to(self.device)

                # Get action and value from current policy
                actions, log_probs = self.actor.get_action(obs_tensor, deterministic=False)
                values = self.critic.get_value(obs_tensor)

                # Step environments with actions
                actions_np = actions.cpu().numpy()
                next_obs, rewards, dones, infos = vec_env.step(actions_np)

                # Store transition in buffer
                # Note: obs_tensor is already in CHW format which is what the buffer expects
                self.rollout_buffer.add(
                    obs=obs_tensor.cpu().numpy(),  # Convert back to numpy for buffer storage
                    action=actions,
                    reward=rewards,
                    done=dones,
                    value=values,
                    log_prob=log_probs
                )

                # Update observation
                obs = next_obs

        # Compute final values for bootstrapping
        with torch.no_grad():
            next_obs_tensor = torch.from_numpy(obs['image']).permute(0, 3, 1, 2).to(self.device)
            last_values = self.critic.get_value(next_obs_tensor)
            last_dones = np.zeros(self.config.num_envs)  # Assume not done

        # Compute returns and advantages using GAE
        self.rollout_buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_dones=last_dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        return obs

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout

        This implements the optimization phase of PPO (Algorithm 1, lines 6-9).
        The policy and value function are updated using mini-batch SGD for
        multiple epochs.

        Paper Reference: Schulman et al. (2017), Algorithm 1

        Algorithm:
        1. For K epochs:
            2. Randomly shuffle data
            3. For each mini-batch:
                a. Compute policy loss L^{CLIP}
                b. Compute value loss L^{VF}
                c. Compute entropy bonus H
                d. Update networks with total loss

        Returns:
            Dictionary with training statistics:
            - 'policy_loss': Mean policy loss
            - 'value_loss': Mean value loss
            - 'entropy': Mean entropy
            - 'approx_kl': Mean approximate KL divergence
            - 'clipfrac': Fraction of clipped policy updates
        """
        self.actor.train()
        self.critic.train()

        # Statistics accumulators
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clipfracs = []

        # Perform multiple epochs of updates
        for epoch in range(self.update_epochs):
            # Iterate over mini-batches
            for batch in self.rollout_buffer.get_batches(self.batch_size):
                # Extract batch data
                obs = batch['observations']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']

                # === Compute Policy Loss L^{CLIP} ===
                # Get current policy log probabilities and entropy
                new_log_probs, entropy = self.actor.evaluate_actions(obs, actions)

                # Compute probability ratio: r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
                # In log space: log r_t(θ) = log π_θ(a_t|s_t) - log π_{θ_old}(a_t|s_t)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # Compute clipped surrogate objective
                # L^{CLIP}(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
                policy_loss_unclipped = ratio * advantages
                policy_loss_clipped = torch.clamp(
                    ratio,
                    1 - self.clip_coef,
                    1 + self.clip_coef
                ) * advantages

                # Take minimum for pessimistic bound
                policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()

                # === Compute Value Loss L^{VF} ===
                # MSE between predicted values and target returns
                # L^{VF}(ϕ) = E[(V_ϕ(s_t) - V_t^{targ})^2]
                new_values = self.critic.get_value(obs)
                value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                # === Compute Entropy Bonus ===
                # Entropy bonus encourages exploration
                # Total objective includes: -c_2 * H(π_θ(·|s_t))
                entropy_loss = -entropy.mean()

                # === Total Loss ===
                # L(θ,ϕ) = L^{CLIP}(θ) - c_1 * L^{VF}(ϕ) - c_2 * H(π_θ)
                # where c_1 = vf_coef, c_2 = ent_coef
                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                # === Optimization Step ===
                # Update actor
                self.actor_optimizer.zero_grad()
                policy_loss_total = policy_loss + self.ent_coef * entropy_loss
                policy_loss_total.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # === Logging ===
                # Track statistics for monitoring
                with torch.no_grad():
                    # Approximate KL divergence: KL ≈ (r - 1) - log r
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    # Fraction of updates that were clipped
                    clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                approx_kls.append(approx_kl.item())
                clipfracs.append(clipfrac.item())

        # Clear buffer for next rollout
        self.rollout_buffer.reset()
        self.update_count += 1

        # Return mean statistics
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'approx_kl': np.mean(approx_kls),
            'clipfrac': np.mean(clipfracs),
            'update_count': self.update_count
        }

    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> int:
        """
        Get action from policy for a single observation

        This method is used during evaluation or single-environment interaction.

        Args:
            obs: Single observation
                 Shape: (H, W, C)
            deterministic: If True, use greedy action selection
                          If False, sample from policy

        Returns:
            action: Action index
                   Scalar integer in [0, num_actions-1]
        """
        self.actor.eval()

        # Add batch dimension and transpose: (H, W, C) -> (1, C, H, W)
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Get action from policy
        actions, _ = self.actor.get_action(obs_tensor, deterministic=deterministic)

        return int(actions[0].cpu().numpy())

    def save(self, path: str):
        """
        Save agent state to disk

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'update_count': self.update_count
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """
        Load agent state from disk

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
