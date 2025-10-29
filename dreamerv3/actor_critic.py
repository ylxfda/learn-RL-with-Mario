"""
Actor-Critic Learning for DreamerV3

Implements policy (actor) and value function (critic) learning using
imagination in the learned world model. The agent:
1. Imagines trajectories using the world model
2. Computes returns using the value function
3. Updates the actor to maximize returns
4. Updates the critic to predict returns accurately

Paper Reference: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
Section 2.2 "Actor-Critic Learning"
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Callable, Any

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

from dreamerv3.networks import MLP
from dreamerv3.world_model import RewardEMA
from dreamerv3.utils import tools


class ActorCritic(nn.Module):
    """
    Actor-Critic for Policy Learning via Imagination

    Uses the learned world model to imagine trajectories and learn
    a policy (actor) and value function (critic) without environment
    interaction.

    Process:
    1. Start from real latent states
    2. Imagine forward using current policy: π(a_t | z_t, h_t)
    3. Predict rewards and values along imagined trajectory
    4. Compute λ-returns as targets
    5. Update actor to maximize returns
    6. Update critic to predict returns

    Paper Reference: Section 2.2, Algorithm 1
    """

    def __init__(self, config: Any, world_model: nn.Module):
        """
        Initialize actor-critic

        Args:
            config: Configuration object
            world_model: Learned world model for imagination
        """
        super(ActorCritic, self).__init__()

        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model

        # Calculate feature size (stoch + deter) - discrete only
        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter

        # === Actor Network (Policy) ===
        # Maps latent features to action distribution (discrete actions only)
        # Paper: Section 2.2, policy π(a_t | z_t, h_t)
        self.actor = MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.actor["dist"],  # Should be "onehot" for discrete actions
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor"
        )

        # === Critic Network (Value Function) ===
        # Estimates expected return from latent state
        # Paper: Section 2.2, value function V(z_t, h_t)
        self.value = MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value"
        )

        # === Slow Target Network ===
        # Stabilizes value learning with slowly updated target
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0

        # === Optimizers ===
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)

        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw
        )
        print(
            f"Optimizer actor_opt has "
            f"{sum(param.numel() for param in self.actor.parameters())} variables."
        )

        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw
        )
        print(
            f"Optimizer value_opt has "
            f"{sum(param.numel() for param in self.value.parameters())} variables."
        )

        # === Reward Normalization ===
        # Track reward statistics for normalization
        if self._config.reward_EMA:
            # Register as buffer so it's saved/loaded with model
            self.register_buffer(
                "ema_vals",
                torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start: Dict[str, torch.Tensor],
        objective: Callable
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor, torch.Tensor, Dict]:
        """
        Train actor and critic using imagined trajectories

        Process (Algorithm 1 in paper):
        1. Imagine trajectories from start states using current policy
        2. Compute rewards and values along trajectories
        3. Compute λ-returns as targets
        4. Update actor to maximize returns
        5. Update critic to predict returns

        Args:
            start: Starting latent states (from real experience)
                   Dictionary with 'stoch', 'deter', etc.
            objective: Function to compute rewards (usually from world model)

        Returns:
            Tuple of (imag_feat, imag_state, imag_action, weights, metrics)
        """
        # Update slow target network
        self._update_slow_target()

        metrics = {}

        # === 1. Imagine Trajectories ===
        # Roll out policy in latent space
        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # Imagine forward using policy
                # Output shapes: (horizon+1, batch, ...)
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )

                # Compute rewards from world model
                # Shape: (horizon+1, batch, 1)
                reward = objective(imag_feat, imag_state, imag_action)

                # Compute entropy for regularization
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()

                # === 2. Compute Target Returns ===
                # Uses λ-returns for bias-variance tradeoff
                # Paper: Section 2.2, Equation for returns
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )

                # === 3. Compute Actor Loss ===
                # Policy gradient to maximize returns
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base
                )

                # Add entropy regularization
                # Encourages exploration
                actor_loss = actor_loss - self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)

                metrics.update(mets)
                value_input = imag_feat

        # === 4. Compute Critic Loss ===
        # Train value function to predict returns
        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                # Predict values
                # Shape: (horizon, batch, 1)
                value = self.value(value_input[:-1].detach())

                # Stack targets: List[(batch, 1)] -> (horizon, batch, 1)
                target = torch.stack(target, dim=1)

                # Negative log-likelihood loss
                value_loss = -value.log_prob(target.detach())

                # Regularization with slow target
                if self._config.critic["slow_target"]:
                    slow_target = self._slow_value(value_input[:-1].detach())
                    value_loss = value_loss - value.log_prob(slow_target.mode().detach())

                # Weight by trajectory probabilities (for off-policy correction)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        # === 5. Log Metrics ===
        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))

        # Log actions (discrete only)
        metrics.update(
            tools.tensorstats(
                torch.argmax(imag_action, dim=-1).float(),
                "imag_action"
            )
        )

        metrics["actor_entropy"] = tools.to_np(torch.mean(actor_ent))

        # === 6. Optimize ===
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(
        self,
        start: Dict[str, torch.Tensor],
        policy: nn.Module,
        horizon: int
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Imagine trajectories by rolling out policy in latent space

        Starting from real latent states, repeatedly:
        1. Sample action from policy
        2. Predict next latent state using world model dynamics
        3. Extract features for downstream predictions

        Args:
            start: Initial latent states, each tensor shape (batch, ...)
            policy: Policy network (actor)
            horizon: Number of steps to imagine

        Returns:
            Tuple of:
            - imag_feat: Features, shape (horizon+1, batch, feat_dim)
            - imag_state: Latent states, dict with (horizon+1, batch, ...)
            - imag_action: Actions, shape (horizon, batch, act_dim)
        """
        dynamics = self._world_model.dynamics

        # Flatten batch dimensions for processing
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev: Tuple, _) -> Tuple:
            """
            Single imagination step

            Args:
                prev: (prev_state, prev_feat, prev_action)
                _: Unused (timestep index)

            Returns:
                (next_state, features, action)
            """
            state, _, _ = prev

            # Extract features from state
            feat = dynamics.get_feat(state)

            # Detach to stop gradients through previous timesteps
            # (for computational efficiency)
            inp = feat.detach()

            # Sample action from policy
            action = policy(inp).sample()

            # Predict next state
            succ = dynamics.img_step(state, action)

            return succ, feat, action

        # Roll out for 'horizon' steps
        # tools.static_scan applies 'step' sequentially
        succ, feats, actions = tools.static_scan(
            step,
            [torch.arange(horizon)],  # Dummy input for iteration
            (start, None, None)
        )

        # Prepend initial state to state sequence
        # states: (horizon+1, batch, ...)
        states = {
            k: torch.cat([start[k][None], v[:-1]], dim=0)
            for k, v in succ.items()
        }

        return feats, states, actions

    def _compute_target(
        self,
        imag_feat: torch.Tensor,
        imag_state: Dict[str, torch.Tensor],
        reward: torch.Tensor
    ) -> Tuple[list, torch.Tensor, torch.Tensor]:
        """
        Compute target returns using λ-returns (TD(λ))

        The λ-return interpolates between 1-step TD and Monte Carlo:
        G^λ_t = r_t + γ[(1-λ)V(s_{t+1}) + λG^λ_{t+1}]

        This provides a bias-variance tradeoff.

        Paper Reference: Section 2.2, uses λ=0.95

        Args:
            imag_feat: Imagined features, (horizon+1, batch, feat_dim)
            imag_state: Imagined latent states
            reward: Predicted rewards, (horizon+1, batch, 1)

        Returns:
            Tuple of:
            - target: List of returns for each timestep
            - weights: Importance weights for off-policy correction
            - base: Value estimates (for advantage computation)
        """
        # Predict continuation probability (1 - terminal)
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            # Shape: (horizon+1, batch, 1)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)

        # Predict values for all imagined states
        # Shape: (horizon+1, batch, 1)
        value = self.value(imag_feat).mode()

        # Compute λ-returns recursively from end to start
        # Paper: λ=0.95 for balance between bias and variance
        target = tools.lambda_return(
            reward[1:],           # Rewards from t=1 to T
            value[:-1],           # Values from t=0 to T-1
            discount[1:],         # Discounts
            bootstrap=value[-1],  # Bootstrap from final value
            lambda_=self._config.discount_lambda,
            axis=0
        )

        # Compute importance weights for trajectories
        # Used for off-policy correction and policy gradient
        # weights[t] = γ^0 * γ^1 * ... * γ^{t-1}
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], dim=0),
            dim=0
        ).detach()

        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat: torch.Tensor,
        imag_action: torch.Tensor,
        target: list,
        weights: torch.Tensor,
        base: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute actor (policy) loss

        DreamerV3 supports three gradient estimators:
        1. 'dynamics': Backprop through dynamics (default)
        2. 'reinforce': REINFORCE estimator
        3. 'both': Interpolation between the two

        Paper Reference: Section 2.2, uses dynamics gradients

        Args:
            imag_feat: Imagined features, (horizon+1, batch, feat_dim)
            imag_action: Sampled actions, (horizon, batch, act_dim)
            target: Target returns (list of tensors)
            weights: Importance weights, (horizon+1, batch, 1)
            base: Value baseline, (horizon, batch, 1)

        Returns:
            Tuple of (actor_loss, metrics)
        """
        metrics = {}

        # Detach features to stop gradients through world model during policy update
        inp = imag_feat.detach()

        # Recompute policy distribution
        policy = self.actor(inp)

        # Stack targets to tensor
        target = torch.stack(target, dim=1)

        # === Reward Normalization ===
        # Normalize returns for stable learning
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale

            # Advantage = normalized_return - normalized_baseline
            adv = normed_target - normed_base

            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = tools.to_np(self.ema_vals[0])
            metrics["EMA_095"] = tools.to_np(self.ema_vals[1])

        # === Compute Policy Gradient ===
        if self._config.imag_gradient == "dynamics":
            # Backprop through dynamics (default)
            # Most efficient, leverages differentiable world model
            actor_target = adv

        elif self._config.imag_gradient == "reinforce":
            # REINFORCE estimator
            # log π(a|s) * (return - baseline)
            # Convert imag_action to strict one-hot (remove straight-through gradients)
            imag_action_onehot = F.one_hot(
                torch.argmax(imag_action, dim=-1),
                imag_action.shape[-1]
            ).float()
            actor_target = (
                policy.log_prob(imag_action_onehot)[:-1][:, :, None] *
                (target - self.value(imag_feat[:-1]).mode()).detach()
            )

        elif self._config.imag_gradient == "both":
            # Interpolation between dynamics and REINFORCE
            # Convert imag_action to strict one-hot (remove straight-through gradients)
            imag_action_onehot = F.one_hot(
                torch.argmax(imag_action, dim=-1),
                imag_action.shape[-1]
            ).float()
            actor_target = (
                policy.log_prob(imag_action_onehot)[:-1][:, :, None] *
                (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix

        else:
            raise NotImplementedError(self._config.imag_gradient)

        # Weighted negative objective (loss to minimize)
        actor_loss = -weights[:-1] * actor_target

        return actor_loss, metrics

    def _update_slow_target(self):
        """
        Update slow target network for critic

        Uses exponential moving average (EMA) to slowly update target.
        This stabilizes value learning by providing consistent targets.

        Paper: Common practice in RL (e.g., DDPG, TD3)
        """
        if self._config.critic["slow_target"]:
            # Update every N steps
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]

                # EMA update: target = mix * source + (1-mix) * target
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data

            self._updates += 1
