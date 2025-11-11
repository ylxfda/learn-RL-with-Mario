"""
World Model for DreamerV3

The world model learns to predict future observations, rewards, and episode
continuations from past experience. It consists of:
- Encoder: Maps observations to embeddings
- RSSM: Recurrent State Space Model for dynamics
- Decoder: Reconstructs observations from latent states
- Reward Head: Predicts rewards
- Continuation Head: Predicts episode continuation

Paper Reference: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
Section 2.1 "World Model Learning"
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

from dreamerv3.networks import RSSM, MultiEncoder, MultiDecoder, MLP
from dreamerv3.utils import tools


class WorldModel(nn.Module):
    """
    World Model - Learns Environment Dynamics

    The world model predicts future states, observations, rewards, and episode
    continuations. It enables the agent to "imagine" future trajectories for
    planning without interacting with the environment.

    Components:
    1. Encoder: o_t -> e_t
    2. RSSM Dynamics: (h_{t-1}, z_{t-1}, a_{t-1}, e_t) -> (h_t, z_t)
    3. Decoder: (h_t, z_t) -> ô_t
    4. Reward predictor: (h_t, z_t) -> r̂_t
    5. Continuation predictor: (h_t, z_t) -> ĉ_t

    Training uses the following loss:
    L = E[reconstruction_loss + reward_loss + continuation_loss + KL_loss]

    Paper Reference: Algorithm 1, Section 2.1
    """

    def __init__(
        self,
        obs_space: Any,
        act_space: Any,
        step: int,
        config: Any
    ):
        """
        Initialize world model

        Args:
            obs_space: Observation space (gym.spaces.Dict)
            act_space: Action space
            step: Current training step (for logging)
            config: Configuration object with hyperparameters
        """
        super(WorldModel, self).__init__()

        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config

        # Extract observation shapes
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        # === Encoder ===
        # Maps observations to latent embeddings
        # Paper: Section 2.1, encoder f_enc
        self.encoder = MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim

        # === RSSM Dynamics ===
        # Recurrent State Space Model for temporal dynamics
        # Paper: Section 2.1, dynamics model
        self.dynamics = RSSM(
            stoch_dim=config.dyn_stoch,
            deter_dim=config.dyn_deter,
            hidden_dim=config.dyn_hidden,
            rec_depth=config.dyn_rec_depth,
            discrete_dim=config.dyn_discrete,
            activation=config.act,
            use_layer_norm=config.norm,
            mean_activation=config.dyn_mean_act,
            std_activation=config.dyn_std_act,
            min_std=config.dyn_min_std,
            unimix_ratio=config.unimix_ratio,
            initial_state=config.initial,
            num_actions=config.num_actions,
            embed_dim=self.embed_size,
            device=config.device
        )

        # Calculate feature size (concatenated stoch + deter)
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        # === Prediction Heads ===
        self.heads = nn.ModuleDict()

        # Decoder: Reconstructs observations
        # Paper: Section 2.1, decoder f_dec
        self.heads["decoder"] = MultiDecoder(
            feat_size, shapes, **config.decoder
        )

        # Reward Head: Predicts rewards
        # Paper: Section 2.1, reward predictor r̂_t
        self.heads["reward"] = MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward"
        )

        # Continuation Head: Predicts episode continuation (1 - terminal)
        # Paper: Section 2.1, continuation predictor ĉ_t
        self.heads["cont"] = MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont"
        )

        # Verify all grad_heads exist
        for name in config.grad_heads:
            assert name in self.heads, f"grad_head '{name}' not found in heads"

        # === Optimizer ===
        # Single optimizer for all world model components
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp
        )

        print(
            f"Optimizer model_opt has "
            f"{sum(param.numel() for param in self.parameters())} variables."
        )

        # Loss scales
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"]
        )

    def _train(self, data: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict, Dict]:
        """
        Train world model on a batch of experience

        This implements one update step of Algorithm 1 in the paper.

        Training process:
        1. Encode observations: e_t = f_enc(o_t)
        2. Compute posterior and prior: z_t ~ q(z_t|h_t,e_t), z_t ~ p(z_t|h_t)
        3. Compute KL loss: KL[q||p] for representation learning
        4. Predict observations: ô_t ~ p(o_t|h_t,z_t)
        5. Predict rewards: r̂_t ~ p(r_t|h_t,z_t)
        6. Predict continuation: ĉ_t ~ p(c_t|h_t,z_t)
        7. Optimize all components jointly

        Paper Reference: Section 2.1 "World Model Learning", Algorithm 1

        Args:
            data: Dictionary containing:
                - 'action': (batch_size, batch_length, act_dim)
                - 'image': (batch_size, batch_length, H, W, C)
                - 'reward': (batch_size, batch_length)
                - 'discount': (batch_size, batch_length)
                - 'is_first': (batch_size, batch_length)
                - 'is_terminal': (batch_size, batch_length)

        Returns:
            Tuple of (posterior_states, context_dict, metrics_dict)
        """
        # Preprocess data (normalize images, add dimensions)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                # === 1. Encode observations ===
                # o_t -> e_t
                embed = self.encoder(data)

                # === 2. Compute dynamics (posterior and prior) ===
                # Posterior: z_t ~ q(z_t | h_t, e_t) using observations
                # Prior: z_t ~ p(z_t | h_t) for imagination
                posterior, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

                # === 3. Compute KL divergence loss ===
                # Balances representation and dynamics learning
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale

                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    posterior, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                # === 4. Compute prediction losses ===
                preds = {}
                losses = {}

                for name, head in self.heads.items():
                    # Get features from posterior
                    # Some heads train with gradients, others don't
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_latent_state_feature(posterior)
                    feat = feat if grad_head else feat.detach()

                    # Predict
                    pred = head(feat)

                    # Handle multi-output heads
                    if isinstance(pred, dict):
                        preds.update(pred)
                    else:
                        preds[name] = pred

                # Compute negative log-likelihood for each prediction
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss

                # === 5. Scale and combine losses ===
                scaled_losses = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }

                # Total loss
                model_loss = sum(scaled_losses.values()) + kl_loss

            # === 6. Optimize ===
            metrics = self._model_opt(
                torch.mean(model_loss),
                self.parameters()
            )

        # === 7. Collect metrics for logging ===
        metrics.update({
            f"{name}_loss": tools.to_np(loss)
            for name, loss in losses.items()
        })
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = tools.to_np(dyn_loss)
        metrics["rep_loss"] = tools.to_np(rep_loss)
        metrics["kl"] = tools.to_np(torch.mean(kl_value))

        with torch.amp.autocast('cuda', enabled=self._use_amp):
            # Distribution entropies
            metrics["prior_ent"] = tools.to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = tools.to_np(
                torch.mean(self.dynamics.get_dist(posterior).entropy())
            )

            # Context for exploration (if used)
            context = dict(
                embed=embed,
                feat=self.dynamics.get_latent_state_feature(posterior),
                kl=kl_value,
                postent=self.dynamics.get_dist(posterior).entropy()
            )

        # Detach posterior for policy training
        posterior = {k: v.detach() for k, v in posterior.items()}

        return posterior, context, metrics

    def preprocess(self, obs: Dict[str, any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess observations for training

        Converts to tensors, normalizes images, and adds necessary flags.

        Args:
            obs: Dictionary of observations (NumPy arrays or tensors)

        Returns:
            Dictionary of preprocessed tensors
        """
        # Convert to tensors
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            if not isinstance(v, torch.Tensor) else v.to(self._config.device)
            for k, v in obs.items()
        }

        # Normalize images from [0, 255] to [0, 1]
        obs["image"] = obs["image"] / 255.0

        # Apply discount factor
        if "discount" in obs:
            obs["discount"] = obs["discount"] * self._config.discount
            # Add feature dimension: (batch, time) -> (batch, time, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)

        # Ensure required flags are present
        assert "is_first" in obs, "'is_first' flag required for RSSM initialization"
        assert "is_terminal" in obs, "'is_terminal' flag required for continuation head"

        # Continuation target: 1 if episode continues, 0 if terminal
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)

        return obs

    def video_pred(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate video prediction for visualization

        Shows the model's ability to:
        1. Reconstruct observations from posterior (using observations)
        2. Imagine future observations from prior (without observations)

        Creates a concatenated video showing:
        - Ground truth observations
        - Posterior reconstructions (first 5 steps)
        - Prior predictions (remaining steps)
        - Prediction error

        Args:
            data: Batch of experience

        Returns:
            Video tensor: (batch_size, time_steps, height*3, width, channels)
            The height is tripled to show [truth | model | error] stacked vertically
        """
        data = self.preprocess(data)
        embed = self.encoder(data)

        # Use first 6 trajectories, observe first 5 steps
        # Compute posterior states for conditioning
        states, _ = self.dynamics.observe(
            embed[:6, :5],
            data["action"][:6, :5],
            data["is_first"][:6, :5]
        )

        # Reconstruct from posterior (with observations)
        recon = self.heads["decoder"](
            self.dynamics.get_latent_state_feature(states)
        )["image"].mode()[:6]

        # Predict rewards from posterior
        reward_post = self.heads["reward"](
            self.dynamics.get_latent_state_feature(states)
        ).mode()[:6]

        # Initialize imagination from last posterior state
        init = {k: v[:, -1] for k, v in states.items()}

        # Imagine future without observations (open-loop prediction)
        # FIXED: Use action[4:-1] to maintain time alignment
        # The last posterior state (at index 4) was computed using obs[4] and action[4]
        # So next prediction should use action[4] to predict obs[5]
        # Also pass is_first flags to handle episode boundaries correctly
        # Note: action[4:-1] and is_first[5:] must have the same length
        prior = self.dynamics.imagine_with_action(
            data["action"][:6, 4:-1],     # Exclude last action to match is_first length
            init,
            data["is_first"][:6, 5:]      # is_first for steps 5 onwards
        )

        # Decode imagined states
        openl = self.heads["decoder"](
            self.dynamics.get_latent_state_feature(prior)
        )["image"].mode()

        # Predict rewards from imagination
        reward_prior = self.heads["reward"](
            self.dynamics.get_latent_state_feature(prior)
        ).mode()

        # Combine posterior reconstruction and prior imagination
        # First 5 steps: reconstruction, remaining: imagination
        # Skip first imagined frame as it corresponds to step 5 (already have reconstruction)
        model = torch.cat([recon[:, :5], openl[:, 1:]], dim=1)

        # Match truth length to model (exclude last frame since we don't predict it)
        truth = data["image"][:6, :-1]

        # Compute prediction error (shifted to [0, 1] range)
        error = (model - truth + 1.0) / 2.0

        # Concatenate vertically: [truth | model | error] (stacked top to bottom)
        return torch.cat([truth, model, error], dim=2)


class RewardEMA:
    """
    Exponential Moving Average for Reward Normalization

    Tracks the 5th and 95th percentiles of rewards using EMA,
    then normalizes rewards to have consistent scale. This helps
    the value function handle rewards of varying magnitudes.

    Paper Reference: Section 2.2 "Actor-Critic Learning"
    """

    def __init__(self, device: str, alpha: float = 1e-2):
        """
        Initialize reward EMA

        Args:
            device: Device for tensors
            alpha: EMA coefficient (higher = faster adaptation)
        """
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(
        self,
        x: torch.Tensor,
        ema_vals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update EMA and compute normalization parameters

        Args:
            x: Reward values, shape (...)
            ema_vals: Current EMA values [low, high], shape (2,)

        Returns:
            Tuple of (offset, scale) for normalization:
            normalized = (x - offset) / scale
        """
        # Compute current quantiles
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)

        # Update EMA (in-place)
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals

        # Compute normalization parameters
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]

        return offset.detach(), scale.detach()
