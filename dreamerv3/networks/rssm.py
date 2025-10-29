"""
Recurrent State Space Model (RSSM) for DreamerV3

This module implements the core world model dynamics as described in:
"Mastering Diverse Domains through World Models" (Hafner et al., 2023)
https://arxiv.org/abs/2301.04104

The RSSM consists of:
- Deterministic path: GRU recurrent network that processes actions and previous states
- Stochastic path: Categorical distributions that capture environment stochasticity
- Posterior: Incorporates observations to update beliefs
- Prior: Predicts next state without observations (used for imagination)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as td
from typing import Dict, Tuple, Optional

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from dreamerv3.utils import tools


class RSSM(nn.Module):
    """
    Recurrent State Space Model (RSSM) - The dynamics model of DreamerV3

    The RSSM maintains a latent state composed of:
    - Deterministic state h_t (deter): Recurrent hidden state from GRU
    - Stochastic state z_t (stoch): Sampled from categorical distribution

    State transitions follow:
    1. Deterministic: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  [GRU]
    2. Prior: z_t ~ p(z_t | h_t)  [Imagination without observations]
    3. Posterior: z_t ~ q(z_t | h_t, o_t)  [Belief update with observations]

    Paper Reference: Section 2 "Method", Algorithm 1
    """

    def __init__(
        self,
        stoch_dim: int = 32,           # Number of categorical distributions (stoch in paper)
        deter_dim: int = 512,          # Deterministic state dimension (deter in paper)
        hidden_dim: int = 512,         # Hidden layer dimension
        rec_depth: int = 1,            # GRU recurrence depth
        discrete_dim: int = 32,        # Classes per categorical (discrete in paper)
        activation: str = "SiLU",      # Activation function
        use_layer_norm: bool = True,   # Whether to use layer normalization
        unimix_ratio: float = 0.01,    # Uniform mixing ratio for categorical
        initial_state: str = "learned",  # Initial state: "learned" or "zeros"
        num_actions: int = None,       # Action space dimension
        embed_dim: int = None,         # Observation embedding dimension
        device: str = None,            # Device for tensors
        # Unused parameters kept for compatibility
        mean_activation: str = "none",
        std_activation: str = "softplus",
        min_std: float = 0.1
    ):
        """
        Initialize the RSSM dynamics model for discrete actions only

        Args:
            stoch_dim: Number of categorical distributions (default: 32)
            deter_dim: Dimension of deterministic recurrent state (default: 512)
            hidden_dim: Hidden layer dimension for MLPs (default: 512)
            rec_depth: Depth of recurrent processing (default: 1)
            discrete_dim: Number of classes per categorical distribution (default: 32)
            activation: Activation function name (default: "SiLU")
            use_layer_norm: Whether to apply layer normalization (default: True)
            unimix_ratio: Uniform mixing ratio for exploration (default: 0.01)
            initial_state: How to initialize recurrent state (default: "learned")
            num_actions: Dimension of action space (required)
            embed_dim: Dimension of observation embeddings (required)
            device: Device to place tensors on (required)
        """
        super(RSSM, self).__init__()

        # Store configuration (only discrete actions supported)
        self._stoch_dim = stoch_dim
        self._deter_dim = deter_dim
        self._hidden_dim = hidden_dim
        self._discrete_dim = discrete_dim
        self._rec_depth = rec_depth
        self._unimix_ratio = unimix_ratio
        self._initial_state = initial_state
        self._num_actions = num_actions
        self._embed_dim = embed_dim
        self._device = device

        # Get activation function
        activation_fn = getattr(torch.nn, activation)

        # === Imagination Network ===
        # Input: [stoch_{t-1}, action_{t-1}] -> Hidden
        # Maps previous stochastic state and action to hidden representation
        # before GRU processing (only discrete stochastic states supported)
        img_input_dim = self._stoch_dim * self._discrete_dim + num_actions

        img_input_layers = []
        img_input_layers.append(nn.Linear(img_input_dim, self._hidden_dim, bias=False))
        if use_layer_norm:
            img_input_layers.append(nn.LayerNorm(self._hidden_dim, eps=1e-03))
        img_input_layers.append(activation_fn())
        self._img_input_net = nn.Sequential(*img_input_layers)
        self._img_input_net.apply(tools.weight_init)

        # === GRU Cell ===
        # Processes hidden representation to update deterministic state
        # h_t = GRU(h_{t-1}, hidden)
        self._gru_cell = GRUCell(self._hidden_dim, self._deter_dim, norm=use_layer_norm)
        self._gru_cell.apply(tools.weight_init)

        # === Imagination Output Network ===
        # Deter -> Hidden (for computing prior)
        # Projects deterministic state to hidden representation for prior prediction
        img_output_layers = []
        img_output_layers.append(nn.Linear(self._deter_dim, self._hidden_dim, bias=False))
        if use_layer_norm:
            img_output_layers.append(nn.LayerNorm(self._hidden_dim, eps=1e-03))
        img_output_layers.append(activation_fn())
        self._img_output_net = nn.Sequential(*img_output_layers)
        self._img_output_net.apply(tools.weight_init)

        # === Observation Network ===
        # Input: [deter_t, embed_t] -> Hidden (for computing posterior)
        # Combines deterministic state with observation embedding
        # to compute posterior distribution
        obs_output_layers = []
        obs_input_dim = self._deter_dim + self._embed_dim
        obs_output_layers.append(nn.Linear(obs_input_dim, self._hidden_dim, bias=False))
        if use_layer_norm:
            obs_output_layers.append(nn.LayerNorm(self._hidden_dim, eps=1e-03))
        obs_output_layers.append(activation_fn())
        self._obs_output_net = nn.Sequential(*obs_output_layers)
        self._obs_output_net.apply(tools.weight_init)

        # === Distribution Parameters Layers ===
        # Predict distribution parameters for prior and posterior
        # Categorical distribution (DreamerV3 discrete representation)
        # Output logits for each class in each categorical distribution
        self._img_dist_layer = nn.Linear(
            self._hidden_dim, self._stoch_dim * self._discrete_dim
        )
        self._img_dist_layer.apply(tools.uniform_weight_init(1.0))

        self._obs_dist_layer = nn.Linear(
            self._hidden_dim, self._stoch_dim * self._discrete_dim
        )
        self._obs_dist_layer.apply(tools.uniform_weight_init(1.0))

        # === Learned Initial State ===
        # Learnable parameter for initial deterministic state
        if self._initial_state == "learned":
            self.W_initial = torch.nn.Parameter(
                torch.zeros((1, self._deter_dim), device=torch.device(self._device)),
                requires_grad=True
            )

    def initial(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Create initial state for the RSSM (discrete stochastic states only)

        Args:
            batch_size: Number of parallel sequences

        Returns:
            Dictionary containing initial state with keys:
            - 'deter': Deterministic state h, shape (batch_size, deter_dim)
            - 'stoch': Stochastic state z, shape (batch_size, stoch_dim, discrete_dim)
            - 'logit': Logits, shape (batch_size, stoch_dim, discrete_dim)
        """
        # Initialize deterministic state
        deter = torch.zeros(batch_size, self._deter_dim, device=self._device)

        # Discrete (categorical) stochastic state
        state = dict(
            logit=torch.zeros(
                [batch_size, self._stoch_dim, self._discrete_dim],
                device=self._device
            ),
            stoch=torch.zeros(
                [batch_size, self._stoch_dim, self._discrete_dim],
                device=self._device
            ),
            deter=deter
        )

        # Apply learned initialization if specified
        if self._initial_state == "zeros":
            return state
        elif self._initial_state == "learned":
            # Use learned initial deterministic state h (take tanh to bound values and match GRU)
            state["deter"] = torch.tanh(self.W_initial).repeat(batch_size, 1)
            # Sample initial stochastic state from prior given initial deter p(z_t | h_t)
            state["stoch"] = self._get_initial_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial_state)

    def _get_initial_stoch(self, deter: torch.Tensor) -> torch.Tensor:
        """
        Sample initial stochastic state from prior given deterministic state

        Args:
            deter: Deterministic state, shape (batch_size, deter_dim)

        Returns:
            stoch: Sampled stochastic state
        """
        hidden = self._img_output_net(deter)
        stats = self._compute_distribution_params("prior", hidden)
        dist = self._get_distribution(stats)
        return dist.mode()

    def observe(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        is_first: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Observe a sequence of observations and compute posterior and prior states

        This implements the sequential state inference described in Algorithm 1 of the paper.
        For each timestep:
        1. Compute prior: p(z_t | h_t)
        2. Compute posterior: q(z_t | h_t, o_t)
        3. Update h_{t+1} using posterior sample

        Args:
            embed: Observation embeddings, shape (batch_size, time_steps, embed_dim)
            action: Actions taken, shape (batch_size, time_steps, action_dim)
            is_first: Episode start flags, shape (batch_size, time_steps)
            state: Initial state (optional), if None uses initial()

        Returns:
            Tuple of (posterior, prior) dictionaries, each containing:
            - 'deter': shape (batch_size, time_steps, deter_dim)
            - 'stoch': shape (batch_size, time_steps, stoch_dim, discrete_dim) or (batch_size, time_steps, stoch_dim)
            - 'logit' or 'mean'/'std': Distribution parameters
        """
        # Transpose to (time_steps, batch_size, ...) for sequential processing
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed = swap(embed)
        action = swap(action)
        is_first = swap(is_first)

        # Process sequence using static_scan
        # prev_state[0] selects posterior from obs_step return (posterior, prior)
        posterior, prior = tools.static_scan(
            lambda prev_state, prev_action, embed, is_first: self.obs_step(
                prev_state[0], prev_action, embed, is_first
            ),
            (action, embed, is_first),
            (state, state)
        )

        # Transpose back to (batch_size, time_steps, ...)
        posterior = {k: swap(v) for k, v in posterior.items()}
        prior = {k: swap(v) for k, v in prior.items()}

        return posterior, prior

    def obs_step(
        self,
        prev_state: Optional[Dict[str, torch.Tensor]],
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
        sample: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Single observation step: compute prior, then posterior

        This implements one step of the RSSM:
        1. Reset state if episode start (is_first=True)
        2. Compute prior: h_t = f(h_{t-1}, z_{t-1}, a_{t-1}), z_t ~ p(z_t | h_t)
        3. Compute posterior: z_t ~ q(z_t | h_t, o_t)

        Args:
            prev_state: Previous state dictionary or None
            prev_action: Previous action, shape (batch_size, action_dim)
            embed: Current observation embedding, shape (batch_size, embed_dim)
            is_first: Episode start flag, shape (batch_size,)
            sample: Whether to sample or use mode

        Returns:
            Tuple of (posterior, prior) state dictionaries
        """
        batch_size = is_first.shape[0]

        # Handle initialization and episode resets
        if prev_state is None or torch.sum(is_first) == len(is_first):
            # Initialize all states
            prev_state = self.initial(batch_size)
            prev_action = torch.zeros(
                (batch_size, self._num_actions), device=self._device
            )
        elif torch.sum(is_first) > 0:
            # Partial reset: reset only episodes that are starting
            is_first = is_first[:, None].float()  # Add dimension and convert to float for broadcasting
            prev_action = prev_action * (1.0 - is_first)

            init_state = self.initial(batch_size)
            for key, val in prev_state.items():
                # Broadcast is_first to match value shape
                is_first_broadcast = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape))
                )
                prev_state[key] = (
                    val * (1.0 - is_first_broadcast) +
                    init_state[key] * is_first_broadcast
                )

        # Compute prior: imagine next state without observation
        prior = self.img_step(prev_state, prev_action, sample=sample)

        # Compute posterior: incorporate observation
        # Concatenate deterministic state with observation embedding
        x = torch.cat([prior["deter"], embed], dim=-1)
        x = self._obs_output_net(x)  # (batch_size, hidden_dim)

        # Compute posterior distribution parameters
        stats = self._compute_distribution_params("posterior", x)

        # Sample stochastic state
        if sample:
            stoch = self._get_distribution(stats).sample()
        else:
            stoch = self._get_distribution(stats).mode()

        # Combine into posterior state
        posterior = {"stoch": stoch, "deter": prior["deter"], **stats}

        return posterior, prior

    def img_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        sample: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Imagination step: predict next state without observation (prior)

        Implements the deterministic and stochastic state transitions:
        1. h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        2. z_t ~ p(z_t | h_t)

        This is used during imagination for planning.

        Args:
            prev_state: Previous state dictionary with 'stoch' and 'deter'
            prev_action: Previous action, shape (batch_size, action_dim)
            sample: Whether to sample or use mode

        Returns:
            Prior state dictionary with updated 'deter', 'stoch', and distribution params
        """
        # Get previous stochastic state and flatten if discrete
        prev_stoch = prev_state["stoch"]
        if self._discrete_dim:
            # Flatten (batch_size, stoch_dim, discrete_dim) -> (batch_size, stoch_dim * discrete_dim)
            shape = list(prev_stoch.shape[:-2]) + [self._stoch_dim * self._discrete_dim]
            prev_stoch = prev_stoch.reshape(shape)

        # Concatenate previous stochastic state and action
        x = torch.cat([prev_stoch, prev_action], dim=-1)

        # Process through input network
        x = self._img_input_net(x)  # (batch_size, hidden_dim)

        # Update deterministic state with GRU
        # Note: rec_depth > 1 not correctly implemented in original, so we use 1
        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            x, deter = self._gru_cell(x, [deter])
            deter = deter[0]  # Unwrap from list

        # Compute prior distribution
        x = self._img_output_net(deter)  # (batch_size, hidden_dim)
        stats = self._compute_distribution_params("prior", x)

        # Sample stochastic state
        if sample:
            stoch = self._get_distribution(stats).sample()
        else:
            stoch = self._get_distribution(stats).mode()

        # Combine into prior state
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def imagine_with_action(
        self,
        action: torch.Tensor,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine future states given a sequence of actions

        Used for planning: rolls out the learned dynamics forward in time
        given a policy's action sequence.

        Args:
            action: Action sequence, shape (batch_size, time_steps, action_dim)
            state: Initial state dictionary

        Returns:
            Imagined state dictionary with time dimension
        """
        # Transpose to (time_steps, batch_size, action_dim)
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        action = swap(action)

        # Roll out dynamics
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]

        # Transpose back to (batch_size, time_steps, ...)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract feature vector from state for downstream prediction

        Concatenates stochastic and deterministic components into a single
        feature vector used by reward, value, and decoder networks.

        Args:
            state: State dictionary with 'stoch' and 'deter'

        Returns:
            Feature vector, shape (..., stoch_dim * discrete_dim + deter_dim)
        """
        stoch = state["stoch"]

        # Flatten discrete stochastic state
        shape = list(stoch.shape[:-2]) + [self._stoch_dim * self._discrete_dim]
        stoch = stoch.reshape(shape)

        # Concatenate stochastic and deterministic
        return torch.cat([stoch, state["deter"]], dim=-1)

    def get_dist(
        self,
        state: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype] = None
    ) -> td.Distribution:
        """
        Get probability distribution from state parameters (discrete only)

        Args:
            state: State dictionary with distribution parameters
            dtype: Optional data type (unused, for compatibility)

        Returns:
            PyTorch distribution object
        """
        return self._get_distribution(state)

    def _get_distribution(self, state: Dict[str, torch.Tensor]) -> td.Distribution:
        """
        Create distribution from state parameters (discrete only)

        Creates Independent OneHotCategorical over all stoch_dim categoricals

        Args:
            state: Dictionary with 'logit' for discrete representation

        Returns:
            Distribution object
        """
        # Categorical distribution with uniform mixing for exploration
        logit = state["logit"]
        dist = td.independent.Independent(
            tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio),
            reinterpreted_batch_ndims=1
        )
        return dist

    def _compute_distribution_params(
        self,
        name: str,
        hidden: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distribution parameters from hidden representation (discrete only)

        Args:
            name: "prior" or "posterior" to select the appropriate layer
            hidden: Hidden representation, shape (batch_size, hidden_dim)

        Returns:
            Dictionary with 'logit' for categorical distribution
        """
        # Predict logits for categorical distribution
        if name == "prior":
            x = self._img_dist_layer(hidden)
        elif name == "posterior":
            x = self._obs_dist_layer(hidden)
        else:
            raise NotImplementedError(f"Unknown distribution name: {name}")

        # Reshape to (batch_size, stoch_dim, discrete_dim)
        logit = x.reshape(
            list(x.shape[:-1]) + [self._stoch_dim, self._discrete_dim]
        )
        return {"logit": logit}

    def kl_loss(
        self,
        posterior: Dict[str, torch.Tensor],
        prior: Dict[str, torch.Tensor],
        free_nats: float,
        dyn_scale: float,
        rep_scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence loss for representation and dynamics learning

        DreamerV3 uses two separate KL terms:
        1. Dynamics loss: KL(sg(posterior) || prior) - encourages prior to match posterior
        2. Representation loss: KL(posterior || sg(prior)) - encourages posterior to match prior

        Both are clipped at free_nats to allow some stochasticity.

        Paper Reference: Section 2.1 "World Model Learning", Equation 1

        Args:
            posterior: Posterior state with distribution parameters
            prior: Prior state with distribution parameters
            free_nats: Free bits threshold (no penalty below this)
            dyn_scale: Scale for dynamics loss (typically 0.5)
            rep_scale: Scale for representation loss (typically 0.1)

        Returns:
            Tuple of:
            - loss: Combined weighted loss, shape (batch_size, time_steps)
            - kl_value: Actual KL divergence (unclipped)
            - dyn_loss: Dynamics loss (clipped)
            - rep_loss: Representation loss (clipped)
        """
        kl_divergence = td.kl.kl_divergence
        sg = lambda x: {k: v.detach() for k, v in x.items()}  # Stop gradient

        # Get distributions
        post_dist = self._get_distribution(posterior)
        prior_dist = self._get_distribution(prior)

        # For discrete distributions, use the distribution directly
        # For continuous, use the base distribution
        if self._discrete_dim:
            post_for_kl = post_dist
            prior_for_kl = prior_dist
        else:
            post_for_kl = post_dist._dist
            prior_for_kl = prior_dist._dist

        # Representation loss: KL(post || sg(prior))
        # Encourages encoder to be consistent with dynamics
        rep_loss = kl_divergence(
            post_for_kl if self._discrete_dim else post_dist._dist,
            self._get_distribution(sg(prior)) if self._discrete_dim
                else self._get_distribution(sg(prior))._dist
        )

        # Dynamics loss: KL(sg(post) || prior)
        # Encourages dynamics to be consistent with observations
        dyn_loss = kl_divergence(
            self._get_distribution(sg(posterior)) if self._discrete_dim
                else self._get_distribution(sg(posterior))._dist,
            prior_for_kl if self._discrete_dim else prior_dist._dist
        )

        # Store unclipped KL for logging
        kl_value = rep_loss

        # Apply free nats (don't penalize KL below this threshold)
        rep_loss = torch.clip(rep_loss, min=free_nats)
        dyn_loss = torch.clip(dyn_loss, min=free_nats)

        # Weighted combination
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, kl_value, dyn_loss, rep_loss


class GRUCell(nn.Module):
    """
    Gated Recurrent Unit (GRU) Cell

    Standard GRU implementation with optional layer normalization.
    Used as the deterministic component of the RSSM.

    Update equations:
    r_t = σ(W_r [h_{t-1}, x_t])           # Reset gate
    u_t = σ(W_u [h_{t-1}, x_t])           # Update gate
    h~_t = tanh(W_h [r_t ⊙ h_{t-1}, x_t]) # Candidate
    h_t = (1 - u_t) ⊙ h_{t-1} + u_t ⊙ h~_t # New hidden state
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        norm: bool = True,
        activation = torch.tanh,
        update_bias: float = -1.0
    ):
        """
        Initialize GRU cell

        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            norm: Whether to use layer normalization
            activation: Activation function for candidate state
            update_bias: Bias initialization for update gate
        """
        super(GRUCell, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._update_bias = update_bias

        # Linear layer for all three gates
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear",
            nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=False)
        )
        if norm:
            self.layers.add_module(
                "GRU_norm",
                nn.LayerNorm(3 * hidden_size, eps=1e-03)
            )

    @property
    def state_size(self) -> int:
        """Return hidden state size"""
        return self._hidden_size

    def forward(
        self,
        inputs: torch.Tensor,
        state: list
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through GRU cell

        Args:
            inputs: Input tensor, shape (batch_size, input_size)
            state: List containing previous hidden state [h_{t-1}]

        Returns:
            Tuple of (output, [new_state])
            - output: New hidden state, shape (batch_size, hidden_size)
            - new_state: Same as output, wrapped in list
        """
        state = state[0]  # Unwrap from list (Keras compatibility)

        # Compute all gates
        parts = self.layers(torch.cat([inputs, state], dim=-1))
        reset, candidate, update = torch.split(
            parts, [self._hidden_size] * 3, dim=-1
        )

        # Apply gate activations
        reset = torch.sigmoid(reset)
        candidate = self._activation(reset * candidate)
        update = torch.sigmoid(update + self._update_bias)

        # Compute new hidden state
        output = update * candidate + (1 - update) * state

        return output, [output]
