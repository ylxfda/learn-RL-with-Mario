"""
Actor and Critic Networks for PPO

This module implements the policy (π_θ) and value function (V_ϕ) networks
used in Proximal Policy Optimization.

Paper Reference: Schulman et al. (2017), Section 3
"Proximal Policy Optimization Algorithms"

Network Architecture:
- Both networks use a shared CNN feature extractor for visual observations
- Actor outputs action probabilities: π_θ(a|s)
- Critic outputs state value estimate: V_ϕ(s)

Naming Conventions (from paper):
- θ: Actor network parameters
- ϕ: Critic network parameters
- π_θ(a|s): Policy (action distribution given state)
- V_ϕ(s): Value function (expected return from state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple


def init_orthogonal(layer: nn.Module, gain: float = np.sqrt(2)):
    """
    Orthogonal initialization for neural network layers

    This initialization scheme is commonly used in PPO implementations
    as it helps with gradient flow and training stability.

    Paper Reference: Saxe et al. (2013) "Exact solutions to the nonlinear
    dynamics of learning in deep linear neural networks"

    Args:
        layer: Neural network layer (Linear or Conv2d)
        gain: Scaling factor for the weights (default: sqrt(2))
              sqrt(2) is recommended for ReLU activations
    """
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network for extracting features from visual observations

    Architecture inspired by the Nature DQN paper and commonly used in
    PPO implementations for Atari games.

    Paper Reference: Mnih et al. (2015) "Human-level control through deep
    reinforcement learning"

    Input: RGB image observations
    Output: Flattened feature vector

    Network Structure:
        Conv1: 32 filters, 8x8 kernel, stride 4
        Conv2: 64 filters, 4x4 kernel, stride 2
        Conv3: 64 filters, 3x3 kernel, stride 1
        Flatten + Linear: 512 features
    """

    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        """
        Initialize CNN feature extractor

        Args:
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
            feature_dim: Dimension of output feature vector (default: 512)
        """
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute size after convolutions (for 64x64 input)
        # After conv1 (8x8, s=4): (64-8)/4+1 = 15
        # After conv2 (4x4, s=2): (15-4)/2+1 = 6
        # After conv3 (3x3, s=1): (6-3)/1+1 = 4
        # Total: 64 * 4 * 4 = 1024
        self.feature_size = 64 * 4 * 4

        self.fc = nn.Linear(self.feature_size, feature_dim)

        # Initialize weights with orthogonal initialization
        init_orthogonal(self.conv1)
        init_orthogonal(self.conv2)
        init_orthogonal(self.conv3)
        init_orthogonal(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from visual observation

        Args:
            x: Input image tensor
               Shape: (batch_size, channels, height, width)
               Range: [0, 255] or [0, 1]

        Returns:
            Feature vector
            Shape: (batch_size, feature_dim)
        """
        # Normalize to [0, 1] if needed
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))

        return x


class Actor(nn.Module):
    """
    Policy Network π_θ(a|s)

    The Actor network outputs a probability distribution over actions given
    the current state. In PPO, this is used to:
    1. Sample actions during rollout collection
    2. Compute log probabilities for the policy gradient
    3. Compute policy entropy for the entropy bonus

    Paper Reference: Schulman et al. (2017), Section 3
    The policy is updated using the clipped surrogate objective:
        L^{CLIP}(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
    where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

    Architecture:
        Feature Extractor: Shared CNN (512 features)
        Policy Head: Linear layer -> Softmax

    Output: Categorical distribution over discrete actions
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_actions: int = 7,
        feature_dim: int = 512
    ):
        """
        Initialize Actor network

        Args:
            input_channels: Number of input channels (3 for RGB)
            num_actions: Number of discrete actions (7 for simple Mario controls)
            feature_dim: Dimension of feature vector (default: 512)
        """
        super().__init__()

        self.num_actions = num_actions

        # Shared feature extractor
        self.feature_extractor = CNNFeatureExtractor(input_channels, feature_dim)

        # Policy head: features -> action logits
        self.policy_head = nn.Linear(feature_dim, num_actions)
        init_orthogonal(self.policy_head, gain=0.01)  # Small init for policy

    def forward(self, obs: torch.Tensor) -> Categorical:
        """
        Forward pass through policy network

        Args:
            obs: Observation tensor
                 Shape: (batch_size, channels, height, width)
                 For Mario: (batch_size, 3, 64, 64)

        Returns:
            action_dist: Categorical distribution over actions
                        Can be used to:
                        - Sample actions: action_dist.sample()
                        - Get log probs: action_dist.log_prob(action)
                        - Get entropy: action_dist.entropy()
        """
        features = self.feature_extractor(obs)
        action_logits = self.policy_head(features)
        action_dist = Categorical(logits=action_logits)
        return action_dist

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy

        This method is used during:
        1. Rollout collection (deterministic=False): stochastic sampling for exploration
        2. Evaluation (deterministic=True): greedy action selection

        Args:
            obs: Observation tensor, shape (batch_size, C, H, W)
            deterministic: If True, select most probable action (mode)
                          If False, sample from distribution (default)

        Returns:
            Tuple of (actions, log_probs):
            - actions: Selected actions, shape (batch_size,)
            - log_probs: Log probabilities log π_θ(a|s), shape (batch_size,)
        """
        action_dist = self.forward(obs)

        if deterministic:
            actions = action_dist.probs.argmax(dim=-1)
        else:
            actions = action_dist.sample()

        log_probs = action_dist.log_prob(actions)

        return actions, log_probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given state-action pairs

        This method is used during PPO updates to compute:
        1. Log probabilities log π_θ(a|s) for the policy gradient
        2. Entropy H(π_θ(·|s)) for the entropy bonus

        The entropy bonus encourages exploration by preventing the policy
        from becoming too deterministic.

        Args:
            obs: Observation tensor, shape (batch_size, C, H, W)
            actions: Action tensor, shape (batch_size,)

        Returns:
            Tuple of (log_probs, entropy):
            - log_probs: Log π_θ(a|s), shape (batch_size,)
                        Used to compute probability ratio r_t(θ)
            - entropy: H(π_θ(·|s)), shape (batch_size,)
                      Entropy bonus term: c_2 * H(π_θ(·|s))
        """
        action_dist = self.forward(obs)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_probs, entropy


class Critic(nn.Module):
    """
    Value Network V_ϕ(s)

    The Critic network estimates the expected return (value) from a given state.
    In PPO, this is used to:
    1. Compute advantage estimates: Â_t = δ_t + (γλ)δ_{t+1} + ... (GAE)
    2. Provide a baseline to reduce variance in policy gradient
    3. Train the value function with MSE loss

    Paper Reference: Schulman et al. (2017), Section 3
    The value function is updated by minimizing:
        L^{VF}(ϕ) = E[(V_ϕ(s_t) - V_t^{targ})^2]
    where V_t^{targ} is the target return (from GAE or n-step returns)

    Architecture:
        Feature Extractor: Shared CNN (512 features)
        Value Head: Linear layer -> scalar value

    Output: Scalar value estimate V_ϕ(s)
    """

    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 512
    ):
        """
        Initialize Critic network

        Args:
            input_channels: Number of input channels (3 for RGB)
            feature_dim: Dimension of feature vector (default: 512)
        """
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = CNNFeatureExtractor(input_channels, feature_dim)

        # Value head: features -> scalar value
        self.value_head = nn.Linear(feature_dim, 1)
        init_orthogonal(self.value_head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network

        Args:
            obs: Observation tensor
                 Shape: (batch_size, channels, height, width)
                 For Mario: (batch_size, 3, 64, 64)

        Returns:
            values: State value estimates V_ϕ(s)
                   Shape: (batch_size,)
        """
        features = self.feature_extractor(obs)
        values = self.value_head(features).squeeze(-1)
        return values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for given observation

        This is a convenience method equivalent to forward(), used during:
        1. Rollout collection: computing V(s_t) for GAE
        2. Evaluation: monitoring value predictions

        Args:
            obs: Observation tensor, shape (batch_size, C, H, W)

        Returns:
            values: Value estimates V_ϕ(s), shape (batch_size,)
        """
        return self.forward(obs)
