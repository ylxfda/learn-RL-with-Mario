"""
Custom Probability Distributions for DreamerV3

This module implements custom distribution classes used throughout DreamerV3:
- OneHotDist: Categorical distribution with straight-through gradients
- DiscDist: Discretized continuous distribution in symlog space
- ContDist: Continuous distribution with absolute maximum clamping
- MSEDist: MSE-based deterministic "distribution"
- SymlogDist: Distribution in symlog space
- Other utility distributions

Paper Reference: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
"""

import torch
import torch.nn.functional as F
from torch import distributions as td
from typing import Tuple, Optional


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symlog transformation: sign(x) * log(|x| + 1)

    Compresses large values while preserving sign and being smooth at zero.
    Used for reward and value predictions in DreamerV3.

    Paper Reference: Section 2.3 "Symlog Predictions"

    Args:
        x: Input tensor

    Returns:
        Symlog-transformed tensor
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of symlog: sign(x) * (exp(|x|) - 1)

    Args:
        x: Input tensor in symlog space

    Returns:
        Tensor in original space
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class OneHotDist(td.one_hot_categorical.OneHotCategorical):
    """
    One-Hot Categorical Distribution with Straight-Through Gradients

    This distribution is used for discrete actions and discrete latent variables.
    Key features:
    - Straight-through gradients: sample is one-hot, but gradients flow through softmax
    - Uniform mixing: adds small uniform probability for exploration

    Paper Reference: Used for discrete actions (Section 2.2) and discrete representations
    """

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        unimix_ratio: float = 0.0
    ):
        """
        Initialize one-hot categorical distribution

        Args:
            logits: Logits for each class, shape (..., num_classes)
            probs: Probabilities for each class, shape (..., num_classes)
            unimix_ratio: Ratio of uniform distribution to mix in [0, 1]
        """
        if logits is not None and unimix_ratio > 0.0:
            # Mix with uniform distribution for exploration
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self) -> torch.Tensor:
        """
        Return mode (argmax) with straight-through gradients

        Returns:
            One-hot tensor with gradients from softmax, shape (..., num_classes)
        """
        # === Straight-Through Estimator ===
        # Create one-hot encoding of argmax (discrete, non-differentiable)
        # We use argmax to get the most likely class
        _mode = F.one_hot(
            torch.argmax(super().logits, dim=-1),
            super().logits.shape[-1]
        )

        # Straight-through trick: _mode.detach() + logits - logits.detach()
        # Forward pass:  returns discrete one-hot mode (argmax result)
        # Backward pass: gradients flow through continuous logits directly
        #
        # Math: Let f = _mode.detach() + logits - logits.detach()
        #   Forward:  f = _mode + logits - logits = _mode (discrete one-hot)
        #   Backward: ∂f/∂logits = 0 + 1 - 0 = 1 (gradients flow through logits)
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(
        self,
        sample_shape: Tuple = (),
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample from distribution with straight-through gradients

        Args:
            sample_shape: Shape of samples to draw
            seed: Random seed (not implemented)

        Returns:
            One-hot sampled tensor with gradients from probs, shape (*sample_shape, ..., num_classes)
        """
        if seed is not None:
            raise ValueError("Seeding not implemented")

        # === Straight-Through Estimator ===
        # Sample using standard categorical sampling (discrete, non-differentiable)
        # We detach() because sampling is a discrete operation with no gradient
        sample = super().sample(sample_shape).detach()

        # Straight-through trick: sample + probs - probs.detach()
        # Forward pass:  returns discrete one-hot sample
        # Backward pass: gradients flow through continuous probs (via softmax)
        #
        # This allows us to:
        # - Use discrete samples in forward pass (preserving discrete nature)
        # - Get continuous gradients in backward pass (enabling learning)
        #
        # Math: Let f = sample + probs - probs.detach()
        #   Forward:  f = sample + probs - probs = sample (discrete)
        #   Backward: ∂f/∂logits = 0 + ∂probs/∂logits - 0 = ∂softmax/∂logits (continuous)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample = sample + probs - probs.detach()

        return sample

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability with automatic one-hot conversion

        This overrides the parent's log_prob to handle straight-through gradients.
        Automatically converts non-strict one-hot vectors to strict one-hot.

        Args:
            value: Action tensor (may have straight-through gradients)

        Returns:
            Log probabilities, shape value.shape[:-1]
        """
        # Convert to strict one-hot if needed
        # Check if value has gradients or is not strictly binary
        if value.requires_grad or torch.any((value != 0) & (value != 1)):
            # Has gradients or not strictly binary - convert to one-hot
            value = F.one_hot(
                torch.argmax(value, dim=-1),
                value.shape[-1]
            ).float()

        # Call parent's log_prob (will now pass validation)
        return super().log_prob(value)


class DiscDist:
    """
    Discretized Continuous Distribution in Symlog Space

    Represents continuous values (e.g., rewards, values) as a categorical
    distribution over discrete buckets in symlog space. This allows:
    1. Handling rewards/values of varying magnitudes
    2. Using classification loss for regression
    3. Better gradient flow for extreme values

    Paper Reference: Section 2.3 "Symlog Predictions"
    """

    def __init__(
        self,
        logits: torch.Tensor,
        low: float = -20.0,
        high: float = 20.0,
        transfwd = symlog,
        transbwd = symexp,
        device: str = "cuda"
    ):
        """
        Initialize discretized distribution

        Args:
            logits: Logits for each bucket, shape (..., num_buckets)
            low: Lower bound in symlog space
            high: Upper bound in symlog space
            transfwd: Forward transformation function (default: symlog)
            transbwd: Backward transformation function (default: symexp)
            device: Device for bucket locations
        """
        self.logits = logits
        self.probs = torch.softmax(logits, dim=-1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self) -> torch.Tensor:
        """
        Compute expected value

        Returns:
            Mean in original space, shape (..., 1)
        """
        # Expected bucket value
        _mean = self.probs * self.buckets
        # Transform back to original space
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self) -> torch.Tensor:
        """
        Compute mode (same as mean for this implementation)

        Returns:
            Mode in original space, shape (..., 1)
        """
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of continuous value

        Uses two-bucket interpolation: spreads probability of x across
        the two nearest buckets based on distance.

        Args:
            x: Values to compute log probability for, shape (..., 1)

        Returns:
            Log probabilities, shape (...)
        """
        # Transform to symlog space
        x = self.transfwd(x)

        # Find buckets below and above x
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )

        # Clip to valid bucket indices
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)

        # Check if x falls exactly on a bucket
        equal = (below == above)

        # Compute distances for interpolation
        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above

        # Interpolation weights (inverse distance)
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        # Create soft target distribution
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None] +
            F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )

        # Compute log probability
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)


class MSEDist:
    """
    Mean Squared Error "Distribution"

    Treats MSE as a log probability for deterministic predictions.
    Used for image reconstruction in DreamerV3.
    """

    def __init__(self, mode: torch.Tensor, agg: str = "sum"):
        """
        Initialize MSE distribution

        Args:
            mode: Predicted values
            agg: Aggregation method: "sum" or "mean"
        """
        self._mode = mode
        self._agg = agg

    def mode(self) -> torch.Tensor:
        """Return predicted mode"""
        return self._mode

    def mean(self) -> torch.Tensor:
        """Return predicted mean (same as mode)"""
        return self._mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute negative MSE as log probability

        Args:
            value: Target values

        Returns:
            Negative MSE, shape (batch, time) if inputs are (batch, time, ...)
        """
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)

        # Compute squared error
        distance = (self._mode - value) ** 2

        # Aggregate over spatial dimensions
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)

        return -loss


class SymlogDist:
    """
    Distribution for Symlog-space Predictions

    Similar to MSEDist but operates in symlog space for better handling
    of values with varying magnitudes.
    """

    def __init__(
        self,
        mode: torch.Tensor,
        dist: str = "mse",
        agg: str = "sum",
        tol: float = 1e-8
    ):
        """
        Initialize symlog distribution

        Args:
            mode: Predicted values in symlog space
            dist: Distance metric: "mse" or "abs"
            agg: Aggregation: "sum" or "mean"
            tol: Tolerance for zero distance
        """
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self) -> torch.Tensor:
        """Return mode in original space"""
        return symexp(self._mode)

    def mean(self) -> torch.Tensor:
        """Return mean in original space"""
        return symexp(self._mode)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability (negative distance in symlog space)

        Args:
            value: Target values in original space

        Returns:
            Negative distance
        """
        assert self._mode.shape == value.shape

        # Compute distance in symlog space
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)

        # Aggregate
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)

        return -loss


class ContDist:
    """
    Continuous Distribution Wrapper with Absolute Maximum Clamping

    Wraps continuous distributions (e.g., Normal) and clamps samples/modes
    to an absolute maximum value. Used for continuous action spaces.
    """

    def __init__(
        self,
        dist: Optional[td.Distribution] = None,
        absmax: Optional[float] = None
    ):
        """
        Initialize continuous distribution wrapper

        Args:
            dist: Base distribution
            absmax: Absolute maximum value for clamping
        """
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name: str):
        """Forward attribute access to base distribution"""
        return getattr(self._dist, name)

    def entropy(self) -> torch.Tensor:
        """Compute entropy"""
        return self._dist.entropy()

    def mode(self) -> torch.Tensor:
        """
        Compute mode with clamping

        Returns:
            Mode clamped to [-absmax, absmax] if absmax is set
        """
        out = self._dist.mean
        if self.absmax is not None:
            # Clamp while preserving gradients
            out = out * (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape: Tuple = ()) -> torch.Tensor:
        """
        Sample with clamping

        Args:
            sample_shape: Shape of samples

        Returns:
            Samples clamped to [-absmax, absmax] if absmax is set
        """
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            # Clamp while preserving gradients
            out = out * (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability"""
        return self._dist.log_prob(x)


class Bernoulli:
    """
    Bernoulli Distribution with Straight-Through Gradients

    Used for binary predictions (e.g., episode continuation).
    """

    def __init__(self, dist: Optional[td.Distribution] = None):
        """
        Initialize Bernoulli distribution

        Args:
            dist: Base Bernoulli distribution
        """
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name: str):
        """Forward attribute access to base distribution"""
        return getattr(self._dist, name)

    def entropy(self) -> torch.Tensor:
        """Compute entropy"""
        return self._dist.entropy()

    def mode(self) -> torch.Tensor:
        """
        Compute mode (rounded mean) with straight-through gradients

        Returns:
            Binary mode with gradients from mean
        """
        _mode = torch.round(self._dist.mean)
        # Straight-through gradients
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape: Tuple = ()) -> torch.Tensor:
        """Sample from Bernoulli"""
        return self._dist.rsample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability manually for numerical stability

        Args:
            x: Binary values

        Returns:
            Log probabilities
        """
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, dim=-1)


class SafeTruncatedNormal(td.normal.Normal):
    """
    Truncated Normal Distribution with Safe Clipping

    Normal distribution with values clipped to [low, high] using
    straight-through gradients.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float,
        high: float,
        clip: float = 1e-6,
        mult: float = 1
    ):
        """
        Initialize safe truncated normal

        Args:
            loc: Mean
            scale: Standard deviation
            low: Lower bound
            high: Upper bound
            clip: Clipping margin from boundaries
            mult: Multiplicative factor
        """
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape: Tuple) -> torch.Tensor:
        """
        Sample and clip to bounds

        Args:
            sample_shape: Shape of samples

        Returns:
            Clipped samples
        """
        event = super().sample(sample_shape)

        if self._clip:
            clipped = torch.clip(
                event,
                self._low + self._clip,
                self._high - self._clip
            )
            # Straight-through: forward uses clipped, backward uses original
            event = event - event.detach() + clipped.detach()

        if self._mult:
            event = event * self._mult

        return event


class TanhBijector(td.Transform):
    """
    Tanh Bijective Transformation

    Used for squashing actions to [-1, 1] range while maintaining
    a proper probability distribution.
    """

    def __init__(self, validate_args: bool = False, name: str = "tanh"):
        """Initialize tanh bijector"""
        super().__init__()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply tanh transformation"""
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse tanh (arctanh)"""
        # Clip to valid range for numerical stability
        y = torch.where(
            (torch.abs(y) <= 1.0),
            torch.clamp(y, -0.99999997, 0.99999997),
            y
        )
        return torch.atanh(y)

    def _forward_log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log determinant of Jacobian

        For tanh: log|dy/dx| = log(1 - tanh²(x))
        """
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


class SampleDist:
    """
    Distribution Approximated by Sampling

    Estimates distribution properties (mean, mode, entropy) using samples.
    Used when analytical forms are intractable.
    """

    def __init__(self, dist: td.Distribution, samples: int = 100):
        """
        Initialize sample-based distribution

        Args:
            dist: Base distribution
            samples: Number of samples for estimation
        """
        self._dist = dist
        self._samples = samples

    @property
    def name(self) -> str:
        """Distribution name"""
        return "SampleDist"

    def __getattr__(self, name: str):
        """Forward attribute access to base distribution"""
        return getattr(self._dist, name)

    def mean(self) -> torch.Tensor:
        """Estimate mean by sampling"""
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, dim=0)

    def mode(self) -> torch.Tensor:
        """Estimate mode as sample with highest log probability"""
        samples = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(samples)
        return samples[torch.argmax(logprob)][0]

    def entropy(self) -> torch.Tensor:
        """Estimate entropy by sampling"""
        samples = self._dist.sample(self._samples)
        logprob = self.log_prob(samples)
        return -torch.mean(logprob, dim=0)


class UnnormalizedHuber(td.normal.Normal):
    """
    Unnormalized Huber Loss as Distribution

    Uses Huber loss (combination of L1 and L2) as a log probability.
    More robust to outliers than MSE.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        threshold: float = 1,
        **kwargs
    ):
        """
        Initialize Huber distribution

        Args:
            loc: Location (mean)
            scale: Scale
            threshold: Threshold for switching between L1 and L2
            **kwargs: Additional arguments for Normal
        """
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss as log probability

        Args:
            event: Values to evaluate

        Returns:
            Negative Huber loss
        """
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold ** 2) -
            self._threshold
        )

    def mode(self) -> torch.Tensor:
        """Return mode (mean)"""
        return self.mean
