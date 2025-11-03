"""
Utility Functions and Helper Classes for DreamerV3

This module contains various utility functions used throughout DreamerV3:
- Weight initialization
- Sequence processing (static_scan, lambda_return)
- Optimizer wrapper
- Tensor statistics
- Gradient management

Paper Reference: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Callable, Any, Tuple, Optional
import random
import os
import pathlib
import collections


# Re-export distributions for convenience
from .distributions import (
    OneHotDist, DiscDist, ContDist, MSEDist, SymlogDist, Bernoulli,
    SafeTruncatedNormal, TanhBijector, SampleDist, UnnormalizedHuber,
    symlog, symexp
)


def to_np(x: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array

    Args:
        x: PyTorch tensor

    Returns:
        NumPy array on CPU
    """
    return x.detach().cpu().numpy()


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, recursively merging nested dicts

    This is useful for merging configuration dictionaries where you want
    to override specific nested values while preserving others.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary (new dict, does not modify inputs)

    Example:
        >>> base = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> override = {"a": {"c": 3, "e": 4}}
        >>> result = deep_merge(base, override)
        >>> result
        {"a": {"b": 1, "c": 3, "e": 4}, "d": 3}  # b kept, c overridden, e added
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Direct override for non-dict values or new keys
            result[key] = value
    return result


# === Weight Initialization ===

def weight_init(m: nn.Module):
    """
    Initialize network weights using truncated normal distribution

    Uses the initialization scheme from DreamerV3:
    - For linear/conv layers: Truncated normal with std based on fan-in/fan-out
    - For layer norm: Initialize weights to 1, biases to 0

    Args:
        m: PyTorch module to initialize
    """
    if isinstance(m, nn.Linear):
        # Compute scale based on average of input and output dimensions
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978  # Adjust for truncation

        # Truncated normal initialization
        nn.init.trunc_normal_(
            m.weight.data,
            mean=0.0,
            std=std,
            a=-2.0 * std,
            b=2.0 * std
        )

        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # For convolutional layers, account for kernel size
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978

        nn.init.trunc_normal_(
            m.weight.data,
            mean=0.0,
            std=std,
            a=-2.0 * std,
            b=2.0 * std
        )

        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

    elif isinstance(m, nn.LayerNorm):
        # Layer norm: weights to 1, biases to 0
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale: float) -> Callable:
    """
    Create uniform weight initializer with given scale

    Returns a function that initializes weights uniformly in a range
    determined by the scale and layer dimensions.

    Args:
        given_scale: Scale factor for initialization

    Returns:
        Initialization function
    """
    def f(m: nn.Module):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)  # Uniform variance = scale

            nn.init.uniform_(m.weight.data, a=-limit, b=limit)

            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)

            nn.init.uniform_(m.weight.data, a=-limit, b=limit)

            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


# === Sequence Processing ===

def static_scan(
    fn: Callable,
    inputs: List[torch.Tensor],
    start: Any
) -> List[Any]:
    """
    Apply a function sequentially over time dimension

    Similar to jax.lax.scan or tf.scan, but implemented imperatively.
    Processes sequences one timestep at a time, maintaining state.

    Used for:
    - RSSM observation processing (computing posterior/prior sequentially)
    - Imagination rollouts

    Args:
        fn: Function (state, *inputs_t) -> new_state
        inputs: List of tensors with time as first dimension
        start: Initial state

    Returns:
        List of outputs collected over time
    """
    last = start
    indices = range(inputs[0].shape[0])
    flag = True

    for index in indices:
        # Extract inputs for current timestep
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))

        # Initialize output structure on first iteration
        if flag:
            if isinstance(last, dict):
                # Dictionary state (e.g., RSSM state)
                outputs = {
                    key: value.clone().unsqueeze(0)
                    for key, value in last.items()
                }
            else:
                # Tuple/list state
                outputs = []
                for _last in last:
                    if isinstance(_last, dict):
                        outputs.append({
                            key: value.clone().unsqueeze(0)
                            for key, value in _last.items()
                        })
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            # Append to existing outputs
            if isinstance(last, dict):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)],
                        dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if isinstance(last[j], dict):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)],
                                dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)],
                            dim=0
                        )

    # Wrap dict output in list for consistency
    if isinstance(last, dict):
        outputs = [outputs]

    return outputs


def static_scan_for_lambda_return(
    fn: Callable,
    inputs: Tuple[torch.Tensor],
    start: torch.Tensor
) -> List[torch.Tensor]:
    """
    Specialized static_scan for computing lambda returns in reverse

    Processes sequence in reverse order for backward computation of returns.

    Args:
        fn: Function (state, *inputs_t) -> new_state
        inputs: Tuple of input tensors
        start: Initial state (bootstrap value)

    Returns:
        List of returns for each timestep
    """
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)  # Process backwards
    flag = True

    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))

        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)

    # Reshape and reverse back to forward order
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)

    return outputs


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    pcont: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
    axis: int
) -> torch.Tensor:
    """
    Compute λ-returns for advantage estimation

    Implements the λ-return from TD(λ), which interpolates between
    1-step TD (λ=0) and Monte Carlo (λ=1):

    G^λ_t = r_t + γ[(1-λ)V(s_{t+1}) + λG^λ_{t+1}]

    Used for computing target values for the critic in actor-critic learning.

    Paper Reference: Section 2.2 "Actor-Critic Learning"

    Args:
        reward: Rewards, shape (time, batch, 1)
        value: Value estimates, shape (time, batch, 1)
        pcont: Continuation probability (discount), shape (time, batch, 1)
        bootstrap: Bootstrap value for final step, shape (batch, 1)
        lambda_: λ parameter in [0, 1]
        axis: Time axis (typically 0)

    Returns:
        λ-returns, shape same as reward
    """
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)

    # Convert scalar pcont to tensor
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)

    # Permute to put time axis first if needed
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]

    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)

    # Bootstrap final value if not provided
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])

    # Compute TD targets: r_t + γV(s_{t+1})
    next_values = torch.cat([value[1:], bootstrap.unsqueeze(0)], dim=0)
    inputs = reward + pcont * next_values * (1 - lambda_)

    # Recursively compute λ-returns backwards
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
        (inputs, pcont),
        bootstrap
    )

    # Permute back to original axis order
    if axis != 0:
        returns = returns.permute(dims)

    return returns


# === Gradient Management ===

class RequiresGrad:
    """
    Context manager for temporarily enabling gradients

    Used to selectively enable gradient computation for specific modules
    during training, while keeping others frozen.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize context manager

        Args:
            model: Module to enable gradients for
        """
        self._model = model

    def __enter__(self):
        """Enable gradients"""
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        """Disable gradients"""
        self._model.requires_grad_(requires_grad=False)


# === Optimizer Wrapper ===

class Optimizer:
    """
    Optimizer Wrapper with Gradient Clipping and AMP Support

    Wraps PyTorch optimizers with:
    - Gradient clipping
    - Weight decay
    - Automatic Mixed Precision (AMP)
    - Metric logging
    """

    def __init__(
        self,
        name: str,
        parameters,
        lr: float,
        eps: float = 1e-4,
        clip: Optional[float] = None,
        wd: Optional[float] = None,
        wd_pattern: str = r".*",
        opt: str = "adam",
        use_amp: bool = False
    ):
        """
        Initialize optimizer wrapper

        Args:
            name: Name for logging
            parameters: Model parameters to optimize
            lr: Learning rate
            eps: Epsilon for numerical stability
            clip: Gradient clipping threshold (None for no clipping)
            wd: Weight decay coefficient
            wd_pattern: Regex pattern for weight decay (not implemented)
            opt: Optimizer type: "adam", "adamax", "sgd", "momentum"
            use_amp: Whether to use automatic mixed precision
        """
        assert 0 <= wd < 1 if wd is not None else True
        assert not clip or 1 <= clip

        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern

        # Create optimizer
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()

        # Gradient scaler for AMP
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(
        self,
        loss: torch.Tensor,
        params,
        retain_graph: bool = True
    ) -> Dict[str, float]:
        """
        Perform optimization step

        Args:
            loss: Scalar loss to minimize
            params: Parameters to update
            retain_graph: Whether to retain computation graph

        Returns:
            Dictionary of metrics (loss, grad_norm)
        """
        assert len(loss.shape) == 0, f"Loss must be scalar, got shape {loss.shape}"

        metrics = {}
        metrics[f"{self._name}_loss"] = to_np(loss)

        # Compute gradients
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)

        # Clip gradients
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)

        # Apply weight decay
        if self._wd:
            self._apply_weight_decay(params)

        # Update parameters
        self._scaler.step(self._opt)
        self._scaler.update()
        self._opt.zero_grad()

        metrics[f"{self._name}_grad_norm"] = to_np(norm)

        return metrics

    def _apply_weight_decay(self, params):
        """Apply weight decay to parameters"""
        if self._wd_pattern != r".*":
            raise NotImplementedError("Pattern-based weight decay not implemented")

        for var in params:
            var.data = (1 - self._wd) * var.data


# === Tensor Statistics ===

def tensorstats(tensor: torch.Tensor, prefix: Optional[str] = None) -> Dict[str, float]:
    """
    Compute statistics of a tensor for logging

    Args:
        tensor: Input tensor
        prefix: Optional prefix for metric names

    Returns:
        Dictionary with mean, std, min, max
    """
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }

    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

    return metrics


# === Reproducibility ===

def set_seed_everywhere(seed: int):
    """
    Set random seed for reproducibility

    Sets seeds for:
    - PyTorch (CPU and CUDA)
    - NumPy
    - Python random

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    """
    Enable deterministic mode for reproducibility

    Warning: May reduce performance
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# === Optimizer State Management ===

def recursively_collect_optim_state_dict(
    obj: Any,
    path: str = "",
    optimizers_state_dicts: Optional[Dict] = None,
    visited: Optional[set] = None
) -> Dict[str, Dict]:
    """
    Recursively collect all optimizer state dicts from an object

    Used for saving optimizer states in checkpoints.

    Args:
        obj: Object to search (typically a model)
        path: Current path (for naming)
        optimizers_state_dicts: Accumulator dict
        visited: Set of visited object IDs (prevents cycles)

    Returns:
        Dictionary mapping paths to optimizer state dicts
    """
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()

    # Avoid cyclic references
    if id(obj) in visited:
        return optimizers_state_dicts
    visited.add(id(obj))

    # Get attributes
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update({
            k: attr
            for k, attr in obj.named_modules()
            if "." not in k and obj != attr
        })

    # Recursively search
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name

        if isinstance(attr, torch.optim.Optimizer):
            # Found an optimizer
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            # Recurse into nested objects
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )

    return optimizers_state_dicts


def recursively_load_optim_state_dict(
    obj: Any,
    optimizers_state_dicts: Dict[str, Dict]
):
    """
    Recursively load optimizer state dicts into an object

    Used for loading optimizer states from checkpoints.

    Args:
        obj: Object to load into (typically a model)
        optimizers_state_dicts: Dictionary mapping paths to state dicts
    """
    for path, state_dict in optimizers_state_dicts.items():
        # Navigate to optimizer using path
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)

        # Load state dict
        obj_now.load_state_dict(state_dict)


# === Scheduling Utilities ===

class Every:
    """
    Trigger action every N steps

    Used for periodic logging, evaluation, etc.
    """

    def __init__(self, every: int):
        """
        Initialize scheduler

        Args:
            every: Period (0 means never)
        """
        self._every = every
        self._last = None

    def __call__(self, step: int) -> int:
        """
        Check if action should trigger

        Args:
            step: Current step

        Returns:
            Number of times to trigger (typically 0 or 1)
        """
        if not self._every:
            return 0

        if self._last is None:
            self._last = step
            return 1

        count = int((step - self._last) / self._every)
        self._last += self._every * count

        return count


class Once:
    """
    Trigger action exactly once

    Used for pretraining, initialization, etc.
    """

    def __init__(self):
        """Initialize once trigger"""
        self._once = True

    def __call__(self) -> bool:
        """
        Check if action should trigger

        Returns:
            True only on first call
        """
        if self._once:
            self._once = False
            return True
        return False


class Until:
    """
    Trigger action until a certain step

    Used for exploration scheduling, etc.
    """

    def __init__(self, until: int):
        """
        Initialize until scheduler

        Args:
            until: Step to stop triggering (0 means always)
        """
        self._until = until

    def __call__(self, step: int) -> bool:
        """
        Check if action should trigger

        Args:
            step: Current step

        Returns:
            True if step < until
        """
        if not self._until:
            return True
        return step < self._until


# === Argument Parsing ===

def args_type(default: Any) -> Callable:
    """
    Create argument parser based on default value type

    Used for flexible command-line argument parsing.

    Args:
        default: Default value (determines type)

    Returns:
        Parsing function
    """
    def parse_string(x: str):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


# === Episode Management ===

def convert(value: any, precision: int = 32) -> np.ndarray:
    """
    Convert value to numpy array with appropriate dtype

    Args:
        value: Value to convert
        precision: Bit precision (16, 32, or 64)

    Returns:
        NumPy array
    """
    value = np.array(value)

    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)

    return value.astype(dtype)


def add_to_cache(cache: dict, id: str, transition: dict):
    """
    Add transition to episode cache

    Args:
        cache: Cache dictionary
        id: Episode identifier
        transition: Transition dictionary
    """
    if id not in cache:
        cache[id] = {}
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # Fill missing data (e.g., action) at second timestep
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def save_episodes(directory: pathlib.Path, episodes: dict) -> bool:
    """
    Save episodes to disk

    Args:
        directory: Directory to save to
        episodes: Dictionary mapping episode IDs to episode data

    Returns:
        True on success
    """
    import io

    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"

        # Save compressed
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())

    return True


def load_episodes(
    directory: pathlib.Path,
    limit: Optional[int] = None,
    reverse: bool = True
) -> collections.OrderedDict:
    """
    Load episodes from disk

    Args:
        directory: Directory to load from
        limit: Maximum number of steps to load
        reverse: Whether to load newest episodes first

    Returns:
        OrderedDict mapping episode IDs to episode data
    """
    import os
    import collections

    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0

    # Get file list
    files = list(directory.glob("*.npz"))
    if reverse:
        files = reversed(sorted(files))
    else:
        files = sorted(files)

    for filename in files:
        try:
            with filename.open("rb") as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f"Could not load episode: {e}")
            continue

        # Extract filename without extension
        episode_id = str(os.path.splitext(os.path.basename(filename))[0])
        episodes[episode_id] = episode
        total += len(episode["reward"]) - 1

        if limit and total >= limit:
            break

    return episodes


def erase_over_episodes(cache: dict, dataset_size: int, directory: pathlib.Path = None) -> int:
    """
    Remove oldest episodes to stay under size limit

    Args:
        cache: Episode cache
        dataset_size: Maximum dataset size in steps
        directory: Directory where episodes are stored (if provided, will delete files from disk)

    Returns:
        Current dataset size
    """
    step_in_dataset = 0

    # Count steps from newest to oldest
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size or
            step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            # Remove oldest episode from memory
            del cache[key]

            # Also remove from disk if directory is provided
            if directory:
                episode_length = len(ep["reward"])
                filename = directory / f"{key}-{episode_length}.npz"
                if filename.exists():
                    filename.unlink()
                    print(f"Deleted old episode from disk: {filename.name}")

    return step_in_dataset


def sample_episodes(
    episodes: collections.OrderedDict,
    length: int,
    seed: int = 0
):
    """
    Sample sequences from episodes

    Yields random subsequences of specified length from the episode buffer.
    Can span multiple episodes if a single episode is too short.

    Args:
        episodes: OrderedDict of episodes
        length: Sequence length
        seed: Random seed

    Yields:
        Dictionary of sequences with shape (length, ...)
    """
    np_random = np.random.RandomState(seed)

    while True:
        size = 0
        ret = None

        # Sample proportional to episode length
        p = np.array([
            len(next(iter(episode.values())))
            for episode in episodes.values()
        ])
        p = p / np.sum(p)

        while size < length:
            # Sample episode
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))

            # Must have at least one transition
            if total < 2:
                continue

            if not ret:
                # Start new sequence
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index:min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # Continue sequence from another episode
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k],
                        v[index:min(index + possible, total)].copy(),
                        axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True

            size = len(next(iter(ret.values())))

        yield ret


def from_generator(generator, batch_size: int):
    """
    Create batched iterator from generator

    Args:
        generator: Generator yielding individual sequences
        batch_size: Number of sequences per batch

    Yields:
        Batched sequences with shape (batch_size, length, ...)
    """
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))

        # Stack into batch
        data = {}
        for key in batch[0].keys():
            data[key] = np.stack([b[key] for b in batch], axis=0)

        yield data


# === Configuration Management ===

def save_config(config: Any, logdir: pathlib.Path, verbose: bool = True):
    """
    Save configuration to YAML file in log directory

    Converts config object to YAML format, handling tuples and nested structures.
    Used to save training configuration for later use (e.g., during evaluation).

    Args:
        config: Configuration object with attributes
        logdir: Log directory path
        verbose: Whether to print save confirmation

    Returns:
        Path to saved config file
    """
    from ruamel.yaml import YAML

    config_save_path = pathlib.Path(logdir) / "config.yaml"

    # Skip if config already exists
    if config_save_path.exists():
        if verbose:
            print(f"Config already exists at {config_save_path}")
        return config_save_path

    # Convert tuples to lists for YAML compatibility
    def convert_tuples(obj):
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_tuples(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tuples(item) for item in obj]
        else:
            return obj

    # Convert config attributes to dict
    config_dict = {
        k: convert_tuples(v)
        for k, v in vars(config).items()
        if not k.startswith('_')
    }

    # Save to YAML
    yaml = YAML()
    yaml.default_flow_style = False
    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f)

    if verbose:
        print(f"Saved config to {config_save_path}")

    return config_save_path
