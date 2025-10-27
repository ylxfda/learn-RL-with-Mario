"""
Encoder and Decoder Networks for DreamerV3

This module implements the observation encoding and decoding networks:
- ConvEncoder: CNN encoder for visual observations
- ConvDecoder: CNN decoder for image reconstruction
- MultiEncoder: Handles multiple input modalities (images, vectors)
- MultiDecoder: Handles multiple output modalities
- MLP: General-purpose multi-layer perceptron

Paper Reference: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
Section 2 "Method" - describes encoder/decoder architecture
"""

import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as td
from typing import Dict, Tuple, Optional

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from dreamerv3.utils import tools


class MultiEncoder(nn.Module):
    """
    Multi-Modal Encoder

    Combines multiple input modalities (e.g., images, proprioception) into
    a single latent embedding. Uses separate encoders for different modalities:
    - CNN encoder for images
    - MLP encoder for vector observations

    Paper Reference: Section 2.1 "World Model Learning"
    The encoder maps observations to latent embeddings: o_t -> e_t
    """

    def __init__(
        self,
        shapes: Dict[str, Tuple],
        mlp_keys: str,
        cnn_keys: str,
        act: str,
        norm: bool,
        cnn_depth: int,
        kernel_size: int,
        minres: int,
        mlp_layers: int,
        mlp_units: int,
        symlog_inputs: bool
    ):
        """
        Initialize multi-modal encoder

        Args:
            shapes: Dictionary mapping observation keys to shapes
                   e.g., {'image': (64, 64, 3), 'velocity': (2,)}
            mlp_keys: Regex pattern for MLP-encoded observations
            cnn_keys: Regex pattern for CNN-encoded observations
            act: Activation function name (e.g., 'SiLU')
            norm: Whether to use layer normalization
            cnn_depth: Base number of CNN channels
            kernel_size: Convolution kernel size
            minres: Minimum spatial resolution before stopping downsampling
            mlp_layers: Number of MLP layers
            mlp_units: Number of units in MLP hidden layers
            symlog_inputs: Whether to apply symlog to vector inputs
        """
        super(MultiEncoder, self).__init__()

        # Filter out non-observation keys (metadata)
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }

        # Separate image and vector observations
        self.cnn_shapes = {
            k: v for k, v in shapes.items()
            if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }

        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0

        # Create CNN encoder for images
        if self.cnn_shapes:
            # Concatenate all image channels
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim

        # Create MLP encoder for vectors
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,  # No output shape (returns hidden activations)
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder"
            )
            self.outdim += mlp_units

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observations to latent embedding

        Input:
            obs: Dictionary of observations
                - Images: (batch_size, time_steps, height, width, channels)
                - Vectors: (batch_size, time_steps, feature_dim)

        Output:
            Latent embedding: (batch_size, time_steps, embed_dim)
        """
        outputs = []

        # Encode images with CNN
        if self.cnn_shapes:
            # Concatenate all image inputs
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], dim=-1)
            outputs.append(self._cnn(inputs))

        # Encode vectors with MLP
        if self.mlp_shapes:
            # Concatenate all vector inputs
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], dim=-1)
            outputs.append(self._mlp(inputs))

        # Concatenate all embeddings
        outputs = torch.cat(outputs, dim=-1)
        return outputs


class MultiDecoder(nn.Module):
    """
    Multi-Modal Decoder

    Decodes latent features into multiple output modalities (images, vectors).
    Used for reconstructing observations from latent states.

    Paper Reference: Section 2.1 "World Model Learning"
    The decoder reconstructs observations: z_t, h_t -> ô_t
    """

    def __init__(
        self,
        feat_size: int,
        shapes: Dict[str, Tuple],
        mlp_keys: str,
        cnn_keys: str,
        act: str,
        norm: bool,
        cnn_depth: int,
        kernel_size: int,
        minres: int,
        mlp_layers: int,
        mlp_units: int,
        cnn_sigmoid: bool,
        image_dist: str,
        vector_dist: str,
        outscale: float
    ):
        """
        Initialize multi-modal decoder

        Args:
            feat_size: Size of input features (stoch_dim * discrete_dim + deter_dim)
            shapes: Dictionary mapping observation keys to shapes
            mlp_keys: Regex pattern for MLP-decoded observations
            cnn_keys: Regex pattern for CNN-decoded observations
            act: Activation function name
            norm: Whether to use layer normalization
            cnn_depth: Base number of CNN channels
            kernel_size: Convolution kernel size
            minres: Minimum spatial resolution
            mlp_layers: Number of MLP layers
            mlp_units: Number of units in MLP hidden layers
            cnn_sigmoid: Whether to apply sigmoid to image outputs
            image_dist: Distribution type for images ('mse' or 'normal')
            vector_dist: Distribution type for vectors
            outscale: Output layer initialization scale
        """
        super(MultiDecoder, self).__init__()

        # Filter out metadata
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}

        # Separate image and vector observations
        self.cnn_shapes = {
            k: v for k, v in shapes.items()
            if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }

        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        # Create CNN decoder for images
        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            # Output channels = sum of all image channels
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid
            )

        # Create MLP decoder for vectors
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder"
            )

        self._image_dist = image_dist

    def forward(self, features: torch.Tensor) -> Dict[str, td.Distribution]:
        """
        Decode features to observation distributions

        Input:
            features: Latent features (batch_size, time_steps, feat_size)

        Output:
            Dictionary mapping observation keys to distributions
            Each distribution can sample reconstructed observations
        """
        dists = {}

        # Decode images with CNN
        if self.cnn_shapes:
            outputs = self._cnn(features)
            # Split into separate image channels
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, dim=-1)
            dists.update({
                key: self._make_image_dist(output)
                for key, output in zip(self.cnn_shapes.keys(), outputs)
            })

        # Decode vectors with MLP
        if self.mlp_shapes:
            dists.update(self._mlp(features))

        return dists

    def _make_image_dist(self, mean: torch.Tensor) -> td.Distribution:
        """
        Create distribution for image reconstruction

        Args:
            mean: Predicted image mean

        Returns:
            Distribution (MSE or Normal)
        """
        if self._image_dist == "normal":
            return tools.ContDist(
                td.independent.Independent(
                    td.normal.Normal(mean, 1),
                    reinterpreted_batch_ndims=3  # Spatial + channel dims
                )
            )
        elif self._image_dist == "mse":
            return tools.MSEDist(mean)
        else:
            raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder for Images

    Downsamples images through a series of convolutional layers.
    Architecture: Conv -> LayerNorm -> Activation -> ... -> Flatten

    Each conv layer:
    - Doubles the number of channels
    - Halves the spatial resolution (stride=2)
    - Uses "same" padding to maintain shape

    Paper Reference: Uses standard CNN architecture as in DreamerV2/V3
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        depth: int = 32,
        act: str = "SiLU",
        norm: bool = True,
        kernel_size: int = 4,
        minres: int = 4
    ):
        """
        Initialize convolutional encoder

        Args:
            input_shape: (height, width, channels)
            depth: Base number of output channels
            act: Activation function name
            norm: Whether to use layer normalization
            kernel_size: Convolution kernel size
            minres: Minimum spatial resolution (stop downsampling at this size)
        """
        super(ConvEncoder, self).__init__()

        act_fn = getattr(torch.nn, act)
        h, w, input_ch = input_shape

        # Calculate number of downsampling stages
        # Stop when resolution reaches minres
        stages = int(np.log2(h) - np.log2(minres))

        # Build encoder layers
        in_dim = input_ch
        out_dim = depth
        layers = []

        for i in range(stages):
            # Convolution with stride 2 (downsampling)
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False
                )
            )
            # Layer normalization (applied to channel dimension)
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            # Activation
            layers.append(act_fn())

            # Update dimensions for next layer
            in_dim = out_dim
            out_dim *= 2  # Double channels each stage
            h, w = h // 2, w // 2  # Halve spatial dimensions

        # Output dimension after flattening
        self.outdim = (out_dim // 2) * h * w

        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode image observations

        Input:
            obs: Images in [0, 1], shape (batch_size, time_steps, H, W, C)

        Output:
            Encoded features, shape (batch_size, time_steps, outdim)
        """
        # Center images around 0
        obs = obs - 0.5

        # Reshape: (batch, time, H, W, C) -> (batch * time, H, W, C)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))

        # PyTorch conv2d expects (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Apply convolutions
        x = self.layers(x)

        # Flatten spatial dimensions: (batch * time, C, H, W) -> (batch * time, C*H*W)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])

        # Reshape back: (batch * time, outdim) -> (batch, time, outdim)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    """
    Convolutional Decoder for Images

    Upsamples latent features to reconstruct images.
    Architecture: Linear -> Reshape -> ConvTranspose -> ... -> ConvTranspose

    Each transpose conv layer:
    - Halves the number of channels
    - Doubles the spatial resolution (stride=2)

    Paper Reference: Uses standard CNN decoder as in DreamerV2/V3
    """

    def __init__(
        self,
        feat_size: int,
        shape: Tuple[int, int, int] = (3, 64, 64),
        depth: int = 32,
        act: str = "SiLU",
        norm: bool = True,
        kernel_size: int = 4,
        minres: int = 4,
        outscale: float = 1.0,
        cnn_sigmoid: bool = False
    ):
        """
        Initialize convolutional decoder

        Args:
            feat_size: Size of input features
            shape: Output shape (channels, height, width)
            depth: Base number of channels
            act: Activation function name
            norm: Whether to use layer normalization
            kernel_size: Convolution kernel size
            minres: Initial spatial resolution
            outscale: Output layer initialization scale
            cnn_sigmoid: Whether to apply sigmoid to outputs
        """
        super(ConvDecoder, self).__init__()

        act_fn = getattr(torch.nn, act)
        self._shape = shape  # (C, H, W)
        self._cnn_sigmoid = cnn_sigmoid
        self._minres = minres

        # Calculate number of upsampling stages
        layer_num = int(np.log2(shape[1]) - np.log2(minres))

        # Initial channel dimension at minres x minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        # Linear layer to expand features
        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))

        # Build decoder layers
        in_dim = out_ch // (minres**2)  # Channels after reshaping
        out_dim = in_dim // 2
        layers = []

        for i in range(layer_num):
            # Special handling for final layer
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]  # Output channels
                act_fn = None  # No activation on final layer
                bias = True
                norm = False

            # Update in_dim for all but first layer
            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth

            # Calculate padding for "same" output size with stride 2
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)

            # Transpose convolution (upsampling)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias
                )
            )

            # Layer norm and activation (except final layer)
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act_fn:
                layers.append(act_fn())

            # Update dimensions
            in_dim = out_dim
            out_dim //= 2

        # Initialize all layers except the last with standard init
        for m in layers[:-1]:
            m.apply(tools.weight_init)
        # Initialize last layer with outscale
        layers[-1].apply(tools.uniform_weight_init(outscale))

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k: int, s: int, d: int) -> Tuple[int, int]:
        """
        Calculate padding for "same" output size with stride > 1

        Args:
            k: Kernel size
            s: Stride
            d: Dilation

        Returns:
            (padding, output_padding)
        """
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(
        self,
        features: torch.Tensor,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Decode features to images

        Input:
            features: Latent features, shape (batch_size, time_steps, feat_size)
            dtype: Optional output dtype (unused)

        Output:
            Reconstructed images in [0, 1], shape (batch_size, time_steps, H, W, C)
        """
        # Expand features to initial spatial resolution
        x = self._linear_layer(features)

        # Reshape: (batch, time, embed) -> (batch * time, minres, minres, channels)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )

        # PyTorch conv2d expects (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Apply transpose convolutions
        x = self.layers(x)

        # Reshape: (batch * time, C, H, W) -> (batch, time, C, H, W)
        mean = x.reshape(features.shape[:-1] + self._shape)

        # Convert to (batch, time, H, W, C)
        mean = mean.permute(0, 1, 3, 4, 2)

        # Map to [0, 1] range
        if self._cnn_sigmoid:
            mean = torch.sigmoid(mean)
        else:
            mean = mean + 0.5

        return mean


class MLP(nn.Module):
    """
    Multi-Layer Perceptron

    General-purpose feedforward network used throughout DreamerV3:
    - Actor network (policy)
    - Critic network (value function)
    - Reward predictor
    - Continuation predictor
    - Vector encoder/decoder

    Architecture: Linear -> LayerNorm -> Activation -> ... -> Linear

    Supports various output distributions for different tasks.
    """

    def __init__(
        self,
        inp_dim: int,
        shape: Optional[any],
        layers: int,
        units: int,
        act: str = "SiLU",
        norm: bool = True,
        dist: str = "normal",
        std: any = 1.0,
        min_std: float = 0.1,
        max_std: float = 1.0,
        absmax: Optional[float] = None,
        temp: float = 0.1,
        unimix_ratio: float = 0.01,
        outscale: float = 1.0,
        symlog_inputs: bool = False,
        device: str = "cuda",
        name: str = "NoName"
    ):
        """
        Initialize MLP

        Args:
            inp_dim: Input dimension
            shape: Output shape(s) - int, tuple, or dict of shapes (None for encoder)
            layers: Number of hidden layers
            units: Hidden layer size
            act: Activation function name
            norm: Whether to use layer normalization
            dist: Output distribution type:
                  - 'normal': Gaussian (continuous actions)
                  - 'onehot': Categorical (discrete actions)
                  - 'symlog_disc': Discretized symlog (rewards, values)
                  - 'binary': Bernoulli (continuation)
                  - 'mse': Deterministic MSE
            std: Standard deviation (for normal dist) - 'learned' or float
            min_std: Minimum std for normal distribution
            max_std: Maximum std for normal distribution
            absmax: Absolute maximum for clamping outputs
            temp: Temperature for categorical sampling
            unimix_ratio: Uniform mixing ratio for exploration
            outscale: Output layer initialization scale
            symlog_inputs: Whether to apply symlog to inputs
            device: Device for tensors
            name: Name for layer identification
        """
        super(MLP, self).__init__()

        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)

        act_fn = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        # Build hidden layers
        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}",
                nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}",
                    nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act_fn())

            if i == 0:
                inp_dim = units

        self.layers.apply(tools.weight_init)

        # Build output layers
        if isinstance(self._shape, dict):
            # Multiple outputs (e.g., vector decoder)
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))

            # Learned std for continuous distributions
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

        elif self._shape is not None:
            # Single output
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))

            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(
        self,
        features: torch.Tensor,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Forward pass through MLP

        Input:
            features: Input features, shape (..., inp_dim)
            dtype: Optional output dtype

        Output:
            - If shape is None: hidden activations (for encoder)
            - If shape is dict: dict of distributions
            - Otherwise: single distribution
        """
        x = features

        # Apply symlog to inputs if specified
        if self._symlog_inputs:
            x = tools.symlog(x)

        # Pass through hidden layers
        out = self.layers(x)

        # Return hidden activations (for encoder)
        if self._shape is None:
            return out

        # Multiple outputs
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists[name] = self._make_dist(self._dist, mean, std, shape)
            return dists

        # Single output
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self._make_dist(self._dist, mean, std, self._shape)

    def _make_dist(
        self,
        dist_type: str,
        mean: torch.Tensor,
        std: any,
        shape: Tuple
    ) -> td.Distribution:
        """
        Create output distribution

        Args:
            dist_type: Distribution type
            mean: Mean or logits
            std: Standard deviation (for continuous)
            shape: Output shape

        Returns:
            Distribution object
        """
        if dist_type == "tanh_normal":
            # Normal distribution squashed through tanh to [-1, 1]
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = td.normal.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = td.independent.Independent(dist, reinterpreted_batch_ndims=1)
            dist = tools.SampleDist(dist)

        elif dist_type == "normal":
            # Normal distribution with learned/fixed std
            std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
            dist = td.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                td.independent.Independent(dist, reinterpreted_batch_ndims=1),
                absmax=self._absmax
            )

        elif dist_type == "normal_std_fixed":
            # Normal with fixed std
            dist = td.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                td.independent.Independent(dist, reinterpreted_batch_ndims=1),
                absmax=self._absmax
            )

        elif dist_type == "trunc_normal":
            # Truncated normal to [-1, 1]
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                td.independent.Independent(dist, reinterpreted_batch_ndims=1),
                absmax=self._absmax
            )

        elif dist_type == "onehot":
            # Categorical distribution for discrete actions
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)

        elif dist_type == "onehot_gumble":
            # Gumbel-softmax for differentiable discrete sampling
            dist = tools.ContDist(
                td.gumbel.Gumbel(mean, 1 / self._temp),
                absmax=self._absmax
            )

        elif dist_type == "huber":
            # Huber loss (robust to outliers)
            dist = tools.ContDist(
                td.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax
                )
            )

        elif dist_type == "binary":
            # Bernoulli for binary predictions
            dist = tools.Bernoulli(
                td.independent.Independent(
                    td.bernoulli.Bernoulli(logits=mean),
                    len(shape)
                )
            )

        elif dist_type == "symlog_disc":
            # Discretized distribution in symlog space
            dist = tools.DiscDist(logits=mean, device=self._device)

        elif dist_type == "symlog_mse":
            # MSE in symlog space
            dist = tools.SymlogDist(mean)

        else:
            raise NotImplementedError(dist_type)

        return dist


class Conv2dSamePad(torch.nn.Conv2d):
    """
    Conv2d with 'SAME' padding (like TensorFlow)

    Automatically calculates padding to maintain spatial dimensions
    when stride=1, or to properly downsample when stride>1.
    """

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """
        Calculate padding for dimension

        Args:
            i: Input size
            k: Kernel size
            s: Stride
            d: Dilation

        Returns:
            Required padding
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with automatic padding

        Input:
            x: Input tensor (batch, channels, height, width)

        Output:
            Convolved tensor
        """
        ih, iw = x.size()[-2:]

        # Calculate padding for height and width
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        # Apply convolution
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class ImgChLayerNorm(nn.Module):
    """
    Layer Normalization for Image Channels

    Applies LayerNorm to the channel dimension of images (C, H, W format).
    Handles the permutation needed for PyTorch's LayerNorm.
    """

    def __init__(self, ch: int, eps: float = 1e-03):
        """
        Initialize channel-wise layer norm

        Args:
            ch: Number of channels
            eps: Epsilon for numerical stability
        """
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm to channel dimension

        Input:
            x: Image tensor (batch, channels, height, width)

        Output:
            Normalized tensor (same shape)
        """
        # Permute to (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)
        # Apply layer norm to channel dimension
        x = self.norm(x)
        # Permute back to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        return x
