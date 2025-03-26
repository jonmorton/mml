from math import e
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class FullyConnected(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr_mult=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty([out_features, in_features]))
        nn.init.uniform_(self.weight, -1.0 / lr_mult, 1.0 / lr_mult)
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
        self.weight_gain = lr_mult / np.sqrt(in_features)

    def forward(self, x):
        w = self.weight
        w = w * self.weight_gain
        return torch.nn.functional.linear(x, w, self.bias)


class AFT(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        # self.norm = nn.RMSNorm(hidden_dim)
        self.norm = nn.Identity()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_qkv = FullyConnected(dim, hidden_dim * 3, bias=False)
        self.project = FullyConnected(hidden_dim, dim, bias=False)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x[..., 0], dtype=torch.bool)

        B, A, F = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)

        k = k.masked_fill(
            mask.view(B, A, 1).expand(-1, -1, self.hidden_dim) == 0, float("-inf")
        )
        probs = torch.softmax(k, 1)
        probs = nn.functional.dropout(probs, p=0.15, training=self.training)
        weights = torch.mul(probs, v)
        weights = weights.sum(dim=1, keepdim=True)
        yt = torch.mul(torch.sigmoid(q), weights)
        yt = yt.view(B, A, self.hidden_dim)
        return x + self.project(yt)


class GEGLU(nn.Module):
    def forward(self, x):
        x, y = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(y)


def calculate_output_size_from_sequential(input_size, sequential_model):
    """
    Written by: Grok

    Calculate output size after processing through an nn.Sequential model

    Args:
        input_size (tuple): Input dimensions (height, width)
        sequential_model (nn.Sequential): PyTorch Sequential model containing Conv2d and MaxPool2d layers

    Returns:
        tuple: Final output size (height, width)
    """
    h, w = input_size

    # Iterate through each layer in the sequential model
    for layer in sequential_model:
        if isinstance(layer, nn.Conv2d):
            # Extract conv parameters
            kernel_size = (
                layer.kernel_size[0]
                if isinstance(layer.kernel_size, tuple)
                else layer.kernel_size
            )
            stride = (
                layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            )
            padding = (
                layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
            )

            h = ((h + 2 * padding - kernel_size) // stride) + 1
            w = ((w + 2 * padding - kernel_size) // stride) + 1

        elif isinstance(layer, nn.MaxPool2d):
            # Extract pooling parameters
            kernel_size = (
                layer.kernel_size
                if isinstance(layer.kernel_size, int)
                else layer.kernel_size[0]
            )
            stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            padding = (
                layer.padding if isinstance(layer.padding, int) else layer.padding[0]
            )

            h = ((h + 2 * padding - kernel_size) // stride) + 1
            w = ((w + 2 * padding - kernel_size) // stride) + 1

        elif hasattr(layer, "calc_out_shape"):
            h, w = layer.calc_out_shape((h, w))

        # Check if dimensions become invalid
        if h <= 0 or w <= 0:
            raise ValueError(
                f"Output dimensions became invalid ({h}, {w}) at layer {layer}"
            )

    return (h, w)


class Block2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_h=1,
        downsample_w=1,
        use_bn=True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels // 4, kernel_size=(1, 5), padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels // 4, kernel_size=(5, 1), padding=2
        )
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv4 = nn.Conv2d(
            in_channels, out_channels - out_channels // 4 * 3, kernel_size=3
        )

        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()

        self.relu = nn.ReLU()
        if downsample_h > 1 or downsample_w > 1:
            self.downsample = nn.MaxPool2d(
                kernel_size=(downsample_h, downsample_w),
                stride=(downsample_h, downsample_w),
            )
        else:
            self.downsample = nn.Identity()

        super().__init__(
            self.bn,
            self.relu,
            self.downsample,
        )

    def forward(self, x):
        x = torch.cat(
            [self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=1
        )
        super().forward(x)
        return x

    def calc_out_shape(self, input_shape):
        return calculate_output_size_from_sequential(input_shape, self)


class AttentionPool1D(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        """
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int, optional): Dimension of hidden layer for attention computation.
                                      Defaults to input_dim if not specified
        """
        super(AttentionPool1D, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim

        # Attention mechanism layers
        self.attention = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Pooled output of shape (batch_size, input_dim)
        """
        # Compute attention weights
        # Shape: (batch_size, seq_len, 1)
        attention_weights = self.attention(x)

        # Apply softmax to get normalized weights
        # Shape: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Compute weighted sum
        # Shape: (batch_size, input_dim)
        output = torch.sum(x * attention_weights, dim=1)

        return output


class AssetTablePolicy(ActorCriticPolicy):
    """
    expected obseveration space: Box(num_feats, num_assets)"
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self.num_feats = observation_space.shape[0]
        self.num_assets = observation_space.shape[1]
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            log_std_init=log_std_init,
            net_arch=None,
            ortho_init=False,
            use_sde=use_sde,
            full_std=True,
            use_expln=False,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        # if obs.ndim == 2:
        #     obs = obs.unsqueeze(0)

        # (Batch,Feature,Asset) -> (Batch,Asset,Feature)

        x = self.bn(obs)

        x = x.swapaxes(-1, -2)

        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        x = self.fe(x)

        # y = self.asset_mlp(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        # return
        #  torch.cat([x, y], dim=-1)

        return x

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        values = self.value_net(features)
        distribution = self._get_action_dist_from_latent(features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(obs).get_actions(deterministic=deterministic)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        distribution = self._get_action_dist_from_latent(features)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(features)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        return self._get_action_dist_from_latent(self.extract_features(obs))

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        return self.value_net(features)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        super()._build(lr_schedule)

        latent_dim_pi = self.features_dim

        if isinstance(self.action_dist, DiagGaussianDistribution):
            _, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            _, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            _ = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        asset_embed_dim = 32
        out_latent_dim = 128

        # round up to the nearest multiple of 32
        asset_expand_dim = int((self.num_feats * 1.75) + 31) // 32 * 32
        print("Asset expand dim:", asset_expand_dim)

        norm_style = "bn"

        if norm_style == "rmsnorm":
            norm_layer = nn.RMSNorm
        elif norm_style == "layernorm":
            norm_layer = nn.LayerNorm
        elif norm_style == "bn":
            norm_layer = lambda nfeat: nn.BatchNorm1d(nfeat, affine=False)
        else:
            raise ValueError(f"Unknown norm style '{norm_style}'")

        self.bn = nn.BatchNorm1d(self.num_feats)

        self.fe = nn.Sequential(
            FullyConnected(self.num_feats + 1, asset_expand_dim),
            # nn.RMSNorm(asset_expand_dim),
            nn.ReLU(),
            FullyConnected(asset_expand_dim, asset_embed_dim),
            nn.ReLU(),
            nn.Flatten(1),
            FullyConnected(asset_embed_dim * self.num_assets, out_latent_dim),
        )

        self.asset_mlp = nn.Sequential(
            FullyConnected(self.num_assets, self.num_assets * 4),
            GEGLU(),
            FullyConnected(self.num_assets * 2, self.num_assets),
        )

        self.action_net = nn.Sequential(
            FullyConnected(out_latent_dim, get_action_dim(self.action_space)),
            # nn.ReLU(),
            # FullyConnected(
            #     out_latent_dim, get_action_dim(self.action_space), bias=False
            # ),
            nn.Tanh(),
        )

        self.value_net = nn.Sequential(
            FullyConnected(out_latent_dim, 1),
        )

        self.features_dim = out_latent_dim

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, FullyConnected):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
