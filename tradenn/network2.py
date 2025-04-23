from math import prod
from typing import Optional

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from tradenn.config import NetworkConfig


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class FullyConnected(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr_mult=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_mult = lr_mult
        self.weight = torch.nn.Parameter(torch.empty([out_features, in_features]))
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None
        self.weight_gain = lr_mult / np.sqrt(in_features)

        self.init_weights()

    def init_weights(self, gain=1.0):
        nn.init.orthogonal_(self.weight, gain=gain / self.lr_mult)  # type: ignore
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w = self.weight * self.weight_gain
        return torch.nn.functional.linear(x, w, self.bias)


class Conv1d(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        lr_mult=1.0,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.in_features = in_features
        self.out_features = out_features
        self.lr_mult = lr_mult
        self.weight = torch.nn.Parameter(
            torch.empty([out_features, in_features, kernel_size])
        )
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None
        self.weight_gain = lr_mult / self.weight[0].numel()

        self.init_weights()

    def init_weights(self, gain=1.0):
        nn.init.orthogonal_(self.weight, gain=gain / self.lr_mult)  # type: ignore
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w = self.weight * self.weight_gain
        return torch.nn.functional.conv1d(
            x, w, self.bias, stride=self.stride, padding=self.padding
        )


class AttentionPool1d(nn.Module):
    """
    Performs attention pooling along the sequence dimension (dim=-2).
    """

    def __init__(self, dim, lr_mult=1.0):
        super().__init__()
        self.query = FullyConnected(dim, 1, bias=False, lr_mult=lr_mult)

    def forward(self, x, keepdim=False):
        # x shape: (..., seq_len, dim)
        attn_scores = self.query(x)  # (..., seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=-2)  # (..., seq_len, 1)
        pooled = torch.sum(
            x * attn_weights, dim=-2, keepdim=keepdim
        )  # (..., dim) or (..., 1, dim)
        return pooled


class TradeNetSimple(nn.Module):
    def __init__(
        self,
        config: NetworkConfig,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        recurrent: bool = False,
    ) -> None:
        super().__init__()

        def activation():
            if config.activation == "relu":
                return nn.ReLU()
            elif config.activation == "tanh":
                return nn.Tanh()
            elif config.activation == "leaky_relu":
                return nn.LeakyReLU(0.2)
            elif config.activation == "silu":
                return nn.SiLU()
            elif config.activation == "gelu":
                return nn.GELU()
            else:
                raise ValueError(f"Unknown activation function: {config.activation}")

        f_shape = observation_space["feats"].shape
        fa_shape = observation_space["asset_feats"].shape
        dfa_shape = observation_space["day_asset_feats"].shape

        assert f_shape is not None and fa_shape is not None and dfa_shape is not None

        asset_embed_dim = config.asset_embed_dim
        hdim = config.hdim
        global_dim = 8

        self.mlp_global = nn.Sequential(
            FullyConnected(f_shape[0], global_dim),
            activation(),
        )

        self.mlp_asset = nn.Sequential(
            FullyConnected(fa_shape[0], hdim),
            activation(),
            FullyConnected(hdim, asset_embed_dim, bias=False),
            activation(),
        )

        self.conv = nn.Sequential(
            Conv1d(dfa_shape[1], config.conv_dim, kernel_size=3, padding=1, stride=2),
            activation(),
            Conv1d(
                config.conv_dim,
                config.conv_dim,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            ),
            activation(),
            Conv1d(
                config.conv_dim,
                asset_embed_dim // 4,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            ),
            activation(),
        )

        # Aggregate per-asset embeddings to support variable number of assets
        # latent_input_dim = concatenated size of per-asset features and global features
        latent_input_dim = asset_embed_dim * 2
        if recurrent:
            # Recurrent across time steps, input size independent of num_assets
            self.latent_extractor = nn.GRU(latent_input_dim, hdim, num_layers=1)
        else:
            self.latent_extractor = nn.Sequential(
                FullyConnected(latent_input_dim, hdim // 2, bias=False),
                activation(),
            )

        self.portfolio_latent = nn.Sequential(
            FullyConnected(latent_input_dim, hdim // 2, bias=False),
            activation(),
        )

        # Attention pooling for portfolio features
        self.portfolio_pool = AttentionPool1d(hdim // 2)

        assert action_space.shape is not None

        self.policy_net = nn.Sequential(
            FullyConnected(hdim + global_dim, config.policy_dim, bias=False),
            activation(),
            FullyConnected(config.policy_dim, 1, bias=False),
        )

        self.value_net = nn.Sequential(
            FullyConnected(
                hdim + global_dim,
                config.value_dim,
                bias=False,
                lr_mult=config.value_lr_mult,
            ),
            activation(),
            AttentionPool1d(config.value_dim, lr_mult=config.value_lr_mult),
            FullyConnected(
                config.value_dim, 1, bias=False, lr_mult=config.value_lr_mult
            ),
        )

        for m in self.modules():
            if isinstance(m, FullyConnected) or isinstance(m, Conv1d):
                m.init_weights(config.init_gain)

        self.value_net[-1].init_weights(config.value_proj_init_gain)
        self.policy_net[-1].init_weights(config.action_proj_init_gain)

    def _forward_policy(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features).squeeze(-1)

    def _forward_value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features).squeeze(-1)

    def _forward_common(
        self,
        features: dict[str, torch.Tensor],
        lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x_global = self.mlp_global(features["feats"])

        x_f = self.mlp_asset(features["asset_feats"].swapaxes(-1, -2))

        x_d = features["day_asset_feats"].swapaxes(-1, -3)
        orig_shape = x_d.shape
        x_d = x_d.reshape(prod(orig_shape[:-2]), *orig_shape[-2:])
        x_d = self.conv(x_d)
        x_d = x_d.reshape(*orig_shape[:2], -1)

        x = torch.cat([x_d, x_f], dim=-1)

        x_asset = self.latent_extractor(x)

        # Apply attention pooling instead of mean pooling
        portfolio_latent = self.portfolio_latent(x)
        portfolio_latent = self.portfolio_pool(portfolio_latent, keepdim=True)
        portfolio_latent = torch.cat([portfolio_latent, x_global.unsqueeze(-2)], dim=-1)

        x_portfolio = portfolio_latent.expand(
            *([-1] * (len(x.shape) - 2)), x_asset.shape[-2], -1
        )

        x = torch.cat([x_asset, x_portfolio], dim=-1)

        return x, lstm_states

    def forward(
        self,
        features: dict[str, torch.Tensor],
        lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        asset_latents, lstm_states = self._forward_common(
            features, lstm_states, episode_starts
        )
        return (
            self._forward_policy(asset_latents),
            self._forward_value(asset_latents),
            lstm_states,
        )

    def forward_actor(
        self,
        features: dict[str, torch.Tensor],
        lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        asset_latents, lstm_states = self._forward_common(
            features, lstm_states, episode_starts
        )
        return self._forward_policy(asset_latents), lstm_states

    def forward_critic(
        self,
        features: dict[str, torch.Tensor],
        lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        asset_latents, lstm_states = self._forward_common(
            features, lstm_states, episode_starts
        )
        return self._forward_value(asset_latents), lstm_states


NETWORKS = {
    "simple": TradeNetSimple,
}
