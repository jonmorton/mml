import math
from math import prod
from typing import Optional, Union

import numpy as np
import torch
from gymnasium import spaces
from torch import nn, norm


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class Norm(torch.nn.Module):
    def __init__(self, dim, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return normalize(x, dim=self.dim, eps=self.eps)


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
        self.weight = torch.nn.Parameter(
            torch.empty([out_features, in_features, kernel_size])
        )
        nn.init.uniform_(self.weight, -1.0 / lr_mult, 1.0 / lr_mult)
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
        self.weight_gain = lr_mult / self.weight[0].numel()

    def forward(self, x):
        w = self.weight * self.weight_gain
        return torch.nn.functional.conv1d(
            x, w, self.bias, stride=self.stride, padding=self.padding
        )


class AFT(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        # self.norm = nn.RMSNorm(dim)
        self.norm = nn.Identity()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_qkv = FullyConnected(dim, hidden_dim * 3, bias=False)
        self.project = FullyConnected(hidden_dim, dim, bias=False)

    def forward(self, x):
        B, A, F = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)

        probs = torch.softmax(k, 1)
        weights = torch.mul(probs, v)
        weights = weights.sum(dim=1, keepdim=True)
        yt = torch.mul(torch.sigmoid(q), weights)
        yt = yt.view(B, A, self.hidden_dim)
        return self.project(yt)


class SimpleAttn(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.scale = hidden_dim**-0.5
        self.kq = FullyConnected(dim, hidden_dim, bias=False)
        self.v = FullyConnected(dim, hidden_dim, bias=False)
        self.out = FullyConnected(hidden_dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim)

    def forward(self, x):
        c = self.kq(x)
        c = normalize(c)
        S = torch.bmm(c, c.transpose(1, 2))
        S = torch.softmax(S * self.scale, dim=-1)
        # S = nn.functional.dropout(S, p=0.2, training=self.training)
        v = self.v(x)
        v = torch.bmm(S, v)
        return self.out(v)


class ScaledTanh(nn.Module):
    """
    Scales the output of a tanh activation function to the range [low, high].

    :param low: The lower bound of the output range
    :param high: The upper bound of the output range
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = 1 + self.alpha
        gamma = 1 + self.gamma
        return torch.tanh(x * alpha) * gamma + self.beta


class BaseNet(nn.Module):
    def __init__(
        self,
        recurrent: bool = False,
    ) -> None:
        super().__init__()
        self.recurrent = recurrent

    def _forward_common(
        self,
        features: dict[str, torch.Tensor],
        lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError("Must be implemented in child class.")

    @staticmethod
    def _process_sequence(
        rnn: nn.RNNBase,
        features: torch.Tensor,
        lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if lstm_states is None:
            # Initialize the LSTM state
            lstm_states = (
                torch.zeros(
                    1, features.shape[0], rnn.hidden_size, device=features.device
                ),
                torch.zeros(
                    1, features.shape[0], rnn.hidden_size, device=features.device
                ),
            )
        if episode_starts is None:
            episode_starts = torch.zeros(features.shape[0], device=features.device)

        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        features_sequence = features.reshape((n_seq, -1, rnn.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if torch.all(episode_starts == 0.0):
            lstm_output, lstm_states = rnn(features_sequence, lstm_states[0])
            lstm_output = torch.flatten(
                lstm_output.transpose(0, 1), start_dim=0, end_dim=1
            )
            return lstm_output, (lstm_states, lstm_states)

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip(
            features_sequence, episode_starts, strict=True
        ):
            hidden, lstm_states = rnn(
                features.unsqueeze(dim=0),
                (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = torch.flatten(
            torch.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1
        )
        return lstm_output, (lstm_states, lstm_states)

    def _forward_policy(self, features: torch.Tensor):
        raise NotImplementedError("Must be implemented in child class.")

    def _forward_value(self, features: torch.Tensor):
        raise NotImplementedError("Must be implemented in child class.")

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
        latents, lstm_states = self._forward_common(
            features, lstm_states, episode_starts
        )
        return (
            self._forward_policy(latents),
            self._forward_value(latents),
            lstm_states,
        )

    def forward_actor(
        self,
        features: dict[str, torch.Tensor],
        lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        latents, lstm_states = self._forward_common(
            features, lstm_states, episode_starts
        )
        return self._forward_policy(latents), lstm_states

    def forward_critic(
        self,
        features: dict[str, torch.Tensor],
        lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        latents, lstm_states = self._forward_common(
            features, lstm_states, episode_starts
        )
        return self._forward_value(latents), lstm_states


class TradeNetSimple(BaseNet):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        recurrent: bool = False,
    ) -> None:
        super().__init__(recurrent=recurrent)

        f_shape = observation_space["feats"].shape
        fa_shape = observation_space["asset_feats"].shape
        dfa_shape = observation_space["day_asset_feats"].shape

        assert f_shape is not None and fa_shape is not None and dfa_shape is not None

        num_assets = fa_shape[1]

        asset_embed_dim = 16
        hdim = 128
        global_dim = 8

        self.mlp_global = nn.Sequential(
            FullyConnected(f_shape[0], global_dim),
        )

        self.mlp_asset = nn.Sequential(
            FullyConnected(fa_shape[0], hdim),
            nn.ReLU(),
            FullyConnected(hdim, asset_embed_dim, bias=False),
        )

        # self.conv = nn.Sequential(
        #     Conv1d(dfa_shape[1], 32, kernel_size=3, padding=1, stride=1),
        #     nn.ReLU(),
        #     nn.AvgPool1d(2, 2, 0),
        #     Conv1d(32, 32, kernel_size=3, padding=1, stride=1),
        #     nn.ReLU(),
        #     nn.AvgPool1d(2, 2, 0),
        #     Conv1d(32, asset_embed_dim // 4, kernel_size=3, padding=1, stride=1),
        #     nn.AvgPool1d(2, 2, 0),
        # )

        self.conv = nn.Sequential(
            Conv1d(dfa_shape[1], 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            Conv1d(32, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(),
            Conv1d(
                32, asset_embed_dim // 4, kernel_size=3, padding=1, stride=2, bias=False
            ),
        )

        # Aggregate per-asset embeddings to support variable number of assets
        # latent_input_dim = concatenated size of per-asset features and global features
        latent_input_dim = asset_embed_dim * 2 * num_assets + global_dim
        if recurrent:
            # Recurrent across time steps, input size independent of num_assets
            self.latent_extractor = nn.GRU(latent_input_dim, hdim, num_layers=1)
        else:
            self.latent_extractor = nn.Sequential(
                FullyConnected(latent_input_dim, hdim, bias=False),
            )

        self.latent_dim_pi = 128
        self.latent_dim_vf = 128

        assert action_space.shape is not None

        self.policy_net = nn.Sequential(
            FullyConnected(hdim, hdim, bias=False),
            nn.ReLU(),
            FullyConnected(hdim, action_space.shape[0], bias=False),
        )

        self.value_net = nn.Sequential(
            FullyConnected(hdim, hdim, bias=False),
            nn.ReLU(),
            FullyConnected(hdim, 1, bias=False),
        )

    def _forward_policy(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def _forward_value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

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

        x = torch.cat([x_d.flatten(-2), x_f.flatten(-2), x_global], dim=-1)
        x = nn.functional.relu(x)

        if self.recurrent:
            assert isinstance(self.latent_extractor, nn.RNNBase)
            x, lstm_states = BaseNet._process_sequence(
                self.latent_extractor, x, lstm_states, episode_starts
            )
        else:
            x = self.latent_extractor(x)

        return x, lstm_states


class TradeNet(BaseNet):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        recurrent: bool = False,
    ) -> None:
        super().__init__(recurrent=recurrent)

        f_shape = observation_space["feats"].shape
        fa_shape = observation_space["asset_feats"].shape
        dfa_shape = observation_space["day_asset_feats"].shape

        assert f_shape is not None and fa_shape is not None and dfa_shape is not None

        num_assets = fa_shape[1]

        asset_embed_dim = 64
        hdim = 128
        global_dim = 8

        self.mlp_global = nn.Sequential(
            FullyConnected(f_shape[0], global_dim),
        )

        self.mlp_asset = nn.Sequential(
            FullyConnected(fa_shape[0], hdim),
            nn.ReLU(),
            FullyConnected(hdim, asset_embed_dim, bias=False),
        )

        self.conv = nn.Sequential(
            Conv1d(dfa_shape[1], 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            Conv1d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            Conv1d(32, asset_embed_dim // 4, kernel_size=3, padding=1, stride=2),
        )

        if recurrent:
            self.latent_extractor = nn.GRU(
                num_assets * asset_embed_dim * 2, hdim, num_layers=1
            )
        else:
            self.latent_extractor = nn.Sequential(
                nn.Identity(),
                FullyConnected(
                    asset_embed_dim * 2 + global_dim, asset_embed_dim, bias=False
                ),
                Norm(-1),
                # nn.ReLU(),
            )

        self.attn = SimpleAttn(asset_embed_dim, 64)

        self.latent_dim_pi = 128
        self.latent_dim_vf = 128

        self.policy_net = nn.Sequential(
            FullyConnected(asset_embed_dim, hdim, bias=False),
            nn.ReLU(),
            FullyConnected(hdim, 1, bias=False),
        )

        self.value_net_1 = nn.Sequential(
            FullyConnected(asset_embed_dim, hdim, bias=False),
        )
        self.value_net_2 = nn.Sequential(
            FullyConnected(hdim, 1, bias=False),
        )

    def _forward_policy(self, features: torch.Tensor) -> torch.Tensor:
        x = self.policy_net(features).squeeze(-1)
        return x

    def _forward_value(self, features: torch.Tensor) -> torch.Tensor:
        x = self.value_net_1(features)
        x = x.sum(-2)
        x = self.value_net_2(x)
        return x

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

        x = torch.cat(
            [
                x_d,
                x_f,
                x_global.unsqueeze(-2).expand(
                    *([-1] * (len(x_global.shape) - 1)), x_d.shape[-2], -1
                ),
            ],
            dim=-1,
        )
        x = nn.functional.relu(x)

        if self.recurrent:
            assert isinstance(self.latent_extractor, nn.RNNBase)
            x, lstm_states = BaseNet._process_sequence(
                self.latent_extractor, x, lstm_states, episode_starts
            )
        else:
            x = self.latent_extractor(x)

        # x = x + self.attn(x)

        return x, lstm_states


NETWORKS = {
    "simple": TradeNetSimple,
    "attn": TradeNet,
}
