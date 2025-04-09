"""Policies: abstract base class and concrete implementations."""

import warnings
from functools import partial
from tkinter import Scale
from typing import Any, Optional, Self, Union

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from pyparsing import Opt
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    sum_independent_dims,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import lstm, nn


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


class IdentityExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space: The observation space of the environment
    """

    def __init__(self, observation_space: spaces.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations


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

    def forward(self, x: th.Tensor) -> th.Tensor:
        alpha = 1 + self.alpha
        gamma = 1 + self.gamma
        return torch.tanh(x * alpha) * gamma + self.beta


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        recurrent: bool = False,
    ) -> None:
        super().__init__()

        assert observation_space.shape is not None
        feat_dim = observation_space.shape[0]
        num_assets = observation_space.shape[1]

        asset_embed_dim = 32
        hdim = 128

        self.feat_extractor = nn.Sequential(
            FullyConnected(feat_dim, hdim),
            nn.ReLU(),
            FullyConnected(hdim, asset_embed_dim, bias=False),
        )

        self.asset_extractor = nn.Sequential(
            FullyConnected(num_assets, hdim, bias=False),
            nn.ReLU(),
            FullyConnected(hdim, num_assets, bias=False),
        )

        self.latent_dim_pi = 128
        self.latent_dim_vf = 128

        self.recurrent = recurrent

        if recurrent:
            self.latent_extractor = nn.GRU(
                num_assets * asset_embed_dim, hdim, num_layers=1
            )
        else:
            self.latent_extractor = nn.Sequential(
                FullyConnected(num_assets * asset_embed_dim, hdim),
                nn.ReLU(),
            )

        self.policy_net = nn.Sequential(
            FullyConnected(hdim, self.latent_dim_pi, bias=False),
            nn.ReLU(),
        )

        self.value_net = nn.Sequential(
            FullyConnected(hdim, self.latent_dim_vf, bias=False),
            nn.ReLU(),
        )
        self.norm = nn.RMSNorm(hdim)

    def _forward_common(
        self,
        features: th.Tensor,
        lstm_states: Optional[tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Common forward pass for the policy and value networks.
        :param features: The input features
        :return: The output of the network
        """
        features = features.swapaxes(-1, -2)
        features = self.feat_extractor(features)
        features = features.swapaxes(-1, -2)
        features = features + self.asset_extractor(features)

        features = features.flatten(-2)

        if self.recurrent:
            features, lstm_states = self._process_sequence(
                features, lstm_states, episode_starts
            )
        else:
            features = self.latent_extractor(features)

        return features, lstm_states

    def _process_sequence(
        self,
        features: th.Tensor,
        lstm_states: Optional[tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Do a forward pass in the LSTM network.

        :param features: Input tensor
        :param lstm_states: previous hidden and cell states of the LSTM, respectively
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset LSTM states.
        :param lstm: LSTM object.
        :return: LSTM output and updated LSTM states.
        """
        rnn = self.latent_extractor

        if lstm_states is None:
            # Initialize the LSTM state
            print("None")
            lstm_states = (
                th.zeros(1, features.shape[0], rnn.hidden_size, device=features.device),
                th.zeros(1, features.shape[0], rnn.hidden_size, device=features.device),
            )
        if episode_starts is None:
            episode_starts = th.zeros(features.shape[0], device=features.device)

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
        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = rnn(features_sequence, lstm_states[0])
            lstm_output = self.norm(lstm_output)
            lstm_output = th.flatten(
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
        lstm_output = th.flatten(
            th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1
        )
        lstm_output = self.norm(lstm_output)
        return lstm_output, (lstm_states, lstm_states)

    def forward(
        self,
        features: th.Tensor,
        lstm_states: Optional[tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features, lstm_states = self._forward_common(
            features, lstm_states, episode_starts
        )
        return self.policy_net(features), self.value_net(features), lstm_states

    def forward_actor(
        self,
        features: th.Tensor,
        lstm_states: Optional[tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        features, lstm_states = self._forward_common(
            features, lstm_states, episode_starts
        )
        return self.policy_net(features), lstm_states

    def forward_critic(
        self,
        features: th.Tensor,
        lstm_states: Optional[tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        features, lstm_states = self._forward_common(
            features, lstm_states, episode_starts
        )
        return self.value_net(features), lstm_states


class BetaDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Sequential(
            nn.Linear(latent_dim, self.action_dim * 2),
            nn.Softplus(),
        )
        return mean_actions

    def proba_distribution(self, mean_actions: th.Tensor) -> Self:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        c1, c2 = mean_actions.chunk(2, dim=-1)
        self.distribution = torch.distributions.Beta(c1, c2)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> Optional[th.Tensor]:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(
        self, mean_actions: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class TraderPolicy(RecurrentActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        recurrent: bool = False,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        # Small values to avoid NaN in Adam optimizer
        if optimizer_class in (th.optim.Adam, th.optim.AdamW):
            optimizer_kwargs["eps"] = 1e-5

        self.recurrent = recurrent

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=IdentityExtractor,
            share_features_extractor=True,
            normalize_images=False,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=128,
        )

        if (
            isinstance(net_arch, list)
            and len(net_arch) > 0
            and isinstance(net_arch[0], dict)
        ):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = True
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (squash_output and not use_sde), (
            "squash_output=True is only available when using gSDE (use_sde=True)"
        )
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist: Distribution = DiagGaussianDistribution(
            get_action_dim(action_space)
        )  # BetaDistribution(get_action_dim(action_space))

        self._build(lr_schedule)

    @property
    def lstm_actor(self):
        return self.mlp_extractor.latent_extractor

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.mlp_extractor = MlpExtractor(self.observation_space, self.recurrent).to(
            self.device
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
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
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        elif isinstance(self.action_dist, BetaDistribution):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )  # type: ignore[call-arg]

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: Optional[tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf, lstm_states = self.mlp_extractor(
            features, lstm_states[0], episode_starts
        )
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        if self.recurrent:
            return actions, values, log_prob, RNNStates(lstm_states, lstm_states)
        else:
            return actions, values, log_prob

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(
                obs,
                self.features_extractor
                if features_extractor is None
                else features_extractor,
            )
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            assert isinstance(pi_features, th.Tensor), (
                "Features extractor should return a tensor"
            )
            assert isinstance(vf_features, th.Tensor), (
                "Features extractor should return a tensor"
            )
            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        elif isinstance(self.action_dist, BetaDistribution):
            return self.action_dist.proba_distribution(mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        lstm_states: Optional[RNNStates] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
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
        latent_pi, latent_vf, lstm_states = self.mlp_extractor(
            features,
            lstm_states[0] if lstm_states is not None else None,
            episode_starts,
        )
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(
        self,
        obs: PyTorchObs,
        lstm_states: Optional[tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> Union[Distribution, tuple[Distribution, tuple[th.Tensor, th.Tensor]]]:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        assert isinstance(features, th.Tensor), (
            "Features extractor should return a tensor"
        )
        latent_pi, lstm_states = self.mlp_extractor.forward_actor(
            features,
            lstm_states,
            episode_starts,
        )

        if self.recurrent:
            return self._get_action_dist_from_latent(latent_pi), lstm_states
        else:
            return self._get_action_dist_from_latent(latent_pi)

    def predict_values(
        self,
        obs: PyTorchObs,
        lstm_states: Optional[tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        assert isinstance(features, th.Tensor), (
            "Features extractor should return a tensor"
        )
        latent_vf, lstm_states = self.mlp_extractor.forward_critic(
            features, lstm_states, episode_starts
        )
        return self.value_net(latent_vf)
