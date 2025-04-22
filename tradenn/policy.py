"""Policies: abstract base class and concrete implementations."""

import warnings
from functools import partial
from typing import Any, Optional, Self, Union

import gymnasium
import numpy as np
import torch
import torch as th
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
    sum_independent_dims,
)
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import log_, nn

from tradenn.config import NetworkConfig
from tradenn.network2 import TradeNetSimple


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
        network_config: Optional[NetworkConfig] = None,
    ):
        if network_config is None:
            network_config = NetworkConfig()
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        # Small values to avoid NaN in Adam optimizer
        if optimizer_class in (th.optim.Adam, th.optim.AdamW):
            optimizer_kwargs["eps"] = 1e-5

        self.recurrent = recurrent
        self.network_config = network_config

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
            squash_output=False,
            features_extractor_class=IdentityExtractor,
            share_features_extractor=True,
            normalize_images=False,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=128,
            enable_critic_lstm=False,
        )
        self._squash_output = True

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
        self.action_dist: Distribution = SquashedDiagGaussianDistribution(
            get_action_dim(action_space)
        )

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        assert isinstance(self.observation_space, gymnasium.spaces.Dict)
        self.mlp_extractor = TradeNetSimple(
            self.network_config,
            self.observation_space,
            self.action_space,
            self.recurrent,
        ).to(self.device)
        self.lstm_actor = self.mlp_extractor.latent_extractor

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(
            self.action_dist,
            (DiagGaussianDistribution, SquashedDiagGaussianDistribution),
        ):
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
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

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
        action_params, values, lstm_states = self.mlp_extractor(
            features,
            lstm_states[0] if lstm_states is not None else None,
            episode_starts,
        )
        # Evaluate the values for the given observations
        distribution = self._get_action_dist(action_params)
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
            # assert isinstance(pi_features, th.Tensor), (
            #     "Features extractor should return a tensor"
            # )
            # assert isinstance(vf_features, th.Tensor), (
            #     "Features extractor should return a tensor"
            # )
            return pi_features, vf_features

    def _get_action_dist(self, actions: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if isinstance(
            self.action_dist,
            (DiagGaussianDistribution, SquashedDiagGaussianDistribution),
        ):
            return self.action_dist.proba_distribution(actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=actions)
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
        action_params, values, lstm_states = self.mlp_extractor(
            features,
            lstm_states[0] if lstm_states is not None else None,
            episode_starts,
        )
        distribution = self._get_action_dist(action_params)
        log_prob = distribution.log_prob(actions)
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
        # assert isinstance(features, th.Tensor), (
        #     "Features extractor should return a tensor"
        # )
        latent_pi, lstm_states = self.mlp_extractor.forward_actor(
            features,
            lstm_states,
            episode_starts,
        )

        if self.recurrent:
            return self._get_action_dist(latent_pi), lstm_states
        else:
            return self._get_action_dist(latent_pi)

    def predict_values(
        self,
        obs: PyTorchObs,
        lstm_states: Optional[tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        # assert isinstance(features, th.Tensor), (
        #     "Features extractor should return a tensor"
        # )
        v, lstm_states = self.mlp_extractor.forward_critic(
            features, lstm_states, episode_starts
        )
        return v

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        if self.recurrent:
            if isinstance(obs_tensor, dict):
                n_envs = obs_tensor[next(iter(obs_tensor.keys()))].shape[0]
            else:
                n_envs = obs_tensor.shape[0]
            # state : (n_layers, n_envs, dim)
            if state is None:
                # Initialize hidden states to zeros
                state_ = np.concatenate(
                    [np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)],
                    axis=1,
                )
                state = (state_, state_)

            if episode_start is None:
                episode_start = np.array([False for _ in range(n_envs)])

            with th.no_grad():
                # Convert to PyTorch tensors
                states = (
                    th.tensor(state[0], dtype=th.float32, device=self.device),
                    th.tensor(state[1], dtype=th.float32, device=self.device),
                )
                episode_starts = th.tensor(
                    episode_start, dtype=th.float32, device=self.device
                )
                actions, states = self._predict(
                    obs_tensor,
                    lstm_states=states,
                    episode_starts=episode_starts,
                    deterministic=deterministic,
                )
                states = (states[0].cpu().numpy(), states[1].cpu().numpy())

            # Convert to numpy
            actions = actions.cpu().numpy()

        else:
            # Check for common mistake that the user does not mix Gym/VecEnv API
            # Tuple obs are not supported by SB3, so we can safely do that check
            if (
                isinstance(observation, tuple)
                and len(observation) == 2
                and isinstance(observation[1], dict)
            ):
                raise ValueError(
                    "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                    "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                    "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                    "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                    "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
                )

            with th.no_grad():
                dist = self.get_distribution(obs_tensor)
                assert isinstance(dist, Distribution)
                actions = dist.get_actions(deterministic=deterministic)

            # Convert to numpy, and reshape to the original action shape
            actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)

            else:
                if np.any(
                    np.logical_or(
                        actions < self.action_space.low,
                        actions > self.action_space.high,
                    )
                ):
                    warnings.warn(
                        "Actions are out of bounds. This is likely due to the action distribution not being properly set."
                    )
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state
