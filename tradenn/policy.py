"""Policies: abstract base class and concrete implementations."""

import collections
import warnings
from typing import Any, Optional, Union

import gymnasium
import numpy as np
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
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import (
    get_action_dim,
    get_flattened_obs_dim,
    preprocess_obs,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from tradenn.config import NetworkConfig
from tradenn.network2 import TradeNetSimple


class IdentityExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations


class TraderPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
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

        self.log_std_init = log_std_init
        self.recurrent = recurrent
        self.network_config = network_config

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=IdentityExtractor,
            features_extractor_kwargs={},
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=False,
        )
        self._squash_output = True

        dist_kwargs = None

        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": True,
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

    def _get_constructor_parameters(self) -> dict[str, Any]:
        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self._dummy_schedule,
            use_sde=self.use_sde,
            log_std_init=self.log_std_init,
            full_std=default_none_kwargs["full_std"],
            use_expln=default_none_kwargs["use_expln"],
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            recurrent=self.recurrent,
            network_config=self.network_config,
        )

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

        if isinstance(self.action_dist, DiagGaussianDistribution):
            _, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=128, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            _, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=128,
                latent_sde_dim=128,
                log_std_init=self.log_std_init,
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )  # type: ignore[call-arg]

    def _prepare_obs(self, obs: PyTorchObs) -> PyTorchObs:
        return preprocess_obs(obs, self.observation_space, False)

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        if self.recurrent:
            distribution, lstm_states = self.get_distribution(
                observation, lstm_states, episode_starts
            )
            return distribution.get_actions(deterministic=deterministic), lstm_states
        else:
            return self.get_distribution(observation).get_actions(
                deterministic=deterministic
            )

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
        obs = self._prepare_obs(obs)
        action_params, values, lstm_states = self.mlp_extractor(
            obs,
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

    def _get_action_dist(self, actions: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(actions, self.log_std)
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        lstm_states: Optional[RNNStates] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        # Preprocess the observation if needed
        obs = self._prepare_obs(obs)
        action_params, values, lstm_states = self.mlp_extractor(
            obs,
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
        obs = self._prepare_obs(obs)
        latent_pi, lstm_states = self.mlp_extractor.forward_actor(
            obs,
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
        # assert isinstance(features, th.Tensor), (
        #     "Features extractor should return a tensor"
        # )
        obs = self._prepare_obs(obs)
        v, lstm_states = self.mlp_extractor.forward_critic(
            obs, lstm_states, episode_starts
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
