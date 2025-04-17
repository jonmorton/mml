import os
from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from tradenn.config import Config
from tradenn.trainer import evaluate


@dataclass
class FakeLogger:
    dir: str


class FakeAgent:
    def __init__(self, env: VecEnv, out_dir):
        self.env = env
        self.out_dir = out_dir

    @property
    def logger(self):
        return FakeLogger(os.path.join(self.out_dir, "tb"))

    def predict(self, obs, lstm_states=None, episode_starts=None, deterministic=False):
        pass

    def save(self, filename):
        with open(filename, "w") as f:
            f.write("")
        return self

    def load(self, filename, env=None, device=None):
        return self


class BuyAndHoldAgent(FakeAgent):
    def __init__(self, env, out_dir):
        super().__init__(env, out_dir)

    def predict(self, obs, *args, **kwargs):
        return np.ones(
            (obs["feats"].shape[0],) + self.env.action_space.shape,
            dtype=self.env.action_space.dtype,
        ), None


class RandomAgent(FakeAgent):
    def __init__(self, env, out_dir):
        super().__init__(env, out_dir)

    def predict(self, obs, *args, **kwargs):
        return np.stack(
            [self.env.action_space.sample() for _ in range(obs["feats"].shape[0])],
            axis=0,
        ), None


def eval_fake_agents(config: Config, env: VecEnv):
    agent = BuyAndHoldAgent(env, os.path.join(config.out_dir, "buy_and_hold"))
    evaluate(config, agent, env)
    agent = RandomAgent(env, os.path.join(config.out_dir, "random"))
    evaluate(config, agent, env)
