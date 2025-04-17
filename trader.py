import argparse
import copy
import os
import random
import sys

import haven
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize

from tradenn.agents import eval_fake_agents
from tradenn.config import Config
from tradenn.trainer import build_envs, evaluate, train, tune

if __name__ == "__main__":
    torch.set_num_threads(4)

    argparser = argparse.ArgumentParser(description="Train a stock trading agent.")
    argparser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to the configuration file.",
    )
    argparser.add_argument(
        "--tune",
        type=int,
        default=0,
        help="How many tuning iterations to run. 0 means no tuning.",
    )
    argparser.add_argument("overrides", nargs="*", help="Config overrides.")
    args = argparser.parse_args()

    if args.config == "":
        config = haven.load(Config, dotlist_overrides=args.overrides)
    else:
        with open(args.config, "r") as f:
            config = haven.load(Config, stream=f, dotlist_overrides=args.overrides)

    run_name = "trader"

    config.seed = max(1, config.seed)
    random.seed(config.seed - 1)
    np.random.seed(config.seed - 1)
    torch.manual_seed(config.seed - 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed - 1)

    os.makedirs(config.out_dir, exist_ok=True)

    train_env, eval_env = build_envs(config)

    if not os.path.exists("models/fake/buy_and_hold") or not os.path.exists(
        "models/fake/random"
    ):
        print("Eval fake agents...")
        config2 = copy.deepcopy(config)
        config2.run_name = "fake"
        eval_fake_agents(config2, eval_env)

    if args.tune:
        config = tune(config, train_env, eval_env, n_trials=args.tune)
        train_env, eval_env = build_envs(config)

    with open(f"{config.out_dir}/config.yaml", "w") as f:
        f.write(haven.dump(config, "yaml"))

    agent = train(config, train_env)

    evaluate(config, agent, eval_env)

    print(config)
