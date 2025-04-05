import argparse
import os
import random

import haven
import numpy as np
import torch

from tradenn.config import Config
from tradenn.trainer import build_envs, evaluate, train, tune

if __name__ == "__main__":
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

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    os.makedirs(config.out_dir, exist_ok=True)

    train_env, eval_env = build_envs(config)
    if args.tune:
        config = tune(config, train_env, eval_env, n_trials=args.tune)
        train_env, eval_env = build_envs(config)

    with open(f"{config.out_dir}/config.yaml", "w") as f:
        f.write(haven.dump(config, "yaml"))

    agent = train(config, train_env)
    evaluate(config, agent, eval_env)

    print(config)
