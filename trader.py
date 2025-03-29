import argparse
import os
import random

import haven
import numpy as np
import torch

from tradenn.config import Config
from tradenn.trainer import create_envs, evaluate, train, tune

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a stock trading agent.")
    argparser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to the configuration file.",
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

    with open(f"{config.out_dir}/config.yaml", "w") as f:
        f.write(haven.dump(config, "yaml"))

    train_env, eval_env = create_envs(config)
    # config = tune(config, train_env, eval_env)
    agent = train(config, train_env)
    evaluate(config, agent, eval_env)
