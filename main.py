import sys
sys.path.append("/home/per/IdeaProjects/deep_logistics")
sys.path.append("/home/per/GIT/code/deep_logistics")
sys.path.append("/root")
from ray import tune

import alg_config
import collections
import os


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]



import argparse
import ray
from ray.rllib.agents import ppo, a3c, impala

from ray.tune.logger import pretty_print
from envs import DeepLogisticsA10M20x20D4
import numpy as np
if __name__ == "__main__":
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgx", help="Use DGX cpu and gpu", default=False, action="store_true")
    args = parser.parse_args()

    def on_episode_end(info):
        episode = info["episode"]
        env = info["env"].envs[0]

        episode.custom_metrics["deliveries"] = np.mean(env.statistics.deliveries_before_crash)
        episode.custom_metrics["pickups"] = np.mean(env.statistics.pickups_before_crash)

    config = ppo.DEFAULT_CONFIG.copy()
    dict_merge(config, alg_config.ppo["v1"])
    config["callbacks"]["on_episode_end"] = tune.function(on_episode_end)

    if args.dgx:
        config["num_workers"] = os.cpu_count() - 4
        config["num_gpus"] = 16

    # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    tune.run_experiments({
        "experiment-1-ppo": {
            "run": "PPO",
            "env": DeepLogisticsA10M20x20D4,
            "stop": {"episode_reward_mean": 500},
            "config": config
        },
    })


