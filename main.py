import sys

from ray import tune

import alg_config
import collections


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

sys.path.append("/home/per/IdeaProjects/deep-logistics")
sys.path.append("/home/per/GIT/code/deep-logistics")
sys.path.append("/root/deep_logistics")

import argparse
import ray
from ray.rllib.agents import ppo, a3c, impala

from ray.tune.logger import pretty_print
from envs import DeepLogisticsA10M20x20D4
import numpy as np
if __name__ == "__main__":
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_only", help="Only train the AI single process...", default=False, action="store_true")
    parser.add_argument("--train", help="Train the AI", default=False, action="store_true")
    parser.add_argument("--ppo", help="Train the AI", default=False, action="store_true")
    parser.add_argument("--random", help="Random Agent, default=False", action="store_true")
    parser.add_argument("--manhattan", help="Manhattan Agent", default=False, action="store_true")
    args = parser.parse_args()

    def on_episode_end(info):
        episode = info["episode"]
        env = info["env"].envs[0]

        episode.custom_metrics["deliveries"] = np.mean(env.statistics.deliveries_before_crash)
        episode.custom_metrics["pickups"] = np.mean(env.statistics.pickups_before_crash)

    config = ppo.DEFAULT_CONFIG.copy()
    dict_merge(config, alg_config.ppo["v1"])
    config["callbacks"]["on_episode_end"] = tune.function(on_episode_end)

    # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    tune.run_experiments({
        "experiment-1-ppo": {
            "run": "PPO",
            "env": DeepLogisticsA10M20x20D4,
            "stop": {"episode_reward_mean": 500},
            "config": config
        },
    })


