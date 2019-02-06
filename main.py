import sys
sys.path.append("/home/per/IdeaProjects/deep-logistics")
sys.path.append("/home/per/GIT/code/deep-logistics")
sys.path.append("/root/deep-logistics")

import argparse
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from envs import DeepLogisticsA10M20x20D4
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_only", help="Only train the AI single process...", default=False, action="store_true")
    parser.add_argument("--train", help="Train the AI", default=False, action="store_true")
    parser.add_argument("--ppo", help="Train the AI", default=False, action="store_true")
    parser.add_argument("--random", help="Random Agent, default=False", action="store_true")
    parser.add_argument("--manhattan", help="Manhattan Agent", default=False, action="store_true")
    args = parser.parse_args()

    def on_episode_end(info):
        env = info['env']
        episode = info["episode"]

        pickups = []
        deliveries = []
        for e in env.envs:

            pickups.extend(e.statistics.pickups_before_crash)
            deliveries.extend(e.statistics.deliveries_before_crash)

        episode.custom_metrics["average_pickups"] = np.mean(pickups)
        episode.custom_metrics["average_deliveries"] = np.mean(deliveries)

    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 6
    config["callbacks"] = {
        'on_episode_end': on_episode_end
    }
    config["train_batch_size"] = 100000
    print(config)
    agent = ppo.PPOAgent(config=config, env=DeepLogisticsA10M20x20D4)
    for i in range(1000):
        print("== Iteration", i, "==")
        result = agent.train()
        print(result)

        if i % 100 == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)

