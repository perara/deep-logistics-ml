import sys
sys.path.append("/home/per/IdeaProjects/deep-logistics")
sys.path.append("/home/per/GIT/code/deep-logistics")
sys.path.append("/root/deep-logistics")

import argparse
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from envs import DeepLogisticsA1M20x20D4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_only", help="Only train the AI single process...", default=False, action="store_true")
    parser.add_argument("--train", help="Train the AI", default=False, action="store_true")
    parser.add_argument("--ppo", help="Train the AI", default=False, action="store_true")
    parser.add_argument("--random", help="Random Agent, default=False", action="store_true")
    parser.add_argument("--manhattan", help="Manhattan Agent", default=False, action="store_true")
    args = parser.parse_args()


    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    agent = ppo.PPOAgent(config=config, env=DeepLogisticsA1M20x20D4)
    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = agent.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)

