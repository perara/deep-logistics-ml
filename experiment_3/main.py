import ray
import argparse
import os
import subprocess
import sys

from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

from deep_logistics_ml.experiment_3.env import DeepLogisticsMultiEnv1


def install_dependencies():
    with open("requirements.txt", "r") as f:
        for package in f.readlines():
            subprocess.call([sys.executable, "-m", "pip", "install", "--user", package])


def experiment_1():
    env = DeepLogisticsMultiEnv1(config=dict(
        graphics_render=True
    ))

    """Create distinct policy graphs for each agent."""
    policy_graphs = {
        k: (PPOPolicyGraph, env.observation_space, env.action_space, dict(
            gamma=0.95
        )) for k, a in env.agents.items()
    }

    policy_ids = list(policy_graphs.keys())

    trainer = ppo.APPOAgent(env="DeepLogisticsMultiEnv1",
                            config=dict(
                                multiagent=dict(
                                    policy_graphs=policy_graphs,
                                    policy_mapping_fn=tune.function(
                                        lambda agent_id: agent_id
                                    )
                                ),
                                callbacks=dict(
                                    on_episode_end=tune.function(DeepLogisticsMultiEnv1.on_episode_end)
                                )

                                # num_envs_per_worker=4,
                                # num_workers=2
                            ))

    while True:
        trainer.train()

        env.reset()
        terminal = False
        prev_action = {agent_id: None for agent_id in env.agents.keys()}
        prev_reward = {agent_id: None for agent_id in env.agents.keys()}
        prev_state = {agent_id: env.state_representation.generate(agent) for agent_id, agent in env.agents.items()}
        terminal_dict = {agent_id: False for agent_id in env.agents.keys()}
        while not terminal:

            for agent_id, agent in env.agents.items():
                if terminal_dict[agent_id]:
                    del prev_action[agent_id]
                    del prev_reward[agent_id]
                    del prev_state[agent_id]

                action = trainer.compute_action(prev_state[agent_id],
                                                prev_action=prev_action[agent_id],
                                                prev_reward=prev_reward[agent_id],
                                                policy_id=policy_ids[0]
                                                )
                prev_action[agent_id] = action

            prev_state, prev_reward, terminal_dict, info_dict = env.step(action_dict=prev_action)

            terminal = terminal_dict["__all__"]



if __name__ == "__main__":

    """Argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgx", type=bool, default=False, help="Run the experiment on DGX infrastructure.")
    parser.add_argument("--install", type=bool, default=False, help="Install dependencies automatically and exit")
    args = parser.parse_args()

    if args.dgx:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8,9,10,11,12"

    if args.install:
        install_dependencies()
        exit(0)

    ray.init()

    experiment_1()
