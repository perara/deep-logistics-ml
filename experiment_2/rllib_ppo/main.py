from ray.rllib import MultiAgentEnv
import os

from ray.rllib.agents import ppo
from ray.tune import register_env

from deep_logistics import DeepLogistics
from deep_logistics import SpawnStrategies
from gym.spaces import Tuple, Discrete

from deep_logistics_ml.experiment_2.rllib_ppo.reward_functions import Reward0
from deep_logistics_ml.experiment_2.rllib_ppo.state_representations import State0


class DeepLogisticsMultiEnv(MultiAgentEnv):

    def __init__(self, state, reward, width, height, depth, taxi_n, group_type="individual",graphics_render=False, delivery_locations=None):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = DeepLogistics(width=width,
                                 height=height,
                                 depth=depth,
                                 taxi_n=taxi_n,
                                 ups=None,
                                 graphics_render=graphics_render,
                                 delivery_locations=delivery_locations,
                                 spawn_strategy=SpawnStrategies.RandomSpawnStrategy
                                 )
        self.state_representation = state(self.env)
        self.reward_function = reward
        self.observation_space = self.state_representation.generate(self.env.agents[0])
        self.action_space = Discrete(self.env.action_space.N_ACTIONS)

        self.agents = {"agent_%s" % i: self.env.agents[i] for i in range(taxi_n)}

        """Set up grouping for the environments."""
        if group_type == "individual":
            self.grouping = {
                "group_%s" % x: ["agent_%s" % x] for x in range(taxi_n)
            }
        elif group_type == "grouped":
            self.grouping = {
                'group_1': ["agent_%s" % x for x in range(taxi_n)]
            }
        else:
            raise NotImplementedError("The group type %s is not implemented." % group_type)

        self.with_agent_groups(
                groups=self.grouping,
                obs_space=Tuple([self.observation_space for _ in range(taxi_n)]),
                act_space=Tuple([self.action_space for _ in range(taxi_n)])
            )

    def step(self, action_dict):
        info_dict = {}
        reward_dict = {}
        terminal_dict = {"__all__": False}
        state_dict = {}

        """Perform actions in environment."""
        for agent_name, action in action_dict.items():
            self.agents[agent_name].do_action(action=action)

        """Update the environment"""
        self.env.update()

        """Evaluate score"""
        t__all__ = True
        for agent_name, agent in action_dict.items():
            reward, terminal = self.reward_function(self.agents[agent_name])

            reward_dict[agent_name] = reward
            terminal_dict[agent_name] = terminal

            if not terminal:
                t__all__ = terminal

            state_dict[agent_name] = self.state_representation.generate(self.agents[agent_name])

        """Update terminal dict"""
        terminal_dict["__all__"] = t__all__

        return state_dict, reward_dict, terminal_dict, info_dict

    def reset(self):
        self.env.reset()

        return {
            agent_name: self.state_representation.generate(agent) for agent_name, agent in self.agents.items()
        }


if __name__ == "__main__":

    register_env("DeepLogisticsMultiEnv", lambda config: DeepLogisticsMultiEnv(**config))

    trainer = ppo.PPOAgent(env="DeepLogisticsMultiEnv")
