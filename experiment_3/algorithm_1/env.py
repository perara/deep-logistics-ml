from gym.spaces import Tuple, Discrete, Box

from deep_logistics import DeepLogistics
from deep_logistics import SpawnStrategies
from deep_logistics_ml.experiment_3.algorithm_1.reward_functions import Reward0
import numpy as np
from deep_logistics_ml.experiment_3.algorithm_1.state_representations import State0



class DLTestEnv:

    def __init__(self, state, reward, width, height, depth, taxi_n, group_type="individual", graphics_render=False, delivery_locations=None):
        self.env = DeepLogistics(width=width,
                                 height=height,
                                 depth=depth,
                                 taxi_n=taxi_n,
                                 ups=None,
                                 graphics_render=True,
                                 delivery_locations=delivery_locations,
                                 spawn_strategy=SpawnStrategies.RandomSpawnStrategy
                                 )
        self.state_representation = state(self.env)
        self.reward_function = reward
        self._render = graphics_render

        self.observation_space = Box(low=-1, high=1, shape=self.state_representation.generate(self.env.agents[0]).shape, dtype=np.float32)
        self.action_space = Discrete(self.env.action_space.N_ACTIONS)

        self.agents = {"agent_%s" % i: self.env.agents[i] for i in range(taxi_n)}

        self.total_steps = 0

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

        #self.with_agent_groups(
        #    groups=self.grouping,
        #    obs_space=Tuple([self.observation_space for _ in range(taxi_n)]),
        #    act_space=Tuple([self.action_space for _ in range(taxi_n)])
        #)

    def step(self, action_dict):
        self.total_steps += 1

        # TODO this loop does not make sense when using multiple policies. Now we do 1 action for all taxis with a single policy (i think) instead of 1 action per policy
        # Cluster: https://ray.readthedocs.io/en/latest/install-on-docker.html#launch-ray-in-docker
        info_dict = {}
        reward_dict = {}
        terminal_dict = {"__all__": False}
        state_dict = {}

        """Perform actions in environment."""
        for agent_name, action in action_dict.items():
            self.agents[agent_name].do_action(action=action)

        """Update the environment"""
        self.env.update()
        if self._render:
            self.env.render()

        """Evaluate score"""
        t__all__ = False
        for agent_name, agent in action_dict.items():
            reward, terminal = self.reward_function(self.agents[agent_name])

            reward_dict[agent_name] = reward
            terminal_dict[agent_name] = terminal

            if terminal:
                t__all__ = terminal

            state_dict[agent_name] = self.state_representation.generate(self.agents[agent_name])

        """Update terminal dict"""
        terminal_dict["__all__"] = t__all__

        return state_dict, reward_dict, terminal_dict, info_dict

    def reset(self):
        self.env.reset()

        self.total_steps = 0

        return {
            agent_name: self.state_representation.generate(agent) for agent_name, agent in self.agents.items()
        }
