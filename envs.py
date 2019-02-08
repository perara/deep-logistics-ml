from collections import deque
import sys
sys.path.append("/home/per/GIT/deep-logistics")
sys.path.append("/home/per/IdeaProjects/deep-logistics")
sys.path.append("/home/per/GIT/code/deep-logistics")
sys.path.append("/root/deep-logistics")

from deep_logistics.environment import Environment
from deep_logistics.agent import Agent, ManhattanAgent
from ray.rllib import MultiAgentEnv

from state_representations import State0
from gym.spaces import Tuple, Discrete


class Statistics:

    def __init__(self):
        self.deliveries_before_crash = deque(maxlen=100)
        self.pickups_before_crash = deque(maxlen=100)

class DeepLogisticBase(MultiAgentEnv):

    def __init__(self, height, width, ai_count, agent_count, agent, ups, delivery_points, state, render_screen=False):
        self.render_screen = render_screen
        self.env = Environment(
            height=height,
            width=width,
            depth=3,
            agents=0,
            agent_class=agent,
            draw_screen=self.render_screen,
            tile_height=32,
            tile_width=32,
            #scheduler=RandomScheduler,
            ups=ups,
            ticks_per_second=1,
            spawn_interval=1,  # In steps
            task_generate_interval=1,  # In steps
            task_assign_interval=1,  # In steps
            delivery_points=delivery_points
        )

        self.statistics = Statistics()

        assert ai_count < agent_count

        for i in range(ai_count):
            self.env.add_agent(Agent)

        self.state_representation = state(self.env)
        self.observation_space = self.state_representation.generate(self.env.agents[0])
        self.action_space = Discrete(self.env.action_space.N_ACTIONS)

        self.grouping = {'group_1': ["agent_%s" % x for x in range(ai_count)]}
        self.agents = {k: self.env.agents[i] for i, k in enumerate(self.grouping["group_1"])}
        obs_space = Tuple([self.observation_space for _ in range(ai_count)])
        act_space = Tuple([self.action_space for _ in range(ai_count)])

        self.with_agent_groups(
            groups=self.grouping,
            obs_space=obs_space,
            act_space=act_space
        )

        print(dir(self))



        """Spawn all agents etc.."""
        self.env.deploy_agents()
        self.env.task_assignment()


        self.episode = 0

    def get_agents(self):
        return self.env.agents

    def player_evaluate(self, player):
        if player.state in [Agent.IDLE, Agent.MOVING]:
            reward = -0.001
            terminal = False
        elif player.state in [Agent.PICKUP]:
            reward = 0.2
            terminal = False
        elif player.state in [Agent.DELIVERY]:
            reward = 0.8
            terminal = False
        elif player.state in [Agent.DESTROYED]:
            self.statistics.pickups_before_crash.append(player.total_pickups)
            self.statistics.deliveries_before_crash.append(player.total_deliveries)
            #print("Episode: %s | Pickups: %s, Deliveries: %s" % (self.episode, player.total_pickups, player.total_deliveries))
            self.episode += 1
            reward = -1
            terminal = True
        elif player.state in [Agent.INACTIVE]:
            terminal = True
            reward = 0
        else:
            raise NotImplementedError("Should never happen. all states should be handled somehow")

        return reward, terminal

    def step(self, action_dict):
        info_dict = {}
        reward_dict = {}
        terminal_dict = {
            "__all__": False
        }
        state_dict = {}

        for agent_name, action in action_dict.items():
            self.agents[agent_name].do_action(action=action)

        """Update the environment"""
        self.env.update()

        """Evaluate score"""
        t__all__ = True
        for agent_name, agent in self.agents.items():
            reward, terminal = self.player_evaluate(self.agents[agent_name])

            reward_dict[agent_name] = reward
            terminal_dict[agent_name] = terminal
            if not terminal:
                t__all__ = terminal

            info_dict[agent_name] = {}
            state_dict[agent_name] = self.state_representation.generate(agent)

        """Update terminal dict"""
        terminal_dict["__all__"] = t__all__

        if self.render_screen:
            self.render()

        return state_dict, reward_dict, terminal_dict, info_dict

    def reset(self):
        return {
            agent_name: self.state_representation.generate(agent) for agent_name, agent in self.agents.items()
        }

    def render(self, mode='human', close=False):
        if self.render_screen:
            self.env.render()
        return self.env


class DeepLogisticsA10M20x20D4(DeepLogisticBase):

    def __init__(self, args):
        DeepLogisticBase.__init__(self,
                                  height=10,
                                  width=10,
                                  ai_count=1,
                                  agent_count=15,
                                  agent=ManhattanAgent,
                                  ups=None,
                                  delivery_points=[
                                      (3, 7),
                                      (3, 7),
                                      (7, 3),
                                      (7, 7)
                                  ],
                                  state=State0)

