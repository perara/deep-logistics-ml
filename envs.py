from deep_logistics.environment import Environment
from deep_logistics.agent import Agent, ManhattanAgent
from state_representations import State0
from gym.spaces import Tuple, Discrete

class Statistics:

    def __init__(self):
        self.episode_start = 0
        self.episode_end = 0
        self.episode_durations = []

        self.episode_pickup_count = 0
        self.episode_delivery_count = 0
        self.episode_cumulative_reward = 0
        self.rewards = []


class DeepLogisticBase:

    def __init__(self, height, width, ai_count, agent_count, agent, ups, delivery_points, state):
        self.env = Environment(
            height=height,
            width=width,
            depth=3,
            agents=0,
            agent_class=agent,
            renderer=None,
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
        self.state_representation = state(self.env)
        self.observation_space = self.state_representation.generate()
        self.action_space = Discrete(self.env.action_space.N_ACTIONS)

        self.statistics = Statistics()

        assert ai_count < agent_count

        self.agent_names = ['agent_%s' % i for i in range(ai_count)]
        [self.env.add_agent(Agent) for i in range(ai_count)]

        self.p_i = 0  # The player iterator
        self.p_f = ai_count #  When to stop iterating

        grouping = {'group_1': ["agent_%s" % x for x in range(ai_count)]}
        obs_space = Tuple([self.observation_space for _ in range(ai_count)])
        act_space = Tuple([self.action_space for _ in range(ai_count)])

        register_env(
            "grouped_twostep",
            lambda config: TwoStepGame(config).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space))

        """Spawn all agents etc.."""
        self.env.deploy_agents()
        self.env.task_assignment()

    def get_agents(self):
        return self.env.agents

    def player_evaluate(self, player):
        if player.state in [Agent.IDLE, Agent.MOVING]:
            reward = -0.01
            terminal = False
        elif player.state in [Agent.PICKUP]:
            reward = 1
            terminal = False
        elif player.state in [Agent.DELIVERY]:
            reward = 10
            terminal = False
        elif player.state in [Agent.DESTROYED]:
            reward = -1
            terminal = True
        elif player.state in [Agent.INACTIVE]:
            terminal = True
            reward = 0
        else:
            raise NotImplementedError("Should never happen. all states should be handled somehow")

        return reward, terminal

    def step(self, action_dict):
        info = {}

        player = self.env.agents[self.p_i]

        for action in enumerate(action_dict):
            pass


        """Perform the action."""
        self.player.do_action(action=action)

        """Update the environment"""
        self.env.update()



        self.render()
        return self.state_representation.generate(), reward, terminal, info

    def reset(self):
        return self.state_representation.generate()

    def render(self, mode='human', close=False):
        self.env.render()
        return self.env


class DeepLogisticsA10M20x20D4(DeepLogisticBase):

    def __init__(self, args):
        DeepLogisticBase.__init__(self,
                                  height=20,
                                  width=20,
                                  ai_count=1,
                                  agent_count=10,
                                  agent=ManhattanAgent,
                                  ups=None,
                                  delivery_points=[
                                      (4, 4),
                                      (4, 14),
                                      (14, 4),
                                      (14, 14)
                                  ],
                                  state=State0)

