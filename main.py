import sys

import state_representations

sys.path.append("/home/per/IdeaProjects/deep-logistics/deep-logistics")
sys.path.append("/workspace/deep-logistics")

from environment import Environment
from agent import Agent
from tensorforce.agents import PPOAgent
import numpy as np


class AIAgent(Agent):

    def __init__(self, env):
        super().__init__(env)


class Env:

    def __init__(self, state_representation, fps=60, ups=None):
        self.env = Environment(
            height=12,
            width=10,
            depth=3,
            agents=1,
            agent_class=AIAgent,
            renderer=None,
            tile_height=32,
            tile_width=32,
            #scheduler=RandomScheduler,
            ups=ups,
            ticks_per_second=1,
            spawn_interval=1,  # In seconds
            task_generate_interval=1,  # In seconds
            task_assign_interval=1  # In seconds
        )

        self.state_representation = state_representation(self.env)

        # Assumes that all agnets have spawned already and that all tasks are assigned.
        self.env.deploy_agents()
        self.env.task_assignment()

        #env.daemon = True
        #env.start()

        self.player = self.env.agent
        print(self.player)

    def step(self, action):
        state = self.player.state
        self.player.do_action(action=action)
        self.env.update()
        new_state = self.player.state
        #print("%s => %s" % (state, new_state))

        """Fast-forward the game until the player is respawned."""
        while self.player.state == Agent.INACTIVE:
            self.env.update()

        state = self.state_representation.generate()

        if self.player.state in [Agent.IDLE, Agent.MOVING]:
            reward = -0.01
            terminal = False
        elif self.player.state in [Agent.PICKUP]:
            reward = 1
            terminal = False
            print("Pickup", state, self.player.task.c_1)
        elif self.player.state in [Agent.DELIVERY]:
            reward = 10
            terminal = True
            print("Delivery", state)
        elif self.player.state in [Agent.DESTROYED]:
            reward = -1
            terminal = True

        else:
            raise NotImplementedError("Should never happen. all states should be handled somehow")

        return state, reward, terminal, {}

    def reset(self):
        pass  # TODO Implement

    def render(self):
        self.env.render()
        return self.state_representation.generate()




if __name__ == "__main__":

    env = Env(state_representation=state_representations.State0)

    # Instantiate a Tensorforce agent
    """

    """
    agent = PPOAgent(
        states=dict(type='float', shape=env.state_representation.get_shape()),
        actions=dict(type='int', num_actions=env.env.action_space.N_ACTIONS),
        network=[
            dict(type='dense', size=64),
            dict(type='dense', size=64),
            dict(type='dense', size=64)
        ],
        step_optimizer=dict(type='adam', learning_rate=1e-4),
        summarizer=dict(directory="/home/per/IdeaProjects/deep-logistics_ml/board",
                        labels=[    "bernoulli", "beta", "categorical", "distributions", "dropout", "entropy", "gaussian", "graph",
                                    "loss", "losses", "objective-loss", "regularization-loss", "relu", "updates", "variables", "actions", "states", "rewards"]
                        ),
    )

    while True:
        # Retrieve the latest (observable) environment state
        state = env.render()  # (float array of shape [10])

        # Query the agent for its action decision
        action = agent.act(states=state)  # (scalar between 0 and 4)

        # Execute the decision and retrieve the current performance score
        state_1, reward, terminal, _ = env.step(action)  # (any scalar float)

        # Pass feedback about performance (and termination) to the agent
        agent.observe(reward=reward, terminal=terminal)
