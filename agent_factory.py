from tensorforce.agents import PPOAgent

from agents import RandomAgent


class AgentFactory:

    @staticmethod
    def create(t, env):
        try:
            return getattr(AgentFactory, t)(env)
        except Exception as e:
            raise NotImplementedError("The agent type %s does not exist!" % t)

    @staticmethod
    def random(env):
        return RandomAgent(env)

    @staticmethod
    def ppo(env):
        return PPOAgent(
            states=dict(type='float', shape=env.state_representation.get_shape()),
            actions=dict(type='int', num_actions=env.env.action_space.N_ACTIONS),
            #batching_capacity=10000,
            network=[
                dict(type='dense', size=128),
                dict(type='dense', size=128),
                dict(type='dense', size=128)
            ],
            step_optimizer=dict(type='adam', learning_rate=1e-4),

        )
"""
summarizer=dict(directory="./board",
                            labels=[
                                "bernoulli",
                                "beta",
                                "categorical",
                                "distributions",
                                "dropout",
                                "entropy",
                                "gaussian",
                                "graph",
                                "loss",
                                "losses",
                                "objective-loss",
                                "regularization-loss",
                                "relu",
                                "updates",
                                "variables",
                                "actions",
                                "states",
                                "rewards"
                            ]
                            ),
"""
