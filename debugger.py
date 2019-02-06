
from deep_logistics.environment import Environment
from deep_logistics.agent import InputAgent

if __name__ == "__main__":
    env = Environment(
        height=20,
        width=20,
        depth=3,
        agents=2,
        agent_class=InputAgent,
        renderer=None,
        tile_height=32,
        tile_width=32,
        #scheduler=RandomScheduler,
        ups=60,
        ticks_per_second=1,
        spawn_interval=1,  # In steps
        task_generate_interval=1,  # In steps
        task_assign_interval=1,  # In steps
        delivery_points=None
    )

    env.deploy_agents()
    env.task_assignment()


    while True:
            env.update()
            env.render()
