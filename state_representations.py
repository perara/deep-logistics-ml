import numpy as np


class BaseState:

    def __init__(self, env):
        self.env = env

    def generate(self, player):
        raise NotImplementedError("state.*generate()* must be implemented.")

    def get_shape(self):
        return self.generate(self.env.agent).shape

    def norm_cord_x(self, x):
        return x / self.env.grid.width

    def norm_cord_y(self, y):
        return y / self.env.grid.height


class State0(BaseState):
    """Generate state representation of the environment."""

    def generate(self, player):
        state_features = []

        """
        1. State looks like:
        []
        """
        if player.cell:
            state_features.append(self.norm_cord_x(player.cell.x) + 0.0001)
            state_features.append(self.norm_cord_y(player.cell.y) + 0.0001)
        else:
            state_features.append(0)
            state_features.append(0)

        """
        2. State looks like:
        [x, y]
        """

        if not player.task:
            state_features.append(0)
            state_features.append(0)
        elif not player.task.has_picked_up:
            state_features.append(self.norm_cord_y(player.task.x_0))
            state_features.append(self.norm_cord_y(player.task.y_0))
        else:
            state_features.append(self.norm_cord_x(player.task.x_1))
            state_features.append(self.norm_cord_y(player.task.y_1))

        """
        3. State looks like:
        [x, y, target_x, target_y]
        """
        if not player.task:
            """Hint is no movement (NOOP)"""
            state_features.append(0)
            state_features.append(0)
        else:
            task_coords = player.task.get_coordinates()
            d_x = player.cell.x - task_coords.x
            d_y = player.cell.y - task_coords.y
            d_x = 0 if d_x == 0 else d_x / abs(d_x)  # Normalize
            d_y = 0 if d_y == 0 else d_y / abs(d_y)  # Normalize

            if d_x == -1:
                d_x = 0
            elif d_x == 0:
                d_x = 0.5

            if d_y == -1:
                d_y = 0
            elif d_y == 0:
                d_y = 0.5

            state_features.append(d_x)
            state_features.append(d_y)

        """
        4. State looks like:
        [x, y, target_x, target_y, direction_hint_x, direction_hint_y]
        """

        if not player.cell:
            state_features.extend([-1, -1, -1, -1])
        else:
            d_l = self.norm_cord_x(player.cell.x)
            d_r = self.norm_cord_x(player.environment.grid.width - player.cell.x)
            d_u = self.norm_cord_y(player.cell.y)
            d_d = self.norm_cord_y(player.environment.grid.height - player.cell.y)
            state_features.extend([d_l, d_r, d_u, d_d])

        """
        5. State looks like: ^ 
        [x, y, target_x, target_y, direction_hint_x, direction_hint_y, distance_border_left, distance_border_right, distance_border_up, distance_border_down]
        """

        """
        y. State looks like: ^ 
        [x, y, target_x, target_y, direction_hint_x, direction_hint_y, distance_border_left, distance_border_right, distance_border_up, distance_border_down, state]
        """
        state_features.append(player.state / len(player.ALL_STATES))

        """
          7. Add speed
        """
        state_features.append(player.action_intensity)

        return np.array(state_features)


class State1(State0):
    """Generate state representation of the environment."""

    """
       8. Add proximity sensor (LEFT RIGHT, UP DOWN)
      """
    def generate(self, player):
        state_features = np.array(player.get_proximity_sensors())

        return np.concatenate((super().generate(player), state_features))



class State2(BaseState):
    """Generate state representation of the environment."""
    pass


class State3(BaseState):
    """Generate state representation of the environment."""
    pass
