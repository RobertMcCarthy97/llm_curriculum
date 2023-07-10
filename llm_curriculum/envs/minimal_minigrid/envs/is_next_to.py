from __future__ import annotations

from typing import Tuple

from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key
from minigrid.minigrid_env import MiniGridEnv


def get_obj_pos(grid: Grid, type: str, color: str) -> Tuple[int, int]:
    for obj in grid.grid:
        if obj is None:
            continue
        if obj.type == type and obj.color == color:
            return obj.cur_pos
    raise ValueError(f"Object {type} {color} not found")


def is_next_to(coord_a, coord_b):
    dx = coord_a[0] - coord_b[0]
    dy = coord_a[1] - coord_b[1]
    return abs(dx) + abs(dy) <= 1


class IsNextToEnv(MiniGridEnv):

    """
    Same as PutNearEnv but fixed object types.
    Also, doesn't matter which object you pick up.
    """

    def __init__(self, size=6, max_steps: int | None = None, **kwargs):
        self.size = size
        mission_space = spaces.Text(max_length=256)

        if max_steps is None:
            max_steps = 5 * size

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return f"red ball is next to green key"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        objs_to_place = [("ball", "red"), ("key", "green")]
        objs = []
        objPos = []

        def near_obj(env, p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False

        # Until we have placed all the objects
        for obj in objs_to_place:
            objType, objColor = obj
            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            pos = self.place_obj(obj, reject_fn=near_obj)
            objs.append((objType, objColor))
            objPos.append(pos)

        # Randomize the agent start position and orientation
        self.place_agent()

        self.obj0_type = objs[0][0]
        self.obj0_color = objs[0][1]
        self.obj1_type = objs[1][0]
        self.obj1_color = objs[1][1]

        self.mission = self._gen_mission()

    def step(self, action):
        preCarrying = self.carrying
        obs, reward, terminated, truncated, info = super().step(action)

        if preCarrying is not None and self.carrying is None:
            green_key_pos = get_obj_pos(self.grid, "key", "green")
            red_ball_pos = get_obj_pos(self.grid, "ball", "red")
            if is_next_to(green_key_pos, red_ball_pos):
                reward = self._reward()
                terminated = True
        else:
            reward = 0
            terminated = False

        return obs, reward, terminated, truncated, info
