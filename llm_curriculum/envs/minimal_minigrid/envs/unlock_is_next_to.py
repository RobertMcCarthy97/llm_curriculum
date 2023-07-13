from __future__ import annotations

from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.roomgrid import RoomGrid, Grid
from typing import Tuple


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


class UnlockIsNextToEnv(RoomGrid):
    def __init__(self, room_size=6, max_steps: int | None = None, **kwargs):
        mission_space = spaces.Text(max_length=256)

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return f"blue ball is next to grey box"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box", color="grey")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True, color="green")
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)
        self.add_object(0, 0, "ball", "blue")

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"blue ball is next to grey box"

    def step(self, action):
        preCarrying = self.carrying
        obs, _, _, truncated, info = super().step(action)

        reward = 0
        terminated = False
        if preCarrying is not None and self.carrying is None:
            box_pos = get_obj_pos(self.grid, "box", "grey")
            ball_pos = get_obj_pos(self.grid, "ball", "blue")
            if is_next_to(box_pos, ball_pos):
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
