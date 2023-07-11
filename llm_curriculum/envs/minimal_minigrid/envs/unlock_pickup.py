from __future__ import annotations

from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.roomgrid import RoomGrid


class UnlockPickupEnv(RoomGrid):
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
        return f"pick up the blue box"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box", color="blue")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True, color="yellow")
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
