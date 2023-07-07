""" Functions to describe the environment """


import numpy as np
from typing import Tuple, List, Dict, Any
from copy import deepcopy
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

import minigrid
from minigrid.core.roomgrid import Room, RoomGrid
from minigrid.core.constants import (
    STATE_TO_IDX,
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
)

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


def parse_field_of_view(img_obs: np.ndarray) -> Dict[str, Any]:
    """Describe the field of view of the agent"""

    dict_obs = {}
    for y in range(img_obs.shape[0]):
        for x in range(img_obs.shape[1]):
            obj_idx, color_idx, state_idx = img_obs[y, x]
            obj = IDX_TO_OBJECT[obj_idx]
            color = IDX_TO_COLOR[color_idx]

            if obj in ("empty", "unseen", "wall"):
                continue

            obj_dict = {"position": (x, y)}

            if obj == "door":
                state = IDX_TO_STATE[state_idx]
                obj_dict["state"] = state

            dict_obs[f"{color}_{obj}"] = obj_dict

    return dict_obs


def parse_agent(env) -> Dict[str, Any]:
    """Describe the agent"""
    dict_obs = {
        "position": env.agent_pos,
        "direction": env.agent_dir,
        "carrying": env.carrying,
    }
    return dict_obs


def describe_agent(env):
    text_obs = ""

    # Describe agent
    directions = ("east", "south", "west", "north")
    text_obs += f"The agent is facing {directions[env.agent_dir]}.\n"

    # Describe objects adjacent to agent
    directions = ("east", "south", "west", "north")
    delta_pos = ((1, 0), (0, 1), (-1, 0), (0, -1))
    text_obs += "The agent is next to objects: "
    for dir, dpos in zip(directions, delta_pos):
        obj_pos = (env.agent_pos[0] + dpos[0], env.agent_pos[1] + dpos[1])
        obj = env.grid.get(*obj_pos)
        if obj is not None and not obj.type in ("unseen", "empty"):
            text_obs += f"{obj.color} {obj.type} to the {dir}, "
    text_obs += "\n"

    # TODO: describe agent field of view

    # Describe inventory
    if env.carrying:
        text_obs += (
            f"The agent is carrying: {env.carrying.color} {env.carrying.type}.\n"
        )

    return text_obs


def describe_roomgrid(env: RoomGrid) -> str:
    """Describe all rooms in a RoomGrid env"""
    text_obs = ""
    print("The rooms are labelled as (col, row).")
    for row in range(env.num_rows):
        for col in range(env.num_cols):
            room = env.get_room(col, row)
            text_obs += "Room ({}, {}):\n".format(col, row)
            text_obs += describe_room(room)
    return text_obs


def room_idx_from_pos(env: RoomGrid, x: int, y: int) -> Tuple[int, int]:
    """Get the room a given position maps to"""

    assert x >= 0
    assert y >= 0

    i = x // (env.room_size - 1)
    j = y // (env.room_size - 1)

    assert i < env.num_cols
    assert j < env.num_rows

    return (i, j)


class RoomGridTextPartialObsWrapper(ObservationWrapper):
    @property
    def spec(self):
        return self.env.spec

    def __init__(self, env, fully_obs=False):
        assert isinstance(
            env.unwrapped, minigrid.roomgrid.RoomGrid
        ), "This wrapper only works with RoomGrid environments"
        super().__init__(env)

        self.fully_obs = fully_obs
        self.observation_space = deepcopy(self.env.observation_space)
        self.observation_space.spaces["text"] = spaces.Text(max_length=4096)

    def observation(self, observation) -> dict:
        """Describes objects at a higher level than the Minigrid wrapper"""

        env = self.unwrapped
        text_obs = ""

        if self.fully_obs:
            room_idx = room_idx_from_pos(env, *env.agent_pos)
            text_obs += f"The agent is in room ({room_idx[0]}, {room_idx[1]}).\n"
            text_obs += describe_roomgrid(env)
        else:
            room = env.room_from_pos(*env.agent_pos)
            text_obs += "The agent is in a room."
            text_obs += describe_room(room)

        text_obs += describe_agent(env)

        observation["text"] = text_obs
        return observation


class RoomGridTextFullyObsWrapper(RoomGridTextPartialObsWrapper):
    def __init__(self, env):
        super().__init__(env, fully_obs=True)
