""" Functions to describe the environment """


import numpy as np
from typing import Tuple, List, Dict, Any
from copy import deepcopy
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

import minigrid
from minigrid.core.roomgrid import Room, RoomGrid
from minigrid.core.world_object import WorldObj
from minigrid.core.constants import (
    STATE_TO_IDX,
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
)

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


def obj_to_str(obj: WorldObj) -> str:
    """Convert a WorldObj to a string"""
    return f"{obj.color}_{obj.type}"


def room_id_from_pos(env: RoomGrid, x: int, y: int) -> Room:
    """Get the room a given position maps to"""

    assert x >= 0
    assert y >= 0

    i = x // (env.room_size - 1)
    j = y // (env.room_size - 1)

    assert i < env.num_cols
    assert j < env.num_rows

    return (j, i)


def get_num_rooms(env: RoomGrid) -> int:
    """Get the number of rooms in the environment"""
    return env.num_rows * env.num_cols


def describe_env(env: minigrid.minigrid_env.MiniGridEnv) -> str:
    assert isinstance(env, minigrid.minigrid_env.MiniGridEnv)
    if isinstance(env, RoomGrid):
        return describe_room_grid(env)
    else:
        return describe_single_room(env)


def describe_single_room(env: minigrid.minigrid_env.MiniGridEnv) -> str:
    obj_str = ""
    for obj in env.grid.grid:
        if obj is None or obj.type in ("empty", "unseen", "wall"):
            continue
        obj_str += f"{obj_to_str(obj)}, "

    text_obs = " a single room. " f"It contains: {obj_str}"
    return text_obs


def describe_room_grid(env: RoomGrid) -> str:
    """Describe a RoomGrid environment"""
    text_obs = f"{get_num_rooms(env)} rooms in a {env.num_rows}x{env.num_cols} grid.\n"
    for i in range(env.num_cols):
        for j in range(env.num_rows):
            room = env.get_room(i, j)
            room_str = f"The room ({j}, {i}) contains: \n"

            # Describe objects in room
            obj_str = ""
            for obj in room.objs:
                if obj is None or obj.type in ("empty", "unseen", "wall"):
                    continue
                obj_str += f"{obj_to_str(obj)}, "
            room_str += f"objects: {obj_str}\n"

            # Describe doors in room
            door_str = ""
            for i, door in enumerate(room.doors):
                if door is None:
                    continue
                direction = ("east", "south", "west", "north")[i]
                door_state = (
                    "open" if door.is_open else "locked" if door.is_locked else "closed"
                )
                door_str += f"{door_state} {door.color} door leading {direction}, "
            room_str += f"doors: {door_str}\n"
            text_obs += room_str

    # Describe agent state
    agent_str = f"The agent is in room {room_id_from_pos(env, *env.agent_pos)}.\n"
    text_obs += agent_str

    return text_obs


def parse_field_of_view(img_obs: np.ndarray) -> Dict[str, Any]:
    """Describe the field of view of the agent"""

    dict_obs = {}
    for y in range(img_obs.shape[1]):
        for x in range(img_obs.shape[0]):
            obj_idx, color_idx, state_idx = img_obs[x, y]
            obj = IDX_TO_OBJECT[obj_idx]
            color = IDX_TO_COLOR[color_idx]

            if obj in ("empty", "unseen", "wall", "agent"):
                continue

            obj_dict = {"position": (int(x), int(y))}

            if obj == "door":
                state = IDX_TO_STATE[state_idx]
                obj_dict["state"] = state

            dict_obs[f"{color}_{obj}"] = obj_dict

    return dict_obs


def parse_agent(env) -> Dict[str, Any]:
    """Describe the agent"""
    carrying = env.carrying
    if carrying:
        carrying = obj_to_str(carrying)
    else:
        carrying = "nothing"

    assert isinstance(env, minigrid.minigrid_env.MiniGridEnv)
    if isinstance(env, RoomGrid):
        room = room_id_from_pos(env, *env.agent_pos)
    else:
        # Default to assuming single room
        room = (0, 0)

    dict_obs = {
        "position": (int(env.agent_pos[0]), int(env.agent_pos[1])),
        "direction": int(env.agent_dir),
        "carrying": carrying,
        "room": room,
    }
    return dict_obs
