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


def describe_env(env: minigrid.minigrid_env.MiniGridEnv) -> str:
    obj_str = ""
    for obj in env.grid.grid:
        if obj is None or obj.type in ("empty", "unseen", "wall"):
            continue
        obj_str += f"{obj_to_str(obj)}, "

    text_obs = " a single room. " f"It contains: {obj_str}"
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
    dict_obs = {
        "position": (int(env.agent_pos[0]), int(env.agent_pos[1])),
        "direction": int(env.agent_dir),
        "carrying": carrying,
    }
    return dict_obs
