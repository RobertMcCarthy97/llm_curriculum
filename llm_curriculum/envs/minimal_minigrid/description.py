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
