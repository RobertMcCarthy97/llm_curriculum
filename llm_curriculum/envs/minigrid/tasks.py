import abc
import llm_curriculum.envs.minigrid.grid_utils as grid_utils

from dataclasses import dataclass
from llm_curriculum.envs.minigrid.grid_utils import ObjectDescription


class BaseTask(abc.ABC):
    @abc.abstractmethod
    def check_success(self, state) -> bool:
        pass


class GoToObjectTask(BaseTask):
    """Task to go to an object"""

    def __init__(self, object_desc: ObjectDescription):
        self.object_desc = object_desc

    def check_success(self, env) -> bool:
        agent_desc = ObjectDescription("agent", "red")
        agent_pos = grid_utils.get_object_pos(env.grid, agent_desc)
        object_pos = grid_utils.get_object_pos(env.grid, self.object_desc)
        agent_y, agent_x = agent_pos
        object_y, object_x = object_pos
        return abs(agent_x - object_x) + abs(agent_y - object_y) <= 1

    def to_string(self):
        return f"Go to {self.object_desc.to_string()}"


class PickUpObjectTask(BaseTask):
    """Task to pick up an object"""

    def __init__(self, object_desc: ObjectDescription):
        self.object_desc = object_desc

    def check_success(self, env) -> bool:
        carrying = env.unwrapped.carrying
        if carrying is None:
            return False
        return self.object_desc.match(carrying)

    def to_string(self):
        return f"Pick up {self.object_desc.to_string()}"


class OpenDoorTask(BaseTask):
    """Task to open a door"""

    def __init__(self, object_desc: ObjectDescription):
        self.object_desc = object_desc
        assert self.object_desc.type == "door"

    def check_success(self, env) -> bool:
        pos_y, pos_x = grid_utils.get_object_pos(env.grid, self.object_desc)
        door = env.grid.grid[pos_y * env.grid.width + pos_x]
        return door.is_open

    def to_string(self):
        return f"Open {self.object_desc.to_string()}"
