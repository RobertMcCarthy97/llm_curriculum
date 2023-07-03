from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ObjectDescription:
    type: str = ""
    color: str = ""

    def to_string(self):
        return f"{self.color} {self.type}"

    @staticmethod
    def from_string(string):
        color, type, _, location = string.split()
        return ObjectDescription(type, color, location)

    def match(self, object: WorldObj):
        return self.type == object.type and self.color == object.color


def get_object_pos(grid: Grid, object_desc: ObjectDescription) -> Tuple[int, int]:
    """Get position of object matching description.

    If there are multiple objects matching the description,
    return the first one.

    If there are no objects matching the description,
    return (-1, -1)
    """
    for idx, object in enumerate(grid.grid):
        if object is None:
            continue
        if object_desc.match(object):
            row = idx // grid.width
            col = idx % grid.width
            return (row, col)
    return (-1, -1)


def grid_to_str(grid: Grid) -> str:
    """Get string description of grid."""
    grid_str = ""
    for idx, object in enumerate(grid.grid):
        if object is None:
            continue
        else:
            y = idx // grid.width
            x = idx % grid.width
            cell_str = f"{object.color} {object.type} at {(x,y)}\n"
            if object.type == "door":
                if object.state == 2:
                    cell_str = "locked " + cell_str
                elif object.state == 1:
                    cell_str = "closed " + cell_str
                elif object.state == 0:
                    cell_str = "open " + cell_str
                else:
                    raise ValueError(f"Unknown door state {object.state}")
            grid_str += cell_str
    return grid_str
