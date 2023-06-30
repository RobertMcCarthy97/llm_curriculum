import abc

from dataclasses import dataclass


class BaseTask(abc.ABC):
    @abc.abstractmethod
    def check_success(self, state) -> bool:
        pass


@dataclass
class ObjectDescription:
    type: str
    color: str
    location: str

    def to_string(self):
        return f"{self.color} {self.type} at {self.location}"

    @staticmethod
    def from_string(string):
        color, type, _, location = string.split()
        return ObjectDescription(type, color, location)


class GoToObjectTask(BaseTask):
    """Task to go to an object"""

    def __init__(self, object_desc: ObjectDescription):
        self.object_desc = object_desc

    def check_success(self, env) -> bool:
        pass

    def to_string(self):
        return f"Go to {self.object_desc.to_string()}"


class PickUpObjectTask(BaseTask):
    """Task to pick up an object"""

    def __init__(self, object_desc: ObjectDescription):
        self.object_desc = object_desc

    def check_success(self, env) -> bool:
        pass

    def to_string(self):
        return f"Pick up {self.object_desc.to_string()}"


class PlaceObjectTask(BaseTask):
    """Task to place an object at a specific location"""

    def __init__(self, object_desc: ObjectDescription):
        self.object_desc = object_desc

    def check_success(self, env) -> bool:
        pass

    def to_string(self):
        return f"Place {self.object_desc.to_string()}"
