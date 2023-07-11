from llm_curriculum.envs.tasks.core_tasks import (
    MoveCubeToTargetTask,
    PickUpCubeTask,
    MoveGripperToCubeTask,
    GraspCubeTask,
    CubeBetweenGripperTask,
    CloseGripperCubeTask,
    LiftCubeTask,
    PlaceCubeAtTargetTask,
    MoveCubeTowardsTargetGraspTask,
)

from llm_curriculum.envs.tasks.drawer_tasks import (
    OpenDrawerTask,
    MoveGripperToDrawerTask,
    GraspHandleTask,
    PullHandleToOpenTask,
    CloseDrawerTask,
    PushHandleToCloseTask,
    PlaceCubeDrawerTask,
    PlaceGraspedCubeDrawerTask,
    MoveCubeOverDrawerTask,
    ReleaseCubeInDrawerTask,
    PlaceCubeOnDrawerTopTask,
    PlaceGraspedCubeOnDrawerTopTask,
    MoveCubeOverDrawerTopTask,
    ReleaseCubeOnDrawerTopTask,
    MoveGripperToCubeInDrawerTask,
)

"""
TODO:
- Add assertions about intial state of envirtonmnet for each task? (e.g. some tasks assume drawer is open, cube is in drawer, etc.)

"""

######################
# Define high-level task trees
######################

## Cube only ##

pick_up_cube_mini_tree = {
    PickUpCubeTask: {
        MoveGripperToCubeTask: None,
        CubeBetweenGripperTask: None,
        CloseGripperCubeTask: None,
        LiftCubeTask: None,
    }
}

pick_up_cube_tree = {
    PickUpCubeTask: {
        MoveGripperToCubeTask: None,
        GraspCubeTask: {
            CubeBetweenGripperTask: None,
            CloseGripperCubeTask: None,
        },
        LiftCubeTask: None,
    },
}

move_cube_to_target_tree = {
    MoveCubeToTargetTask: {
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceCubeAtTargetTask: {
            MoveCubeTowardsTargetGraspTask: None,
        },
    }
}

## Drawer only ##

open_drawer_tree = {
    OpenDrawerTask: {
        MoveGripperToDrawerTask: None,
        GraspHandleTask: None,
        PullHandleToOpenTask: None,
    }
}

close_drawer_tree = {
    CloseDrawerTask: {
        MoveGripperToDrawerTask: None,
        GraspHandleTask: None,
        PushHandleToCloseTask: None,
    }
}

## Cube and drawer ##

place_cube_open_drawer_tree = {
    PlaceCubeDrawerTask: {
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceGraspedCubeDrawerTask: {
            MoveCubeOverDrawerTask: None,
            ReleaseCubeInDrawerTask: None,
        },
    }
}

place_cube_drawer_top_tree = {
    PlaceCubeOnDrawerTopTask: {
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceGraspedCubeOnDrawerTopTask: {
            MoveCubeOverDrawerTopTask: None,
            ReleaseCubeOnDrawerTopTask: None,
        },
    }
}

open_then_place_in_drawer_tree = {
    PlaceCubeDrawerTask: {
        OpenDrawerTask: {
            MoveGripperToDrawerTask: None,
            GraspHandleTask: None,
            PullHandleToOpenTask: None,
        },
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceGraspedCubeDrawerTask: {
            MoveCubeOverDrawerTask: None,
            ReleaseCubeInDrawerTask: None,
        },
    }
}

open_then_place_drawer_then_close_tree = {
    PlaceCubeDrawerTask: {
        OpenDrawerTask: {
            MoveGripperToDrawerTask: None,
            GraspHandleTask: None,
            PullHandleToOpenTask: None,
        },
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceGraspedCubeDrawerTask: {
            MoveCubeOverDrawerTask: None,
            ReleaseCubeInDrawerTask: None,
        },
        CloseDrawerTask: {
            MoveGripperToDrawerTask: None,
            GraspHandleTask: None,
            PushHandleToCloseTask: None,
        },
    }
}

pick_up_cube_in_drawer_tree = {
    PickUpCubeTask: {
        MoveGripperToCubeInDrawerTask: None,
        GraspCubeTask: {
            CubeBetweenGripperTask: None,
            CloseGripperCubeTask: None,
        },
        LiftCubeTask: None,
    },
}

"""
TODO: create these trees

self.str_description = "grasp cube mini"
self.subtask_cls_seq = [MoveGripperToCubeTask, CubeBetweenGripperTask]

self.str_description = "pick and place mini"
self.subtask_cls_seq = [MoveGripperToCubeTask, CubeBetweenGripperTask, CloseGripperCubeTask, LiftCubeTask, MoveCubeTowardsTargetGraspTask]
"""

##############################
# 0-shot pretrianing trees
##############################

### V1
cube_on_table_to_cube_at_target_tree = {
    MoveCubeToTargetTask: {
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceCubeAtTargetTask: {
            MoveCubeTowardsTargetGraspTask: None,
        },
    }
}

cube_on_drawer_to_cube_in_drawer_tree = {
    PlaceCubeDrawerTask: {
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceGraspedCubeDrawerTask: {
            MoveCubeOverDrawerTask: None,
            ReleaseCubeInDrawerTask: None,
        },
    }
}

### V2
cube_on_table_to_cube_in_drawer_tree = {
    PlaceCubeDrawerTask: {
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceGraspedCubeDrawerTask: {
            MoveCubeOverDrawerTask: None,
            ReleaseCubeInDrawerTask: None,
        },
    }
}

cube_on_drawer_to_cube_at_target_tree = {
    MoveCubeToTargetTask: {
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceCubeAtTargetTask: {
            MoveCubeTowardsTargetGraspTask: None,
        },
    }
}

### V3

cube_on_table_to_cube_on_drawer_tree = {
    PlaceCubeOnDrawerTopTask: {
        PickUpCubeTask: {
            MoveGripperToCubeTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceGraspedCubeOnDrawerTopTask: {
            MoveCubeOverDrawerTopTask: None,
            ReleaseCubeOnDrawerTopTask: None,
        },
    }
}

cube_in_drawer_to_cube_at_target_tree = {
    MoveCubeToTargetTask: {
        PickUpCubeTask: {
            MoveGripperToCubeInDrawerTask: None,
            GraspCubeTask: {
                CubeBetweenGripperTask: None,
                CloseGripperCubeTask: None,
            },
            LiftCubeTask: None,
        },
        PlaceCubeAtTargetTask: {
            MoveCubeTowardsTargetGraspTask: None,
        },
    }
}

######################################
# Define some 0-shot adaptation trees
######################################

## V1

cube_on_drawer_to_cube_at_target_adapt = {
    # Mix 'cube_on_table -> cube_at_target' with 'cube_in_open_drawer -> 'cube on drawer'
    MoveCubeToTargetTask: {PickUpCubeTask: None, PlaceCubeAtTargetTask: None},
}

cube_on_table_to_cube_in_drawer_adapt = {
    PlaceCubeDrawerTask: {
        PickUpCubeTask: None,
        PlaceGraspedCubeDrawerTask: None,
    }
}

cube_in_closed_drawer_to_cube_at_target_adapt = {
    # Mix 'cube_on_table -> cube_at_target' with 'cube_in_open_drawer -> 'cube on drawer'
    MoveCubeToTargetTask: {
        OpenDrawerTask: None,
        PickUpCubeTask: None,
        PlaceCubeAtTargetTask: None,
    },
}

open_drawer_then_cube_in_drawer_adapt = {
    PlaceCubeDrawerTask: {
        OpenDrawerTask: None,
        PlaceCubeDrawerTask: None,  # TODO: this should be different from parent
    }
}

open_drawer_then_pick_cube_adapt = {
    PickUpCubeTask: {
        OpenDrawerTask: None,
        PickUpCubeTask: None,  # TODO: this should be different from parent
    }
}

## V2

cube_on_drawer_to_cube_in_drawer_adapt = {
    PlaceCubeDrawerTask: {
        PickUpCubeTask: None,
        PlaceGraspedCubeDrawerTask: None,
    }
}

cube_on_table_to_cube_at_target_adapt = {
    MoveCubeToTargetTask: {PickUpCubeTask: None, PlaceCubeAtTargetTask: None},
}

## V3

cube_in_drawer_to_cube_on_drawer_adapt = {
    PlaceCubeOnDrawerTopTask: {
        PickUpCubeTask: None,
        PlaceGraspedCubeOnDrawerTopTask: None,
    }
}

######################################
# record all valid trees in this dict
######################################

TASK_TREES = {
    ## high-level tasks
    # cube only
    "pick_up_cube_mini": pick_up_cube_mini_tree,
    "pick_up_cube": pick_up_cube_tree,
    "move_cube_to_target": move_cube_to_target_tree,
    # drawer only
    "open_drawer": open_drawer_tree,
    "close_drawer": close_drawer_tree,
    # cube and drawer
    "place_cube_open_drawer": place_cube_open_drawer_tree,
    "place_cube_drawer_top": place_cube_drawer_top_tree,
    "open_then_place_in_drawer": open_then_place_in_drawer_tree,
    "open_then_place_drawer_then_close": open_then_place_drawer_then_close_tree,
    "pick_up_cube_in_drawer": pick_up_cube_in_drawer_tree,
    ## 0-shot Pretraining tasks
    "cube_on_table_to_cube_at_target": cube_on_table_to_cube_at_target_tree,
    "cube_on_drawer_to_cube_in_drawer": cube_on_drawer_to_cube_in_drawer_tree,
    "cube_on_table_to_cube_in_drawer": cube_on_table_to_cube_in_drawer_tree,
    "cube_on_drawer_to_cube_at_target": cube_on_drawer_to_cube_at_target_tree,
    "cube_on_table_to_cube_on_drawer": cube_on_table_to_cube_on_drawer_tree,
    "cube_in_drawer_to_cube_at_target": cube_in_drawer_to_cube_at_target_tree,
    ## 0-shot Adapt tasks
    # V1
    "cube_on_drawer_to_cube_at_target_adapt": cube_on_drawer_to_cube_at_target_adapt,
    "cube_on_table_to_cube_in_drawer_adapt": cube_on_table_to_cube_in_drawer_adapt,
    # V2
    "cube_on_drawer_to_cube_in_drawer_adapt": cube_on_drawer_to_cube_in_drawer_adapt,
    "cube_on_table_to_cube_at_target_adapt": cube_on_table_to_cube_at_target_adapt,
    # V3
    "cube_in_drawer_to_cube_on_drawer_adapt": cube_in_drawer_to_cube_on_drawer_adapt,
    # Other
    "cube_in_closed_drawer_to_cube_at_target_adapt": cube_in_closed_drawer_to_cube_at_target_adapt,
    "open_drawer_then_cube_in_drawer_adapt": open_drawer_then_cube_in_drawer_adapt,
    "open_drawer_then_pick_cube_adapt": open_drawer_then_pick_cube_adapt,
}

######################
# Build trees
######################


class TaskTreeBuilder:
    def __init__(self, use_dense_reward_lowest_level=False, **kwargs):
        self.hparams = {"use_dense_reward_lowest_level": use_dense_reward_lowest_level}
        self.hparams.update(kwargs)

    def build_from_name_list(self, tree_name_list):
        high_level_tasks = []
        for tree_name in tree_name_list:
            assert tree_name in TASK_TREES.keys(), "invalid task tree name!"
            task_tree = TASK_TREES[tree_name]
            task = self.build_task_tree(task_tree)
            high_level_tasks.append(task)
        return high_level_tasks

    def build_task_tree(self, task_tree):
        assert len(task_tree) == 1, "only 1 high-level task!"
        first_key, first_value = next(iter(task_tree.items()))
        assert first_value is not None, "high-level task must have subtasks!"

        def build_task_tree_inner(task_tree, parent=None, level=0):
            task_seq = []
            for task_cls, child_tree in task_tree.items():
                # init task and append
                task = task_cls(parent_task=parent, level=level, **self.hparams)
                task_seq.append(task)
                # set task children
                if child_tree is None:
                    child_task_seq = []  # TODO: is this correct
                else:
                    child_task_seq = build_task_tree_inner(
                        child_tree, parent=task, level=level + 1
                    )
                task.set_subtask_sequence(child_task_seq)
            return task_seq

        return build_task_tree_inner(task_tree)[0]
