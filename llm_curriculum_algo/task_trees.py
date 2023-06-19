from llm_curriculum_algo.tasks import (
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

######################
# Define trees
######################

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

# TODO: create these trees
# self.str_description = "grasp cube mini"
# self.subtask_cls_seq = [MoveGripperToCubeTask, CubeBetweenGripperTask]

# self.str_description = "Pick up cube mini"
# self.subtask_cls_seq = [MoveGripperToCubeTask, CubeBetweenGripperTask, CloseGripperCubeTask, LiftCubeTask]

# self.str_description = "pick and place mini"
# self.subtask_cls_seq = [MoveGripperToCubeTask, CubeBetweenGripperTask, CloseGripperCubeTask, LiftCubeTask, MoveCubeTowardsTargetGraspTask]

# record all valid trees in this dict
TASK_TREES = {
    'move_cube_to_target': move_cube_to_target_tree,
}

######################
# Build trees
######################

class TaskTreeBuilder():
    def __init__(self, use_dense_reward_lowest_level=False):
        self.hparams = {'use_dense_reward_lowest_level': use_dense_reward_lowest_level}

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
                    child_task_seq = [] # TODO: is this correct
                else:
                    child_task_seq = build_task_tree_inner(child_tree, parent=task, level=level+1)
                task.set_subtask_sequence(child_task_seq)
            return task_seq
        
        return build_task_tree_inner(task_tree)[0]