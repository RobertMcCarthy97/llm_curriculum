from env_wrappers import make_env
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import util
import torch

hparams = {
        'seed': 0,
        # env
        'manual_decompose_p': 1,
        'dense_rew_lowest': False,
        'dense_rew_tasks': [],
        'use_language_goals': True,
        'render_mode': 'rgb_array',
        'use_oracle_at_warmup': False,
        'max_ep_len': 50,
        'use_baseline_env': False,
        'single_task_names': ["move_gripper_to_cube", "cube_between_grippers", "close_gripper_cube", "lift_cube"],
        'high_level_task_names': ['move_cube_to_target'],
        'contained_sequence': False,
}

env = make_env(
            manual_decompose_p=hparams['manual_decompose_p'],
            dense_rew_lowest=hparams['dense_rew_lowest'],
            dense_rew_tasks=hparams['dense_rew_tasks'],
            use_language_goals=hparams['use_language_goals'],
            render_mode=hparams['render_mode'],
            max_ep_len=hparams['max_ep_len'],
            single_task_names=hparams['single_task_names'],
            high_level_task_names=hparams['high_level_task_names'],
            contained_sequence=hparams['contained_sequence'],
            )

task_dict = env.agent_conductor.task_embeddings_dict

# Convert your dictionary values (i.e., the embeddings) to a tensor
embeddings = torch.stack([torch.tensor(e) for e in task_dict.values()])

# Calculate the cosine similarity matrix
cosine_scores = util.cos_sim(embeddings, embeddings)

# Convert tensor to numpy for plotting
cosine_scores_np = cosine_scores.numpy()

# Create a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(cosine_scores_np, annot=True, cmap='viridis', xticklabels=list(task_dict.keys()), yticklabels=list(task_dict.keys()))

# Rotate x-axis labels
plt.xticks(rotation=45)

plt.title('Cosine Similarity Between Task Embeddings')
plt.tight_layout()
plt.show()