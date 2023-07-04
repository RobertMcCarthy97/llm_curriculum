import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


scenarios = {
    "both_low": {"parent": 0.1, "child": 0.1},
    "parent_low_child_high": {"parent": 0.1, "child": 0.9},
    "parent_high_child_low": {"parent": 0.9, "child": 0.1},
    "both_high": {"parent": 0.9, "child": 0.9},
    "both_mid": {"parent": 0.5, "child": 0.5},
}


def mean_combo(parent, child):
    p = (parent + child) / 2
    decompose_p = 1 - p
    return decompose_p


"""
Issues here:

Scenario: parent_high_child_low
Parent: 0.9, Child: 0.1
Mean: 0.5
Proportional: 0.08999999999999998

- Want to choose parent more when parent high!
"""


def proportional_combo(parent, child):
    decompose_p = (1 - parent) * (1 - child)
    return decompose_p


"""
Issues here:
- favours parent too much

Scenario: parent_high_child_low
Parent: 0.9, Child: 0.1
Mean: 0.5
Proportional: 0.08999999999999998

- should choose child more in this scenrio - child useful for collecting more data for parent!
"""


def parent_clip(parent, child):
    if parent > child:
        decompose_p = 1 - parent
    else:
        decompose_p = mean_combo(parent, child)
    return decompose_p


for scenario_key, items in scenarios.items():
    parent = items["parent"]
    child = items["child"]
    combo_decompose_p = mean_combo(parent, child)
    proportional_decompose_p = proportional_combo(parent, child)
    print(f"\nScenario: {scenario_key}")
    print(f"Parent: {parent}, Child: {child}")
    print("Decompose_p's:")
    print("---------------")
    print(f"Mean: {combo_decompose_p:.3f}")
    print(f"Proportional: {proportional_decompose_p:.3f}")
    print(f"Parent Clip: {parent_clip(parent, child):.3f}")


"""
Scenario: both_low
Parent: 0.1, Child: 0.1
Mean: 0.9
Proportional: 0.81

Scenario: parent_low_child_high
Parent: 0.1, Child: 0.9
Mean: 0.5
Proportional: 0.08999999999999998

Scenario: parent_high_child_low
Parent: 0.9, Child: 0.1
Mean: 0.5
Proportional: 0.08999999999999998

Scenario: both_high
Parent: 0.9, Child: 0.9
Mean: 0.09999999999999998
Proportional: 0.009999999999999995

Scenario: both_mid
Parent: 0.5, Child: 0.5
Mean: 0.5
Proportional: 0.25
"""


# Define range of x and y values
x_values = np.arange(0, 1.01, 0.01)
y_values = np.arange(0, 1.01, 0.01)

# Initialize empty arrays to hold the function values
data_mean = np.empty((len(x_values), len(y_values)))
data_proportional = np.empty((len(x_values), len(y_values)))
data_clip = np.empty((len(x_values), len(y_values)))

# Iterate over x and y values
for i, x in enumerate(x_values):
    for j, y in enumerate(y_values):
        data_mean[i, j] = mean_combo(x, y)
        data_proportional[i, j] = proportional_combo(x, y)
        data_clip[i, j] = parent_clip(x, y)

# Set up the figure with 3 subplots
fig, axs = plt.subplots(ncols=3, figsize=(18, 6))

# Create the heatmaps
sns.heatmap(
    data_mean,
    cmap="YlGnBu",
    ax=axs[0],
    cbar_kws={"label": "decompose_p"},
    xticklabels=20,
    yticklabels=20,
)
sns.heatmap(
    data_proportional,
    cmap="YlGnBu",
    ax=axs[1],
    cbar_kws={"label": "decompose_p"},
    xticklabels=20,
    yticklabels=20,
)
sns.heatmap(
    data_clip,
    cmap="YlGnBu",
    ax=axs[2],
    cbar_kws={"label": "decompose_p"},
    xticklabels=20,
    yticklabels=20,
)

# Set the labels
axs[0].set_title("Mean Combo")
axs[1].set_title("Proportional Combo")
axs[2].set_title("Parent Clip")
for ax in axs:
    ax.set_ylabel("Parent success")
    ax.set_xlabel("Child success")

# Display the plots
plt.tight_layout()
plt.show()
