import numpy as np


def get_user_action():
    prompt = "Enter action: \n"
    prompt += "[F]orward, [B]ackward, [L]eft, [R]ight, [U]p, [D]own, [O]pen, [C]lose: "
    user_action = input(prompt)
    if user_action == "F":
        action = np.array([1, 0, 0, 0])
    elif user_action == "B":
        action = np.array([-1, 0, 0, 0])
    elif user_action == "L":
        action = np.array([0, 1, 0, 0])
    elif user_action == "R":
        action = np.array([0, -1, 0, 0])
    elif user_action == "U":
        action = np.array([0, 0, 1, 0])
    elif user_action == "D":
        action = np.array([0, 0, -1, 0])
    elif user_action == "O":
        action = np.array([0, 0, 0, 1])
    elif user_action == "C":
        action = np.array([0, 0, 0, -1])
    else:
        action = np.array([0, 0, 0, 0])
    return action * 0.5
