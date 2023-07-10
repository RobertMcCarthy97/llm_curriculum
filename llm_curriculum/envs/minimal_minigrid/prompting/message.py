import json


def save_obj(obj, filename):
    """
    Saves an object to a JSON file.

    Args:
    obj: The object to be saved.
    filename: The path to the file where the object will be saved.
    """
    with open(filename, "w") as f:
        json.dump(obj, f)


def load_obj(filename):
    """
    Loads an object from a JSON file.

    Args:
    filename: The path to the file from which the object will be loaded.

    Returns:
    The object that was loaded from the file.
    """
    with open(filename, "r") as f:
        obj = json.load(f)
    return obj


def save_messages(messages, filename):
    """
    Saves a list of message dictionaries to a JSON file.

    Args:
    messages: A list of dictionaries. Each dictionary represents a message and contains two keys: 'role' and 'content'.
    filename: The path to the file where the messages will be saved.
    """
    with open(filename, "w") as f:
        json.dump(messages, f)


def load_messages(filename):
    """
    Loads a list of message dictionaries from a JSON file.

    Args:
    filename: The path to the file from which the messages will be loaded.

    Returns:
    A list of dictionaries. Each dictionary represents a message and contains two keys: 'role' and 'content'.
    """
    with open(filename, "r") as f:
        messages = json.load(f)
    return messages
