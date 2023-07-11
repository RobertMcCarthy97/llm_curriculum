""" Script to rename files """

import os
import glob


def rename_files():
    files = glob.glob("is_next_to/*MiniGrid-IsNextTo-6x6-N2-v0*.json")

    for file in files:
        new_name = file.replace("MiniGrid-IsNextTo-6x6-N2-v0", "IsNextTo")
        os.rename(file, new_name)
        print(f"Renamed {file} to {new_name}")


if __name__ == "__main__":
    rename_files()
