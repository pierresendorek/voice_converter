import os
from pprint import pprint
import re

def get_file_and_path_list(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        # print("root : ", root)
        # print("directories : ", directories)
        # print("files : ", files)
        for filename in files:
            # Join the two strings in order to form the full filepath.
            if re.match('(.)+\.wav$', filename):
                filepath = os.path.join(root, filename)
                file_paths.append([root, filename])  # Add it to the list.

    return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.

from params.params import get_params
import os

if __name__ == "__main__":

    params = get_params()
    project_base_path = params["project_base_path"]
    full_file_paths = get_file_and_path_list(os.path.join(project_base_path, "data/bruce_willis/Studio"))
    pprint(full_file_paths)