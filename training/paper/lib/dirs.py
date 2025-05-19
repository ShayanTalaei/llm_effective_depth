import os

def get_dirs():
    curr_dir = os.getcwd()
    main_dir = os.path.abspath(curr_dir+"/../../")
    my_rel_dir = os.path.relpath(curr_dir, main_dir)

    return curr_dir, main_dir, my_rel_dir
