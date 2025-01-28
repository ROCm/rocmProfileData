import pandas as pd
import os
import shlex

def setup(argv, script_name):
    """
    Perform common setup for the cookbooks
    Designed for running ipython using one of these two syntaxes:

    1. Args after double-dash:
    $ ipython -i COOKBOOK_SCRIPT -- [COOKBOOK_SCRIPT_ARGS]

    2. Args from file:
    $ ipython -i COOKBOOK_SCRIPT 
    (or when running interactive python without arguments)
    if no arguments are specified, read arguments from a "./args" file if it exists.
    """

    pd.set_option('display.max_rows', 100)
    pd.options.display.max_colwidth = 40 
    pd.set_option('display.float_format', '{:.1f}'.format)

    if len(argv) == 1:
        args_file = os.path.abspath("args")
        if os.path.exists(args_file):
            print("info: reading args from '" + args_file + "'")
            try:
                with open(args_file, 'r') as args_file:
                    data = args_file.read()
                    argv += shlex.split(data)
            except FileNotFoundError:
                pass


    print("info: args='" + " ".join(argv) + "'")

    return argv
        
