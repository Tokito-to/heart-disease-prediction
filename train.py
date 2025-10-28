import os

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

script_dir = os.path.dirname(__file__)
os.chdir(script_dir)
