# Complete Playground
Complete Playground is a lightweight Python module for creating and testing your own Classic Control problems. 

## Sample Run
Use the following command in bash to try it out. Currently hard-coded for CartPole.
'''bash
python3 complete_playground/rl_training_scripts/c51.py --save-model True --total-timesteps 200000
'''


## Ongoing Issues
ModuleNotFoundError. Look in c51.py and you will see the current workaround of appending to sys.path. Alternatives to add this on the Python path is to do it at the command line or export to the shell configuration. See (stackoverflow)[https://stackoverflow.com/questions/5875810/importerror-when-trying-to-import-a-custom-module-in-python].