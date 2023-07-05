# Complete Playground
Complete Playground is a lightweight Python module for creating and testing your own Classic Control problems. 

## Sample Run
Use the following command in bash to try it out. Currently hard-coded for CartPole.
'''bash
python3 complete_playground/rl_training_scripts/c51.py --save-model True --total-timesteps 200000
'''

## Currently Working On
There are two types of ways for cartpole to end. 
One way is through termination. This is when the pole falls according to the experiment parameters.
The other is through truncation. This is when the experiment exceeds max timesteps allowed or something not defined by the problem. 
In CleanRL c51, they handle these two on lines 202-210 (termination) as well as 216-218 (truncation).
It seems there are 4 keys added to infos: 'final_observation', '_final_observation', 'final_info', '_final_info'.
- Final Observation: seems like it is the last observation before termination/truncation
- _Final_Observation: True/False, np.ndarray
- Final Info: See below
- _Final_Info: True/False, np.ndarray

It seems there is a 'final_info' key that is added to infos. This tells the program it is finished.
This 'final_info' key of 'infos' has a value of a numpy.ndarray. 
The np.array has a dictionary inside with a key of 'episode'. 
This 'episode' has three keys: 'r', 'l', and 't'.
Example values:
'r': np.ndarray, array([13.], dtype=float32): return
'l': np.ndarray, array([13.], dtype=int32): length
't': np.ndarray, array([4.043945], dtype=float32): ???

## Ongoing Issues
ModuleNotFoundError. Look in c51.py and you will see the current workaround of appending to sys.path. Alternatives to add this on the Python path is to do it at the command line or export to the shell configuration. See (stackoverflow)[https://stackoverflow.com/questions/5875810/importerror-when-trying-to-import-a-custom-module-in-python].