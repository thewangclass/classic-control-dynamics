# Complete Playground
Complete Playground is a lightweight Python module for creating and testing your own Classic Control problems. 

## Sample Run
Use the following command in bash to try it out. Currently hard-coded for CartPole.
'''bash
python3 complete_playground/rl_training_scripts/c51.py --save-model True --total-timesteps 200000
'''

## Currently Working On
Infos, rewards
Look at Wrappers -> RecordEpisodeStatistics
https://github.com/Farama-Foundation/Gymnasium/blob/9bc0bf308dcb5b2baead896e91fa6b3170b2405d/gymnasium/wrappers/record_episode_statistics.py

This may have something to do with why acrobot doesn't 'learn' - cartpole terminates more than truncates early on, especially when compared to acrobot. 
Maybe I should not be setting infos['final_observation'] as the last observation? Maybe it should be what reset returns? I could call reset here?

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

Reset.
VecEnv resets automatically when a done signal is encountered. 
Where should we reset? At the end of the episode, we should call reset. 
https://github.com/Farama-Foundation/Gymnasium/blob/9bc0bf308dcb5b2baead896e91fa6b3170b2405d/gymnasium/vector/vector_env.py
https://github.com/Farama-Foundation/Gymnasium/blob/9bc0bf308dcb5b2baead896e91fa6b3170b2405d/gymnasium/wrappers/autoreset.py
https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb#scrollTo=rYzDXA9vJfz1

Normalizing observations + rewards
sb3 ReplayBuffers normalizes the observations and rewards when sampling is done.
This is inherited from parent buffer class, which calls the env.normalize_obs (or env.normalize_reward).
This is from sb3 vec_env -> vec_normalize.py file. 
Maybe acrobot requires this?