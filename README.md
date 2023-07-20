# Complete Playground
Complete Playground is a lightweight Python module for creating and testing your own Classic Control problems. It is meant to be well-documented and flexible, making it easy to implement your own learning algorithms and classical control environments. 

Complete Playground draws the majority of its own code from [Gymnasium](https://github.com/Farama-Foundation/Gymnasium). One major update of gymnasium is detailing the difference between termination and truncation in the reinforcement learning process. Gymnasium also heavily relies on wrappers for additional functionality. Complete Playground currently does not support rendering, which Gymnasium does.
Many of the reinforcement learning algorithms come from the deep reinforcement learning library [CleanRL](https://github.com/vwxyzjn/cleanrl); however, work has been done to extensively comment each section to make it understandable for someone with a beginner's understanding of reinforcement learning. 
Additional code, such as the ReplayBuffer, comes from another reinforcement learning library [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). 


## Importing the Code
To import the code, navigate to the folder you want the project in. Then, clone the git repository using the following command: 
```bash
git clone https://github.com/fedetask/categorical-dqn.git
```

## Sample Run
```bash
python3 complete_playground/rl_training_scripts/c51.py --env-id Cartpole-v1 --total-timesteps 500000 
```

This will run the train the Cartpole environment using the c51 reinforcement learning algorithm for a total of 500K timesteps.
Other additional arguments can be found in each reinforcement learning algorithm script. 

## How to Add Your Own Environment
There are primarily two locations to pay attention to.
The environment you create should be placed in complete_playground/envs. 
Register your environment in complete_playground/envs/__init__.py, following the format provided.

Make sure your environment inherits from the Env class. This means it must implement step() and reset().

## Currently Working On
### Box Space -> Discrete Space
Some algorithms only work for discrete spaces. Could we convert a Box space to a Discrete space?
Could we discretize the continuous Box Space?

Example: Continuous_MountainCarEnv
Box -> Discrete
Action space is 1D, ranging from -1 to 1. 
Shape is (1,)
Let n = 100.
Subdivide -1 to 1 into n discrete steps. 
Let action space = Discrete(n). 
n would have to be >= 2 (to account for the lower and upper range)

Example: PendulumEnv
Box -> Discrete
Action space is 1D, ranging from 2 to 2.
Shape is (1,)
Let n = 100.
Subdivide -2 to 2 into n discrete steps.
Let action space = Discrete(n).

Example: quadrotor_2D
Box -> Discrete.
Action space is 2D, with thrust on each propeller ranging from -constant to constant.
Shape is (2,). [[-constant, constant], [-constant, constant]]. 
Let n = 100.
Either divide n by shape[0] which equals 10. Then for each variable in the array, you divide it by that amount (10). Action 0 would correspond with -constant, -constant. Action 100 would correspond with constant, constant.
Would need to create tuples (left, right) to represent this. 
OR
Create environment as 9 possible actions.
Left: Down Right: Down
Left: Down Right: Neutral
Left: Down Right: Up
Left: Neutral Right: Down
Left: Neutral Right: Neutral
Left: Neutral Right: Up 
Left: Up Right: Down
Left: Up Right: Neutral
Left: Up Right: Up

### Reset.
VecEnv resets automatically when a done signal is encountered. 
Where should we reset? At the end of the episode, we should call reset. 
https://github.com/Farama-Foundation/Gymnasium/blob/9bc0bf308dcb5b2baead896e91fa6b3170b2405d/gymnasium/vector/vector_env.py
https://github.com/Farama-Foundation/Gymnasium/blob/9bc0bf308dcb5b2baead896e91fa6b3170b2405d/gymnasium/wrappers/autoreset.py
https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb#scrollTo=rYzDXA9vJfz1

### Normalizing observations + rewards
sb3 ReplayBuffers normalizes the observations and rewards when sampling is done.
This is inherited from parent buffer class, which calls the env.normalize_obs (or env.normalize_reward).
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py#L144
This is from sb3 vec_env -> vec_normalize.py file. 
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py
Maybe acrobot requires this?

Gymnasium (save and return normalized results, original data lost)
create RunningMeanStd for obs and returns in init of each environment (or in algo? or in acrobot only?)
in step, do all the calculations, then right before return, normalize everything
https://github.com/Farama-Foundation/Gymnasium/blob/9bc0bf308dcb5b2baead896e91fa6b3170b2405d/gymnasium/wrappers/normalize.py

SB3 (keeps underlying data, returns normalized results)
ReplayBuffer -> 
relies on BaseBuffer with the _normalize_obs and _normalize_reward ->
calls upon normalize_obs and normalize_reward from VecEnv/VecNormalize class, which is Wrapper -> 
if not dictionary then calls _normalize_obs in VecNormalize class, which uses a common/RunningMeanStd -> 
created line 54/68 of VecNormalize (dict or not) for observations and line 85 for returns ->
which are updated in 171 step_wait on line 185/187, and then normalized again

### Infos, rewards
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

### Autotesting
When modifying an environment, reinforcement learning algo, buffer, or other component, there should be a way to test the change and see if it is compatible with everything else. 

## Ongoing Issues
### Acrobot does not work.
Possible reasons: 
- Acrobot itself is coded wrong 
    - unlikely, have my own implementation and pretty much copy+pasted the acrobot from gymnasium
- c51 algorithm has a defect
    - reset() returns an observation that should be assigned to state (obs)
- Buffer is not correctly made
    - check how dones is calculated? 
- Environments with high truncation to termination overall just don't work?
    - Cartpole is high termination to truncation, it works
    - Acrobot is high truncation to termination, does not work
    - Create another env with high truncation to termination (mountain-car?) to check
        - If this fails, then not just an acrobot problem but something to do with high truncation to termination
    - This relates to the other bullet points, namely reset() returning an observation that should be assigned to state, infos['final_observation'], and buffer

