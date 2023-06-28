"""
Categorical DQN algo. 
Resources used
1) https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py
2) https://github.com/fedetask/categorical-dqn/blob/master/categorical_dqn.py

"""
import sys
sys.path.append("/home/thewangclass/projects/classic-control-dynamics/")
import argparse
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from complete_playground.envs import cartpole, acrobot


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    # parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="if toggled, this experiment will be tracked with Weights and Biases")
    # parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
    #     help="the wandb's project name")
    # parser.add_argument("--wandb-entity", type=str, default=None,
    #     help="the entity (team) of wandb's project")
    # parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="whether to capture videos of the agent performances (check out `videos` folder)")
    # parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="whether to save model into the `runs/{run_name}` folder")
    # parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="whether to upload the saved model to huggingface")
    # parser.add_argument("--hf-entity", type=str, default="",
    #     help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="cartpole",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    # parser.add_argument("--num-envs", type=int, default=1,
    #     help="the number of parallel game environments")
    parser.add_argument("--n-atoms", type=int, default=101,
        help="the number of atoms")
    parser.add_argument("--v-min", type=float, default=-100,
        help="the return lower bound")
    parser.add_argument("--v-max", type=float, default=100,
        help="the return upper bound")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()

    return args


"""
Later separate this into agent (learner) and network?
"""
class CategoricalDQN(nn.Module):
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100, df=0.99, buffer_size=1e6, batch_size=128, lr=2.5e-4, network_update=500,training_freq=10, start_eps=1, end_eps=0.05, start_train_at=10000):
        super(CategoricalDQN, self).__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.dz = (v_max-v_min) / (n_atoms - 1)     # width of each "category"
        
        self.df = df                                # discount factor for future rewards
        self.epsilon = start_eps                    # episilon value dictates greedy/explore
        self.end_epsilon = end_eps

        self.start_train_at = start_train_at
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = training_freq

        # dependent on env >> modify code later to be dynamic, hardcode for now
        self.n = len(env.action_space)      # number of possible actions, 2 actions for 
        print(self.n)
        print(env.observation_space.shape)

        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),       # 
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms)
        )

    def get_action(self, x, action=None):
        pass


    def forward(self, x):
        pass

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"   # create experiment run name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)

    # setup environment
    env = cartpole.CartPole()
    # env = args.env_id   # cartpole/acrobot for now
    print(env.state)          # cartpole -> somehow grab the constructor Cartpole() and call it
    env.reset()
    print(env.state)

    # Setup network
    n_atoms = args.n_atoms
    v_min = args.v_min
    v_max = args.v_max
    df = args.gamma
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    lr = args.learning_rate
    network_update = args.target_network_frequency
    training_freq = args.train_frequency
    start_eps = args.start_e
    end_eps = args.end_e
    start_train_at = args.learning_starts

    c51_network = CategoricalDQN(env, n_atoms=n_atoms, v_min=v_min, v_max=v_max, df=df, buffer_size=buffer_size, batch_size=batch_size, 
                                 lr=lr, network_update=network_update, training_freq=training_freq, start_eps=start_eps, end_eps=end_eps, start_train_at=start_train_at).to(device)

    # Setup replay buffer

    # Begin trial!



    pass