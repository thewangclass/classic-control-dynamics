"""
Categorical DQN algo. 
Resources used
1) https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py
2) https://github.com/fedetask/categorical-dqn/blob/master/categorical_dqn.py

"""
import os
import sys
sys.path.append("/home/thewangclass/projects/classic-control-dynamics/")
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from complete_playground.utils.buffer import ReplayBuffer
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
    parser.add_argument("--track", type=bool, default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    # --wandb-project-name
    # --wandb-entity
    # --capture-video
    parser.add_argument("--save-model", type=bool, default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    # --upload-model
    # --hf-entity


    parser.add_argument("--env-id", type=str, default="cartpole",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    # --num-envs
    parser.add_argument("--n-atoms", type=int, default=101,
        help="the number of atoms")
    parser.add_argument("--v-min", type=float, default=-100,
        help="the return lower bound")
    parser.add_argument("--v-max", type=float, default=100,
        help="the return upper bound")
    parser.add_argument("--buffer-size", type=int, default=100000,
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
class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100, df=0.99, buffer_size=1e6, batch_size=1, lr=2.5e-4, network_update=500,training_freq=10, start_eps=1, end_eps=0.05, start_train_at=10000):
        super(QNetwork, self).__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        # dependent on env >> modify code later to be dynamic, hardcode for now
        self.n = len(env.action_space)      # number of possible actions, 2 actions for 

        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),       # 
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms)             # 2 action_space x 101 number_atoms
        )

    def get_action(self, x, action=None):
        # x = x.unsqueeze(0)              # cleanrl passes in tensor of [1, obs_shape], moved unsqueeze to exploit choosing
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)).long(), action]        # tensor used as indices must be long/byte/bool tensor


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



def sample_transitions(replay_buffer, batch_size):
    batch = np.random.permutation(len(replay_buffer))[:batch_size]
    trans = np.array(replay_buffer)[batch]
    return trans


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"   # create experiment run name

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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

    # Initialize Network
    # Create Model and Choose Optimizer
    q_network = QNetwork(env, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(env, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """
    c51_network = CategoricalDQN(env, n_atoms=n_atoms, v_min=v_min, v_max=v_max, df=df, buffer_size=buffer_size, batch_size=batch_size, 
                                 lr=lr, network_update=network_update, training_freq=training_freq, start_eps=start_eps, end_eps=end_eps, start_train_at=start_train_at).to(device)
    """
    

    # Setup replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_type,
        device
        )
    start_time = time.time()

    # Begin trial!
    obs, _ = env.reset()
    for global_step in range(args.total_timesteps):

        # Choose next action according to Explore vs Exploit
        # epsilon will decrease over time
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:   # choose random action
            actions = np.array([random.choice(tuple(env.action_space))])
        else:                           # choose action with highest future rewards
            actions, pmf = q_network.get_action(torch.Tensor(obs).unsqueeze(dim=0).to(device))      # unsqueeze because only one env?
            actions = actions.cpu().numpy()

        # Execute a step in the game
        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        # Record rewards for plotting purposes
        if "final_info" in infos:
            print(f"global_step={global_step}, episodic_return={infos['final_info']['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos['final_info']['episode']['r'], global_step)
            writer.add_scalar("charts/episodic_length", infos['final_info']['episode']['r'], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

        # Handle final observation due to truncation
        real_next_obs = next_obs.copy()
        if truncated:
                real_next_obs = infos["final_observation"]

        # Save data to replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # Set state to next state
        obs = next_obs

        # Start training when the timestep is > time to start training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(data.next_observations)        # _ is batch size
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                    # projection to an atom
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network according to specified frequency
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        # TODO: Check if terminated or truncated, if so then record stats, reset and start over
        if terminated or truncated:
            env.reset()


    if args.save_model:
        model_path = f"{os.getcwd()}/runs/{run_name}_{args.exp_name}_{args.total_timesteps}timesteps.model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")