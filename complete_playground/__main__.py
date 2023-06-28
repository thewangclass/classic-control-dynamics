import sys
sys.path.append("/home/thewangclass/projects/classic-control-dynamics/")

from complete_playground.envs import cartpole, acrobot

if __name__ == '__main__':
    env = cartpole.CartPole()
    print(env.state)
    env.reset()
    print(env.state)