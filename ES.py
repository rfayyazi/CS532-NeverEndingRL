import argparse
import os
import copy
import json
from tqdm import tqdm
import wandb
import gym
import torch
import torch.nn.functional as F
from torch import nn

from utils import init_wandb, build_args


class PolicyNet(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        p = F.softmax(x, dim=0)
        return p


def get_cumsum_reward(env, policy):
    S = torch.tensor(env.reset())
    terminal = False
    cumsum = 0
    while not terminal:
        ps = policy(S)
        pi = torch.distributions.Categorical(ps)  # policy distribution
        A = pi.sample()
        S, R, terminal, _ = env.step(A.numpy())
        S = torch.tensor(S)
        cumsum += R
    return cumsum


def train(args, env, policy):
    wandb.log({"cumulative-reward": get_cumsum_reward(env, policy),
               "generation": 0,
               "max-reward": args.max_reward})
    Z = 1.0 / (args.N * args.sigma)
    for g in tqdm(range(args.G)):

        theta_dims = [theta.shape for theta in policy.parameters()]
        population, noises = [], []
        for _ in range(args.N):
            population.append(copy.deepcopy(policy))
            noises.append([torch.normal(0.0, 1.0, size=shape) for shape in theta_dims])

        theta_updates = [torch.zeros(d) for d in theta_dims]
        for i, net in enumerate(population):
            for j, theta in enumerate(net.parameters()):
                theta += args.sigma * noises[i][j]

            cumsum_reward = get_cumsum_reward(env, net)

            for j, theta in enumerate(theta_updates):
                theta += cumsum_reward * noises[i][j]

        for i, theta in enumerate(policy.parameters()):
            theta += args.lr * Z * theta_updates[i]

        wandb.log({"cumulative-reward": get_cumsum_reward(env, policy),
                   "generation": g+1,
                   "max-reward": args.max_reward})

        if args.track_param:
            wandb.log({"policy-param": policy.fc3.weight.data[0, 10].item()})

    return policy


def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--G", default=1000, help="number of generations")
    parser.add_argument("--N", default=100, help="population size")
    parser.add_argument("--lr", default=0.001, help="learning rate for policy network")
    parser.add_argument("--sigma", default=0.1, help="parameter noise standard deviation")
    parser.add_argument("--hidden_dims", default=[64, 64], help="list of 2 hidden dims of policy network", nargs="+")
    parser.add_argument("--track_param", default=False, help="wandb log a parameter from final layer of actor network")
    return parser.parse_args()


def main():
    env = gym.make("CartPole-v0")

    cmd_args = get_cmd_args()
    args = build_args("ES", cmd_args, env)
    os.mkdir(args.results_folder)
    init_wandb(args)

    policy = PolicyNet(args.state_dim, args.hidden_dims, args.n_actions)
    for theta in policy.parameters():
        theta.requires_grad = False
    policy = train(args, env, policy)

    env.close()

    torch.save(policy, os.path.join(args.results_folder, "policy_net.pt"))
    with open(os.path.join(args.results_folder, "args.json"), "w") as f:
        args.wandb_run_id = wandb.run.id
        json.dump(vars(args), f)


if __name__ == "__main__":
    main()