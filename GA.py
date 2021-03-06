import argparse
import os
import copy
import json
from tqdm import tqdm
import wandb
import gym
import numpy as np
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


def train(args, env):
    elite = None
    population = []
    for g in tqdm(range(args.G)):

        # build new population and get performance for each genotype (policy)
        new_population = []
        performance = []
        for _ in range(args.N-1):
            if g == 0:
                policy = PolicyNet(args.state_dim, args.hidden_dims, args.n_actions)
                for theta in policy.parameters():
                    theta.requires_grad = False
                population.append(policy)
            else:
                k = np.random.randint(args.T)
                policy = population[k]
                for theta in policy.parameters():
                    theta += args.sigma * torch.normal(0.0, 1.0, size=theta.shape)
                new_population.append(policy)
            performance.append(get_cumsum_reward(env, policy))

        if g > 0:
            population = [copy.deepcopy(policy) for policy in new_population]

        # sort population by performance
        order = np.argsort(performance)[::-1]
        population = [population[i] for i in order]

        # get candidates, i.e. best performing genotypes plus previous generation's elite
        if g == 0:
            C = [population[i] for i in range(args.n_candidates)]
        else:
            C = [population[i] for i in range(args.n_candidates-1)]
            C.append(elite)

        # evaluate average performance of each candidate over 30 repeats
        candidate_performance = []
        for c in C:
            candidate_performance.append(sum([get_cumsum_reward(env, c) for _ in range(30)]) / 30.0)

        # remove best candidate from population and make it elite (if it's in the population i.e. not elite)
        elite_idx = np.argmax(candidate_performance)
        wandb.log({"cumulative-reward-elite": candidate_performance[elite_idx],
                   "generation": g + 1,
                   "max-reward": args.max_reward})
        if args.track_param:
            wandb.log({"elite-param": elite.fc3.weight.data[0, 10].item()})

        if elite_idx != args.n_candidates-1:  # i.e. elite didn't stayed the same
            elite = C[elite_idx]
            del population[elite_idx]

        # insert elite at head of population
        population.insert(0, elite)

    return elite


def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--G", default=1000, help="number of generations", type=int)
    parser.add_argument("--N", default=100, help="population size", type=int)  # 1000
    parser.add_argument("--T", default=20, help="truncation size", type=int)
    parser.add_argument("--n_candidates", default=10, help="num of best performers to consider candidates", type=int)
    parser.add_argument("--sigma", default=0.005, help="parameter mutation standard deviation", type=float)
    parser.add_argument("--hidden_dims", default=[64, 64], help="list of 2 hidden dims of policy network", nargs="+")
    parser.add_argument("--track_param", default=False, help="wandb log a parameter from final layer of actor network")
    return parser.parse_args()


def main():
    env = gym.make("CartPole-v0")

    cmd_args = get_cmd_args()
    assert cmd_args.N > cmd_args.T, "population size (N) must be greater than truncation size (T)"
    args = build_args("GA", cmd_args, env)
    os.mkdir(args.results_folder)
    init_wandb(args)

    elite = train(args, env)

    env.close()

    torch.save(elite, os.path.join(args.results_folder, "elite_net.pt"))
    with open(os.path.join(args.results_folder, "args.json"), "w") as f:
        args.wandb_run_id = wandb.run.id
        json.dump(vars(args), f)


if __name__ == "__main__":
    main()