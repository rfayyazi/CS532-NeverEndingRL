import argparse
import os
import copy
import json
from tqdm import tqdm
import wandb
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch import nn

from utils import init_wandb
from apple_island.environment import AppleIsland
from apple_island.agent import Agent


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


def get_performance(args, AI, population):
    states = [torch.tensor(state) for state in AI.get_states(population)]
    rewards = np.zeros(len(population))
    for _ in range(args.T_episode):
        pols = [agent.policy(states[i].float()) for i, agent in enumerate(population)]
        pis = [torch.distributions.Categorical(pol) for pol in pols]
        actions = [pi.sample() for pi in pis]
        rewards += AI.transition(population, actions)
        states = [torch.tensor(state) for state in AI.get_states(population)]
    return rewards


def train(args, AI):
    elite = None
    population = []
    for g in tqdm(range(args.G)):

        # build new population and get performance for each genotype (policy)
        new_population = []
        for _ in range(args.N-1):
            if g == 0:
                agent = Agent()
                agent.policy = PolicyNet(args.state_dim, args.hidden_dims, args.n_actions)
                for theta in agent.policy.parameters():
                    theta.requires_grad = False
                population.append(agent)
            else:
                k = np.random.randint(args.T)
                new_agent = population[k]
                for theta in new_agent.policy.parameters():
                    theta += args.sigma * torch.normal(0.0, 1.0, size=theta.shape)
                new_population.append(new_agent)
            # performance.append(get_cumsum_reward(env, policy))

        if g > 0:
            population = [copy.deepcopy(policy) for policy in new_population]

        rewards = get_performance(args, AI, population)

        # sort population by performance
        order = np.argsort(rewards)[::-1]
        population = [population[i] for i in order]

        # get candidates, i.e. best performing genotypes plus previous generation's elite
        if g == 0:
            C = [population[i] for i in range(args.n_candidates)]
        else:
            C = [population[i] for i in range(args.n_candidates-1)]
            C.append(elite)

        # evaluate average performance of each candidate over 30 repeats
        candidate_performance = np.zeros(args.n_candidates)
        for i in range(30):
            candidate_performance += get_performance(args, AI, C)
        candidate_performance = candidate_performance / 30

        # remove best candidate from population and make it elite (if it's in the population i.e. not elite)
        elite_idx = np.argmax(candidate_performance)
        wandb.log({"cumulative-reward-elite": candidate_performance[elite_idx],
                   "generation": g + 1})
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
    parser.add_argument("--D", default=10, help="dimension of Apple Island", type=int)
    parser.add_argument("--growth_times", default=[5, 5, 5], help="apple replenishment times for 3 trees")
    parser.add_argument("--T_episode", default=100, help="number of time steps per episode")
    parser.add_argument("--G", default=1000, help="number of generations", type=int)
    parser.add_argument("--N", default=10, help="population size", type=int)  # 1000
    parser.add_argument("--T", default=7, help="truncation size", type=int)
    parser.add_argument("--n_candidates", default=4, help="num of best performers to consider candidates", type=int)
    parser.add_argument("--sigma", default=0.005, help="parameter mutation standard deviation", type=float)
    parser.add_argument("--hidden_dims", default=[64, 64], help="list of 2 hidden dims of policy network", nargs="+")
    parser.add_argument("--track_param", default=False, help="wandb log a parameter from final layer of actor network")
    return parser.parse_args()


def main():

    args = get_cmd_args()
    assert args.N > args.T, "population size (N) must be greater than truncation size (T)"

    args.n_actions = 3  # [left, right, pick]
    args.state_dim = args.D + 1  # number of agents in each grid square, plus agent's current position
    args.exp_tag = "AI"
    args.run_name = args.exp_tag + "-" + time.strftime("%y%m%d") + "-" + time.strftime("%H%M%S")
    args.results_folder = os.path.join("results", args.run_name)

    os.mkdir(args.results_folder)
    init_wandb(args)

    AI = AppleIsland(args.D, args.growth_times)
    elite = train(args, AI)

    torch.save(elite, os.path.join(args.results_folder, "elite_net.pt"))
    with open(os.path.join(args.results_folder, "args.json"), "w") as f:
        args.wandb_run_id = wandb.run.id
        json.dump(vars(args), f)


if __name__ == "__main__":
    main()