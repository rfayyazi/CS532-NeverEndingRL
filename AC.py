import argparse
import os
import json
from tqdm import tqdm
import wandb

import torch
import torch.nn.functional as F
from torch import nn
import gym

from utils import init_wandb, build_args


class ActorNet(nn.Module):
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


class CriticNet(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(args, env, actor, critic):
    for ep in tqdm(range(args.N)):
        S = torch.tensor(env.reset())
        I = 1
        cumsum_reward = 0
        terminal = False
        while not terminal:
            if args.render and ep % args.render_step == 0:
                env.render()
            ps = actor(S)
            pi = torch.distributions.Categorical(ps)  # policy distribution
            A = pi.sample()
            log_p = pi.log_prob(A)
            if args.track_logp:
                wandb.log({"logp-chosen-action": log_p.item()})
            log_p = log_p.unsqueeze(0)  # log probability of chosen action under policy

            S_new, R, terminal, _ = env.step(A.numpy())
            S_new = torch.tensor(S_new)
            cumsum_reward += R

            if not terminal:
                v_new = critic(S_new).detach()
                ret = R + args.gamma * v_new
            else:
                ret = R

            v = critic(S)
            delta = (ret - v).detach()

            actor.zero_grad()
            critic.zero_grad()
            log_p.backward()
            v.backward()

            for theta in actor.parameters():
                theta.data += args.actor_lr * I * delta * theta.grad.data

            for w in critic.parameters():
                w.data += args.critic_lr * delta * w.grad.data

            I = args.gamma * I
            S = S_new

        wandb.log({
            "episode": ep,
            "cumulative-reward": cumsum_reward,
            "max-reward": args.max_reward
        })

        if args.track_param:
            wandb.log({"actor-param": actor.fc3.weight.data[0, 10].item()})

    return actor, critic


def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=7500, type=int)
    parser.add_argument("--gamma", default=0.97, help="discount factor", type=float)
    parser.add_argument("--actor_lr", default=0.001, help="learning rate for actor network", type=float)
    parser.add_argument("--critic_lr", default=0.001, help="learning rate for critic network", type=float)
    parser.add_argument("--actor_dims", default=[64, 64], help="list of 2 hidden dims of actor network", nargs="+")
    parser.add_argument("--critic_dims", default=[64, 64], help="list of 2 hidden dims of critic network", nargs="+")
    parser.add_argument("--track_param", default=False, help="wandb log a parameter from final layer of actor network")
    parser.add_argument("--track_logp", default=True, help="wandb log the log probability of the chosen action")
    parser.add_argument("--render", default=False, help="render episodes during training")
    parser.add_argument("--render_step", default=1000, help="render every render_step episodes, if render True")
    return parser.parse_args()


def main():
    env = gym.make("CartPole-v0")

    cmd_args = get_cmd_args()
    args = build_args("AC", cmd_args, env)
    os.mkdir(args.results_folder)
    init_wandb(args)

    actor = ActorNet(args.state_dim, args.actor_dims, args.n_actions)
    critic = CriticNet(args.state_dim, args.critic_dims, 1)
    actor, critic = train(args, env, actor, critic)

    env.close()

    torch.save(actor, os.path.join(args.results_folder, "actor_net.pt"))
    torch.save(critic, os.path.join(args.results_folder, "critic_net.pt"))
    with open(os.path.join(args.results_folder, "args.json"), "w") as f:
        args.wandb_run_id = wandb.run.id
        json.dump(vars(args), f)


if __name__ == "__main__":
    main()