import sys
import os
import time
import wandb


def build_args(exp_tag, args, env):
    args.max_reward = env.spec.max_episode_steps
    args.n_actions = env.action_space.n  # 2  Discrete(2)
    args.state_dim = env.observation_space.shape[0]  # 4 (pole angle, vel(pole angle), cart pos, vel(cart pos)
    args.exp_tag = exp_tag
    args.run_name = args.exp_tag + "-" + time.strftime("%y%m%d") + "-" + time.strftime("%H%M%S")
    args.results_folder = os.path.join("results", args.run_name)
    return args


def init_wandb(args):
    if "--unobserve" in sys.argv:
        sys.argv.remove("--unobserve")
        os.environ["WANDB_MODE"] = "dryrun"
    os.environ["WANDB_CONSOLE"] = "off"  # Stops wandb from capturing stdout/stderr
    # noinspection PyTypeChecker
    wandb.init(project="CS532J_A2", entity="rfayyazi", config=args, name=args.run_name,  # os.environ['WANDB_ENTITY']
               tags=[args.exp_tag])