from robot_gym.envs import *
from robot_gym.utils import get_args, task_registry, class_to_dict
import wandb
import os

"""
Example training command (command line call) with all arguments specified:

    python -m robot_gym.scripts.train \
        --task dodo \                                -> Task name defined in task_registry envs/__init__.py
        --experiment_name daimao_walking \           -> Name of the experiment, used for logging and wandb run name
        --run_name run_01 \                          -> Name of the run. Overrides config file if provided.
        --num_envs 4096 \                            -> Number of parallel environments to use for training.
        --max_iterations 1500 \                      -> Maximum number of training iterations. 
        --seed 1 \                                   -> Random seed for reproducibility.
        --rl_device cuda:0 \                         -> Device used by the RL algorithm, (cpu, cuda, etc..)
        --headless \                                 -> Force display off at all times (no rendering, faster training).
        --resume \                                   -> Resume training from a checkpoint.
        --load_run daimao_walking_old \              -> Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.
        --checkpoint -1                              -> Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.

Not all arguments are required. A simple call could look like this:
    python -m robot_gym.scripts.train --task dodo --experiment_name dodo_walking_test --num_envs 4096 --max_iterations 1000 --headless
"""

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
    )

    # optional: TensorBoard -> WandB sync
    wandb.tensorboard.patch(root_logdir=ppo_runner.log_dir)

    wandb.init(
        project=train_cfg.runner.wandb_project,
        name=train_cfg.runner.run_name if train_cfg.runner.run_name else train_cfg.runner.experiment_name,
        config={
            "task": args.task,
            "env_cfg": class_to_dict(env_cfg),
            "train_cfg": class_to_dict(train_cfg),
        },
        mode="online" if train_cfg.runner.log_wandb else "disabled",
    )

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )

    ppo_runner.save(os.path.join(ppo_runner.log_dir, "model_final.pt")) # save final model independent of save frequency

    wandb.finish()


if __name__ == "__main__":
    args = get_args()
    train(args)