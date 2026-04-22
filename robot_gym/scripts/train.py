from robot_gym.envs import *
from robot_gym.utils import get_args, task_registry, class_to_dict
import wandb


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

    wandb.finish()


if __name__ == "__main__":
    args = get_args()
    train(args)