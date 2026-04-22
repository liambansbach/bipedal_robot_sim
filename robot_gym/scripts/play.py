import os

from robot_gym import ROBOT_GYM_ROOT_DIR
from robot_gym.envs import *  # noqa: F401,F403 -> ensures task registration
from robot_gym.utils import get_args, export_policy_as_jit, task_registry


EXPORT_POLICY = True


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ----------------------------------------------------------------------
    # Override some parameters for testing / visualization
    # ----------------------------------------------------------------------
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)

    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False

    # Optional viewer/debug settings for play mode
    env_cfg.viewer.visualize_foot_contacts = True

    # ----------------------------------------------------------------------
    # Prepare environment
    # ----------------------------------------------------------------------
    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )

    obs, _ = env.reset()

    # ----------------------------------------------------------------------
    # Load trained policy
    # ----------------------------------------------------------------------
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )

    policy = ppo_runner.get_inference_policy(device=env.device)

    # ----------------------------------------------------------------------
    # Export policy as JIT
    # ----------------------------------------------------------------------
    if EXPORT_POLICY:
        path = os.path.join(
            ROBOT_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print(f"Exported policy as jit script to: {path}")

    # ----------------------------------------------------------------------
    # Run policy
    # ----------------------------------------------------------------------
    for _ in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == "__main__":
    args = get_args()
    play(args)