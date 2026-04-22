from robot_gym import ROBOT_GYM_ROOT_DIR, ROBOT_GYM_ENVS_DIR


from robot_gym.envs.base.legged_robot import LeggedRobot


from robot_gym.envs.dodo.dodo_env import DodoEnv
from robot_gym.envs.dodo.dodo_config import DodoCfg, DodoCfgPPO


from robot_gym.utils.task_registry import task_registry

task_registry.register(
    "dodo",
    DodoEnv,
    DodoCfg(),
    DodoCfgPPO(),
)