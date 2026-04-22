import os

ROBOT_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ROBOT_GYM_DIR = os.path.join(ROBOT_GYM_ROOT_DIR, 'robot_gym')
ROBOT_GYM_ENVS_DIR = os.path.join(ROBOT_GYM_DIR, 'envs')