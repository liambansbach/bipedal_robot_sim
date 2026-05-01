import numpy as np

from robot_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2Cfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = (0.0, 0.0, 0.35) # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class env(LeggedRobotCfg.env):
        # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
        # + commands(3) + dof_pos(12) + dof_vel(12) + actions(12) = 48
        num_envs = 4096
        num_observations = 48
        num_actions = 12
        num_privileged_obs = None
        episode_length_s = 20.0
        send_timeouts = True

    class terrain(LeggedRobotCfg.terrain):
        # for first stable experiments:
        # mode = "plane"
        # mode = "random_uniform_terrain"
        # mode = "mixed"
        mode = "plane"

        curriculum = False

        spawn_flat_radius_sub = 0
        border_flat = False

        name = "go2_training_terrain"

        class mixed:
            options = [
                "flat_terrain",
                "random_uniform_terrain",
                "wave_terrain",
                "pyramid_sloped_terrain",
                "pyramid_stairs_terrain",
                "stepping_stones_terrain",
                "fractal_terrain",
            ]
            probs = [0.05, 0.2, 0.2, 0.2, 0.1, 0.2, 0.05]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 3
        resampling_time = 10.0

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-1.0, 1.0]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
            "FL_hip_joint": 20.0,   # [N*m/rad]
            "RL_hip_joint": 20.0,  
            "FR_hip_joint": 20.0,  
            "RR_hip_joint": 20.0,   
            "FL_thigh_joint": 20.0,    
            "RL_thigh_joint": 20.0,  
            "FR_thigh_joint": 20.0,    
            "RR_thigh_joint": 20.0,  
            "FL_calf_joint": 20.0,  
            "RL_calf_joint": 20.0,   
            "FR_calf_joint": 20.0, 
            "RR_calf_joint": 20.0,   
        }
        damping = {
            "FL_hip_joint": 0.12 * np.sqrt(20.0),   # [N*m*s/rad],
            "RL_hip_joint": 0.12 * np.sqrt(20.0),        
            "FR_hip_joint": 0.12 * np.sqrt(20.0),        
            "RR_hip_joint": 0.12 * np.sqrt(20.0),
            "FL_thigh_joint": 0.12 * np.sqrt(20.0),
            "RL_thigh_joint": 0.12 * np.sqrt(20.0),
            "FR_thigh_joint": 0.12 * np.sqrt(20.0),  
            "RR_thigh_joint": 0.12 * np.sqrt(20.0),
            "FL_calf_joint": 0.12 * np.sqrt(20.0),
            "RL_calf_joint": 0.12 * np.sqrt(20.0),
            "FR_calf_joint": 0.12 * np.sqrt(20.0),
            "RR_calf_joint": 0.12 * np.sqrt(20.0),
        }
        dof_vel_limits = {
            "FL_hip_joint": 30.1,   # [rad/s]
            "RL_hip_joint": 30.1,
            "FR_hip_joint": 30.1,
            "RR_hip_joint": 30.1,
            "FL_thigh_joint": 30.1,
            "RL_thigh_joint": 30.1,
            "FR_thigh_joint": 30.1,
            "RR_thigh_joint": 30.1,
            "FL_calf_joint": 20.07,
            "RL_calf_joint": 20.07,
            "FR_calf_joint": 20.07,
            "RR_calf_joint": 20.07,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 

    class termination(LeggedRobotCfg.termination):
        base_height_threshold = 0.22
        roll_threshold = 35.0 * np.pi / 180.0
        pitch_threshold = 35.0 * np.pi / 180.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 2
        gravity = (0.0, 0.0, -9.81)
        up_axis = 1
        enable_collision = True
        enable_joint_limit = True
        enable_self_collision = False

    class asset( LeggedRobotCfg.asset ):
        robot_file = "go2.urdf"
        name = "go2"
        robot_name = "go2" 
        file_format = "urdf"
        foot_link_names = ["FL_foot", "RL_foot", "FR_foot", "RR_foot"]
        contact_height = 0.025
        hip_abduction_indices = [0, 3, 6, 9]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.3]

        randomize_base_mass = True
        added_mass_range = [-0.3, 0.5]

        randomize_com = True
        com_shift_range = [-0.03, 0.03]

        push_robots = True
        push_interval_s = 10.0
        max_push_vel_xy = 1.0

        randomize_kp = True
        kp_scale_range = [0.8, 1.2]

        randomize_kd = True
        kd_scale_range = [0.8, 1.2] 

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True

        # base reward hyperparameters
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32
        tracking_sigma = 0.175
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        contact_force_threshold = 8.0 # [Nm]

        # dodo-specific reward hyperparameters
        clearance_target = 0.035 # m, target foot clearance during swing phase
        clearance_sigma = 0.01

        pitch_target = 0.08
        pitch_sigma = 0.05

        flat_foot_sigma = 0.20

        class scales(LeggedRobotCfg.rewards.scales):
            # --- tracking ---
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.8

            # --- general stability penalties from base ---
            lin_vel_z = -0.1
            ang_vel_xy = -0.0
            orientation = -0.5
            base_height = -1.0

            # --- smoothness / effort ---
            torques = -0.0002
            dof_vel = -0.0
            dof_acc = -2.5e-7
            action_rate = -0.01

            # --- limits / termination ---
            termination = -20.0
            dof_pos_limits = -2.0
            dof_vel_limits = -0.0
            torque_limits = -1.0

            # --- gait / base rewards ---
            feet_air_time = 0.0
            stand_still = -5.0

            # --- go2-specific rewards from DodoEnv ---
            foot_swing_clearance = 0.3
            survive = 0.01

            # keep unsupported / unused base reward names disabled
            collision = 0.0
            feet_stumble = 0.0

  
class GO2CfgPPO(LeggedRobotCfgPPO):
    seed = 1

    class actor(LeggedRobotCfgPPO.actor):
        class_name = "MLPModel"
        hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 0.8,
            "std_type": "scalar",
        }

    class critic(LeggedRobotCfgPPO.critic):
        class_name = "MLPModel"
        hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True

    class algorithm(LeggedRobotCfgPPO.algorithm): 
        class_name = "PPO"
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 8
        learning_rate = 5.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 48
        max_iterations = 2000

        save_interval = 50
        experiment_name = "go2"
        run_name = ""

        resume = False
        load_run = -1
        checkpoint = -1

        # wandb logging
        log_wandb = True
        wandb_project = "dodo-birdlike-gait"