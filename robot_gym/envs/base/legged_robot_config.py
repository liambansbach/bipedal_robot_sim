from .base_config import BaseConfig
import numpy as np

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 36
        num_actions = 8
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 15 # episode length in seconds
        num_privileged_obs = None 

    class terrain:
        mode = "plane"
        options = ["plane", "uneven"]
        probs = [0.5, 0.5]
        curriculum = False

        class uneven:
            horizontal_scale = 0.25
            vertical_scale = 0.003
            subterrain_size = (2.0, 2.0)
            n_subterrains = (16, 16)
            spawn_flat_radius_sub = 0
            border_flat = True
            randomize = False

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw,
        resampling_time = 5. # time before command are changed[s]
        class ranges:
            lin_vel_x = [0.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.75, 0.75]    # min max [rad/s]

    class init_state:
        pos = (0.0, 0.0, 0.415) # x,y,z [m]
        rot = (1.0, 0.0, 0.0, 0.0) # w,x,y,z [quat]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        dof_vel_limits = {'joint_a': 6.0, 'joint_b': 6.0}   # [rad/s]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        robot_file = ""
        name = "legged_robot"
        robot_name = None
        file_format = None

        # following values are automatically set by the URDF reader in "legged_robot.py"!
        joint_names = None
        foot_link_names = None


    class domain_rand:
        randomize_friction = False # TODO friction randomization is not implemented in base class yet
        friction_range = [0.5, 1.25]
        randomize_base_mass = False # TODO base mass randomization is not implemented in base class yet
        added_mass_range = [-1., 1.]
        push_robots = False # TODO is implemented but not recommended (it applies torque to the joints instead of pushing the base)
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_kp = False
        kp_scale_range = [0.9, 1.1]

        randomize_kd = False
        kd_scale_range = [0.9, 1.1]

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  0.0
            collision = 0.0
            feet_stumble = 0.0 
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized#
        contact_force_threshold = 7. # [Nm] forces above this value are counted as a contact for stumble and air time rewards

    class termination:
        base_height_threshold = 0.25 # [m], if the robot's base is below this height, the episode is terminated
        roll_threshold = 0.8 # [rad], if the robot rolls more than this, the episode is terminated
        pitch_threshold = 0.8 # [rad], if the robot pitches more than this, the episode is terminated

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 50.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        max_fps = 100
        ref_env = [0]
        pos = (2.0, 0.0, 2.5) # [m]
        lookat = (0.0, 0.0, 0.5)  # [m]
        fov = 40 # [degrees]
        show_world_frame = True
        visualize_foot_contacts = False

        # velocity vectors for debugging:
        visualize_velocity_arrows = False
        velocity_arrow_scale = 0.6
        velocity_arrow_radius = 0.03

    class sim:
        dt =  0.005
        substeps = 2
        gravity = (0., 0. , -9.81)  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        enable_collision = True
        enable_joint_limit = True
        enable_self_collision = False
        batch_links_info = False
        batch_dofs_info = False


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        class_name = 'ActorCritic'
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        class_name = 'PPO'
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 1500

        save_interval = 50
        experiment_name = 'test'
        run_name = ''

        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        # wandb logging
        log_wandb = True
        wandb_project = "bipedal-locomotion"