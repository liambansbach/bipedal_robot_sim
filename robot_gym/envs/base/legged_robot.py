from robot_gym import ROBOT_GYM_ROOT_DIR, envs
from warnings import WarningMessage
import numpy as np
import logging

import torch
from torch import Tensor
from typing import Tuple, Dict
from robot_gym.utils.math import gs_rand_float
from robot_gym.utils.urdf_reader import URDFReader

import genesis as gs
from genesis import Scene
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.entities.base_entity import Entity
from genesis.engine.entities.rigid_entity import RigidEntity

from robot_gym import ROBOT_GYM_ROOT_DIR
from robot_gym.envs.base.base_task import BaseTask
from robot_gym.utils.math import wrap_to_pi
from robot_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        # get robot info from urdf and update asset-config accordingly
        self.urdf_reader = URDFReader(robot_file_name=self.cfg.asset.robot_file)

        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, sim_device, headless)

        # create sim
        self.create_sim()

        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
    
        for _ in range(self.cfg.control.decimation):
            self._control_dofs(self.actions) # control the robot based on the actions
            self.sim.step()

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_robot_state()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        if getattr(self.cfg.viewer, "visualize_velocity_arrows", False):
            if self.common_step_counter % 1 == 0: # maybe add some throttling here if the visualization causes lag
                self._update_velocity_arrows() 

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]


    def check_termination(self):
        """ Check if environments need to be reset
        """

        self.reset_buf = self._compute_fallen_mask()
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf # robot-fallen OR time-out -> reset


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        env_ids_np = env_ids.detach().cpu().numpy()

        if len(env_ids) == 0:
            return
        
        # Reset physics
        self.sim.reset(envs_idx=env_ids_np)
        
        # reset robot states
        self._reset_dofs(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.prev_foot_contacts[env_ids] = False
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.extras["observations"]["critic"] = self.obs_buf.clone()

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self._create_genesis_scene()
        self._create_ground_plane()
        self._create_envs()

    #------------- Callbacks --------------
    def _update_robot_state(self):
        """ Refresh robot state tensors from the simulation
        """
        # read torque from sim
        self.torques[:] = self.robot.get_dofs_control_force(
            dofs_idx_local=self.joint_dof_idx
        )

        # Base-Pos & -Orientation (Quaternions)
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.rpy[:] = quat_to_xyz(self.base_quat) # roll, pitch, yaw (in euler angles)

        # Base velocities in body coordinates
        inv_q = inv_quat(self.base_quat)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_q)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_q)

        # feet rotation
        self.foot_euler[:] = torch.stack(
            [quat_to_xyz(link.get_quat()) for link in self.ankle_links],
            dim=1
        )

        # DOF-Pos & -Vel (only Motor-DOFs)
        self.dof_pos[:] = self.robot.get_dofs_position()[..., self.joint_dof_idx]
        self.dof_vel[:] = self.robot.get_dofs_velocity()[..., self.joint_dof_idx]

        # Ankle Heights (Floor-contact-Check)
        self.current_ankle_heights[:] = torch.stack(
            [link.get_pos()[:, 2] for link in self.ankle_links],
            dim=1
        )

        #Real foot-ground contacts
        self.foot_contacts[:] = self._compute_foot_contacts()


    def _compute_foot_contacts(self):
        """
        Compute binary foot-ground contacts for all feet.

        Returns:
            foot_contacts: Bool tensor of shape (num_envs, num_feet)
        """

        num_feet = self.foot_link_indices.numel()

        contacts = self.robot.get_contacts(
            exclude_self_contact=True,
            with_entity=self.ground_floor_entity,
        )

        valid_mask = contacts["valid_mask"]              # (N, K)
        link_a = contacts["link_a"]                      # (N, K)
        link_b = contacts["link_b"]                      # (N, K)
        force_a = contacts["force_a"]                    # (N, K, 3)
        force_b = contacts["force_b"]                    # (N, K, 3)

        # Stack both contact sides into one common representation
        contact_links = torch.cat([link_a, link_b], dim=1)           # (N, 2K)
        contact_forces = torch.cat([force_a, force_b], dim=1)        # (N, 2K, 3)
        contact_valid = torch.cat([valid_mask, valid_mask], dim=1)   # (N, 2K)

        # Use normal force if you want more stable "ground contact" detection.
        # For a flat ground with z-up world, this is often more stable than full norm.
        normal_force = torch.abs(contact_forces[..., 2])

        force_threshold = self.cfg.rewards.contact_force_threshold
        contact_valid = contact_valid & (normal_force > force_threshold)

        # Invalidate non-contact entries
        invalid_fill = torch.full_like(contact_links, -1)
        contact_links = torch.where(contact_valid, contact_links, invalid_fill)

        # Compare all active contact links against all foot link ids
        # result shape: (N, 2K, num_feet)
        is_foot_contact = (contact_links.unsqueeze(-1) == self.foot_link_indices.view(1, 1, num_feet))

        # Reduce over all contact slots -> (N, num_feet)
        foot_contacts = is_foot_contact.any(dim=1)

        return foot_contacts

    def _create_genesis_scene(self):
        """ Initializes the genesis scene with the provided configuration"""

        self.sim: Scene = Scene(
            show_viewer=not self.headless,
            sim_options=gs.options.SimOptions(
                dt=self.cfg.sim.dt,
                substeps=self.cfg.sim.substeps,
                gravity=self.cfg.sim.gravity,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.cfg.viewer.max_fps,
                camera_pos=self.cfg.viewer.pos,
                camera_lookat=self.cfg.viewer.lookat,
                camera_fov=self.cfg.viewer.fov,
            ),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=self.cfg.sim.enable_collision,
                enable_joint_limit=self.cfg.sim.enable_joint_limit,
                enable_self_collision=self.cfg.sim.enable_self_collision,
                batch_links_info=self.cfg.sim.batch_links_info,
                batch_dofs_info=self.cfg.sim.batch_dofs_info,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=self.cfg.viewer.ref_env,
                show_world_frame=self.cfg.viewer.show_world_frame,
            ),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=False,
            ),
        ) 
    
    def _compute_fallen_mask(self) -> torch.Tensor:
        """
        Robust fallen detection for bipeds:

        An environment is considered fallen if:
        - The base height is below a certain threshold OR   
        - The roll OR pitch exceeds their respective thresholds.

        Returns:
            A boolean tensor of shape (num_envs,), where True indicates the robot has fallen.
        """

        # Height check
        height = self.base_pos[:, 2]
        too_low = height < self.cfg.termination.base_height_threshold  # z.B. 0.33

        # Orientation check (Roll & Pitch)
        roll = self.rpy[:, 0]
        pitch = self.rpy[:, 1]
        roll_thresh = self.cfg.termination.roll_threshold
        pitch_thresh = self.cfg.termination.pitch_threshold
        bad_orientation = (roll.abs() > roll_thresh) | (pitch.abs() > pitch_thresh)

        # fallen if either condition is met
        fallen = too_low | bad_orientation
        return fallen
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = gs_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = gs_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = gs_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _control_dofs(self, actions):
        """ Compute target from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled target.
            [NOTE]: target must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type=="P":
            targets = actions_scaled + self.default_dof_pos
            self.robot.control_dofs_position(targets, self.joint_dof_idx)
        elif control_type=="V":
            targets = actions_scaled + self.dof_vel #TODO has to be changed if you want to use velocity control
            self.robot.control_dofs_velocity(targets, self.joint_dof_idx)
        elif control_type=="T":
            torques = actions_scaled #TODO has to be changed if yo want to use force control
            self.robot.control_dofs_force(torques, self.joint_dof_idx)
        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        randomization of PD gains is also possible if domain randomization is enabled in the config file

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.cfg.domain_rand.randomize_kp or self.cfg.domain_rand.randomize_kd:
            self._randomize_pd_gains(env_ids)

        if len(env_ids) == 0:
            return

        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        env_ids_np = env_ids.detach().cpu().numpy()

        rand_scale = torch.empty((len(env_ids), self.num_dof), device=self.device).uniform_(0.9, 1.1)
        self.dof_pos[env_ids] = self.default_dof_pos * rand_scale
        self.dof_vel[env_ids] = 0.0

        self.robot.set_dofs_position(
            position=self.dof_pos[env_ids].detach().cpu().numpy(),
            dofs_idx_local=self.joint_dof_idx,
            envs_idx=env_ids_np,
            zero_velocity=True,
        )

    def _push_robots(self):
        """
        Apply a short random disturbance as a torque pulse on all controlled joints.
        Generic version that works for arbitrary robots.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[
            self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0
        ]

        if len(push_env_ids) == 0:
            return

        max_push = self.cfg.domain_rand.max_push_vel_xy

        # generic small pulse, scaled by actuator limits if available
        if hasattr(self, "torque_limits"):
            # e.g. 10% of actuator limit scaled by max_push
            amp = 0.10 * max_push * self.torque_limits.unsqueeze(0)
            push_torques = (2.0 * torch.rand(len(push_env_ids), self.num_actions, device=self.device) - 1.0) * amp
            push_torques = torch.clip(push_torques, -self.torque_limits, self.torque_limits)
        else:
            torque_scale = 2.0 * max_push
            push_torques = torch.empty(
                (len(push_env_ids), self.num_actions),
                device=self.device,
                dtype=torch.float,
            ).uniform_(-torque_scale, torque_scale)

        self.robot.control_dofs_force(
            force=push_torques,
            dofs_idx_local=self.joint_dof_idx,
            envs_idx=push_env_ids.detach().cpu().numpy(),
        )

   
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        N, A, C, D = self.num_envs, self.num_actions, self.num_commands, self.num_dof

        self.base_pos = torch.zeros((N,3), device=self.device, requires_grad=False)
        self.base_lin_vel = torch.zeros((N, 3), device=self.device, requires_grad=False)
        self.base_ang_vel = torch.zeros((N, 3), device=self.device, requires_grad=False)

        self.base_quat = torch.zeros((N,4), device=self.device, requires_grad=False)
        self.rpy = torch.zeros((N,3), device=self.device, requires_grad=False) # roll, pitch, yaw (in euler angles)

        self.common_step_counter = 0
        self.extras = {"observations": {}}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.projected_gravity = torch.zeros((N, 3), device=self.device, requires_grad=False)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(N,1)
        self.actions = torch.zeros((N, A), dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.torques = torch.zeros((N,A), dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros((A,), dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros((A,), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.commands = torch.zeros((N, C), dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        
        num_feet = len(self.ankle_links)
        self.current_ankle_heights = torch.zeros((N, num_feet), device=self.device, requires_grad=False)
        self.foot_contacts = torch.zeros((N, num_feet), dtype=torch.bool, device=self.device, requires_grad=False)
        self.prev_foot_contacts = torch.zeros((N, num_feet), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros((N, num_feet), device=self.device, requires_grad=False)
        self.foot_euler = torch.zeros((N, num_feet, 3), device=self.device, requires_grad=False)

        self._check_config_joint_names()
        
        # joint positions 
        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name] for name in self.joint_names],
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)

        self.p_gains = self.base_p_gains.clone()
        self.d_gains = self.base_d_gains.clone()

        # objects for debug visualization (velocity arrows), one pair per rendered env
        self._cmd_vel_arrows = {}
        self._base_vel_arrows = {}
        
        logging.info(f"Initialized buffers: num_envs={N}, num_dof={D}, num_actions={A}, num_commands={C}")

    def _check_config_joint_names(self):
        """ Checks that the joint names provided in the config file for default angles, stiffness, damping and dof_vel_limits match the joint names of the robot."""
        # check that default joint angles are provided for all joints
        missing = [name for name in self.joint_names if name not in self.cfg.init_state.default_joint_angles]
        if missing:
            raise KeyError(f"Missing default_joint_angles values for joints: {missing}")
        
        # check that stiffness values are provided for all joints
        missing = [name for name in self.joint_names if name not in self.cfg.control.stiffness]
        if missing:
            raise KeyError(f"Missing stiffness values for joints: {missing}")
        
        # check that damping values are provided for all joints
        missing = [name for name in self.joint_names if name not in self.cfg.control.damping]
        if missing:
            raise KeyError(f"Missing damping values for joints: {missing}")
        
        #check that dof_vel_limits values are provided for all joints
        missing = [name for name in self.joint_names if name not in self.cfg.control.dof_vel_limits]
        if missing:
            raise KeyError(f"Missing dof_vel_limits values for joints: {missing}")
        

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """

        # Choose terraintype
        if self.cfg.terrain.mode == "random":
            terrain_type = np.random.choice(self.cfg.terrain.options, p=self.cfg.terrain.probs)
        else:
            terrain_type = self.cfg.terrain.mode

        self.current_terrain_type = terrain_type  

        ground_mat = gs.surfaces.Default(color=(0.5, 0.5, 0.5))  # grau

        if terrain_type == "plane":
            self.ground_floor_entity = self.sim.add_entity(
                gs.morphs.Plane(),
                surface=ground_mat,
            )
            return

        if terrain_type == "uneven":
            u = self.cfg.terrain.uneven

            n_x, n_y = u.n_subterrains
            sub_size_x, sub_size_y = u.subterrain_size
            c_x, c_y = n_x // 2, n_y // 2

            subterrain_types = []
            for i in range(n_x):
                row = []
                for j in range(n_y):
                    di = abs(i - c_x)
                    dj = abs(j - c_y)
                    if max(di, dj) <= u.spawn_flat_radius_sub:
                        row.append("flat_terrain")
                    else:
                        row.append("random_uniform_terrain")
                subterrain_types.append(row)

            if u.border_flat:
                for k in range(n_y):
                    subterrain_types[0][k] = "flat_terrain"
                    subterrain_types[n_x - 1][k] = "flat_terrain"
                for k in range(n_x):
                    subterrain_types[k][0] = "flat_terrain"
                    subterrain_types[k][n_y - 1] = "flat_terrain"

            total_x = n_x * sub_size_x
            total_y = n_y * sub_size_y
            terrain_pos = (-0.5 * total_x - 0.5 * sub_size_x, -0.5 * total_y - 0.5 * sub_size_y, 0.0)

            self.ground_floor_entity = self.sim.add_entity(
                gs.morphs.Terrain(
                    pos=terrain_pos,
                    n_subterrains=u.n_subterrains,
                    subterrain_size=u.subterrain_size,
                    horizontal_scale=u.horizontal_scale,
                    vertical_scale=u.vertical_scale,
                    subterrain_types=subterrain_types,
                    randomize=u.randomize,
                ),
                surface=ground_mat,
            )
            return

        raise ValueError(f"Unknown terrain type: {terrain_type}")

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        
        asset_path = self.urdf_reader.robot_file_path_absolute
        asset_file = self.urdf_reader.robot_file_name

        # Add robot entity to the simulation scene
        if self.urdf_reader.robot_file_format == "urdf":
            self.robot: RigidEntity = self.sim.add_entity(
                gs.morphs.URDF(
                    file = str(asset_path),
                    fixed = False,
                    pos = self.cfg.init_state.pos,
                    quat = self.cfg.init_state.rot
                ),
                visualize_contact=self.cfg.viewer.visualize_foot_contacts,
            )
            logging.info(f"URDF file {asset_file} loaded successfully.")
        elif self.urdf_reader.robot_file_format == "xml":
            self.robot: RigidEntity = self.sim.add_entity(
                gs.morphs.MJCF(
                    file = str(asset_path),
                    fixed = False,
                    pos = self.cfg.init_state.pos,
                    quat = self.cfg.init_state.rot
                ),
                visualize_contact=self.cfg.viewer.visualize_foot_contacts,
            )
            logging.info(f"XML file {asset_file} loaded successfully.")
        else:
            raise Exception("Neither 'URDF' nor 'XML' file was loaded. Therefore No robot is loaded into the simulation")
        

        # build genesis scene after adding all entities. -> Must be done before acquiring any tensor (e.g. forces, states, etc.)
        self.sim.build(n_envs=self.cfg.env.num_envs)
        
        # joint names exactly as they appear in robot file -> from the urdf reader file
        self.joint_names = list(self.urdf_reader.joint_names)

        self.cfg.asset.joint_names = list(self.joint_names)
        logging.info(f"Joint names: {self.joint_names}")

        #get index of each joint
        self.joint_dof_idx = [self.robot.get_joint(n).dof_start for n in self.joint_names]

        # get number of dofs the robot has according to the urdf file
        self.num_dof = len(self.joint_names)
        self.num_actions = self.num_dof
        
        # get foot link names
        self.cfg.asset.foot_link_names = self.urdf_reader.foot_link_names
        logging.info(f"Foot link names: {self.cfg.asset.foot_link_names}")
        # get foot link indices (for contact calculations)
        self.ankle_links = [self.robot.get_link(n) for n in self.cfg.asset.foot_link_names]
        self.foot_link_indices = torch.tensor(
            [link.idx for link in self.ankle_links],
            dtype=torch.long,
            device=self.device,
        )

        # get joint limits
        lower_lim, upper_lim = self.robot.get_dofs_limit(dofs_idx_local=self.joint_dof_idx)
        logging.info(f"joint limits: lower: {lower_lim}, upper: {upper_lim}")
        self.dof_pos_limits = torch.stack([lower_lim, upper_lim], dim=1).to(self.device)

        # get force limits
        lower_lim, upper_lim = self.robot.get_dofs_force_range(dofs_idx_local=self.joint_dof_idx)
        logging.info(f"force limits: lower: {lower_lim}, upper: {upper_lim}")
        self.torque_limits = upper_lim.to(self.device)

        # velocity limits from config - Genesis has no direct getter for that :(
        self.dof_vel_limits = torch.tensor(
            [self.cfg.control.dof_vel_limits[name] for name in self.joint_names],
            dtype=torch.float,
            device=self.device,
        )

        # set PD gains
        self._build_pd_gains_from_cfg()
        self._set_pd_gains()

    def _build_pd_gains_from_cfg(self):
        """ Read the stiffness and damping values for each joint from the config file and store them in tensors.
        """
        self.base_p_gains = torch.tensor(
            [self.cfg.control.stiffness[name] for name in self.joint_names],
            dtype=torch.float,
            device=self.device,
        )
        self.base_d_gains = torch.tensor(
            [self.cfg.control.damping[name] for name in self.joint_names],
            dtype=torch.float,
            device=self.device,
        )

    def _set_pd_gains(self, env_ids=None, p_gains=None, d_gains=None):
        """ Setting the PD gains for the robot joints.

            We use this function to set different gains for different environments if needed, e.g. for domain randomization
            If p_gains or d_gains are not provided, the base gains from the config will be used.
        """
        if p_gains is None:
            p_gains = self.base_p_gains
        if d_gains is None:
            d_gains = self.base_d_gains

        self.robot.set_dofs_kp(
            kp=p_gains,
            dofs_idx_local=self.joint_dof_idx,
            envs_idx=env_ids,
        )
        self.robot.set_dofs_kv(
            kv=d_gains,
            dofs_idx_local=self.joint_dof_idx,
            envs_idx=env_ids,
        )

    def _randomize_pd_gains(self, env_ids):
        """ Randomizes the PD gains of selected environments based on the ranges defined in the config file.
        """
        if len(env_ids) == 0:
            return

        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        num_envs = len(env_ids)

        p = self.base_p_gains.unsqueeze(0).repeat(num_envs, 1)
        d = self.base_d_gains.unsqueeze(0).repeat(num_envs, 1)

        if self.cfg.domain_rand.randomize_kp:
            low, high = self.cfg.domain_rand.kp_scale_range
            p = p * gs_rand_float(low, high, (num_envs, self.num_dof), self.device)

        if self.cfg.domain_rand.randomize_kd:
            low, high = self.cfg.domain_rand.kd_scale_range
            d = d * gs_rand_float(low, high, (num_envs, self.num_dof), self.device)

        self.robot.set_dofs_kp(
            kp=p,
            dofs_idx_local=self.joint_dof_idx,
            envs_idx=env_ids.detach().cpu().numpy(),
        )
        self.robot.set_dofs_kv(
            kv=d,
            dofs_idx_local=self.joint_dof_idx,
            envs_idx=env_ids.detach().cpu().numpy(),
        )


    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _update_velocity_arrows(self):
        if self.headless:
            return
        if not getattr(self.cfg.viewer, "visualize_velocity_arrows", False):
            return

        scale = float(getattr(self.cfg.viewer, "velocity_arrow_scale", 0.5))
        radius = float(getattr(self.cfg.viewer, "velocity_arrow_radius", 0.01))

        ref_envs = getattr(self.cfg.viewer, "ref_env", [0])
        if ref_envs is None:
            ref_envs = [0]

        for env_id in ref_envs:
            # skip invalid env ids safely
            if env_id < 0 or env_id >= self.num_envs:
                continue

            base_pos = self.base_pos[env_id].detach().cpu().numpy().copy()
            base_pos[2] += 0.2

            # commanded planar velocity in body frame
            cmd_body = self.commands[env_id, :3].detach().cpu().numpy().copy()
            cmd_vec_body = np.array([cmd_body[0], cmd_body[1], 0.0], dtype=np.float32)

            # actual base linear velocity in body frame
            vel_body = self.base_lin_vel[env_id].detach().cpu().numpy().copy()
            vel_vec_body = np.array([vel_body[0], vel_body[1], 0.0], dtype=np.float32)

            quat = self.base_quat[env_id:env_id + 1]

            cmd_vec_world = transform_by_quat(
                torch.from_numpy(cmd_vec_body).to(self.device).unsqueeze(0),
                quat
            )[0].detach().cpu().numpy() * scale

            vel_vec_world = transform_by_quat(
                torch.from_numpy(vel_vec_body).to(self.device).unsqueeze(0),
                quat
            )[0].detach().cpu().numpy() * scale

            # clear old arrows for this env
            if env_id in self._cmd_vel_arrows and self._cmd_vel_arrows[env_id] is not None:
                self.sim.clear_debug_object(self._cmd_vel_arrows[env_id])
                self._cmd_vel_arrows[env_id] = None

            if env_id in self._base_vel_arrows and self._base_vel_arrows[env_id] is not None:
                self.sim.clear_debug_object(self._base_vel_arrows[env_id])
                self._base_vel_arrows[env_id] = None

            # draw new arrows for this env
            # commanded = blue
            self._cmd_vel_arrows[env_id] = self.sim.draw_debug_arrow(
                pos=base_pos,
                vec=cmd_vec_world,
                radius=radius,
                color=(0.0, 0.0, 1.0, 0.8),
            )

            # actual = green
            self._base_vel_arrows[env_id] = self.sim.draw_debug_arrow(
                pos=base_pos + np.array([0.0, 0.0, 0.0], dtype=np.float32),
                vec=vel_vec_world,
                radius=radius,
                color=(0.0, 1.0, 0.0, 0.8),
            )

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.base_pos[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward for long swing phases, i.e. for feet being in the air for a long time before touchdown.
        contact = self.foot_contacts                     # (N, num_feet), bool
        contact_filt = torch.logical_or(contact, self.prev_foot_contacts)

        first_contact = (self.feet_air_time > 0.0) & contact_filt

        self.feet_air_time += self.dt

        rew_air_time = torch.sum(
            torch.clamp(self.feet_air_time - 0.5, min=0.0) * first_contact.float(),
            dim=1
        )

        # no reward when command is near zero
        rew_air_time *= (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()

        # reset airtime for feet that are now in filtered contact
        self.feet_air_time *= (~contact_filt).float()

        # update previous contact memory
        self.prev_foot_contacts[:] = contact

        return rew_air_time
        
    def _reward_stand_still(self):
        # Reward for standing still (zero velocity commands)
        stand_mask = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()

        pos_err = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        vel_err = torch.sum(torch.abs(self.dof_vel), dim=1)

        return (pos_err + 0.1 * vel_err) * stand_mask