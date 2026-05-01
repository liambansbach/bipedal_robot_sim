import numpy as np
import torch

from genesis.utils.geom import transform_by_quat


class VelocityArrowVisualizer:
    def __init__(self, sim, cfg, device):
        self.sim = sim
        self.cfg = cfg
        self.device = device

        self._cmd_vel_arrows = {}
        self._base_vel_arrows = {}

    @staticmethod
    def _arrow_pose_from_pos_vec(pos, vec, eps=1e-6):
        length = float(np.linalg.norm(vec))

        if length < eps:
            length = eps
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            direction = (vec / length).astype(np.float32)

        z_axis = direction

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(z_axis, ref))) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        x_axis = np.cross(ref, z_axis)
        x_axis /= np.linalg.norm(x_axis) + eps

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis) + eps

        T = np.eye(4, dtype=np.float32)
        T[:3, 0] = x_axis
        T[:3, 1] = y_axis
        T[:3, 2] = z_axis * length
        T[:3, 3] = pos

        return T

    def update(
        self,
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        base_lin_vel: torch.Tensor,
        commands: torch.Tensor,
        num_envs: int,
        headless: bool,
    ):
        if headless:
            return

        if not getattr(self.cfg.viewer, "visualize_velocity_arrows", False):
            return

        scale = float(getattr(self.cfg.viewer, "velocity_arrow_scale", 0.5))
        radius = float(getattr(self.cfg.viewer, "velocity_arrow_radius", 0.01))

        ref_envs = getattr(self.cfg.viewer, "ref_env", [0])
        if ref_envs is None:
            ref_envs = [0]

        ref_envs = [
            int(env_id)
            for env_id in ref_envs
            if 0 <= int(env_id) < num_envs
        ]

        if len(ref_envs) == 0:
            return

        env_ids = torch.tensor(ref_envs, device=self.device, dtype=torch.long)

        arrow_pos = base_pos[env_ids].detach().clone()
        arrow_pos[:, 2] += 0.2

        cmd_vec_body = torch.zeros((len(ref_envs), 3), device=self.device)
        cmd_vec_body[:, :2] = commands[env_ids, :2]

        vel_vec_body = torch.zeros((len(ref_envs), 3), device=self.device)
        vel_vec_body[:, :2] = base_lin_vel[env_ids, :2]

        quat = base_quat[env_ids]

        cmd_vec_world = transform_by_quat(cmd_vec_body, quat) * scale
        vel_vec_world = transform_by_quat(vel_vec_body, quat) * scale

        arrow_pos_np = arrow_pos.detach().cpu().numpy()
        cmd_vec_np = cmd_vec_world.detach().cpu().numpy()
        vel_vec_np = vel_vec_world.detach().cpu().numpy()

        objs = []
        poses = []

        for i, env_id in enumerate(ref_envs):
            pos = arrow_pos_np[i]
            cmd_vec = cmd_vec_np[i]
            vel_vec = vel_vec_np[i]

            if env_id not in self._cmd_vel_arrows or self._cmd_vel_arrows[env_id] is None:
                self._cmd_vel_arrows[env_id] = self.sim.draw_debug_arrow(
                    pos=pos,
                    vec=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    radius=radius,
                    color=(0.0, 0.0, 1.0, 0.8),
                )

            if env_id not in self._base_vel_arrows or self._base_vel_arrows[env_id] is None:
                self._base_vel_arrows[env_id] = self.sim.draw_debug_arrow(
                    pos=pos,
                    vec=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    radius=radius,
                    color=(0.0, 1.0, 0.0, 0.8),
                )

            objs.append(self._cmd_vel_arrows[env_id])
            poses.append(self._arrow_pose_from_pos_vec(pos, cmd_vec))

            objs.append(self._base_vel_arrows[env_id])
            poses.append(self._arrow_pose_from_pos_vec(pos, vel_vec))

        self.sim.update_debug_objects(tuple(objs), tuple(poses))