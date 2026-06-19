"""SO-ARM101 Reach-Avoid — Direct-workflow vision RL environment.

The arm must reach a randomized target pose while avoiding a moving "hand"
obstacle. The policy receives a composite (Dict) observation:

    - "camera":  an (H, W, C) image. By default C=4: RGB (rendered) + a binary
                 hand-mask channel (analytically projected from the hand's 3D
                 position, mirroring what MediaPipe gives us at deploy time).
    - "proprio": joint pos/vel + target position + last action.

This uses the Direct workflow on purpose: the Isaac Lab manager-based workflow
only supports a flat Box observation and cannot expose image+state composite
observations to the policy (confirmed by the Isaac Lab maintainers), so vision
RL must be done here.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import json

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, sample_uniform

from so_arm101_avoid.robots import SO_ARM101_CFG

ASSET_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@configclass
class SoArm101ReachAvoidEnvCfg(DirectRLEnvCfg):
    """Configuration for the SO-ARM101 Reach-Avoid direct environment."""

    # --- experiment knobs (used by the ablations later) ---------------------
    # which camera(s) feed the policy: "overhead" | "wrist" | "both"
    camera_view: str = "overhead"
    # visual modality channels: "rgb+mask" | "rgb" | "mask"
    obs_mode: str = "rgb+mask"
    # how the hand mask channel is produced:
    #   "seg"       -> true silhouette via camera semantic segmentation (matches
    #                  a MediaPipe hand mask at deploy). Requires the obstacle to
    #                  carry a ("class", "hand") semantic tag.
    #   "projected" -> analytic blob from the hand's 3D position (cheaper, no seg).
    mask_source: str = "seg"
    # draw a goal marker at the target pose (visible in the GUI / camera)
    show_goal_marker: bool = True
    # domain randomization master switch
    domain_randomization: bool = True

    # --- image resolution fed to the policy ---------------------------------
    # 16:9 to match the real overhead webcam (kept small for training throughput).
    image_height: int = 144
    image_width: int = 256

    # --- overhead camera, matched to the real rig --------------------------
    # Real cam: ~150 deg diagonal FOV, 16:9, mounted ~60 cm above the robot root
    # looking straight down.
    overhead_height: float = 0.50  # metres above the robot root (env origin)
    overhead_fov_diag_deg: float = 130.0  # diagonal field of view
    # "pinhole"  -> rectilinear; the fast TiledCamera renders this exactly.
    # "fisheye"  -> barrel-distorted wide lens (see note in __post_init__).
    overhead_projection: str = "pinhole"

    # --- timing -------------------------------------------------------------
    decimation = 2  # 60 Hz sim / 2 -> 30 Hz control (matches the Reach task)
    episode_length_s = 8.0
    action_scale = 0.5

    # --- task geometry (robot base frame, metres) ---------------------------
    # target sampling box (relative to the robot base)
    target_x_range = (-0.10, 0.10)
    target_y_range = (-0.25, -0.10)
    target_z_range = (0.10, 0.30)
    # hand spawns on a shell around the workspace and sweeps across it
    hand_radius = 0.07  # ~size of a fist; matches hand_cfg sphere radius below
    hand_speed_range = (0.25, 0.60)  # m/s -- lively approach/retreat
    hand_box_min = (-0.30, -0.40, 0.05)
    hand_box_max = (0.30, 0.40, 0.45)

    # --- reward weights -----------------------------------------------------
    w_track = 0.30
    w_track_fine = 0.15
    track_fine_std = 0.10
    w_clearance = 1.50
    clearance_std = 0.14  # penalty grows as hand gets within ~this distance
    w_action_rate = 1.0e-3
    collision_distance = 0.08  # link-centre to hand-centre -> terminate
    collision_penalty = 10.0

    # --- simulation ---------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
    )

    # --- scene --------------------------------------------------------------
    # NOTE: semantic segmentation disables mesh instancing, and the arm mesh is
    # ~76 MB, so we default to a moderate env count. Bump it for headless runs.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0)

    # --- goal visualization (GUI only; harmless when headless) --------------
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(emissive_color=(0.0, 1.0, 0.2)),
            )
        },
    )

    # --- robot --------------------------------------------------------------
    robot_cfg: ArticulationCfg = SO_ARM101_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    ee_body_name: str = "gripper_link"
    arm_joint_names: tuple = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

    # --- moving hand/arm obstacle (kinematic mesh) --------------------------
    # Realistic human arm mesh (recovered from the RL_UR5_IsaacLab reference).
    # Tagged ("class", "hand") so the camera's semantic segmentation gives a
    # true silhouette mask -- the same signal a MediaPipe hand mask provides at
    # deploy time. ``hand_orient`` lays the forearm roughly horizontal so it
    # sweeps through the workspace pointing at the robot.
    hand_usd_scale = (0.01, 0.01, 0.01)
    hand_orient = (0.7071, 0.7071, 0.0, 0.0)  # +90 deg about X -> lay arm down
    hand_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Hand",
        spawn=sim_utils.UsdFileCfg(
            # low-poly (~8k tri) decimation of arm.usd -- visually identical at the
            # policy's render resolution but ~870x smaller. See scripts/decimate_arm.py.
            usd_path=str(ASSET_DIR / "assets" / "arm_lowpoly.usd"),
            scale=hand_usd_scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            semantic_tags=[("class", "hand")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.3), rot=hand_orient),
    )

    # --- cameras (created on demand depending on camera_view) ---------------
    # Looks straight down at the workspace centre. With the "opengl" convention
    # an identity quaternion makes the camera optical axis (-Z) point at world
    # -Z (straight down), image +Y = world +Y.
    overhead_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/cam_overhead",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.62), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=24.0, clipping_range=(0.05, 5.0)
        ),
        width=100,
        height=100,
    )
    wrist_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/wrist_link/cam_wrist",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.02, 0.0, 0.04), rot=(0.5, -0.5, 0.5, -0.5), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=24.0, clipping_range=(0.02, 5.0)
        ),
        width=100,
        height=100,
    )

    # Observation / action spaces are filled in __post_init__.
    action_space = 5
    observation_space = 0
    state_space = 0

    def __post_init__(self):
        # number of image channels
        n_rgb = 3 if "rgb" in self.obs_mode else 0
        n_mask = 1 if "mask" in self.obs_mode else 0
        c = n_rgb + n_mask
        n_cams = 2 if self.camera_view == "both" else 1
        c *= n_cams
        # proprio: joint_pos(6) + joint_vel(6) + target_pos(3) + last_action(5)
        proprio_dim = 6 + 6 + 3 + len(self.arm_joint_names)
        self.observation_space = gym.spaces.Dict(
            {
                "camera": gym.spaces.Box(low=0.0, high=1.0, shape=(self.image_height, self.image_width, c)),
                "proprio": gym.spaces.Box(low=-math.inf, high=math.inf, shape=(proprio_dim,)),
            }
        )
        # keep camera cfg resolution in sync, and wire up the mask source
        use_seg = ("mask" in self.obs_mode) and (self.mask_source == "seg")
        for cam in (self.overhead_camera_cfg, self.wrist_camera_cfg):
            cam.height = self.image_height
            cam.width = self.image_width
            if use_seg and "semantic_segmentation" not in cam.data_types:
                cam.data_types = list(cam.data_types) + ["semantic_segmentation"]
                # return raw integer ids (not colorized) so mask = (id != 0)
                cam.colorize_semantic_segmentation = False

        # --- match the overhead camera to the real rig ---------------------
        # extrinsics: straight down, `overhead_height` m above the robot root.
        self.overhead_camera_cfg.offset = TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, self.overhead_height), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"
        )
        # intrinsics: derive the focal length that yields the desired *diagonal*
        # FOV at the current 16:9 resolution (rectilinear/pinhole geometry).
        h_ap = 24.0
        diag_px = math.hypot(self.image_width, self.image_height)
        tan_half_h = math.tan(math.radians(self.overhead_fov_diag_deg) / 2.0) * (self.image_width / diag_px)
        focal = h_ap / (2.0 * tan_half_h)
        if self.overhead_projection == "fisheye":
            # NOTE: the fast tiled renderer applies pinhole geometry; true barrel
            # distortion is only produced by the (slower) non-tiled Camera. We still
            # expose the cfg so it can be swapped if you move off tiled rendering.
            self.overhead_camera_cfg.spawn = sim_utils.FisheyeCameraCfg(
                focal_length=focal,
                horizontal_aperture=h_ap,
                clipping_range=(0.05, 5.0),
                projection_type="fisheye_polynomial",
                fisheye_max_fov=self.overhead_fov_diag_deg,
            )
        else:
            self.overhead_camera_cfg.spawn = sim_utils.PinholeCameraCfg(
                focal_length=focal, focus_distance=400.0, horizontal_aperture=h_ap, clipping_range=(0.05, 5.0)
            )


@configclass
class SoArm101ReachAvoidEnvCfg_PLAY(SoArm101ReachAvoidEnvCfg):
    def __post_init__(self):
        self.scene.num_envs = 16
        self.domain_randomization = False
        super().__post_init__()


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
class SoArm101ReachAvoidEnv(DirectRLEnv):
    cfg: SoArm101ReachAvoidEnvCfg

    def __init__(self, cfg: SoArm101ReachAvoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint / body indices
        self._arm_joint_ids, _ = self.robot.find_joints(list(self.cfg.arm_joint_names))
        self._ee_body_id, _ = self.robot.find_bodies(self.cfg.ee_body_name)
        self._ee_body_id = self._ee_body_id[0]

        self._default_joint_pos = self.robot.data.default_joint_pos.clone()

        # buffers
        n = self.num_envs
        self._actions = torch.zeros(n, len(self._arm_joint_ids), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._target_pos_b = torch.zeros(n, 3, device=self.device)  # base frame
        self._hand_pos_b = torch.zeros(n, 3, device=self.device)  # env-local frame
        self._hand_vel = torch.zeros(n, 3, device=self.device)

        self._box_min = torch.tensor(self.cfg.hand_box_min, device=self.device)
        self._box_max = torch.tensor(self.cfg.hand_box_max, device=self.device)
        self._hand_quat = torch.tensor(self.cfg.hand_orient, device=self.device).view(1, 4)

        # which cameras are active
        self._use_overhead = self.cfg.camera_view in ("overhead", "both")
        self._use_wrist = self.cfg.camera_view in ("wrist", "both")

    # ----- scene -----------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.hand = RigidObject(self.cfg.hand_cfg)

        self._cameras = {}
        if self.cfg.camera_view in ("overhead", "both"):
            self._cameras["overhead"] = TiledCamera(self.cfg.overhead_camera_cfg)
        if self.cfg.camera_view in ("wrist", "both"):
            self._cameras["wrist"] = TiledCamera(self.cfg.wrist_camera_cfg)

        # ground + dome light
        spawn_ground = sim_utils.GroundPlaneCfg()
        spawn_ground.func("/World/ground", spawn_ground, translation=(0.0, 0.0, -1.05))

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["hand"] = self.hand
        for name, cam in self._cameras.items():
            self.scene.sensors[name] = cam

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # goal marker (shows the target the EE should reach, like the SO-101 demo)
        self._goal_markers = (
            VisualizationMarkers(self.cfg.goal_marker_cfg) if self.cfg.show_goal_marker else None
        )

    # ----- actions ---------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # advance the hand along its sweep, reflecting off the workspace box
        self._hand_pos_b += self._hand_vel * self.step_dt
        below = self._hand_pos_b < self._box_min
        above = self._hand_pos_b > self._box_max
        self._hand_vel = torch.where(below | above, -self._hand_vel, self._hand_vel)
        self._hand_pos_b = self._hand_pos_b.clamp(self._box_min, self._box_max)
        self._write_hand_pose()

    def _apply_action(self):
        targets = self._default_joint_pos[:, self._arm_joint_ids] + self.cfg.action_scale * self._actions
        self.robot.set_joint_position_target(targets, joint_ids=self._arm_joint_ids)

    # ----- observations ----------------------------------------------------
    def _get_observations(self) -> dict:
        image = self._build_image()

        joint_pos_rel = self.robot.data.joint_pos - self._default_joint_pos
        joint_vel = self.robot.data.joint_vel
        proprio = torch.cat([joint_pos_rel, joint_vel, self._target_pos_b, self._actions], dim=-1)

        if self._goal_markers is not None:
            self._goal_markers.visualize(translations=self._target_pos_b + self.scene.env_origins)

        return {"policy": {"camera": image, "proprio": proprio}}

    def _build_image(self) -> torch.Tensor:
        """Assemble the (N, H, W, C) image: per camera, RGB and/or hand mask."""
        chans = []
        for name in ("overhead", "wrist"):
            if name not in self._cameras:
                continue
            cam = self._cameras[name]
            if "rgb" in self.cfg.obs_mode:
                rgb = cam.data.output["rgb"][..., :3].float() / 255.0
                chans.append(rgb)
            if "mask" in self.cfg.obs_mode:
                if self.cfg.mask_source == "seg":
                    mask = self._seg_hand_mask(cam)  # (N, H, W, 1)
                else:
                    mask = self._project_hand_mask(cam)  # (N, H, W, 1)
                chans.append(mask)
        return torch.cat(chans, dim=-1)

    def _seg_hand_mask(self, cam: TiledCamera) -> torch.Tensor:
        """True hand silhouette from the camera's semantic segmentation. The arm
        carries a ("class", "hand") tag; we look up which integer id(s) map to that
        class via ``idToLabels`` and keep only those pixels. Returns (N,H,W,1)."""
        seg = cam.data.output["semantic_segmentation"]  # (N, H, W, 1) uint32 ids
        if seg.dim() == 3:
            seg = seg.unsqueeze(-1)
        seg = seg[..., :1]

        ids = self._hand_seg_ids(cam)
        if not ids:
            return torch.zeros_like(seg, dtype=torch.float32)
        id_tensor = torch.tensor(ids, device=seg.device, dtype=seg.dtype)
        return torch.isin(seg, id_tensor).float()

    def _hand_seg_ids(self, cam: TiledCamera) -> list[int]:
        """Resolve the segmentation id(s) labelled as the hand class from the
        camera's idToLabels mapping. The mapping only lists a class once it has
        been rendered, so we keep probing until found, then cache. ``cam.data.info``
        is a dict ({'semantic_segmentation': {'idToLabels': {...}}}) but we also
        tolerate a per-env list form."""
        if getattr(self, "_hand_seg_id_cache", None):
            return self._hand_seg_id_cache

        info = getattr(cam.data, "info", None)
        seg_infos = []
        if isinstance(info, dict):
            seg_infos.append(info.get("semantic_segmentation"))
        elif isinstance(info, (list, tuple)):
            seg_infos.extend(ei.get("semantic_segmentation") for ei in info if isinstance(ei, dict))

        found: set[int] = set()
        for seg_info in seg_infos:
            mapping = seg_info.get("idToLabels", seg_info) if isinstance(seg_info, dict) else seg_info
            if isinstance(mapping, str):
                try:
                    mapping = json.loads(mapping)
                except json.JSONDecodeError:
                    continue
            if not isinstance(mapping, dict):
                continue
            for key, val in mapping.items():
                label = val.get("class", "") if isinstance(val, dict) else str(val)
                if "hand" in str(label).lower():
                    try:
                        found.add(int(key))
                    except (TypeError, ValueError):
                        pass

        ids = sorted(found)
        if ids:
            self._hand_seg_id_cache = ids
        return ids

    def _project_hand_mask(self, cam: TiledCamera) -> torch.Tensor:
        """Render a binary hand mask by projecting the hand sphere into the
        camera using its intrinsics + extrinsics. Mirrors the deploy-time mask
        we get from MediaPipe (a hand/not-hand blob)."""
        H, W = self.cfg.image_height, self.cfg.image_width
        hand_w = self._hand_pos_b + self.scene.env_origins  # (N,3) world

        cam_pos = cam.data.pos_w  # (N,3)
        cam_quat = cam.data.quat_w_ros  # (N,4) ROS: x right, y down, z forward
        K = cam.data.intrinsic_matrices  # (N,3,3)

        # point in camera frame
        p_cam = quat_apply_inverse(cam_quat, hand_w - cam_pos)  # (N,3)
        z = p_cam[:, 2].clamp(min=1e-4)
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        u = fx * p_cam[:, 0] / z + cx  # (N,)
        v = fy * p_cam[:, 1] / z + cy
        r = (fx * self.cfg.hand_radius / z).clamp(min=1.0)  # pixel radius (N,)

        in_front = p_cam[:, 2] > 0
        device = self.device
        gv, gu = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        gu = gu.unsqueeze(0)  # (1,H,W)
        gv = gv.unsqueeze(0)
        du = gu - u.view(-1, 1, 1)
        dv = gv - v.view(-1, 1, 1)
        inside = (du * du + dv * dv) <= (r.view(-1, 1, 1) ** 2)
        inside = inside & in_front.view(-1, 1, 1)
        return inside.float().unsqueeze(-1)  # (N,H,W,1)

    # ----- rewards / dones -------------------------------------------------
    def _ee_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._ee_body_id, :]

    def _min_hand_clearance(self) -> torch.Tensor:
        """Minimum distance from the hand to any robot link (world frame)."""
        hand_w = (self._hand_pos_b + self.scene.env_origins).unsqueeze(1)  # (N,1,3)
        body_pos = self.robot.data.body_pos_w  # (N, B, 3)
        d = torch.linalg.norm(body_pos - hand_w, dim=-1)  # (N, B)
        return d.min(dim=1).values  # (N,)

    def _get_rewards(self) -> torch.Tensor:
        target_w = self._target_pos_b + self.scene.env_origins
        dist = torch.linalg.norm(self._ee_pos_w() - target_w, dim=-1)
        r_track = -self.cfg.w_track * dist
        r_track_fine = self.cfg.w_track_fine * (1.0 - torch.tanh(dist / self.cfg.track_fine_std))

        clearance = self._min_hand_clearance()
        r_clear = -self.cfg.w_clearance * torch.exp(-((clearance / self.cfg.clearance_std) ** 2))

        r_action = -self.cfg.w_action_rate * torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)

        collided = clearance < self.cfg.collision_distance
        r_collision = -self.cfg.collision_penalty * collided.float()

        return r_track + r_track_fine + r_clear + r_action + r_collision

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        collided = self._min_hand_clearance() < self.cfg.collision_distance
        return collided, time_out

    # ----- reset -----------------------------------------------------------
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # robot joints -> default (+ small noise)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos += sample_uniform(-0.1, 0.1, joint_pos.shape, joint_pos.device)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # resample target (base frame)
        self._target_pos_b[env_ids, 0] = sample_uniform(*self.cfg.target_x_range, (n,), self.device)
        self._target_pos_b[env_ids, 1] = sample_uniform(*self.cfg.target_y_range, (n,), self.device)
        self._target_pos_b[env_ids, 2] = sample_uniform(*self.cfg.target_z_range, (n,), self.device)

        # resample hand: spawn on the workspace box surface, sweep across it
        self._hand_pos_b[env_ids] = self._sample_box_surface(n)
        # heading roughly toward the opposite side (so it sweeps through), with jitter
        direction = -torch.nn.functional.normalize(self._hand_pos_b[env_ids], dim=-1)
        direction += sample_uniform(-0.4, 0.4, (n, 3), self.device)
        direction = torch.nn.functional.normalize(direction, dim=-1)
        speed = sample_uniform(*self.cfg.hand_speed_range, (n, 1), self.device)
        self._hand_vel[env_ids] = direction * speed
        self._write_hand_pose(env_ids)

    def _sample_box_surface(self, n: int) -> torch.Tensor:
        lo, hi = self._box_min, self._box_max
        pos = sample_uniform(0.0, 1.0, (n, 3), self.device) * (hi - lo) + lo
        # snap one random axis to a face of the box so the hand starts at the edge
        axis = torch.randint(0, 3, (n,), device=self.device)
        face = torch.randint(0, 2, (n,), device=self.device)
        face_val = torch.where(face.bool(), hi[axis], lo[axis])
        pos[torch.arange(n, device=self.device), axis] = face_val
        return pos

    def _write_hand_pose(self, env_ids=None):
        if env_ids is None:
            hand_pos_b = self._hand_pos_b
            origins = self.scene.env_origins
        else:
            hand_pos_b = self._hand_pos_b[env_ids]
            origins = self.scene.env_origins[env_ids]
        pos_w = hand_pos_b + origins
        quat = self._hand_quat.expand(pos_w.shape[0], 4)
        root_pose = torch.cat([pos_w, quat], dim=-1)
        self.hand.write_root_pose_to_sim(root_pose, env_ids=env_ids)
