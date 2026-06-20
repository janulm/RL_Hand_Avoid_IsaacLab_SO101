"""Isaac-side deploy bridge (RUN INSIDE DOCKER, with GUI).

A real->sim mirror: the physical follower's joint state (streamed from
``real_client.py`` over localhost UDP) is written into the Isaac articulation so
the sim is a live digital twin. The green goal marker + base triad are drawn in
the viewport. The sim never steps the policy's physics -- the real robot is the
source of truth.

Modes:
  --mirror   (default) policy OFF: just visualize the real arm. Use this to
             validate the joint mapping and rate before any motion.
  --policy   compute the reach action from the mirrored state and send arm joint
             targets back to the client (which drives the real arm).

Run:
  ./docker/run.sh python deploy/sim_bridge.py --mirror
  ./docker/run.sh python deploy/sim_bridge.py --policy [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import glob
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Real->sim mirror bridge for SO-101 deploy.")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--mirror", action="store_true", help="Visualize only (no policy, no command).")
mode.add_argument("--policy", action="store_true", help="Run the policy and command the real arm.")
parser.add_argument("--exp_config", type=str, default="configs/experiments/reach_only_proprio.yaml")
parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint (.pt). Auto-found if omitted.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# GUI on by default (this is a visualizer); user may still pass --headless.

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib.util  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402


def _load_sibling(name):
    """Load deploy/<name>.py by explicit path and register it in sys.modules.

    A bare ``import config`` resolves to Isaac's bundled cv2/config.py (its dir is
    on sys.path before ours), so we bypass name-based resolution entirely.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # so `from <name> import ...` elsewhere also gets ours
    spec.loader.exec_module(mod)
    return mod


C = _load_sibling("config")
protocol = _load_sibling("protocol")
_load_sibling("policy_runner")

from so_arm101_avoid.tasks.reach_avoid.reach_avoid_env import (  # noqa: E402
    SoArm101ReachAvoidEnv,
    SoArm101ReachAvoidEnvCfg,
)


def _auto_checkpoint() -> str:
    pat = "logs/skrl/so101_reachavoid/reach_only_proprio_*/checkpoints/best_agent.pt"
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(f"No checkpoint found matching {pat}; pass --checkpoint.")
    return hits[-1]


def main():
    use_policy = args.policy and not args.mirror
    print(f"[bridge] mode = {'POLICY' if use_policy else 'MIRROR'}")

    # --- build the env (proprio, 1 env, GUI, markers on, DR off) ------------
    cfg = SoArm101ReachAvoidEnvCfg()
    cfg.use_vision = False
    cfg.hand_spawn_prob = 0.0
    cfg.domain_randomization = False
    cfg.show_goal_marker = True
    cfg.show_frame_axes = True
    cfg.scene.num_envs = 1
    cfg.__post_init__()
    env = SoArm101ReachAvoidEnv(cfg)
    env.reset()
    robot = env.robot
    device = env.device

    default_q = robot.data.default_joint_pos[0].clone()  # (J,) rad, Isaac order
    joint_names = list(robot.data.joint_names)
    arm_names = list(cfg.arm_joint_names)
    arm_ids = env._arm_joint_ids
    n_joints = len(joint_names)
    sim_dt = float(cfg.sim.dt)

    # --- optional policy ----------------------------------------------------
    policy = None
    if use_policy:
        from policy_runner import ProprioPolicy

        exp = yaml.safe_load(open(args.exp_config))
        net0 = exp["agent"]["models"]["policy"]["network"][0]
        ckpt = args.checkpoint or _auto_checkpoint()
        print(f"[bridge] loading policy: {ckpt}")
        policy = ProprioPolicy(
            ckpt,
            in_dim=int(2 * n_joints + 3 + len(arm_names)),
            n_actions=len(arm_names),
            layers=tuple(net0["layers"]),
            activation=str(net0["activations"]),
            device=str(device),
        )

    # --- UDP server ---------------------------------------------------------
    sock = protocol.make_socket(C.BRIDGE_PORT)
    print(f"[bridge] listening on udp/{C.BRIDGE_PORT}, replying to {C.IPC_HOST}:{C.CLIENT_PORT}")

    # --- state --------------------------------------------------------------
    prev_q = None
    vel = torch.zeros(n_joints, device=device)
    last_action = torch.zeros(len(arm_names), device=device)
    target_idx = 0
    target_t0 = time.time()
    seq_out = 0
    last_msg_t = None
    warned_wait = False

    while simulation_app.is_running():
        msg = protocol.recv_latest(sock)

        if msg is not None and "q_rad" in msg:
            q = torch.tensor(msg["q_rad"], dtype=torch.float32, device=device)
            if q.numel() != n_joints:
                print(f"[bridge] WARN: got {q.numel()} joints, expected {n_joints}; ignoring")
            else:
                # finite-difference velocity (EMA-filtered)
                dt = C.CONTROL_DT
                if last_msg_t is not None and msg.get("t") is not None:
                    dt = max(1e-3, float(msg["t"]) - last_msg_t)
                last_msg_t = msg.get("t")
                if prev_q is not None:
                    raw_v = (q - prev_q) / dt
                    vel = (1 - C.VEL_EMA) * raw_v + C.VEL_EMA * vel
                prev_q = q.clone()

                # advance the cycling target
                if time.time() - target_t0 >= C.TARGET_HOLD_S:
                    target_idx = (target_idx + 1) % len(C.TARGETS)
                    target_t0 = time.time()
                target = torch.tensor(C.TARGETS[target_idx], dtype=torch.float32, device=device)

                # ---- mirror the real arm into the sim ----
                env._target_pos_b[0] = target
                _write_pose(env, robot, q, vel)

                # ---- policy ----
                if use_policy:
                    proprio = torch.cat([q - default_q, vel, target, last_action]).cpu().tolist()
                    action = torch.tensor(policy.act(proprio), device=device)
                    last_action = action
                    env._actions[0] = action
                    tgt_rad = default_q[arm_ids] + cfg.action_scale * C.ACTION_SCALE_DEPLOY * action
                    seq_out += 1
                    protocol.send(
                        sock, C.IPC_HOST, C.CLIENT_PORT,
                        {"seq": seq_out, "target_rad": tgt_rad.cpu().tolist(), "estop": False},
                    )
            warned_wait = False
        elif not warned_wait:
            print("[bridge] waiting for real_client packets ...")
            warned_wait = True

        # keep the viewport live + draw overlays
        try:
            env._draw_goal()
            env._draw_axes()
        except Exception:
            pass
        env.sim.step(render=True)

    env.close()


def _write_pose(env, robot, q, vel):
    """Kinematically force the sim articulation to the real joint state."""
    pos = q.unsqueeze(0)
    vel2 = vel.unsqueeze(0)
    robot.write_joint_state_to_sim(pos, vel2)
    # hold there so the PD controller doesn't drift the pose between writes
    robot.set_joint_position_target(pos)
    robot.write_data_to_sim()


if __name__ == "__main__":
    main()
    simulation_app.close()
