"""Host-side deploy client (RUN IN THE `lerobot` CONDA ENV).

Owns the serial bus. Streams the follower's joint state (mapped to Isaac units)
to ``sim_bridge.py`` so the sim mirrors the real arm, and -- in --policy mode --
receives arm joint targets back, applies safety, and commands the arm.

  conda activate lerobot
  python deploy/real_client.py --mirror [--free]   # validate twin; --free = torque off to hand-guide
  python deploy/real_client.py --policy            # closed loop (start the bridge with --policy too)

E-stop: Ctrl-C -> torque is disabled on disconnect.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import mapping_util as M
import protocol

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def read_state(robot, mapping, contract):
    obs = robot.get_observation()
    q_lr = {m: float(obs[f"{m}.pos"]) for m in C.LEROBOT_MOTORS}
    q_isaac = M.lr_to_isaac_q(q_lr, mapping, contract)  # list, Isaac joint order
    return q_lr, q_isaac


def ramp_to_home(robot, mapping, contract, arm_names):
    """Slowly drive the arm to the Isaac default pose (== captured home)."""
    home_deg = {n: mapping["home_lr_deg"][n] for n in arm_names}
    start = robot.get_observation()
    start_deg = {n: float(start[f"{n}.pos"]) for n in arm_names}
    n_steps = max(1, int(C.RAMP_S * C.CONTROL_HZ))
    print(f"[client] ramping to home pose over {C.RAMP_S:.1f}s ...")
    for k in range(1, n_steps + 1):
        a = k / n_steps
        action = {f"{n}.pos": (1 - a) * start_deg[n] + a * home_deg[n] for n in arm_names}
        action["gripper.pos"] = C.GRIPPER_HOLD_POS
        robot.send_action(action)
        time.sleep(C.CONTROL_DT)


def main():
    ap = argparse.ArgumentParser(description="Real SO-101 deploy client.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mirror", action="store_true", help="Stream state only (no commands).")
    g.add_argument("--policy", action="store_true", help="Closed loop: send policy targets.")
    ap.add_argument("--free", action="store_true", help="(mirror) disable torque so you can hand-guide.")
    args = ap.parse_args()

    contract = C.load_contract()
    arm_names = contract["arm_joint_names"]
    try:
        mapping = C.load_mapping()
    except FileNotFoundError:
        if args.policy:
            raise
        mapping = M.default_mapping(contract["joint_names"])
        print("[client] no mapping.json yet -> using identity mapping for the mirror. "
              "Run calibrate_mapping.py to fix directions/offsets.")

    robot_cfg = SO101FollowerConfig(
        port=C.FOLLOWER_PORT,
        id=C.FOLLOWER_ID,
        use_degrees=C.USE_DEGREES,
        max_relative_target=C.MAX_RELATIVE_TARGET,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()
    print(f"[client] connected to {C.FOLLOWER_PORT} (id={C.FOLLOWER_ID})")

    if args.mirror and args.free:
        try:
            robot.bus.disable_torque()
            print("[client] torque DISABLED -- move the arm by hand; watch the sim follow.")
        except Exception as e:  # noqa: BLE001
            print(f"[client] could not disable torque ({e}); arm stays powered.")

    sock = protocol.make_socket(C.CLIENT_PORT)

    if args.policy and not args.free:
        ramp_to_home(robot, mapping, contract, arm_names)

    print("[client] running. Ctrl-C to stop (torque off on disconnect).")
    seq = 0
    try:
        while True:
            t0 = time.perf_counter()
            _, q_isaac = read_state(robot, mapping, contract)
            seq += 1
            protocol.send(sock, C.IPC_HOST, C.BRIDGE_PORT,
                          {"seq": seq, "t": time.time(), "q_rad": q_isaac})

            if args.policy:
                cmd = protocol.recv_latest(sock)
                if cmd is not None and not cmd.get("estop", False) and "target_rad" in cmd:
                    tgt = cmd["target_rad"]
                    tgt_by_name = {arm_names[i]: float(tgt[i]) for i in range(len(arm_names))}
                    goal_deg = M.isaac_targets_to_lr(tgt_by_name, mapping, contract)
                    action = {f"{n}.pos": goal_deg[n] for n in arm_names}
                    action["gripper.pos"] = C.GRIPPER_HOLD_POS
                    robot.send_action(action)
                elif cmd is not None and cmd.get("estop", False):
                    print("[client] E-STOP from bridge.")
                    break

            dt = time.perf_counter() - t0
            time.sleep(max(0.0, C.CONTROL_DT - dt))
    except KeyboardInterrupt:
        print("\n[client] stopping (Ctrl-C).")
    finally:
        robot.disconnect()
        print("[client] disconnected (torque off).")


if __name__ == "__main__":
    main()
