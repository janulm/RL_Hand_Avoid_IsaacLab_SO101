"""Guided lerobot<->Isaac joint mapping (RUN IN THE `lerobot` CONDA ENV).

Derives, per arm joint, the SIGN (does +lerobot match +Isaac?) and the HOME
(lerobot degrees at the Isaac default pose), and writes ``deploy/mapping.json``.

This tool OWNS the serial bus and also streams the live joint state to the bridge
(so you can watch the twin while calibrating). So run it INSTEAD of real_client:

  Terminal A (docker):  ./docker/run.sh python deploy/sim_bridge.py --mirror
  Terminal B (conda):   python deploy/calibrate_mapping.py

Flow:
  1. `sign` - jogs each joint a few degrees; confirm whether the SIM moved the
              same direction as the REAL arm (flips the sign if not).
  2. Pose the real arm to the Isaac DEFAULT pose, watching the sim twin; use
     `nudge <joint> <deg>` (updates the twin live) until it shows the canonical
     default pose, then `home` to capture.
  3. `save` - writes mapping.json.

Type `help` for commands.
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import mapping_util as M
import protocol

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOG_DEG = 8.0


def main():
    contract = C.load_contract()
    arm_names = contract["arm_joint_names"]
    all_names = contract["joint_names"]

    try:
        mapping = C.load_mapping()
        print("[calib] loaded existing mapping.json")
    except FileNotFoundError:
        mapping = M.default_mapping(all_names)
        print("[calib] starting from identity mapping (sign=+1, home=0)")

    robot = SO101Follower(
        SO101FollowerConfig(
            port=C.FOLLOWER_PORT, id=C.FOLLOWER_ID,
            use_degrees=C.USE_DEGREES, max_relative_target=C.MAX_RELATIVE_TARGET,
        )
    )
    robot.connect()
    print(f"[calib] connected to {C.FOLLOWER_PORT}.  Bridge should be running with --mirror.")

    lock = threading.Lock()
    stop = threading.Event()
    sock = protocol.make_socket(C.CLIENT_PORT)

    def read_arm_deg():
        with lock:
            obs = robot.get_observation()
        return {n: float(obs[f"{n}.pos"]) for n in arm_names}, obs

    def send_arm_deg(deg_by_name):
        action = {f"{n}.pos": v for n, v in deg_by_name.items()}
        action["gripper.pos"] = C.GRIPPER_HOLD_POS
        with lock:
            robot.send_action(action)

    def stream():
        seq = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                with lock:
                    obs = robot.get_observation()
                q_lr = {m: float(obs[f"{m}.pos"]) for m in C.LEROBOT_MOTORS}
                q_isaac = M.lr_to_isaac_q(q_lr, mapping, contract)
                seq += 1
                protocol.send(sock, C.IPC_HOST, C.BRIDGE_PORT,
                              {"seq": seq, "t": time.time(), "q_rad": q_isaac})
            except Exception:  # noqa: BLE001
                pass
            time.sleep(max(0.0, C.CONTROL_DT - (time.perf_counter() - t0)))

    streamer = threading.Thread(target=stream, daemon=True)
    streamer.start()
    print("[calib] streaming live state to the sim twin.")
    print("Commands: sign | free | hold | home | nudge <joint> <deg> | show | save | quit")

    try:
        while True:
            try:
                cmd = input("calib> ").strip().split()
            except EOFError:
                break
            if not cmd:
                continue
            op = cmd[0].lower()

            if op in ("quit", "q", "exit"):
                break
            elif op == "help":
                print("sign | free | hold | home | nudge <joint> <deg> | show | save | quit")
            elif op == "free":
                try:
                    with lock:
                        robot.bus.disable_torque()
                    print("  torque OFF -- hand-pose the arm to the Isaac default; watch the twin.")
                except Exception as e:  # noqa: BLE001
                    print(f"  could not disable torque ({e})")
            elif op == "hold":
                try:
                    with lock:
                        robot.bus.enable_torque()
                    print("  torque ON.")
                except Exception as e:  # noqa: BLE001
                    print(f"  could not enable torque ({e})")
            elif op == "sign":
                cur, _ = read_arm_deg()
                for n in arm_names:
                    print(f"\n[sign] jogging {n} by +{JOG_DEG} deg ...")
                    moved = dict(cur)
                    moved[n] = cur[n] + JOG_DEG
                    send_arm_deg(moved)
                    time.sleep(0.7)
                    send_arm_deg(cur)
                    time.sleep(0.5)
                    ans = input(f"  Did the SIM {n} move the SAME way as the REAL {n}? [Y/n] ").strip().lower()
                    mapping["sign"][n] *= -1 if ans == "n" else 1
                    print(f"  -> sign[{n}] = {mapping['sign'][n]:+d}")
            elif op == "home":
                deg, _ = read_arm_deg()
                for n in arm_names:
                    mapping["home_lr_deg"][n] = deg[n]
                print(f"[home] captured: { {n: round(mapping['home_lr_deg'][n], 2) for n in arm_names} }")
            elif op == "nudge" and len(cmd) == 3:
                j, d = cmd[1], float(cmd[2])
                if j in mapping["home_lr_deg"]:
                    mapping["home_lr_deg"][j] += d
                    print(f"  home[{j}] = {mapping['home_lr_deg'][j]:.2f}  (twin updates live)")
                else:
                    print(f"  unknown joint '{j}' (choices: {arm_names})")
            elif op == "show":
                for n in arm_names:
                    print(f"  {n:14s} sign={mapping['sign'][n]:+d}  home={mapping['home_lr_deg'][n]:.2f}")
            elif op == "save":
                M.save_mapping(C.MAPPING_PATH, mapping)
                print(f"  wrote {C.MAPPING_PATH}")
            else:
                print("  ? type `help`")
    finally:
        stop.set()
        streamer.join(timeout=1.0)
        robot.disconnect()
        print("[calib] disconnected (torque off).")


if __name__ == "__main__":
    main()
