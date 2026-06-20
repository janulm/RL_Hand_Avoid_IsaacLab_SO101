# Deploy: real SO-101 ⟷ Isaac sim mirror

Run a trained reach policy on the **real** SO-101 follower while a **live Isaac
twin** mirrors the real arm (the real robot is the source of truth; the sim never
commands it). Two processes talk over localhost UDP:

| Process | Where | Role |
|---|---|---|
| `real_client.py` | host **`lerobot` conda env** | owns the serial bus; maps units; safety |
| `sim_bridge.py` | inside **Docker** (Isaac GUI) | digital twin + policy; draws goal/axes |

```
[lerobot conda] real_client.py  ──UDP q_rad──▶  sim_bridge.py [docker]
                send_action(deg) ◀─UDP target─  policy(obs)
```

Units differ (lerobot = **degrees**, mid-range zero; Isaac = **radians**, URDF
zero), so each joint has a per-joint **sign + home offset** in `mapping.json`,
anchored on the Isaac default pose. The policy needs joint **velocity**, which the
follower doesn't report — it's finite-differenced from positions in the bridge.

## One-time setup

1. **Export the contract** (joint order, defaults, limits, action scale) — in Docker:
   ```bash
   ./docker/run.sh python deploy/export_contract.py      # -> deploy/contract.json
   ```

2. **Validate the mirror** (no motion, no policy). Terminal A (Docker):
   ```bash
   ./docker/run.sh python deploy/sim_bridge.py --mirror
   ```
   Terminal B (conda; `--free` powers off torque so you can hand-guide):
   ```bash
   conda activate lerobot
   python deploy/real_client.py --mirror --free
   ```
   Move the real arm by hand → the sim arm should follow. Directions may be
   mirrored/offset until calibration (next step).

3. **Calibrate the joint mapping** (keep the `--mirror` bridge running in Terminal A).
   The calibrator owns the serial bus *and* streams the twin, so run it **instead
   of** `real_client.py` (only one process can open the port). Terminal B:
   ```bash
   conda activate lerobot
   python deploy/calibrate_mapping.py
   ```
   In the `calib>` prompt:
   - `sign` — jogs each joint; confirm whether the SIM moved the same way (torque on).
   - `free` — torque off so you can hand-pose the arm to the Isaac default pose.
   - `nudge <joint> <deg>` — fine-tune (the twin updates live) until it shows the
     canonical default pose; then `home` to capture.
   - `save` → writes `deploy/mapping.json`. (`hold` re-enables torque.)

## Run the policy (closed loop)

Terminal A (Docker) — train first if you haven't, then:
```bash
./docker/run.sh python deploy/sim_bridge.py --policy     # auto-loads best_agent.pt
```
Terminal B (conda):
```bash
conda activate lerobot
python deploy/real_client.py --policy
```
The client ramps to the home pose, then the bridge cycles through the hardcoded
targets (`config.TARGETS`); the arm reaches each while the sim mirrors. **Ctrl-C**
disables torque on disconnect.

## Safety / tuning (`deploy/config.py`)
- `MAX_RELATIVE_TARGET` (deg/step), `ACTION_EMA`, `ACTION_SCALE_DEPLOY`, `RAMP_S`.
- Start conservative; raise once motion looks correct.
- `TARGETS`, `TARGET_HOLD_S`, `GRIPPER_HOLD_POS`, ports/ids all live here too.
