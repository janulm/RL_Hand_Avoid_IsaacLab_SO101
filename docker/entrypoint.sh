#!/bin/bash
# Entrypoint for the SO-ARM101 Reach-Avoid container.
# Sets up the Isaac Sim python environment, makes `python` point at it, and
# (editable-)installs our extension from the mounted source tree so code edits
# on the host are picked up without rebuilding the image.
set -e

ISAAC_SIM=/workspace/isaaclab/_isaac_sim
export CARB_APP_PATH=$ISAAC_SIM/kit
export ISAAC_PATH=$ISAAC_SIM
export EXP_PATH=$ISAAC_SIM/apps
source "${ISAAC_SIM}/setup_python_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES

# Make a bare `python` resolve to Isaac Sim's interpreter.
cat > /usr/local/bin/python << 'WRAPPER'
#!/bin/bash
exec /workspace/isaaclab/_isaac_sim/python.sh "$@"
WRAPPER
chmod +x /usr/local/bin/python

REPO=/workspace/RL_Hand_Avoid_IsaacLab_SO101
EXT=$REPO/source/so_arm101_avoid

# Editable-install the extension if present (idempotent, fast after first run).
if [ -f "$EXT/setup.py" ] || [ -f "$EXT/pyproject.toml" ]; then
    python -m pip install -e "$EXT" >/tmp/ext_install.log 2>&1 || {
        echo "[entrypoint] extension install failed; see /tmp/ext_install.log"; cat /tmp/ext_install.log; }
fi

exec "$@"
