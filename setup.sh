#!/usr/bin/env bash
# One-time setup for the three-stage jump project.
#   ./setup.sh            # uses the conda env "mujoco"
#   ./setup.sh other_env  # uses another env name
set -euo pipefail

ENV_NAME="${1:-mujoco}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 1. Add the project packages to the existing env
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' not found. Pass the right name: ./setup.sh <env_name>" >&2
  exit 1
fi
echo "==> Updating conda env '$ENV_NAME' from environment.yml"
conda env update -n "$ENV_NAME" -f environment.yml

# 2. Download the H1-2 model (URDF + MJCF + meshes) from unitree_ros, only that folder
DESC="third_party/unitree_ros/robots/h1_2_description"
if [ ! -d "$DESC" ]; then
  echo "==> Downloading H1-2 description"
  mkdir -p third_party
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/unitreerobotics/unitree_ros.git third_party/unitree_ros
  git -C third_party/unitree_ros sparse-checkout set robots/h1_2_description
else
  echo "==> H1-2 description already present"
fi

# 3. Phase 0 gate
echo "==> Running checks"
conda run --no-capture-output -n "$ENV_NAME" python scripts/check_env.py 2>/dev/null
