# Shared settings for the demo. Use it in every terminal, from the logibridge-main folder:
#   source demo/env.sh
export REPO="$PWD"
export PY=python3.12
export PYTHONPATH="$REPO/.deps:$REPO"
export PATH="$REPO/.deps/bin:$PATH"
export ANSIBLE_COLLECTIONS_PATH="$REPO/.deps/collections"
export MPLCONFIGDIR="$REPO/.deps/mpl"
if podman machine inspect >/dev/null 2>&1; then
  export DOCKER_HOST="unix://$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}')"
fi
