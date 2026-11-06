#!/bin/bash
# One-time setup before recording: Python deps, Podman machine, network, brokers, registry, Ansible.
cd "$(dirname "$0")/.."
source demo/env.sh

if ! podman machine inspect --format '{{.State}}' | grep -q running; then
  podman machine start
  source demo/env.sh
fi

if [ ! -d .deps/numpy ]; then
  $PY -m pip install --target .deps numpy==1.26.4 paho-mqtt==2.1.0 scipy==1.13.1 matplotlib==3.9.2 \
    psutil==6.0.0 pytest==8.3.3 ansible-core==2.17.7 docker==7.1.0 requests==2.32.3
  $PY -m pip install --target .deps --no-deps ai-edge-litert==1.2.0
fi
if [ ! -d .deps/collections/ansible_collections/community/docker ]; then
  ansible-galaxy collection install community.docker:==3.12.1 -p .deps/collections
fi

podman network exists logiedge_net || podman network create logiedge_net

podman rm -f mqtt_broker ops_broker >/dev/null 2>&1
podman run -d --name mqtt_broker --network logiedge_net -p 127.0.0.1:1883:1883 \
  -v "$REPO/config/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro" docker.io/library/eclipse-mosquitto:2.0
podman run -d --name ops_broker --network logiedge_net -p 127.0.0.1:1884:1883 \
  -v "$REPO/config/ops_broker.conf:/mosquitto/config/mosquitto.conf:ro" docker.io/library/eclipse-mosquitto:2.0

podman container exists registry || podman run -d --name registry -p 127.0.0.1:5000:5000 docker.io/library/registry:2
podman start registry >/dev/null

podman ps --format '{{.Names}}  {{.Status}}  {{.Ports}}'
echo "Setup done."
