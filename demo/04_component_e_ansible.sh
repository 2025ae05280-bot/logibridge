#!/bin/bash
# Segment 4: Component E2 Ansible deployment, run twice (second run must report changed=0).
cd "$(dirname "$0")/.."
source demo/env.sh
source demo/lib.sh
source demo/stack.sh

PLAY="ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml \
  -e runtime_dir=$HOME/logibridge-opt -e runtime_owner=$(id -u) -e runtime_group=$(id -g) \
  -e ansible_become=false -e ansible_python_interpreter=$(command -v $PY)"

step "The playbook has exactly 7 tasks" 'grep -E "^    - name:" deployment/logibridge_deploy.yml'
step "Push the image to the local registry" 'podman tag logibridge-inference:v1.0.1 localhost:5000/logibridge-inference:latest && podman push --tls-verify=false localhost:5000/logibridge-inference:latest'
step "Fresh truck: no container, empty runtime folder (simulator keeps running)" 'podman rm -f inference_engine >/dev/null; rm -rf "$HOME/logibridge-opt"; sim_start none 10'
step "Ansible run 1: deploys, so it reports changes" '$PLAY'
step "Ansible run 2, nothing changed in between: expect changed=0" '$PLAY'
step "The container Ansible started" 'podman ps --filter name=inference_engine --format "{{.Names}}  {{.Image}}  {{.Status}}"'
say "Segment 4 done."
