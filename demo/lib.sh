# Helper used by the segment scripts: show a command, wait for Enter, then run it.
# Set AUTO=1 to run without pausing (for a rehearsal).

step() {
  local title="$1"
  shift
  echo
  printf '\033[1;36m▶ %s\033[0m\n' "$title"
  printf '\033[0;33m$ %s\033[0m\n' "$*"
  if [ "${AUTO:-0}" != "1" ]; then
    read -r -p "  [Enter to run] " _
  fi
  eval "$@"
}

say() {
  echo
  printf '\033[1;32m%s\033[0m\n' "$*"
}
