#!/usr/bin/env bash

set -e

if command -v tput >/dev/null 2>&1 && [ -n "$(tput colors 2>/dev/null)" ]; then
  RED="$(tput setaf 1)"
  GREEN="$(tput setaf 2)"
  YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"
  BOLD="$(tput bold)"
  RESET="$(tput sgr0)"
else
  RED=""
  GREEN=""
  YELLOW=""
  BLUE=""
  BOLD=""
  RESET=""
fi

usage() {
  echo "${BOLD}Usage:${RESET} $0 {${GREEN}prepare${RESET}|${BLUE}train${RESET}|${YELLOW}infer${RESET}}"
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

COMMAND="$1"
shift

# Show the full command we're about to run
# (include any extra args passed after the subcommand)
FULL_CMD=(uv run)
case "$COMMAND" in
  prepare) FULL_CMD+=(src/prepare.py "$@") ;;
  train)   FULL_CMD+=(src/train.py "$@") ;;
  infer)   FULL_CMD+=(src/infer.py "$@") ;;
  *)
    echo "${RED}Unknown command:${RESET} $COMMAND"
    usage
    exit 1
    ;;
esac

echo "${BOLD}${BLUE}➜${RESET} ${BOLD}Running:${RESET} ${GREEN}${FULL_CMD[*]}${RESET}"

# Execute the selected command
case "$COMMAND" in
  prepare)
    uv run src/prepare.py "$@"
    ;;

  train)
    uv run src/train.py "$@"
    ;;

  infer)
    uv run src/infer.py "$@"
    ;;
esac
