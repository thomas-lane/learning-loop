#!/usr/bin/env bash
# Convenience wrapper: run from anywhere, extra args go to `harbor run`.
#   evaluation/run.sh                         # all tasks, config defaults
#   evaluation/run.sh -p evaluation/tasks/log-triage -k 3   # one task, 3 attempts
#   evaluation/run.sh -m some/other-model --ak api_base=http://gpu-box:8000/v1
# The agent imports `evaluation.*` and `learning_loop` (src/), so both go on PYTHONPATH.
# Prefers the project's virtualenv Harbor (same pinned version) when it exists.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
HARBOR=harbor
[ -x .venv/bin/harbor ] && HARBOR=.venv/bin/harbor
exec "$HARBOR" run -c evaluation/configs/local-llama.yaml -y "$@"
