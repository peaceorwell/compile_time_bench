#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo '{"async": true, "asyncTimeout": 300000}'

pip install --quiet --upgrade pip
pip install --quiet -r "$CLAUDE_PROJECT_DIR/requirements.txt"

echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"
