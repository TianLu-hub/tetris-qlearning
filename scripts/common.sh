#!/usr/bin/env bash
set -euo pipefail
TETRIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TETRIS_ROOT"
if [[ -n "${JAVA_HOME:-}" ]]; then
  JAVA="$JAVA_HOME/bin/java"
  JAVAC="$JAVA_HOME/bin/javac"
else
  JAVA="${JAVA:-java}"
  JAVAC="${JAVAC:-javac}"
fi
export JAVA JAVAC
TETRIS_CP="lib/*:build/classes"
if [[ ! -f lib/tetris-1.0.3.jar || ! -f lib/argparse4j-0.9.0.jar ]]; then
  echo 'Missing dependencies. Run: bash scripts/setup.sh /path/to/tetris-1.0.3.jar' >&2
  exit 1
fi
