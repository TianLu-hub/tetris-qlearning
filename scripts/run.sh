#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ ! -f build/classes/src/pas/tetris/tools/HeadlessRunner.class ]]; then
  bash scripts/build.sh
fi
exec "$JAVA" -Xmx1g -Djava.awt.headless=true -cp "$TETRIS_CP" src.pas.tetris.tools.HeadlessRunner "$@"
