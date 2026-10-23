#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
bash scripts/build.sh
mkdir -p build/test-classes
find tests -name '*.java' -print | sort > build/test-sources.txt
"$JAVAC" -encoding UTF-8 -source 8 -target 8 -cp "$TETRIS_CP" -d build/test-classes @build/test-sources.txt
"$JAVA" -Xmx512m -Djava.awt.headless=true -ea -cp "$TETRIS_CP:build/test-classes" TetrisQAgentTest
