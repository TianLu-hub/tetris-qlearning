#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
mkdir -p build/classes
find src -name '*.java' -print | sort > build/sources.txt
"$JAVAC" -encoding UTF-8 -source 8 -target 8 -cp 'lib/*' -d build/classes @build/sources.txt
echo 'Build passed.'
