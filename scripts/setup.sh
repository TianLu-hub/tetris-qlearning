#!/usr/bin/env bash
set -euo pipefail
TETRIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TETRIS_ROOT"
mkdir -p lib
if [[ $# -gt 0 ]]; then
  if [[ ! -f "$1" ]]; then echo 'Engine JAR not found.' >&2; exit 1; fi
  if [[ "$1" != "$TETRIS_ROOT/lib/tetris-1.0.3.jar" && "$1" != "lib/tetris-1.0.3.jar" ]]; then
    cp "$1" lib/tetris-1.0.3.jar
  fi
fi
if [[ ! -f lib/tetris-1.0.3.jar ]]; then
  echo 'Provide your authorized course engine: bash scripts/setup.sh /path/to/tetris-1.0.3.jar' >&2
  exit 1
fi
expected='9eb3b54043038bc111bf0e15a8f69895244f92e3c2bc2800158578ca711f6117'
check_hash() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d ' ' -f 1
  else shasum -a 256 "$1" | cut -d ' ' -f 1; fi
}
if [[ ! -f lib/argparse4j-0.9.0.jar ]]; then
  tmp="$(mktemp lib/argparse4j.XXXXXX)"
  trap 'rm -f "$tmp"' EXIT
  curl --fail --location --silent --show-error 'https://repo.maven.apache.org/maven2/net/sourceforge/argparse4j/argparse4j/0.9.0/argparse4j-0.9.0.jar' -o "$tmp"
  if [[ "$(check_hash "$tmp")" != "$expected" ]]; then echo 'argparse4j checksum mismatch.' >&2; exit 1; fi
  mv "$tmp" lib/argparse4j-0.9.0.jar
fi
if [[ "$(check_hash lib/argparse4j-0.9.0.jar)" != "$expected" ]]; then echo 'argparse4j checksum mismatch.' >&2; exit 1; fi
echo 'Dependencies ready. The course engine stays local and is excluded from Git.'
