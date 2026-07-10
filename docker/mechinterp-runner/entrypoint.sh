#!/usr/bin/env bash
# Prints one line of machine-readable runtime provenance, then execs the
# container command. Downstream logs can grep this line to capture exactly
# which image and package versions produced a run's artifacts.
set -euo pipefail

/opt/venv/bin/python /usr/local/bin/print_provenance.py

exec "$@"
