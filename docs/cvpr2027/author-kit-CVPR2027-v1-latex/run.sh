#!/usr/bin/env bash
# One-click compile for the CVPR 2027 paper (cvpr.sty + pdflatex).
# Usage: ./run.sh [main-file-basename]   (default: main)
set -euo pipefail

cd "$(dirname "$0")"

MAIN="${1:-main}"

# Deep clean (removes pdf + aux), then full build:
# pdflatex + bibtex + as many reruns as needed, handled by latexmk.
latexmk -C "${MAIN}.tex"
latexmk -pdf "${MAIN}.tex"

echo "Done: ${MAIN}.pdf"
