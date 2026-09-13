#!/bin/zsh
# Resume of upgrade-run.sh Stage 2 after the 2026-09-02 23:50 process loss. Only the three unfinished runs.
cd "$(dirname "$0")" || exit 1
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run() { log "RUN: $*"; python3 run.py --provider ollama "$@"; log "exit=$?"; }
log "=== RESUME (phi4-mini uninstructed, granite both arms) ==="
M=phi4-mini:3.8b; T=phi4-mini-3.8b
run --models $M --block A --n-runs 4 --no-instruction --out results-local-$T-noinstr.jsonl
python3 analyze.py results-local-$T-noinstr.jsonl analysis-local-$T-noinstr.json; ollama stop $M
M=granite3.3:2b; T=granite3.3-2b
run --models $M --block A --n-runs 4 --out results-local-$T.jsonl
python3 analyze.py results-local-$T.jsonl analysis-local-$T.json
run --models $M --block A --n-runs 4 --no-instruction --out results-local-$T-noinstr.jsonl
python3 analyze.py results-local-$T-noinstr.jsonl analysis-local-$T-noinstr.json; ollama stop $M
log "=== ALL DONE ==="
