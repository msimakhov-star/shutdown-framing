#!/bin/zsh
# Three-blade screen on new lineages (paper sample sizes), then full block A on each.
# Sequential, one model loaded at a time. Log: upgrade-run.log
cd "$(dirname "$0")" || exit 1
mkdir -p logs
MODELS=(gemma3:4b llama3.2:3b phi4-mini:3.8b granite3.3:2b)
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
# Do not share the 16 GB with the sibling paper-2 rater (qwen3.5:9b); wait for it.
while pgrep -f "rate.py" >/dev/null; do sleep 60; done
run() { log "RUN: $*"; python3 run.py --provider ollama "$@"; log "exit=$?"; }

# Pull all models in the background (network) while earlier models run (CPU/GPU).
( for M in $MODELS; do ollama pull $M >> logs/pull-${M//:/-}.log 2>&1; done ) &
PULLER=$!

log "=== STAGE 1: screen (S1 96 restatements; S2 A1 40+40; S3 B1 40+40) ==="
for M in $MODELS; do
  T=${M//:/-}
  log "--- $M ---"
  until ollama list | grep -q "^$M"; do kill -0 $PULLER 2>/dev/null || break; sleep 30; done
  ollama list | grep -q "^$M" || { log "PULL FAILED $M, skipping"; continue; }
  log "pulled $M"
  run --models $M --mode manipulation --n-runs 12 --block A --out manipcheck-local-$T.jsonl
  run --models $M --mode manipulation --n-runs 12 --block B --out manipcheck-local-$T.jsonl
  python3 manipcheck.py manipcheck-local-$T.jsonl --out manipcheck-local-$T.json
  run --models $M --block A --conditions A1_bare --n-runs 4 --out results-local-$T-A1.jsonl
  run --models $M --block A --conditions A1_bare --n-runs 4 --no-instruction --out results-local-$T-A1-noinstr.jsonl
  run --models $M --block B --conditions B1_files_deleted --n-runs 4 --out results-local-$T-B1.jsonl
  run --models $M --block B --conditions B1_files_deleted --n-runs 4 --no-instruction --out results-local-$T-B1-noinstr.jsonl
  for f in A1 A1-noinstr B1 B1-noinstr; do
    python3 analyze.py results-local-$T-$f.jsonl analysis-local-$T-$f.json | grep -E "^\| $M|valid rows"
  done
  ollama stop $M
done

log "=== STAGE 2: full block A, instructed + uninstructed (240 + 240 per model) ==="
for M in $MODELS; do
  T=${M//:/-}
  ollama list | grep -q "^$M" || { log "SKIP $M (not pulled)"; continue; }
  log "--- $M ---"
  run --models $M --block A --n-runs 4 --out results-local-$T.jsonl
  python3 analyze.py results-local-$T.jsonl analysis-local-$T.json
  run --models $M --block A --n-runs 4 --no-instruction --out results-local-$T-noinstr.jsonl
  python3 analyze.py results-local-$T-noinstr.jsonl analysis-local-$T-noinstr.json
  ollama stop $M
done
log "=== ALL DONE ==="
