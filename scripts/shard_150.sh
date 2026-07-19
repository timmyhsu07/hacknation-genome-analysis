#!/usr/bin/env bash
# Deterministically split the 150 sorted genome FASTAs into 30 batches of 5.
# Batch k (1-indexed NN) gets sorted genomes [5(k-1) .. 5(k-1)+4].
# Read-only with respect to sequence data: only symlinks are created, never copies
# (the volume sits at 98% capacity, so duplicating ~742MB of FASTA is not safe).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GENOMES=data/bvbrc_150/genomes
BATCHES=data/bvbrc_150/batches
PARTS=data/bvbrc_150/parts

# Portable across bash 3.2 (macOS system bash), which has no `mapfile`.
FILES=()
while IFS= read -r line; do
  FILES+=("$line")
done < <(cd "$GENOMES" && ls -1 *.fna | LC_ALL=C sort)
N=${#FILES[@]}
# The cohort is whatever survived the fetcher's QC gate, not a hardcoded 150:
# 562.7763 was dropped as a 3.48 Mbp / 16-contig assembly (too short for a
# viable E. coli), leaving 149. Batches stay contiguous chunks of 5 over the
# sorted list; the final batch simply carries the remainder. Never pad a batch
# with a duplicate or invented genome to hit a round number.
if [ "$N" -lt 1 ]; then
  echo "ABORT: no genomes found in $GENOMES" >&2
  exit 1
fi
echo "cohort size: $N genomes"

mkdir -p "$BATCHES"
for k in $(seq 0 29); do
  NN=$(printf '%02d' $((k + 1)))
  list="$BATCHES/batch-$NN.txt"
  : > "$list"
  indir="$PARTS/batch-$NN/in"
  mkdir -p "$indir"
  for j in $(seq 0 4); do
    idx=$((k * 5 + j))
    [ "$idx" -ge "$N" ] && break
    f="${FILES[$idx]}"
    echo "$f" >> "$list"
    ln -sf "$ROOT/$GENOMES/$f" "$indir/$f"
  done
done

echo "sharded $N genomes into 30 batches"
for k in $(seq 1 30); do
  NN=$(printf '%02d' "$k")
  printf 'batch-%s: %s\n' "$NN" "$(wc -l < "$BATCHES/batch-$NN.txt" | tr -d ' ')"
done
echo "total filenames across batch lists: $(cat "$BATCHES"/batch-*.txt | wc -l | tr -d ' ')"
echo "distinct filenames: $(cat "$BATCHES"/batch-*.txt | sort -u | wc -l | tr -d ' ')"
