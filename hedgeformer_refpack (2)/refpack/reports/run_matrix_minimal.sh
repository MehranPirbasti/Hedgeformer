#!/usr/bin/env bash
set -euo pipefail

if [ ! -f "./refpack_out/runs/Simple_f0.25_s42/DONE" ]; then
  echo "=== Simple_f0.25_s42"
  mkdir -p "./refpack_out/runs/Simple_f0.25_s42"
  python -m refpack.train_one \
    --attention-type Simple \
    --train-csv "./refpack_out/splits/train_frac0.25.csv" \
    --seed 42 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/Simple_f0.25_s42"
  touch "./refpack_out/runs/Simple_f0.25_s42/DONE"
fi

if [ ! -f "./refpack_out/runs/Simple_f0.25_s1337/DONE" ]; then
  echo "=== Simple_f0.25_s1337"
  mkdir -p "./refpack_out/runs/Simple_f0.25_s1337"
  python -m refpack.train_one \
    --attention-type Simple \
    --train-csv "./refpack_out/splits/train_frac0.25.csv" \
    --seed 1337 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/Simple_f0.25_s1337"
  touch "./refpack_out/runs/Simple_f0.25_s1337/DONE"
fi

if [ ! -f "./refpack_out/runs/Simple_f0.25_s2024/DONE" ]; then
  echo "=== Simple_f0.25_s2024"
  mkdir -p "./refpack_out/runs/Simple_f0.25_s2024"
  python -m refpack.train_one \
    --attention-type Simple \
    --train-csv "./refpack_out/splits/train_frac0.25.csv" \
    --seed 2024 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/Simple_f0.25_s2024"
  touch "./refpack_out/runs/Simple_f0.25_s2024/DONE"
fi

if [ ! -f "./refpack_out/runs/SRA_f0.25_s42/DONE" ]; then
  echo "=== SRA_f0.25_s42"
  mkdir -p "./refpack_out/runs/SRA_f0.25_s42"
  python -m refpack.train_one \
    --attention-type SRA \
    --train-csv "./refpack_out/splits/train_frac0.25.csv" \
    --seed 42 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/SRA_f0.25_s42"
  touch "./refpack_out/runs/SRA_f0.25_s42/DONE"
fi

if [ ! -f "./refpack_out/runs/SRA_f0.25_s1337/DONE" ]; then
  echo "=== SRA_f0.25_s1337"
  mkdir -p "./refpack_out/runs/SRA_f0.25_s1337"
  python -m refpack.train_one \
    --attention-type SRA \
    --train-csv "./refpack_out/splits/train_frac0.25.csv" \
    --seed 1337 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/SRA_f0.25_s1337"
  touch "./refpack_out/runs/SRA_f0.25_s1337/DONE"
fi

if [ ! -f "./refpack_out/runs/SRA_f0.25_s2024/DONE" ]; then
  echo "=== SRA_f0.25_s2024"
  mkdir -p "./refpack_out/runs/SRA_f0.25_s2024"
  python -m refpack.train_one \
    --attention-type SRA \
    --train-csv "./refpack_out/splits/train_frac0.25.csv" \
    --seed 2024 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/SRA_f0.25_s2024"
  touch "./refpack_out/runs/SRA_f0.25_s2024/DONE"
fi

if [ ! -f "./refpack_out/runs/TTT_f0.25_s42/DONE" ]; then
  echo "=== TTT_f0.25_s42"
  mkdir -p "./refpack_out/runs/TTT_f0.25_s42"
  python -m refpack.train_one \
    --attention-type TTT \
    --train-csv "./refpack_out/splits/train_frac0.25.csv" \
    --seed 42 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/TTT_f0.25_s42"
  touch "./refpack_out/runs/TTT_f0.25_s42/DONE"
fi

if [ ! -f "./refpack_out/runs/TTT_f0.25_s1337/DONE" ]; then
  echo "=== TTT_f0.25_s1337"
  mkdir -p "./refpack_out/runs/TTT_f0.25_s1337"
  python -m refpack.train_one \
    --attention-type TTT \
    --train-csv "./refpack_out/splits/train_frac0.25.csv" \
    --seed 1337 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/TTT_f0.25_s1337"
  touch "./refpack_out/runs/TTT_f0.25_s1337/DONE"
fi

if [ ! -f "./refpack_out/runs/TTT_f0.25_s2024/DONE" ]; then
  echo "=== TTT_f0.25_s2024"
  mkdir -p "./refpack_out/runs/TTT_f0.25_s2024"
  python -m refpack.train_one \
    --attention-type TTT \
    --train-csv "./refpack_out/splits/train_frac0.25.csv" \
    --seed 2024 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/TTT_f0.25_s2024"
  touch "./refpack_out/runs/TTT_f0.25_s2024/DONE"
fi

if [ ! -f "./refpack_out/runs/Simple_f1.00_s42/DONE" ]; then
  echo "=== Simple_f1.00_s42"
  mkdir -p "./refpack_out/runs/Simple_f1.00_s42"
  python -m refpack.train_one \
    --attention-type Simple \
    --train-csv "./refpack_out/splits/train_frac1.00.csv" \
    --seed 42 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/Simple_f1.00_s42"
  touch "./refpack_out/runs/Simple_f1.00_s42/DONE"
fi

if [ ! -f "./refpack_out/runs/Simple_f1.00_s1337/DONE" ]; then
  echo "=== Simple_f1.00_s1337"
  mkdir -p "./refpack_out/runs/Simple_f1.00_s1337"
  python -m refpack.train_one \
    --attention-type Simple \
    --train-csv "./refpack_out/splits/train_frac1.00.csv" \
    --seed 1337 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/Simple_f1.00_s1337"
  touch "./refpack_out/runs/Simple_f1.00_s1337/DONE"
fi

if [ ! -f "./refpack_out/runs/Simple_f1.00_s2024/DONE" ]; then
  echo "=== Simple_f1.00_s2024"
  mkdir -p "./refpack_out/runs/Simple_f1.00_s2024"
  python -m refpack.train_one \
    --attention-type Simple \
    --train-csv "./refpack_out/splits/train_frac1.00.csv" \
    --seed 2024 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/Simple_f1.00_s2024"
  touch "./refpack_out/runs/Simple_f1.00_s2024/DONE"
fi

if [ ! -f "./refpack_out/runs/SRA_f1.00_s42/DONE" ]; then
  echo "=== SRA_f1.00_s42"
  mkdir -p "./refpack_out/runs/SRA_f1.00_s42"
  python -m refpack.train_one \
    --attention-type SRA \
    --train-csv "./refpack_out/splits/train_frac1.00.csv" \
    --seed 42 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/SRA_f1.00_s42"
  touch "./refpack_out/runs/SRA_f1.00_s42/DONE"
fi

if [ ! -f "./refpack_out/runs/SRA_f1.00_s1337/DONE" ]; then
  echo "=== SRA_f1.00_s1337"
  mkdir -p "./refpack_out/runs/SRA_f1.00_s1337"
  python -m refpack.train_one \
    --attention-type SRA \
    --train-csv "./refpack_out/splits/train_frac1.00.csv" \
    --seed 1337 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/SRA_f1.00_s1337"
  touch "./refpack_out/runs/SRA_f1.00_s1337/DONE"
fi

if [ ! -f "./refpack_out/runs/SRA_f1.00_s2024/DONE" ]; then
  echo "=== SRA_f1.00_s2024"
  mkdir -p "./refpack_out/runs/SRA_f1.00_s2024"
  python -m refpack.train_one \
    --attention-type SRA \
    --train-csv "./refpack_out/splits/train_frac1.00.csv" \
    --seed 2024 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/SRA_f1.00_s2024"
  touch "./refpack_out/runs/SRA_f1.00_s2024/DONE"
fi

if [ ! -f "./refpack_out/runs/TTT_f1.00_s42/DONE" ]; then
  echo "=== TTT_f1.00_s42"
  mkdir -p "./refpack_out/runs/TTT_f1.00_s42"
  python -m refpack.train_one \
    --attention-type TTT \
    --train-csv "./refpack_out/splits/train_frac1.00.csv" \
    --seed 42 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/TTT_f1.00_s42"
  touch "./refpack_out/runs/TTT_f1.00_s42/DONE"
fi

if [ ! -f "./refpack_out/runs/TTT_f1.00_s1337/DONE" ]; then
  echo "=== TTT_f1.00_s1337"
  mkdir -p "./refpack_out/runs/TTT_f1.00_s1337"
  python -m refpack.train_one \
    --attention-type TTT \
    --train-csv "./refpack_out/splits/train_frac1.00.csv" \
    --seed 1337 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/TTT_f1.00_s1337"
  touch "./refpack_out/runs/TTT_f1.00_s1337/DONE"
fi

if [ ! -f "./refpack_out/runs/TTT_f1.00_s2024/DONE" ]; then
  echo "=== TTT_f1.00_s2024"
  mkdir -p "./refpack_out/runs/TTT_f1.00_s2024"
  python -m refpack.train_one \
    --attention-type TTT \
    --train-csv "./refpack_out/splits/train_frac1.00.csv" \
    --seed 2024 --epochs 100 \
    --select-on val_macro_relaxed_f1 \
    --out-dir "./refpack_out/runs/TTT_f1.00_s2024"
  touch "./refpack_out/runs/TTT_f1.00_s2024/DONE"
fi
