# RNN's Revenge — minGRU experiments

minGRU owner's portion of the COMP6242 group project. See
`rnns_revenge_proposal.pdf` for the full proposal; this repo implements the
9-cell run matrix for the minGRU paradigm.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Verify the data layout
#    data/tinyshakespeare/  — auto-downloaded on first run
#    data/longrange_copy/   — train_{short,medium,long}.txt + val_{short,medium,long}.txt
#    data/induction/        — train_*.txt + val_*.txt + *_patterns.txt

# 3. Run one experiment
python scripts/train.py --exp mingru_tinyshakespeare_256

# 4. Run with a specific seed (e.g. for the 3-seed TS-256 baseline)
python scripts/train.py --exp mingru_tinyshakespeare_256 --seed 1337
python scripts/train.py --exp mingru_tinyshakespeare_256 --seed 2025

# 5. Resume a crashed run
python scripts/train.py --exp mingru_tinyshakespeare_256 --resume
```

## Layout

```
rnns_revenge/
├── configs/                  # TrainConfig defaults + per-experiment configs
│   ├── base.py
│   ├── model_mingru.py
│   └── experiments.py        # 9 named experiments — minGRU's share of the matrix
│
├── src/
│   ├── data/                 # task-specific dataset modules
│   │   ├── vocab.py          # char vocabularies
│   │   ├── tinyshakespeare.py
│   │   ├── longrange_copy.py # reads pre-generated .txt files; builds recall masks
│   │   └── induction.py      # reads .txt + *_patterns.txt; builds pattern masks
│   │
│   ├── models/
│   │   └── mingru.py         # Conv4 → minGRU → MLP block, log-space parallel scan
│   │
│   ├── training/
│   │   ├── trainer.py        # main train loop
│   │   ├── checkpointing.py  # best.pt + last.pt with optimizer/scheduler/RNG
│   │   ├── logging_utils.py  # metrics.csv + console.log + meta.txt
│   │   └── schedules.py      # linear warmup + cosine decay
│   │
│   ├── eval/
│   │   ├── metrics.py        # PPL, per-position PPL, pattern-completion accuracy
│   │   └── memory_throughput.py
│   │
│   └── utils/
│       ├── seed.py
│       └── param_count.py
│
├── scripts/
│   └── train.py              # CLI entry point
│
├── runs/                     # per-run output (checkpoints + logs); gitignored
└── data/                     # input data; gitignored
```

## Run matrix (minGRU owner's 9 cells)

| Task              | Length      | Experiment name                  |
|-------------------|-------------|----------------------------------|
| TinyShakespeare   | 256         | `mingru_tinyshakespeare_256`     |
| TinyShakespeare   | 1024        | `mingru_tinyshakespeare_1024`    |
| TinyShakespeare   | 2048        | `mingru_tinyshakespeare_2048`    |
| Long-range copy   | short (~115) | `mingru_longcopy_short`         |
| Long-range copy   | medium (~515)| `mingru_longcopy_medium`        |
| Long-range copy   | long (~2015) | `mingru_longcopy_long`          |
| Induction         | short (~110) | `mingru_induction_short`        |
| Induction         | medium (~410)| `mingru_induction_medium`       |
| Induction         | long (~2010) | `mingru_induction_long`         |

The TinyShakespeare-256 cell is run at 3 seeds (42, 1337, 2025) per proposal §3.1.
All other cells use seed 42 by default; override with `--seed N`.

## Output per run

Each run writes to `runs/<exp_name>__seed<N>__<timestamp>/`:

  * `checkpoints/best.pt` — lowest val loss seen
  * `checkpoints/last.pt` — for `--resume`
  * `metrics.csv`         — one row per eval event; columns include val_ppl,
                            masked_ppl, masked_acc, per-position breakdown,
                            tokens/s, peak GPU memory
  * `console.log`         — text mirror of stdout
  * `meta.txt`            — git SHA, hostname, full config, parameter count
  * `config.txt`          — clean copy of the config repr
