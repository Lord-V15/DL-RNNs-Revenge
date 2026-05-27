## RNN's Revenge: Comparing Recurrent, Attentional, and MLP-Mixing Paradigms

  ▎ COMP6242 Deep Learning · Australian National University · Semester 1, 2026
  ▎ Mayukh Das, Vibhansh Gupta, Daniel, Adam

  Overview

  Feng et al. (2024) argue that recurrent neural networks were displaced from sequence modelling not
  because of inadequate modelling power but because they could not be trained in parallel. Their
  minGRU, a minimal recurrent model trainable via prefix scan, matches Transformer perplexity on
  TinyShakespeare.

  We test this claim and ask: where does a third paradigm — MLP-mixing — fit alongside recurrence and
  attention?

  We compare three token-mixing paradigms at a matched parameter budget of ~750K:
  - minGRU (Feng et al. 2024) — recurrent
  - Transformer (decoder-only, RoPE + RMSNorm) — attention-based
  - Causal gMLP — MLP-mixing with causally-masked Spatial Gating Unit

  All three are evaluated on three autoregressive character-level tasks, each at three sequence
  lengths:
  1. TinyShakespeare — natural English text
  2. Long-range copy — synthetic retrieval requiring exact recall after long distractor spans
  3. Induction — synthetic in-context pattern continuation (Olsson et al. 2022)

  Research Question

  How do recurrence, attention, and MLP-mixing compare on autoregressive character-level language
  modelling at matched parameters, when tested on tasks designed to discriminate between paradigm
  strengths and across varying sequence lengths?

  At ~750K parameters and equivalent training budgets, do these paradigms produce different perplexity
  on tasks emphasizing different capabilities? How does the gap change as sequence length grows from
  short (~128 chars) to long (~2048 chars)?

  Hypotheses
<img width="1094" height="437" alt="image" src="https://github.com/user-attachments/assets/c23da51c-890d-4c90-a735-8a9c52f7beec" />


  Contributions

  1. Reproduction — Verified minGRU matches Transformer on TinyShakespeare: PPL 4.63 ± 0.03 (paper
  reports ≈4.70) ✅
  2. Paradigm comparison — First head-to-head comparison of recurrence, attention, and MLP-mixing at
  matched parameters on these three tasks
  3. Causal gMLP — Autoregressive adaptation of gMLP (Liu et al. 2021) via causally-masked Spatial
  Gating Unit

  Models
<img width="1109" height="290" alt="image" src="https://github.com/user-attachments/assets/091d705a-9e1a-4b26-bca7-f13428e4c4b0" />


  Shared training: AdamW, lr 3e-4 → 3e-5 cosine, batch 64, 5K steps, gradient clip 1.0, dropout 0.2,
  bf16

  Tasks & Datasets

  Task 4.1: TinyShakespeare (Natural Language Baseline)

  Standard character-level LM on 1.1M characters (Karpathy 2022)
  - Lengths: 256, 1024, 2048 context
  - Metric: Best validation perplexity
  - Vocabulary: ~65 unique characters

  Task 4.2: Long-Range Copy (Retrieval Over Distance)

  Format: key: XXXXX | <N distractors> | recall: XXXXX

  - Lengths: 100/500/2000 distractor chars → ~115/~515/~2015 total chars
  - Vocabulary: 55 chars (26 lower + 26 upper + space + : + |)
  - Key metric: PPL at the 5 recall positions (isolates retrieval capability)
  - Dataset: 10K train + 1K val per length = 33K sequences

  Task 4.3: Induction (In-Context Pattern Continuation)

  Format: <M random> <5-char pattern> <M random> <pattern[:4]>

  - Lengths: M=50/200/1000 per side → ~110/~410/~2010 total chars
  - Vocabulary: 27 chars (26 lowercase + space)
  - Key metrics: Pattern-completion PPL & accuracy at final 5 positions
  - Dataset: 10K train + 1K val per length = 33K sequences

  Project Structure

<img width="687" height="421" alt="image" src="https://github.com/user-attachments/assets/3a3f8fb8-1795-4997-9d52-0f9639aab38f" />

  Quick Start

  1. Generate Datasets (Already Done ✅)

  python3 generate_longrange_copy.py
  python3 generate_induction.py
  python3 verify_datasets.py

  2. Use in Training

  from dataset_utils import get_longrange_dataloader, get_induction_dataloader
  from evaluation_metrics import compute_longrange_metrics, compute_induction_metrics

  # Long-range copy
  train_dl, tokenizer = get_longrange_dataloader('short', 'train', batch_size=64)
  val_dl, _ = get_longrange_dataloader('short', 'val', batch_size=64)

  for batch in train_dl:
      logits = model(batch['input_ids'])
      loss = criterion(logits, batch['labels'])
      # ... training ...

  # Validation with discriminating metrics
  for batch in val_dl:
      logits = model(batch['input_ids'])
      metrics = compute_longrange_metrics(
          logits,
          batch['labels'],
          batch['recall_positions']
      )
      print(f"Recall PPL: {metrics['recall_ppl']:.2f}")  # Key metric!

  Run Matrix

  Primary: 27 cells (3 done, 24 to run)
<img width="1073" height="542" alt="image" src="https://github.com/user-attachments/assets/865d697e-06e6-435c-bc8f-e474f44f8f19" />

  Ablations: 4 runs at TinyShakespeare-256
  - Transformer − RoPE, − RMSNorm, − Depth (L=2)
  - Causal gMLP − Tied SGU

  Total: 31 unique runs (3 done, 28 to run)

  Evaluation Metrics

<img width="1103" height="384" alt="image" src="https://github.com/user-attachments/assets/fd9ce477-a646-4f86-8c70-87eb928a57bd" />


  Timeline (2 Weeks)
<img width="1112" height="494" alt="image" src="https://github.com/user-attachments/assets/93b25e94-f48e-45d9-9fbc-36d8cf0936a1" />


  Compute Budget

  Hardware: NVIDIA H100
  Total: ~10-13 H100-hours sequential, ~5-6 hours parallelized wall-clock

  Results (Preliminary)

  minGRU TinyShakespeare Baseline ✅
<img width="800" height="287" alt="image" src="https://github.com/user-attachments/assets/202a9cda-5e00-493f-966f-107bae0d391d" />


  Full results coming after smoke-test gate (Days 3-4).

  Dataset Statistics

  Total Sequences:      66,000 (33K per task)
  Training Sequences:   60,000 (30K per task)
  Validation Sequences:  6,000 (3K per task)
  Total Storage:        ~200 MB
  Generation Time:      ~5 seconds

  References

  - Feng et al. (2024). "Were RNNs All We Needed?" — minGRU architecture
  - Liu et al. (2021). "Pay Attention to MLPs" — gMLP architecture
  - Olsson et al. (2022). "In-context Learning and Induction Heads" — induction task design
  - Karpathy (2022). TinyShakespeare dataset
