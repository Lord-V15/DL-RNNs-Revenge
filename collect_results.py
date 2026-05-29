"""
Paste this into a Colab cell after mounting your Drive / cloning the repo.
It reads all 27 experiment results and prints one table.

Usage:
  %cd /content/DL-RNNs-Revenge
  !python collect_results.py
"""

import json, glob, csv, os

print("=" * 90)
print("27-EXPERIMENT GRID: ALL RESULTS")
print("=" * 90)

# ─── Transformer ───
print("\n>>> TRANSFORMER (from transformer_project/out/*/summary.json)")
trans_files = sorted(glob.glob("transformer_project/out/*/summary.json"))
if not trans_files:
    print("  NOT FOUND. Check path.")
else:
    for p in trans_files:
        with open(p) as f:
            d = json.load(f)
        name = d.get("run_name", os.path.basename(os.path.dirname(p)))
        device = d.get("device", "?")
        iters = d.get("hyperparams", {}).get("max_iters", "?")
        ts = d.get("timestamp", "?")[:16]

        if "best_induction5_accuracy" in d:
            metric = f"induction5_acc = {d['best_induction5_accuracy']:.4f}"
        elif "best_recall_ppl" in d:
            metric = f"recall_ppl = {d['best_recall_ppl']:.4f}"
        else:
            metric = f"val_ppl = {d['best_val_ppl']:.4f}"

        print(f"  {name:<30} {metric:<30} [{device}, {iters} iters, {ts}]")

# ─── gMLP ───
print("\n>>> gMLP (from vibhansh-gMLP/results/gmlp_*.json)")
gmlp_files = sorted(glob.glob("vibhansh-gMLP/results/gmlp_*.json"))
if not gmlp_files:
    print("  NOT FOUND. Check path.")
else:
    for p in gmlp_files:
        with open(p) as f:
            d = json.load(f)
        name = os.path.basename(p).replace(".json", "")
        device = d.get("device", "?")
        iters = d.get("max_iters", d.get("hyperparams", {}).get("max_iters", "?"))
        ts = d.get("timestamp", "?")[:16] if d.get("timestamp") else "?"

        if "shakespeare" in name:
            metric = f"val_ppl = {d['best_val_ppl']:.4f}"
        elif "copy" in name:
            fr = d.get("final_results", {})
            rp = fr.get("discriminating_ppl", fr.get("recall_ppl", "N/A"))
            metric = f"recall_ppl = {rp:.4f}" if isinstance(rp, float) else f"recall_ppl = {rp}"
        elif "induction" in name:
            fr = d.get("final_results", {})
            acc = fr.get("induction5_accuracy", fr.get("accuracy", "N/A"))
            metric = f"induction5_acc = {acc:.4f}" if isinstance(acc, float) else f"accuracy = {acc}"
        else:
            metric = str(d.get("best_val_ppl", "?"))

        print(f"  {name:<35} {metric:<30} [{device}, {iters} iters]")

# ─── minGRU ───
print("\n>>> minGRU (from minGRU/runs/*/metrics.csv)")
mingru_dirs = sorted(glob.glob("minGRU/runs/*/metrics.csv"))
if not mingru_dirs:
    print("  NOT FOUND. Check path.")
else:
    for p in mingru_dirs:
        run_name = os.path.basename(os.path.dirname(p))
        with open(p) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"  {run_name}: EMPTY")
            continue

        best = min(rows, key=lambda r: float(r["val_loss"]))
        n_rows = len(rows)

        if "tinyshakespeare" in run_name:
            metric = f"val_ppl = {float(best['val_ppl']):.4f}"
        elif "longcopy" in run_name:
            metric = f"masked_ppl = {float(best['masked_ppl']):.4f}"
        elif "induction" in run_name:
            metric = f"masked_acc = {float(best['masked_acc']):.4f}"
        else:
            metric = f"val_ppl = {float(best['val_ppl']):.4f}"

        print(f"  {run_name:<55} {metric:<25} [{n_rows} evals]")

# ─── Summary Table ───
print("\n" + "=" * 90)
print("SUMMARY TABLE (for report)")
print("=" * 90)
print(f"{'Model':<12} {'Shak256':>8} {'Shak1024':>8} {'Shak2048':>8} "
      f"{'CpyShrt':>8} {'CpyMed':>8} {'CpyLng':>8} "
      f"{'IndShrt':>8} {'IndMed':>8} {'IndLng':>8}")
print("-" * 90)

# Transformer row
t = {}
for p in trans_files:
    with open(p) as f:
        d = json.load(f)
    t[d["run_name"]] = d

t_row = [
    f"{t['tshake-256-s42-d05']['best_val_ppl']:.2f}",
    f"{t['tshake-1024-s42-d05']['best_val_ppl']:.2f}",
    f"{t['tshake-2048-s42-d05']['best_val_ppl']:.2f}",
    f"{t['lrcopy-short-s42-d05']['best_recall_ppl']:.2f}",
    f"{t['lrcopy-medium-s42-d05']['best_recall_ppl']:.2f}",
    f"{t['lrcopy-long-s42-d05']['best_recall_ppl']:.2f}",
    f"{t['induction-short-s42-d05']['best_induction5_accuracy']:.2f}",
    f"{t['induction-medium-s42-d05']['best_induction5_accuracy']:.2f}",
    f"{t['induction-long-s42-d05']['best_induction5_accuracy']:.2f}",
]
print(f"{'Transformer':<12} " + " ".join(f"{v:>8}" for v in t_row))

# gMLP row
g = {}
for p in gmlp_files:
    with open(p) as f:
        d = json.load(f)
    name = os.path.basename(p).replace(".json", "")
    g[name] = d

g_row = [
    f"{g['gmlp_shakespeare_256_42']['best_val_ppl']:.2f}",
    f"{g['gmlp_shakespeare_1024_42']['best_val_ppl']:.2f}",
    f"{g['gmlp_shakespeare_2048_42']['best_val_ppl']:.2f}",
    f"{g['gmlp_copy_short_42'].get('final_results',{}).get('discriminating_ppl', 0):.2f}",
    f"{g['gmlp_copy_medium_42'].get('final_results',{}).get('discriminating_ppl', 0):.2f}",
    f"{g['gmlp_copy_long_42'].get('final_results',{}).get('discriminating_ppl', 0):.2f}",
    f"{g['gmlp_induction_short_42'].get('final_results',{}).get('induction5_accuracy', g['gmlp_induction_short_42'].get('final_results',{}).get('accuracy', 0)):.2f}",
    f"{g['gmlp_induction_medium_42'].get('final_results',{}).get('induction5_accuracy', g['gmlp_induction_medium_42'].get('final_results',{}).get('accuracy', 0)):.2f}",
    f"{g['gmlp_induction_long_42'].get('final_results',{}).get('induction5_accuracy', g['gmlp_induction_long_42'].get('final_results',{}).get('accuracy', 0)):.2f}",
]
print(f"{'gMLP':<12} " + " ".join(f"{v:>8}" for v in g_row))

# minGRU row
m = {}
for p in mingru_dirs:
    run_name = os.path.basename(os.path.dirname(p))
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if rows:
        best = min(rows, key=lambda r: float(r["val_loss"]))
        m[run_name] = best

def find_mingru(keyword):
    for name, data in m.items():
        if keyword in name:
            return data
    return None

m_shak = [find_mingru(f"tinyshakespeare_{bs}") for bs in [256, 1024, 2048]]
m_copy = [find_mingru(f"longcopy_{l}") for l in ["short", "medium", "long"]]
m_ind = [find_mingru(f"induction_{l}") for l in ["short", "medium", "long"]]

m_row = [
    f"{float(m_shak[0]['val_ppl']):.2f}" if m_shak[0] else "N/A",
    f"{float(m_shak[1]['val_ppl']):.2f}" if m_shak[1] else "N/A",
    f"{float(m_shak[2]['val_ppl']):.2f}" if m_shak[2] else "N/A",
    f"{float(m_copy[0]['masked_ppl']):.2f}" if m_copy[0] else "N/A",
    f"{float(m_copy[1]['masked_ppl']):.2f}" if m_copy[1] else "N/A",
    f"{float(m_copy[2]['masked_ppl']):.2f}" if m_copy[2] else "N/A",
    f"{float(m_ind[0]['masked_acc']):.2f}" if m_ind[0] else "N/A",
    f"{float(m_ind[1]['masked_acc']):.2f}" if m_ind[1] else "N/A",
    f"{float(m_ind[2]['masked_acc']):.2f}" if m_ind[2] else "N/A",
]
print(f"{'minGRU':<12} " + " ".join(f"{v:>8}" for v in m_row))

print("-" * 90)
print("Shakespeare/Copy: lower = better (perplexity). Induction: higher = better (accuracy).")
print("Copy perfect = 1.00, random ~ 26. Induction perfect = 1.00, random ~ 0.04.")
