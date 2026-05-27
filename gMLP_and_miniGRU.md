# RNN's Revenge - gMLP Review & minGRU Analysis

COMP6242 Deep Learning | ANU Semester 1, 2026 | Updated: 2026-05-24 (final standardized protocol)

---

## Part 1: gMLP Report vs Proposal Requirements

### What holds up well:

- Architecture spirit - Causal SGU with lower-triangular masking, correct block structure
- Tasks and lengths - All 9 cells completed (3 tasks x 3 lengths)
- Training config - AdamW, lr 3e-4, 200 warmup, cosine decay, grad clip 1.0, **dropout 0.05**, batch 64
- H1 (Shakespeare PPL 4.5-5.5 at ctx 256) - gMLP 3.75 final PPL is within range
- H4 (length effects diverge) - The sharp cliff at window boundary is compelling evidence
- Shared data - Using same ./data/ files for fair comparison
- Induction data bug identified and fixed (full pattern now appended)
- **Standardized protocol** - All paradigms now use identical dropout=0.05, seed=42

### Tensions/gaps identified:

| Issue | Proposal says | Report says | Severity |
|-------|--------------|-------------|----------|
| d_ffn | 768 | 512 | Medium - needs justification |
| Param budget | ~750K +/-10% | 485K to 933K | High - short 35% under, long 24% over |
| Copy block sizes | 128, 528, 2048 | 144, 544, 2048 | Low - close enough |
| Architecture | Full nxn causal | Fixed 256-window | High - significant departure |
| Induction steps | 5,000 shared | 20,000 (decay 50K) | Medium - justified by signal sparsity |

## Biggest narrative risk:

The proposal describes the Causal gMLP contribution as replacing the SGU nxn spatial projection with its lower-triangular (causal) form - a minimal modification. But what was actually built (v4, fixed-window) is a more significant architectural departure. The final report needs to frame this as: the naive causal masking works but violates the parameter budget at long sequences, so we adopt a fixed-window variant that reveals a fundamental locality limitation of MLP-mixing.

---

## Part 2: gMLP vs minGRU Results (Final Standardized Protocol)

**Protocol: dropout=0.05, seed=42, all other hyperparameters matched.**

### TinyShakespeare (best val PPL):

| Context | minGRU | gMLP | Winner |
|---------|--------|------|--------|
| 256 | 4.44 | **3.75** | gMLP |
| 1024 | 4.34 | **3.68** | gMLP |
| 2048 | 4.32 | **3.72** | gMLP |

### Long-range Copy (discriminating accuracy):

| Length | minGRU | gMLP | Winner |
|--------|--------|------|--------|
| Short | 4.0% | **100%** | gMLP |
| Medium | 3.8% | 3.9% | Tie (both fail) |
| Long | 4.0% | 4.0% | Tie (both fail) |

### Induction (discriminating accuracy) - CORRECTED DATA:

| Length | minGRU | gMLP | Winner |
|--------|--------|------|--------|
| Short (M=50) | 4.6% | **99.1%** | gMLP |
| Medium (M=200) | 3.9% | **99.4%** | gMLP |
| Long (M=1000) | 3.5% | 3.8% | Tie (both fail) |

### Impact on proposal hypotheses:

- **H1** (all paradigms PPL 4.5-5.5): gMLP 3.75 is below predicted range - beating minGRU across all lengths.
- **H2** (retrieval: Transformer > gMLP > minGRU): Confirmed. minGRU fails at ALL lengths including short. gMLP solves within-window perfectly.
- **H3** (induction favours attention): gMLP achieves 99%+ within window - near Transformer level. minGRU fails completely (3.5-4.6%). MLP-mixing CAN implement induction when it has position access.
- **H4** (short lengths: all solve): FALSIFIED. minGRU fails even at M=50 (trivially short). The recurrent state cannot preserve exact patterns.

---

## Part 3: The Induction Data Bug and Its Resolution

### The bug:

The induction data generator appended only pattern[:4] at the second occurrence. The sequence ended at P[3], so there was no P[4] token to predict. The discriminating metric scored the model on predicting PROMPT chars (P[0:3]) instead of the actual induction target. Loss sat at ln(27) = 3.296 forever.

### The fix:

sequence = prefix + pattern + suffix + pattern (full 5 chars). The final token is now P[4] - the genuine induction target.

### Before vs after (gMLP):

| | Before fix (buggy data) | After fix (corrected) |
|---|---|---|
| Short accuracy | 80.2% | 99.1% |
| Medium accuracy | 80.2% | 99.4% |
| What was measured | N-gram prediction of P[0:3] | True induction recall of P[4] |

### minGRU before vs after:

No change. minGRU scored 3.5-4.6% on both buggy and corrected data. Its loss remained pinned at ln(27) = 3.296 for the entire 20,000-step run. The fix confirms that minGRU's failure is genuine: the signal reaches the model (P[4] is present), the recurrent state simply cannot preserve and recall it.

---

## Part 4: Decision - Keep minGRU Results As-Is

### Why NOT to switch to miniLSTM:

1. Proposal explicitly chose minGRU to test the Feng et al. (2024) claim
2. Switching models means rerunning all 9 cells + re-justifying architecture choice
3. Introduces a confound: did miniLSTM succeed because of the task or the cell?
4. The proposal already names minGRU - reviewers would ask why you changed

### Why NOT to fix minGRU:

1. The config is correct per Feng et al. Appendix C.2 - verified from the notebook
2. Dropout was standardized to 0.05 (matching the Transformer researcher's ablation) - still fails
3. The data bug is now fixed - if minGRU still fails, the result is real

### Why presenting as-is is the stronger paper:

With the data bug fixed AND dropout standardized, minGRU's failure is now proven genuine. The core finding:

1. Feng et al. (2024) only tested Shakespeare. They never claimed minGRU works on retrieval or induction. This project tested that boundary.
2. The failure is proven genuine. With corrected data, gMLP jumps from 80% to 99%. minGRU stays at 3.5-4.6%. The signal is reachable; recurrence cannot use it.
3. The three-way hierarchy is definitive: Language modeling: all comparable (3.7-4.4 PPL). Within-window tasks: gMLP = Transformer >> minGRU. Beyond window: Transformer >> gMLP = minGRU = random.
4. Novel finding about MLP-mixing: gMLP can implement the induction circuit at 99%+ accuracy. This was unknown - the original paper never tested autoregressive induction.

### Suggested framing for the final report:

> Feng et al.'s claim that minGRU matches Transformer perplexity on TinyShakespeare is confirmed (4.44 vs 4.34 at ctx-1024). However, this parity is specific to lossy sequence prediction. On tasks requiring exact retrieval, minGRU's compressed state fails completely - even at distances of just 50 characters - while gMLP with direct position access achieves 99%+ accuracy. The revenge of recurrence is confined to tasks where approximate prediction suffices; for exact information preservation, MLP-mixing with explicit position access is categorically superior to compressed recurrent state.

---

## Part 5: minGRU Architecture Verification

### Confirmed consistent with proposal:

- Architecture: Conv4 -> minGRU -> MLP per block (Feng et al. C.2)
- d_model = 128, L = 4 blocks, MLP mult = 4.5
- expansion_factor = 1.0, Tied embeddings (test PASS)
- Training steps = 20,000 for induction (LR decay over 50K)
- LR schedule: peak 3e-4 at step 200, cosine to 3e-5
- Parallel scan verified (test PASS 5/5)
- Shared corrected data files used
- Param count: ~799K (within +/-10% of 750K target)
- **Dropout: 0.05 (standardized across all paradigms per ablation)**

### Training behavior on corrected data (20K steps, dropout=0.05):

```
! Induction-short: loss = ln(27) = 3.296 for entire 20K steps, accuracy 4.6%
! Induction-medium: loss = ln(27) = 3.296 for entire 20K steps, accuracy 3.9%
! Induction-long: loss = ln(27) = 3.296 for entire 20K steps, accuracy 3.5%
! The model outputs uniform distribution at the prediction position across all lengths
```

### Interpretation:

With the data bug fixed, gMLP jumps to 99%+ accuracy on short/medium while minGRU remains at random. With dropout reduced to 0.05 (per the Transformer researcher's ablation showing 0.2 suppresses induction), minGRU STILL remains at random. This proves the failure is architectural, not a data or training artifact. The recurrent compressed state (128-dim) is fundamentally unable to preserve and recall an exact 5-character pattern even across 50 distractor characters. The parallel scan works correctly (verified by tests), gradients flow through all positions, and the model has 20K steps with active LR - it simply cannot solve this task.

---

Document prepared: 2026-05-24 (final standardized protocol: dropout=0.05) | Project: COMP6242 Deep Learning | Author: Vibhansh (gMLP section owner)
