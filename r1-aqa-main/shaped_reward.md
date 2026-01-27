## CGPR (Confidence-Gated Process Rewards) — Summary

### Core Idea

Apply dense token-level rewards **only to entity/bias-list tokens** where correctness is unambiguously verifiable, while using sparse sequence-level rewards for everything else. Weight rewards by confidence to simultaneously improve calibration.

---

### The Problem It Solves

| Challenge | How CGPR Addresses It |
|-----------|----------------------|
| **Sparse reward problem** | Adds dense signal at entity positions |
| **"What's a step?" problem** | Entity boundaries = natural verification points |
| **Reward hacking risk** | Dense rewards only where ground truth is checkable |
| **Calibration degradation** | Confidence weighting penalizes overconfident errors |

---

### Reward Structure

```
r_t = α · (1 - confidence)    if entity token AND correct
r_t = -β · confidence         if entity token AND incorrect  
r_t = 0                       if non-entity token

r_final = -WER - λ·B-WER      at sequence end (terminal reward)
```

**Intuition:**
- **Correct + uncertain → reward** (model should learn this is right)
- **Incorrect + confident → strong penalty** (dangerous mistakes)
- **Incorrect + uncertain → weak penalty** (model already knew it was unsure)
- **Non-entity → no dense signal** (can't verify, avoid hacking)

---

### Why Entity Tokens Are Special

| Property | Entity Tokens | Regular Tokens |
|----------|---------------|----------------|
| Verifiable? | ✅ Match bias list or don't | ❌ No clear criterion |
| High stakes? | ✅ Names, terms matter | Medium |
| Learnable signal? | ✅ Clear right/wrong | Ambiguous |

The bias list $\mathcal{B}$ provides **a priori knowledge** of what matters—no need to know the answer to decide where to apply dense rewards.

---

### Algorithm Sketch

```
for each token t in hypothesis:
    if token ∈ bias_list:
        confidence = 1 - TsallisEntropy(logits[t])
        correct = (token == reference[align(t)])
        
        if correct:
            r[t] = α * (1 - confidence)   # Reward uncertain correctness
        else:
            r[t] = -β * confidence        # Penalize confident errors
    else:
        r[t] = 0  # No dense reward

# Add terminal reward at end
r[T] += -WER - λ_entity * B-WER
```

---

### Key Design Choices

| Choice | Recommendation | Rationale |
|--------|----------------|-----------|
| Confidence metric | Tsallis entropy (q=1/3) | 4× better error detection than raw probs |
| α (correct coefficient) | 0.1 | Modest reward for uncertain correct |
| β (incorrect coefficient) | 0.2 | Stronger penalty for confident wrong |
| Entity weight λ | 4.0 | Emphasize entity errors in terminal |

---

### Theoretical Properties

1. **Additive to terminal reward** — Doesn't break PBRS like correctness-gated ABC would
2. **Sparse by design** — Only ~10-20% of tokens get dense signal (entities)
3. **Self-regularizing** — Confidence weighting discourages overconfident predictions
4. **Verifiable positions only** — Sidesteps reward hacking on non-verifiable tokens

---

### Comparison to Alternatives

| Method | Dense Signal | Verifiable | PBRS Safe | Risk |
|--------|-------------|------------|-----------|------|
| Terminal WER only | None | ✅ | ✅ | Credit assignment |
| Full token-level rewards | All tokens | ❌ | ❌ | Reward hacking |
| ABC (attention redistribution) | All tokens | ❌ | ✅ | Modest gains |
| **CGPR** | Entity tokens | ✅ | ✅ | Low |

---

### One-Sentence Summary

> **CGPR gives dense RL rewards only to entity tokens—where we can verify correctness from the bias list—weighted by inverse confidence to learn from uncertain successes and punish confident failures.**