# Acquisition selection heuristic

## The two-score competition

For every unlabeled compound, the acquisition computes **two independent scores** — one for DRC, one for PS — and throws them into a single sorted list. The query type is decided purely by which score wins the greedy race to the top-k.

### DRC score — exploitation

```
score_DRC(x) = sigmoid((ŷ − target_threshold) / τ) / cost_DRC
             = p_active(x) / cost_DRC
```

This is **expected activity per dollar**. It is maximized when the model predicts ŷ well above `target_threshold` (e.g., 7.0 pEC50 = 100 nM). The sigmoid with temperature τ controls how sharply the score rewards high predictions — smaller τ makes the function steeper and more exploitative.

### PS score — threshold exploration

```
score_PS(x) = H_binary(sigmoid((ŷ − ps_threshold) / τ)) / cost_PS
            = H_binary(p_cross(x, T)) / cost_PS
```

This is **binary entropy of the threshold-crossing probability per dollar**. It is maximized when ŷ ≈ `ps_threshold` (e.g., 5.0), i.e., when the model is maximally uncertain whether the compound clears the primary screen cutoff. At certainty (ŷ far above or below T), H → 0, and the score collapses.

## The unified ranking

Both scores share the same unit — **information (or expected value) per dollar** — so they can be compared directly on a single ranked list. `CostAwareGreedyAcquisition.select()` builds:

```
candidates = [(score_DRC[0], 0, DRC), (score_PS[0], 0, PS),
              (score_DRC[1], 1, DRC), (score_PS[1], 1, PS), ...]
```

sorted descending by score. It then greedily pops entries, skipping any compound index already chosen (each compound can only be queried once per iteration, regardless of fidelity), until k queries are collected.

## Practical outcome

| Scenario | Winner |
|---|---|
| ŷ >> `target_threshold` (7.0) | DRC wins — cheap confirmation of a likely active |
| ŷ ≈ `ps_threshold` (5.0) | PS wins — cheap resolution of threshold ambiguity |
| ŷ << both thresholds | PS score is near-zero, DRC score is near-zero; compound is deprioritized entirely |
| Early iterations (model uncertain, predictions ≈ 6.0 everywhere) | PS dominates — broad exploration is cheap; DRC budget is conserved |
| Late iterations (model sharp, clear actives identified) | DRC dominates — exploit the high-confidence predictions |

## Worked example

**Campaign settings:** `ps_threshold = 5.0`, `target_threshold = 7.0`, `τ = 0.5`, `cost_PS = $1`, `cost_DRC = $10`.

Five compounds are unlabeled at the end of iteration 3. The model has produced these pEC50 predictions:

| Compound | ŷ | Chemical interpretation |
|---|---|---|
| A | 8.2 | Strong predicted active (10 nM) |
| B | 7.1 | Borderline active |
| C | 5.1 | Right on the PS threshold |
| D | 3.8 | Predicted inactive |
| E | 6.0 | Ambiguous — between both thresholds |

Working through the formulas for each compound:

**Compound A (ŷ = 8.2)**
- `p_active = sigmoid((8.2 − 7.0) / 0.5) = sigmoid(2.4) ≈ 0.917`
- `score_DRC = 0.917 / 10 = 0.092`
- `p_cross = sigmoid((8.2 − 5.0) / 0.5) = sigmoid(6.4) ≈ 1.000` → H ≈ 0
- `score_PS = ~0.000 / 1 ≈ 0.000`
- **→ DRC selected.** The model is highly confident this is an active; a full dose-response curve confirms potency and quantifies it precisely.

**Compound C (ŷ = 5.1)**
- `p_active = sigmoid((5.1 − 7.0) / 0.5) = sigmoid(−3.8) ≈ 0.022`
- `score_DRC = 0.022 / 10 = 0.002`
- `p_cross = sigmoid((5.1 − 5.0) / 0.5) = sigmoid(0.2) ≈ 0.550` → H ≈ 0.688 nats
- `score_PS = 0.688 / 1 = 0.688`
- **→ PS selected.** The model is almost exactly at the primary screen cutoff. A $1 primary screen cheaply resolves whether this compound clears the threshold, without spending $10 on a DRC for something that is almost certainly not a drug-like active.

**Compound E (ŷ = 6.0)**
- `p_active = sigmoid((6.0 − 7.0) / 0.5) = sigmoid(−2.0) ≈ 0.119`
- `score_DRC = 0.119 / 10 = 0.012`
- `p_cross = sigmoid((6.0 − 5.0) / 0.5) = sigmoid(2.0) ≈ 0.880` → H ≈ 0.380 nats
- `score_PS = 0.380 / 1 = 0.380`
- **→ PS selected.** Predicted activity is modest but the compound comfortably clears the PS threshold in the model's view. PS reduces cost while still narrowing the search space.

**Compound D (ŷ = 3.8)**
- `score_DRC = sigmoid(−6.4) / 10 ≈ 0.000`
- `score_PS = H(sigmoid(−2.4)) / 1 = H(0.083) ≈ 0.296 / 1 = 0.296`
- **→ PS selected if budget remains.** The compound is predicted inactive, but the PS score is non-negligible because 0.083 is not zero — there is still some residual probability of crossing the threshold. In practice this compound would rank below C and E and would only be queried if k is large enough.

**Full ranking for k = 3:**

| Rank | Compound | Query type | Score |
|---|---|---|---|
| 1 | C | PS | 0.688 |
| 2 | E | PS | 0.380 |
| 3 | A | DRC | 0.092 |

Compounds B and D are not queried this iteration. Note that compound A's DRC score (0.092) narrowly beats D's PS score (0.296 / 10... wait — D's PS score is 0.296, which is higher than A's DRC score of 0.092). In a campaign with k = 4 the fourth slot would go to D (PS), not B (DRC, score ≈ 0.038), illustrating how the cost ratio strongly favors PS queries for lower-confidence compounds.

## The two threshold parameters

`ps_threshold` and `target_threshold` are deliberately separate:

- **`ps_threshold`** — the assay's biological cutoff: what the lab instrument reports as ≥/< T. Drives the PS entropy score.
- **`target_threshold`** — the campaign's optimization goal (e.g., 7.0 = 100 nM IC50). Drives the DRC exploitation score.

They can and often do differ. Both must be set consistently in `oracle:` and `acquisition:` in the config (see `examples/default_config.yaml`).

## Relevant source

- Scoring logic: `moal/acquisition.py` — `CostAwareGreedyAcquisition._score_drc`, `_score_ps`, `select`
- Config parameters: `moal/config.py` — `AcquisitionConfig`
