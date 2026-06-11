# Acquisition selection heuristic

## The three compound pools

The acquisition operates over **three mutually exclusive compound pools** at each iteration:

| Pool | Eligible queries |
|---|---|
| **Unlabeled** (never queried) | PS *or* DRC (first-pass choice) |
| **PS-INTERVAL-labeled** (screened, result ≥ ps_threshold) | DRC upgrade only |
| **DRC-labeled** (or PS-LEFT-labeled) | No further queries |

PS-LEFT compounds (pEC50 < `ps_threshold`) are confirmed inactive and are excluded from the upgrade pool. PS-INTERVAL compounds (pEC50 ≥ `ps_threshold`) are potentially active hits and feed directly into the DRC upgrade pool.

## The two-score competition

For every scorable compound, the acquisition computes **two independent scores** (one for DRC and one for PS) sharing the same unit (information or expected value per dollar).

### DRC score: exploitation

```
score_DRC(x) = sigmoid((ŷ − target_threshold) / τ) / cost_DRC
             = p_active(x) / cost_DRC
```

This is **expected activity per dollar**. It is maximized when the model predicts ŷ well above `target_threshold` (e.g., 7.0 pEC50 = 100 nM). The sigmoid with temperature τ controls how sharply the score rewards high predictions; smaller τ makes the function steeper and more exploitative.

### PS score: threshold exploration

```
score_PS(x) = H_binary(sigmoid((ŷ − ps_threshold) / τ)) / cost_PS
            = H_binary(p_cross(x, T)) / cost_PS
```

This is **binary entropy of the threshold-crossing probability per dollar**. It is maximized when ŷ ≈ `ps_threshold` (e.g., 5.0), i.e., when the model is maximally uncertain whether the compound clears the primary screen cutoff. At certainty (ŷ far above or below T), H → 0, and the score collapses.

## The unified ranking

Both scores share the same unit, so they can be compared on a single ranked list. `CostAwareGreedyAcquisition.select()` merges candidates from both pools:

```
# Unlabeled compounds: contribute a DRC *and* a PS candidate each
candidates = [(score_DRC[0], smi_0, DRC), (score_PS[0], smi_0, PS),
              (score_DRC[1], smi_1, DRC), (score_PS[1], smi_1, PS), ...]

# PS-INTERVAL compounds: contribute a DRC-upgrade candidate only
candidates += [(score_DRC[j], smi_j, DRC) for j in ps_labeled_smiles]
```

The list is sorted descending by score. The greedy procedure pops entries, skipping any SMILES already chosen (each compound is selected at most once regardless of which pool it came from), until k queries are collected.

## The PS → DRC upgrade path

When a compound from the PS-INTERVAL pool is selected, the oracle records a second `LabelRecord` for it (the full dose-response curve) while retaining the original PS record. Both records enter the training set:

- The **PS record** (INTERVAL-censored: pEC50 ≥ `ps_threshold`) constrains the lower bound.
- The **DRC record** (EXACT: measured pEC50) pins the precise value.

This mirrors the standard two-stage HTS funnel: broad primary screen first, full characterization only for hits that clear the threshold.

## Practical outcome

| Scenario | Winner |
|---|---|
| ŷ >> `target_threshold` (7.0) | DRC wins, cheap confirmation of a likely active |
| ŷ ≈ `ps_threshold` (5.0) | PS wins, cheap resolution of threshold ambiguity |
| ŷ << both thresholds | Both scores near-zero; compound deprioritized |
| PS-INTERVAL compound with ŷ >> `target_threshold` | DRC upgrade wins, confirmed hit promoted to full characterization |
| PS-INTERVAL compound with ŷ ≈ threshold | DRC upgrade still competes; score_DRC is low but no PS candidate is generated |
| Early iterations (model uncertain, predictions ≈ 6.0 everywhere) | PS dominates, broad exploration is cheap; DRC budget is conserved |
| Late iterations (model sharp, clear actives identified + PS hits accumulated) | DRC dominates, exploit high-confidence predictions and upgrade PS hits |

## Worked example

### First pass: selecting from the unlabeled pool

**Campaign settings:** `ps_threshold = 5.0`, `target_threshold = 7.0`, `τ = 0.5`, `cost_PS = $1`, `cost_DRC = $10`.

Five compounds are unlabeled at the end of iteration 3. The model has produced these pEC50 predictions:

| Compound | ŷ | Chemical interpretation |
|---|---|---|
| A | 8.2 | Strong predicted active (10 nM) |
| B | 7.1 | Borderline active |
| C | 5.1 | Right on the PS threshold |
| D | 3.8 | Predicted inactive |
| E | 6.0 | Ambiguous, between both thresholds |

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
- **→ PS selected if budget remains.** The compound is predicted inactive, but the PS score is non-negligible because 0.083 is not zero; there is still some residual probability of crossing the threshold. In practice this compound would rank below C and E and would only be queried if k is large enough.

**Full ranking for k = 3:**

| Rank | Compound | Query type | Score |
|---|---|---|---|
| 1 | C | PS | 0.688 |
| 2 | E | PS | 0.380 |
| 3 | A | DRC | 0.092 |

Compounds B and D are not queried this iteration. Note that D's PS score (0.296) is higher than compound A's DRC score (0.092). In a campaign with k = 4 the fourth slot would go to D (PS), not B (DRC, score ≈ 0.038), illustrating how the cost ratio strongly favors PS queries for lower-confidence compounds.

### Subsequent pass: DRC upgrade from the PS-INTERVAL pool

Suppose iteration 4 arrives. Compounds C and E returned INTERVAL PS results (pEC50 ≥ 5.0) and are now in the PS-labeled pool. Compound A has already been DRC-labeled. The remaining unlabeled pool has B and D.

The acquisition scores the combined set:

| Compound | Pool | ŷ | Eligible queries | score_DRC | score_PS |
|---|---|---|---|---|---|
| B | unlabeled | 7.1 | DRC or PS | 0.731 / 10 = 0.073 | H(0.976) / 1 ≈ 0.107 |
| D | unlabeled | 3.8 | DRC or PS | ~0 | 0.296 |
| C | PS-INTERVAL | 5.1 | DRC upgrade only | 0.002 | n/a |
| E | PS-INTERVAL | 6.0 | DRC upgrade only | 0.012 | n/a |

For k = 2 the candidates sort by score as follows:

| Rank | Candidate | Score |
|---|---|---|
| 1 | D (PS) | 0.296 |
| 2 | B (PS) | 0.107 |
| 3 | B (DRC) | 0.073 |
| 4 | E (DRC upgrade) | 0.012 |
| 5 | C (DRC upgrade) | 0.002 |

With k = 2: D gets a PS query, B gets a PS query (its DRC candidate is skipped since B was already chosen). The DRC upgrades for C and E would not fire until a later iteration with larger k or until B and D are exhausted from the unlabeled pool.

## The two threshold parameters

`ps_threshold` and `target_threshold` are deliberately separate:

- **`ps_threshold`**, the assay's biological cutoff: what the lab instrument reports as ≥/< T. Drives the PS entropy score and determines which PS results are INTERVAL-censored (upgrade-eligible) vs. LEFT-censored (confirmed inactive).
- **`target_threshold`**, the campaign's optimization goal (e.g., 7.0 = 100 nM IC50). Drives the DRC exploitation score, including DRC-upgrade scoring for PS-labeled compounds.

They can and often do differ. Both must be set consistently in `oracle:` and `acquisition:` in the config (see `examples/default_config.yaml`).

## Relevant source

- Scoring logic: `moal/acquisition.py`: `CostAwareGreedyAcquisition._score_drc`, `_score_ps`, `select`
- PS upgrade pool: `moal/oracle.py`: `CostAwareOracle.get_ps_labeled_smiles`
- Config parameters: `moal/config.py`: `AcquisitionConfig`

---

## Configuration parameters that modulate PS vs DRC

Seven parameters across three config sections determine which fidelity the acquisition selects for any given compound. They interact through the unified score formula, so no parameter can be tuned in isolation.

```
score_DRC(x) = sigmoid((ŷ − target_threshold) / τ) / cost_DRC
score_PS(x)  = H(sigmoid((ŷ − ps_threshold) / τ))  / cost_PS
```

---

### `acquisition.tau`: sigmoid temperature

**Config section:** `acquisition:`

**Mathematical effect.** τ sets the width of the transition zone around each threshold in both score functions. Smaller τ makes both sigmoids steeper (closer to step functions). Two effects occur simultaneously:

- The DRC sigmoid becomes a sharp spike at `target_threshold`: compounds clearly above it get `p_active → 1`, compounds clearly below get `p_active → 0`.
- The PS entropy collapses faster for compounds far from `ps_threshold`: as `p_cross → 1` rapidly when ŷ > `ps_threshold + a few τ`, entropy → 0 and `score_PS → 0`.

The net effect is that the DRC/PS crossover shifts downward as τ decreases:

| τ | DRC crossover ŷ (default thresholds) |
|---|---|
| 0.5 (default) | ≈ 7.35 |
| 0.3 | ≈ 6.7 |
| 0.2 | ≈ 6.2 |
| 0.1 | ≈ 5.5 |

**Practical effect.** τ = 0.5 is exploratory: many moderately active compounds score similarly and PS almost always wins. τ = 0.2 is exploitative: any compound predicted above ≈6.2 triggers DRC, the acquisition ignores the broad middle of the distribution, and budget concentrates on the predicted top of the activity range.

**Biological interpretation.** τ encodes your confidence in the model's rank-ordering. A large τ says "predictions are noisy; treat the whole predicted-active region as equally worth exploring with cheap PS queries." A small τ says "predictions are reliable; commit early to DRC for anything above the threshold and don't waste DRC budget on borderline cases."

---

### `acquisition.target_threshold`: DRC exploitation target

**Config section:** `acquisition:`

**Mathematical effect.** This is the centre of the DRC sigmoid. The score `sigmoid((ŷ − target_threshold) / τ)` is exactly 0.5 / `cost_DRC` when `ŷ = target_threshold`, rising towards `1.0 / cost_DRC` as ŷ exceeds it and falling toward 0 below it. Lowering `target_threshold` shifts the entire DRC score curve left, making more compounds score above the PS/DRC crossover. Raising it shifts it right, making DRC selection rarer.

**Practical effect.** This is distinct from `oracle.activity_threshold` (which governs evaluation). You can set `acquisition.target_threshold = 6.5` to drive DRC selection toward compounds predicted above 6.5 pEC50 while still only counting compounds above `oracle.activity_threshold = 7.0` as confirmed actives in the dashboard.

**Biological interpretation.** `target_threshold` expresses the minimum potency your campaign is willing to spend a DRC on. At 7.0 (100 nM IC50, the default) you are conservative: only commit a $10 DRC to compounds that look genuinely drug-like. Lowering it to 6.0 (1 µM IC50) says "a µM hit is still worth characterizing fully"; raising it to 8.0 (10 nM) says "only pursue compounds that already look highly potent."

---

### `acquisition.ps_threshold`: PS entropy peak

**Config section:** `acquisition:` (must mirror `oracle.ps_threshold`)

**Mathematical effect.** This is the centre of the PS entropy curve. Binary entropy `H(p_cross)` is maximized at `p_cross = 0.5`, which occurs when `ŷ = ps_threshold`. Any compound predicted near this value gets the maximum PS score of `ln(2) / cost_PS ≈ 0.693`. Compounds far above or below see entropy collapse toward zero. Raising `ps_threshold` shifts the PS peak into higher activity territory, increasing competition with the DRC score there. Lowering it shifts the peak into inactive territory, causing entropy to collapse for all compounds with meaningful predicted activity.

**Practical effect.** If `ps_threshold` is lowered from 5.0 to 3.5, then any compound with ŷ > 4.5 has `p_cross → 1`, `H → 0`, and `score_PS → 0`; DRC would dominate for the entire active region. This is a powerful but blunt lever. **Must be kept in sync** with `oracle.ps_threshold`, which controls the actual assay cutoff; mismatching them causes the acquisition to optimize for a threshold the oracle doesn't use.

**Biological interpretation.** `ps_threshold` is defined by the physical assay: it is typically the highest concentration run in the primary screen (e.g., 10 µM → pEC50 = 5.0). It is not freely tunable as a campaign knob without changing the experimental protocol. Treat it as fixed unless you are designing a new assay.

---

### `oracle.cost_ps` and `oracle.cost_drc`: assay costs

**Config sections:** `oracle:`

**Mathematical effect.** Both scores are divided by their respective costs: `score_DRC / cost_DRC` and `score_PS / cost_PS`. Only the **ratio** `cost_DRC / cost_PS` determines the crossover. With the default ratio of 10, the DRC score must exceed 10 × the PS score to win. The maximum DRC score is `1.0 / cost_DRC = 0.1`; the maximum PS score is `ln(2) / cost_PS ≈ 0.693`. Halving `cost_DRC` to 5 (ratio = 5:1) shifts the crossover from ŷ ≈ 7.35 to ŷ ≈ 6.7, the same gain as lowering τ from 0.5 to 0.3, but through a different mechanism.

**Practical effect.** `cost_drc / cost_ps` is the single most important ratio in the entire acquisition. Campaigns with cheap DRC assays (e.g., fluorescence-based, automated) should reflect that in config; campaigns where DRC is 50× more expensive than PS (manual patch-clamp, for example) will almost never select DRC until late iterations.

**Biological interpretation.** These should reflect real laboratory costs including reagent, instrument time, and analyst overhead. They need not be in dollars; any consistent unit works. The ratio is what matters. A 10:1 DRC:PS ratio is typical for biochemical HTS (PS = single-concentration fluorescence, DRC = 8- or 10-point concentration-response). A 50:1 ratio might reflect cellular assays (PS = viability screen, DRC = full mechanistic profiling).

---

### `oracle.activity_threshold`: confirmed active definition

**Config section:** `oracle:` (evaluation only, does not enter the score formula)

**Mathematical effect.** This parameter does **not** appear in either score equation; it has no effect on which queries are selected. It is used exclusively by `PipelineEvaluator._is_confirmed_active()` to decide whether a labeled record counts toward the "actives found" dashboard metric.

**Practical effect.** Raising `activity_threshold` makes "confirmed active" harder to achieve: a compound needs a high DRC-measured pEC50 to count. Lowering it inflates the confirmed-active count without changing what the campaign actually measures. The important constraint is that `activity_threshold ≥ ps_threshold`: if activity_threshold were below ps_threshold, a PS INTERVAL label could theoretically span an active but would still never confirm it (the INTERVAL lower bound is `ps_threshold`, not the true value). Only EXACT (DRC) labels can produce confirmed actives, which is a structural motivation for the PS → DRC upgrade path.

**Biological interpretation.** This encodes what "hit" means to your project: 100 nM IC50 (pEC50 = 7.0) is a common drug discovery threshold for a lead series. 10 µM (pEC50 = 5.0) is appropriate for a fragment hit. 10 nM (pEC50 = 8.0) suits a late-stage selectivity campaign.

---

### `model.noise_scale`: fast mode prediction noise (fast mode only)

**Config section:** `model:` (only active when `model.fast: true`)

**Mathematical effect.** This shifts each compound's prediction by a draw from `Uniform(−noise_scale, +noise_scale)`. Because DRC/PS selection depends on whether ŷ falls above or below the crossover point, noise creates a stochastic boundary: compounds with true pEC50 within `noise_scale` of the crossover flip between DRC and PS from iteration to iteration. The effective DRC-eligible pool shrinks to compounds where `true_pEC50 > crossover + noise_scale` (reliably above) and grows stochastically for the band `[crossover − noise_scale, crossover + noise_scale]`.

**Practical effect.** Lowering `noise_scale` from 0.5 to 0.1 stabilizes DRC selection for borderline compounds. With `τ = 0.2` (crossover ≈ 6.2) and `noise_scale = 0.1`, any compound with true pEC50 > 6.3 will reliably trigger DRC every iteration. With `noise_scale = 0.5`, the reliable threshold rises to 6.7. In fast mode the PS → DRC upgrade path fires naturally: INTERVAL PS hits accumulated in earlier iterations (where PS dominates) will compete as DRC-upgrade candidates in later iterations once the unlabeled pool shrinks.

**Biological interpretation.** `noise_scale` models the **intra-assay variability** of whatever rapid computational or experimental surrogate you are simulating. It has no direct biological meaning; it is a fast-mode tuning parameter. In a real campaign with CheMeleon, the equivalent uncertainty comes from model prediction error, which naturally decreases as the labeled pool grows.

---

### Crossover summary: how each parameter shifts the DRC/PS boundary

All else equal, DRC selection increases (and the upgrade path fires more readily) when:

| Change | Effect on crossover ŷ | Recommended when |
|---|---|---|
| Lower `tau` (e.g., 0.5 → 0.2) | Shifts down sharply | Model predictions are reliable; campaign should exploit |
| Lower `acquisition.target_threshold` (e.g., 7.0 → 6.0) | Shifts down modestly | Accepting lower-potency hits; broader DRC coverage desired |
| Lower `cost_drc` / raise `cost_ps` | Shifts down (same effect as changing ratio) | DRC assay is cheaper than assumed |
| Lower `ps_threshold` (e.g., 5.0 → 3.5) | Collapses PS for active region entirely | Rarely appropriate (changes assay biology) |
| Lower `noise_scale` (fast mode only) | Stabilizes existing crossover | Fast mode iteration stability |
| Run more iterations / larger k | PS-INTERVAL pool grows → more DRC upgrade candidates | Natural campaign progression |

