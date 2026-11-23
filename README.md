# PaySim Fraud Detection

Machine-learning detection of fraudulent mobile-money transactions on the PaySim dataset, built as a Foundations of Data Science course project.

The project goes beyond "fit a classifier and report accuracy": it answers the questions that actually decide whether a fraud model is deployable — **how much each model overfits, which model is genuinely better and by a measurable margin, why an individual transaction was flagged, and whether the model is biased against any subgroup of transactions.**

---

## Contents

- [Highlights](#highlights)
- [Dataset](#dataset)
- [Method](#method)
- [Results](#results)
  - [Model comparison](#model-comparison-test-all-features)
  - [Ablation study](#ablation-study)
  - [Overfitting](#overfitting)
  - [Alert budget](#alert-budget)
  - [Bias audit](#bias-audit)
  - [Explanations](#explanations)
- [Notes and limitations](#notes-and-limitations)
- [Team](#team)

---

## Highlights

| | |
|---|---|
| **Data** | 6,362,620 transactions, 8,213 fraudulent (0.129%) |
| **Modelling subset** | 2,770,409 TRANSFER/CASH_OUT transactions, 0.296% fraud |
| **Models trained** | 33 (11 model variants × 3 feature-ablation settings) |
| **Best model** | Random Forest — test AUC-PR **1.0000**, MCC **0.9994**, recall **99.9%** |
| **Ablation finding** | Removing the balance features drops XGBoost from 0.9997 → 0.386 test AUC-PR; also removing hour drops it to 0.268 |
| **Overfitting** | RF 0.2% and XGBoost 0.3% relative SD across rolling time folds (limit 10%) |
| **Operational result** | At 250 alerts/day, RF catches 93.6% of frauds and 97.9% of fraud value |
| **Bias** | RF passes all 10 audited subgroups (recall ratio ≈ 1.0) |
| **Explainability** | SHAP-ranked reason codes; `errorBalanceOrig` is the dominant signal |

---

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) simulates mobile-money transactions from a real-world-inspired financial log. One CSV, ~470 MB.

| Property | Value |
|---|---|
| Rows | 6,362,620 |
| Fraudulent | 8,213 (0.129%) |
| Time steps | 743 hourly steps |
| Missing values / duplicates | none |
| Fraud transaction types | TRANSFER (4,097) and CASH_OUT (4,116) only |

**Columns:** `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.

Fraud never occurs in PAYMENT, CASH_IN or DEBIT, so those rows are legitimate by rule and excluded from modelling; results are also reported on the full dataset.

Download and place at:

```
data/raw/PS_20174392719_1491204439457_log.csv
```

---

## Method

### Feature engineering

Nine features are used:

| Feature | Meaning |
|---|---|
| `amount` | Transaction amount |
| `oldbalanceOrg`, `newbalanceOrig` | Sender balance before / after |
| `oldbalanceDest`, `newbalanceDest` | Receiver balance before / after |
| `errorBalanceOrig` | `newbalanceOrig + amount − oldbalanceOrg` — sender's accounting residual |
| `errorBalanceDest` | `oldbalanceDest + amount − newbalanceDest` — receiver's accounting residual |
| `hour` | `step % 24` |
| `is_transfer` | TRANSFER vs CASH_OUT |

`nameOrig` and `nameDest` are dropped (identifiers, not behaviour). `isFlaggedFraud` is kept only as a baseline rule, never as a feature. Raw `step` is used only for ordering and splitting.

### Chronological split (70 / 15 / 15 by row count)

Rows are ordered by `step` and cut at the end of the step where 70% and 85% of the filtered rows are reached:

| Split | Steps | Rows | Frauds | Fraud rate |
|---|---|---|---|---|
| Train | 1–323 | 1,951,895 | 3,643 | 0.19% |
| Validation | 324–377 | 410,437 | 560 | 0.14% |
| Test | 378–743 | 408,077 | 4,010 | 0.98% |

The fraud rate rises sharply over time (legitimate volume collapses after ~step 400 while fraud stays near 250/day), so a random split would leak future information. Scalers, class weights and medians are fit on **train only**; validation and test are never resampled.

### Feature ablation

Every model is trained in three settings:

1. **`all`** — all 9 features.
2. **`no_balance`** — drop the 4 balance columns and 2 error features.
3. **`no_balance_no_hour`** — also drop `hour` (night-time legitimate volume is near zero, so hour is a potential shortcut).

This isolates how much of the performance comes from the balance-bookkeeping signal versus temporal artefacts.

### Models

| Model | Role | Key settings |
|---|---|---|
| **Random Forest** | core | 300 trees, `max_depth` ∈ {10, 20}, `min_samples_leaf` ∈ {5, 20}, `balanced_subsample` |
| **XGBoost** | core | ≤ 500 trees, lr 0.05, depth 6, subsample/colsample 0.8, `scale_pos_weight` from train, early stopping on validation AUC-PR, GPU (`device="cuda"`) when available |
| **Logistic Regression** | core | `class_weight=balanced`, `C` ∈ {0.01, 0.1, 1, 10}, standardised inputs |
| **Isolation Forest** | core | 200 trees, contamination ≈ 0.003 (unsupervised baseline) |
| **Autoencoder** | extra | 16-8-4-8-16, trained on legitimate rows only, early stopping |
| **SMOTE variants** | extra | SMOTENC on training set only, frauds raised to 10% of legitimate rows |
| **Amount-weighted variants** | extra | fraud weight = class weight × min(amount / median fraud amount, 10), rescaled |

### Evaluation protocol

- **Primary metric:** AUC-PR (fraud is rare, so accuracy is meaningless — predicting "legitimate" always scores 99.87%).
- **Secondary:** MCC, precision, recall, F1, Recall@P90, alert rate, amount-weighted recall, AUC-ROC, running time.
- **Uncertainty:** 95% bootstrap intervals, 1,000 resamples.
- **Selection on validation only**, with three guards:
  1. **Overfit guard** — train→validation overfit ratio ≤ 10%, judged on the **upper bound** of its interval.
  2. **AUC-PR difference** — a winner is declared only if the bootstrap interval of the difference excludes 0. Differences below 0.1% relative count as ties.
  3. **Bias guard** — subgroup recall / false-alarm ratios within limits.
  
  Tie-breakers in order: Recall@250 alerts, money caught@250, Recall@P90, MCC, then runtime and explainability. The test set is used **once**, to confirm.
- **Alert budget** replaces a fixed threshold: each day is a 24-step window (test = 15 windows), and the model may raise K = 100 / 250 / 500 / 1,000 alerts per day.
- **Expected-loss ranking** — `calibrated p × amount` via isotonic regression on validation — is compared against pure score ranking.

---

## Results

### Model comparison (test, `all` features)

| Model | AUC-PR | AUC-ROC | Precision | Recall | MCC |
|---|---|---|---|---|---|
| **Random Forest** | **1.0000** | 1.0000 | 1.0000 | 0.9988 | **0.9994** |
| XGBoost | 0.9997 | 1.0000 | 0.9995 | 0.9880 | 0.9937 |
| Logistic Regression | 0.7683 | 0.9827 | 1.0000 | 0.2883 | 0.5350 |
| Isolation Forest | 0.1375 | 0.8850 | 0.2432 | 0.2217 | 0.2249 |

RF and XGBoost are statistically indistinguishable on AUC-PR; the selection rule picks Random Forest because of a 0.45% MCC edge at threshold level. XGBoost is the pragmatic choice: same accuracy, a fraction of the training time.

### Ablation study

Test AUC-PR as features are removed:

| Model | all | no_balance | no_balance_no_hour |
|---|---|---|---|
| Random Forest | 1.0000 | 0.4005 | 0.2558 |
| XGBoost | 0.9997 | 0.3862 | 0.2675 |
| Logistic Regression | 0.7683 | 0.1219 | 0.0979 |
| Isolation Forest | 0.1375 | 0.1766 | 0.1251 |

On validation (where the guards are applied) XGBoost goes 0.99998 → 0.3364 → 0.1660, the same pattern.

**The performance is the balance bookkeeping.** Account balances and their residuals carry essentially all the signal; without them the models fall to AUC-PR ≈ 0.27–0.40, and without `hour` as well they stay in the same range or drop further. Removing the features also destroys *stability*: Random Forest's train→test overfit ratio is 0.0% in `all`, but +20.6% in `no_balance` and +45.3% in `no_balance_no_hour`, and XGBoost's rolling relative SD rises from 0.3% to 23.3%.

This is a finding about PaySim's generative process, not a general claim about fraud detection: the simulator writes fraudulent transfers with sender balances that do not reconcile, so the model is learning that rule.

### Overfitting

Rolling-window validation: 5 equal-size time blocks, 4 folds, each evaluated at the lowest block fraud rate (frauds subsampled, 20 draws). Limit: relative SD of AUC-PR ≤ 10%.

| Model | Mean AUC-PR | Relative SD | Passes |
|---|---|---|---|
| Random Forest | 0.9980 | 0.2% | ✅ |
| XGBoost | 0.9969 | 0.3% | ✅ |
| Logistic Regression | 0.5601 | 4.0% | ✅ |
| Isolation Forest | 0.0215 | 58.2% | ❌ |

Learning curves are flat from 10% of the training data for RF (0.99997 → 1.00000) and XGBoost (0.998 → 1.000), i.e. the models are not data-starved; the remaining error is not a sample-size problem. Logistic Regression is flat and non-monotone (0.567 → 0.568), and Isolation Forest improves only marginally (0.006 → 0.009).

In the ablation settings XGBoost's relative SD rises to 23.3% (`no_balance`) and 9.2% (`no_balance_no_hour`) — the weaker feature set is genuinely less stable over time.

### Alert budget

A fixed probability threshold is unusable at 0.14% prevalence: F1 is maximised by a cut with zero false positives, which catches only a handful of frauds. The budget formulation asks instead: *given K alerts per day, how well does the model rank?*

At **K = 250 alerts/day** on the 15 full test days (one 6-step trailing day is dropped):

| Ranking | Precision@250 | Recall@250 | Fraud value caught |
|---|---|---|---|
| RF by score | 0.983 | 0.936 | 6.05 × 10⁹ of 6.18 × 10⁹ |
| RF by expected loss | 0.982 | 0.935 | 6.17 × 10⁹ of 6.18 × 10⁹ |
| Baseline: `isFlaggedFraud` | 0.253 | 0.241 | 4.68 × 10⁹ |
| Baseline: sender-account-emptied rule | 0.300 | 0.286 | 4.15 × 10⁹ |

At the tighter budget of **K = 100 alerts/day** the ranking criterion matters much more: 1,500 alerts catch 1,500 frauds either way (precision 1.000, recall 0.381), but ranking by expected loss (`p × amount`) catches **5.53 × 10⁹** of fraud value against **2.78 × 10⁹** for pure score ranking — roughly double, because it prefers large transactions over merely suspicious ones.

Net-saving curves are computed for review costs of 0.1%, 1% and 5% of the median train fraud amount.

### Bias audit

Subgroups: transaction type, amount quartile (train cut points), 6-hour time blocks. A group is judged only if it contains ≥ 30 frauds; criteria are recall ratio ≥ 0.8 and false-alarm ratio ≤ 1.25.

**Random Forest passes all 10 judged groups** (recall ratio ≈ 1.0, false-alarm ratio 1.0). The ablation models fail many low-amount and daytime groups — a direct consequence of removing the balance features.

### Explanations

SHAP on ~10,000 test rows gives a stable, interpretable ordering:

```
errorBalanceOrig > newbalanceOrig > oldbalanceOrg > oldbalanceDest
                 > newbalanceDest > amount > errorBalanceDest > hour > is_transfer
```

A depth-3 surrogate tree reproduces the model with fidelity 1.00, and SHAP ranks are stable across resamples (Spearman 1.00 vs a 0.8 threshold).

Each alert also gets human-readable reason codes (top 3):

| Code | Rule | Share of alerts |
|---|---|---|
| R2 | `amount` equals the sender's whole balance | 95.9% |
| R7 | Receiver's balance is 0 before and after | 51.9% |
| R1 | Sender's account was emptied by the transaction | 46.3% |
| R6 | Transaction at an unusual hour (0–7) | 35.5% |
| R4 | Receiver's balance change does not match the amount | 34.7% |
| R5 | Amount above the 75th percentile for its type | 22.6% |
| R3 | Sender's balance change matches the amount exactly | 0.3% (**dropped**) |

R3 is dropped because its rule fires on 1.77M training rows for only 32 frauds (precision 0.002%) and 0% precision on alerts — an *exact* sender balance is the signature of a legitimate transaction, not a fraudulent one. Every remaining code keeps precision above 99% on the alert set.

On the 3,750 alerts raised at the K=250 budget, overall alert precision was 98.3%; 50 alerts (1.3%) fired no code at all.

---

Approximate cost: training is dominated by Random Forest (300 trees on ~1.95M rows). XGBoost trains on the GPU (`device="cuda"`) when one is available; scikit-learn models run on CPU with `n_jobs=-1`.

---

Pages:

**Core** — Overview · Model comparison · Alert budget · Why is it fraud · Overfitting · Bias audit · Ablation
**Extras** — Autoencoder / SMOTE / amount-weighted variants, and cost sensitivity at K = 250

---
## Notes and limitations

- **PaySim is synthetic.** The near-perfect scores reflect the simulator's bookkeeping rules, not real-world fraud. The ablation study makes this explicit: the model is learning that fraudulent transfers leave the balances inconsistent.
- **Random Forest raises zero false positives** at its validation-selected threshold on the test set (precision 1.0000, 4,005 true positives, 0 false positives). This should be read as an upper bound produced by deterministic simulator rules, not as a realistic operating point.
- **The test period is easy in one sense and hard in another.** Legitimate volume collapses late in the dataset, so the test fraud rate (0.98%) is five times the training rate (0.19%). This is why rolling-window validation and prevalence-matched overfitting ratios are used instead of a single train/test comparison.
- **The alert budget is generous on low-volume days.** Test days 2–14 carry only ~2,000–16,000 transactions each, so K = 1,000 alerts/day covers a large share of that day's traffic. K = 100–250 is the realistic operating range; the larger budgets are reported for completeness. The trailing 6-step partial day (70 rows, all fraud) is dropped at every K, leaving 15 full test windows.
- **The max-F1 threshold is degenerate** at this prevalence and is reported only for completeness; the alert budget is the operational rule.
- **`isFlaggedFraud` is broken** — it fires 16 times in 6.3M transactions (all 16 genuinely fraudulent, but that is 0.2% of the 8,213 frauds) and misses almost everything. It is a baseline to beat, not a feature.
- **Overfit ratios use 200 bootstrap resamples** (metric CIs use 1,000); with ratios this close to zero, that is enough to decide the guard but the intervals are coarse.
---

## Team

| Name | Roll No. |
|---|---|
| Akshaj Singh | 2410110038 |
| Sarthak Ranjan | 2410110471 |

