# Data-Driven Insights into Animal Shelter Recidivism: Predicting Pet Returns at Austin Animal Center

## Model Correction Note
An earlier version of this project's Random Forest model measured a narrower question than it claimed to. The original target only included animals that had **already been returned**, splitting them into "fast return" (≤7.9 days) vs. "slow return" (8–30 days) — but it was described as predicting "will this adoption be returned at all," which requires true negative examples (adoptions that were never returned). That mismatch inflated the model's apparent performance and also led to a false conclusion that length of stay was the dominant risk driver.

The target has since been rebuilt as a true binary classifier — did an adopted animal return within 30 days, or not — using real negative examples and correcting for censoring (excluding adoptions too recent in the dataset to know for certain they wouldn't have returned). All results below reflect the corrected model.

A second, separate issue was found after deployment: the Random Forest's `class_weight='balanced'` setting, while correct for classification, also caused `predict_proba()` to output uncalibrated values — the model's raw predicted probabilities averaged ~43% across the test set versus a true base rate of ~6.8%. This has been corrected with isotonic probability calibration (`CalibratedClassifierCV`), so the app's displayed risk percentages now reflect real-world frequencies rather than an inflated relative score.

## Executive Summary
Animal shelter data from the Austin, TX area was cleaned and examined with the intention of reducing recidivism. This project applies Random Forest classification to identify the key risk factors that lead to pets being returned to the shelter within 30 days of adoption.

* **Business Goal:** Help shelter staff proactively identify high-risk adoptions and optimize resource allocation to prevent pet returns.
* **Key Result:** An adopted animal's **age** and **species** are the strongest predictors of return risk — far more than length of shelter stay, which was mistakenly identified as the dominant driver in an earlier version of this model. Dogs return at roughly 2.7x the rate of cats, and return risk climbs steadily with age, independent of how long the animal was in the shelter. Length of stay on its own shows no clean relationship with return risk at all.
* **Actionable Recommendation:** Implement structured follow-up check-ins with new families during the first month, prioritizing older animals and dog adoptions specifically, where return risk is highest.

## The Data
* **Source:** Austin Animal Center Intake and Outcome records.
* **Dataset Size:** **54,408 adoption events** (true Adoption outcomes only, distinct from Return-to-Owner/Rto-Adopt), after excluding records too recent to reliably determine outcome.
* **Base Rate:** **6.8% of adoptions (3,718) resulted in a return within 30 days.**
* **Key Features:** `los_days` (Length of Stay), `age_at_first_visit`, `spp` (species), `akc_group` (breed group for dogs), and `first_reason` (intake condition: routine/medical/behavior/other).
* **Species Distribution:** 60.1% dog, 38.5% cat, remaining 1.3% split across other/bird/livestock/wildlife.

## Methodology & Architecture
1. **Exploratory Data Analysis (EDA):** Investigated whether length of stay was a reliable standalone predictor of return risk. It was not — see Statistical Validation below.
2. **Modeling:** Built a Random Forest binary classifier predicting true 30-day return risk, using tuned hyperparameters (`max_depth=8`, `min_samples_leaf=20`, `class_weight='balanced'`) rather than untuned defaults.
3. **Evaluation:** Evaluated using a held-out 20% test split, prioritizing recall over raw accuracy for the reasons outlined below.

## Statistical Validation
Before finalizing the model's feature set, logistic regression was used to test each candidate feature's *independent* effect on return risk, controlling for the others:

* **Age** is a real, independent risk factor (p < 0.001) — each additional year of age increases return odds, holding length of stay and breed constant.
* **Species** is the strongest single effect found (p < 0.001) — dogs have roughly 2.7x the return odds of cats.
* **Length of stay** has a real but small independent effect, and it points in the *opposite* direction from what an earlier, flawed-target model suggested: once age is controlled for, longer stays are associated with slightly *lower*, not higher, return risk.
* **Intake reason (medical)** is associated with lower return odds relative to routine intakes (p < 0.001).
* **Breed group** mostly doesn't differentiate risk among dogs, with one exception: toy breeds show significantly lower return odds than other groups.

## Evaluation Results
The Random Forest model achieved **55% accuracy**, prioritizing recall to minimize missed at-risk cases:
* **Recall (Class 1 - Returned): 76%** — the model correctly flags 568 of 744 real 30-day returns in the held-out test set.
* **Precision (Class 1 - Returned): 11%** — given the low 6.8% base rate, most "high risk" flags are false alarms; this is an expected and accepted tradeoff (see Metric Prioritization below), not a modeling error.
* **Feature Importance:** age_at_first_visit (58.67%), spp_k9 (20.70%), los_days (11.02%), all other features individually under 3%.

## Technical Decision Rationale

### Data Engineering & SQL Join Rationale
The core datasets required structural restructuring to trace individual animal timelines over multiple shelter stays. To execute this safely without generating duplicating rows or invalid cartesian logic, an explicit multi-layer SQL extraction architecture was built directly within Python using SQLite:

* **Chronological Stay Mapping:** Utilized an algorithmic inner join restricting `o.datetime_out > i.datetime_in`. This ensured outcomes only mapped to relevant, future-facing operational events.
* **Window Partitioning:** Implemented Common Table Expressions (CTEs) combining analytical window ranking functions (`ROW_NUMBER() OVER (PARTITION BY i.apt_id ORDER BY o.datetime_out ASC)`). This isolated the strict, immediate chronological outcome for every specific stay record.
* **True Adoption Filtering:** The raw `outcome_type` is preserved (`outcome_type_raw`) prior to being collapsed into broader categories, so "Adoption" can be distinguished from "Return to Owner" and "Rto-Adopt" — the target only includes true adoptions, matching what the model claims to predict.
* **Pipeline Isolation:** Structured the infrastructure cleanly into dedicated `Extraction`, `Engineering`, and `Evaluation` stages, handling object-to-datetime type-casting systematically to enforce raw schema integrity prior to feature engineering.

### Model Selection & Progression
A separate, standalone analysis modeled `return_time_days` (how long a return takes, *among animals that already returned*) using Ordinary Least Squares regression. This yielded an **R-squared of ~0.002**, meaning length of stay, age, species, and intake reason together explain almost none of the variance in *how fast* a return happens once one occurs. This is a distinct question from *whether* a return happens at all — the classification model above answers the latter, and is the basis for this project's actionable recommendations.

A follow-up Kruskal-Wallis test on this same subset (animals that already returned) found a statistically significant, if weak, association between intake reason and return timing (chi-squared = 9.24, p = 0.026). This is consistent with the OLS result above — a real signal exists, but its practical magnitude is small, which is why intake reason alone shouldn't be read as a meaningful predictor of return speed on its own.

### Recidivism Window Definition
The target threshold for pet recidivism is a 30-day post-adoption horizon. Adoptions within 30 days of the dataset's observed cutoff date are excluded rather than labeled as "did not return" — since a recent adoption may not have had the full 30-day window elapse yet, labeling it a negative would be a guess, not a confirmed outcome. Only adoptions with a fully-observed 30-day window are used for training and evaluation.

### Metric Prioritization
Model optimization intentionally prioritized Recall over strict global Accuracy. In this operational context, a False Negative—failing to flag an animal that will ultimately be returned—results in zero intervention and a failed adoption. Prioritizing Recall ensures the shelter proactively catches the maximum number of true-risk cases, at the accepted cost of a high false-positive rate given the low base rate of returns.

### Confusion Matrix Insights
On a held-out test set of 10,882 adoptions (744 real returns):
* **True Positives: 568** — real returns correctly flagged.
* **False Negatives: 176** — real returns missed by the model.
* **False Positives: 4,742** — the majority of "high risk" flags; an accepted tradeoff given the model's recall-first design and the low 6.8% base rate.
* **True Negatives: 5,396**

### Length of Stay Is Not a Reliable Standalone Predictor
An earlier version of this analysis claimed length of stay showed "a steep, continuing upward trend" in return risk. Re-examined against the corrected target, this does not hold: return rate by week of stay is noisy and ranges narrowly between roughly 4% and 9%, with no clear directional trend — consistent with the logistic regression finding that LOS has only a small, slightly negative independent effect once age is controlled for. This is retained here as a documented correction, since it materially changes the project's earlier headline claim.

## Cross-Shelter Validation (Bloomington, IN)

To test whether this model's findings are specific to Austin or reflect something more general about shelter-return risk, the same recidivism definition and feature pipeline were applied to a second, independent dataset: Bloomington Animal Care & Control (Bloomington, IN).

### Data Quality Correction: Recidivism Target
Bloomington's raw data includes `returndate`/`returnedreason` fields, but a naive `returndate.notna()` check is not a valid recidivism signal on its own. Cross-tabulating against `movementtype` revealed that `returnedreason == 'Stray'` on **100% of the 2,352 rows** where `movementtype == 'Foster'` and `returndate` is populated — a default/placeholder value in the source software, not a real answer, and a foster placement ending is a normal, expected event rather than a failed adoption. A raw `returndate.notna()` check also has no time-window constraint, conflating true recidivism with returns that happened months or years later.

The target was corrected to match Austin's exact definition: an animal adopted, then returned within 30 days of that adoption's start date, checked across an animal's full movement history (not just its final row, since collapsing to one row per animal would lose any failed adoption that wasn't the animal's last placement attempt). This changed the measured recidivism rate from a naive **7.53%** ("ever returned") to a corrected **3.90%** (true 30-day recidivism) — a large enough difference to materially change any downstream analysis if left uncorrected.

### Breed Group Mapping
Bloomington's `breedname` field uses different formatting than Austin's (space/comma-separated vs. underscore-joined) and includes breed identities rare in Austin's data (`Pitbull`, `Bully Breed Mix` — together over 400 records). A dedicated mapping function was built mirroring Austin's exact AKC group categories and priority order, adapted for these formatting and vocabulary differences, achieving 99.97% classification coverage (1 unmapped record out of 3,900 dogs).

### Cross-Shelter Transfer Test
Austin's trained model (`recidivism_model.pkl`) was applied directly to Bloomington's corrected data as a holdout set:

| Metric | Austin (own test set) | Bloomington (Austin model) |
|---|---|---|
| ROC AUC | 0.7116 | 0.6418 |
| Recall (Returned), best threshold | 76% | 16% |

The model retains some genuine discriminative signal at Bloomington (AUC well above the 0.50 no-skill baseline), but the drop is substantial, and even after tuning the decision threshold specifically for Bloomington's data, recall remains far below Austin's own performance. **Austin's model shows partial but not practically deployable transfer to Bloomington as-is.**

### Why Transfer Is Weak: Divergent Feature Importance
A fresh Random Forest was trained on Bloomington's own data (same hyperparameters, same train/test methodology) to test whether recidivism is simply harder to predict at Bloomington, or whether the two shelters are driven by different underlying factors. The Bloomington-native model achieved **0.8641 AUC** on its own held-out test set — *higher* than Austin's own model achieves on Austin's data — with 74% recall on returned cases, closely matching Austin's 76%. This rules out "Bloomington data is inherently noisier"; a model fit to Bloomington's actual patterns performs comparably well.

Comparing feature importances directly explains the transfer gap:

| Feature | Austin | Bloomington |
|---|---|---|
| age_at_first_visit | **58.67%** | 17.33% |
| spp_k9 (species) | 20.70% | 9.40% |
| los_days (length of stay) | 11.02% | **61.89%** |
| all breed/intake-reason features (combined) | ~9.6% | ~11.4% |

At Austin, *who the animal is* (age, species) drives ~79% of the model's predictions. At Bloomington, *what happened during the stay* (length of stay) drives ~62% on its own — length of stay is over 5x more influential at Bloomington than at Austin, while age's influence drops by more than half. These aren't just different weights on the same underlying pattern; they represent genuinely different theories of what causes an adoption to fail, which is why a model trained on one shelter's pattern transfers only partially to the other.

**Caveat:** Bloomington's test set contains only 57 true "Returned" cases (3.9% of 1,440 held-out rows), so its performance metrics — particularly recall — should be read as directionally informative rather than highly precise; a different random split could shift these numbers somewhat.

### Cross-Shelter Notebooks
```text
├── bloomington-data.ipynb                      # Cleaning, schema mapping, breed mapping, recidivism target
├── 2-bloomington-cross-shelter-evaluation.ipynb # Austin model tested against Bloomington data
└── 3-bloomington-model-comparison.ipynb         # Bloomington-native model + feature importance comparison
```

### Project Directory Structure:
```text
├── data/               # Raw and processed datasets
│   ├── Austin_Animal_Center_Intakes.csv
│   ├── Austin_Animal_Center_Outcomes.csv
│   ├── X_test.csv                    # Held-out test features, for reproducibility
│   ├── y_test.csv                    # Held-out test labels, for reproducibility
│   ├── animal-data-1.csv             # Raw Bloomington source data
│   └── bloomington_model_ready.csv   # Cleaned Bloomington data, Austin-schema-aligned
├── images/             # Generated analytical visualizations
├── src/                # Modular Python production scripts
│   ├── recidivism-data-extraction.ipynb
│   ├── recidivism-data-engineering.ipynb
│   ├── recidivism-model-and-evaluation.ipynb
│   ├── bloomington-data.ipynb
│   ├── 2-bloomington-cross-shelter-evaluation.ipynb
│   └── 3-bloomington-model-comparison.ipynb
├── requirements.txt    # Unified library dependencies
└── README.md
```

## Getting Started & Installation

### Prerequisites
Set up your virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
