# Data-Driven Insights into Animal Shelter Recidivism: Predicting Pet Returns at Austin Animal Center

## Model Correction Note
An earlier version of this project's Random Forest model measured a narrower question than it claimed to. The original target only included animals that had **already been returned**, splitting them into "fast return" (≤7.9 days) vs. "slow return" (8–30 days) — but it was described as predicting "will this adoption be returned at all," which requires true negative examples (adoptions that were never returned). That mismatch inflated the model's apparent performance, since distinguishing fast-vs-slow return timing is a much easier problem than predicting whether a return happens in the first place.

The target has since been rebuilt from scratch as a true binary classifier — did an adopted animal return within 30 days, or not — using real negative examples and correcting for censoring (excluding adoptions too recent in the dataset to know for certain they wouldn't have returned). All results below reflect the corrected model.

## Executive Summary
Animal shelter data from the Austin, TX area was cleaned and examined with the intention of reducing recidivism. This project applies Random Forest classification to identify the key risk factors that lead to pets being returned to the shelter within 30 days of adoption.

* **Business Goal:** Help shelter staff proactively identify high-risk adoptions and optimize resource allocation to prevent pet returns.
* **Key Result:** An adopted animal's **age** and **species** are the strongest predictors of return risk — far more than length of shelter stay, which was mistakenly identified as the dominant driver in an earlier version of this model. Dogs return at roughly 2.7x the rate of cats, and return risk climbs steadily with age, independent of how long the animal was in the shelter.
* **Actionable Recommendation:** Implement structured follow-up check-ins with new families during the first month, prioritizing older animals and dog adoptions specifically, where return risk is highest.

## The Data
* **Source:** Austin Animal Center Intake and Outcome records.
* **Dataset Size:** **54,408 adoption events** (true Adoption outcomes only, distinct from Return-to-Owner/Rto-Adopt), after excluding records too recent to reliably determine outcome.
* **Base Rate:** **6.8% of adoptions (3,718) resulted in a return within 30 days.**
* **Key Features:** `los_days` (Length of Stay), `age_at_first_visit`, `spp` (species), `akc_group` (breed group for dogs), and `first_reason` (intake condition: routine/medical/behavior/other).
* **Species Breakdown:** k9 (32,716), fel (20,962), other/bird/livestock/wildlife (730 combined).

## Methodology & Architecture
1. **Exploratory Data Analysis (EDA):** Investigated whether length of stay was a reliable standalone predictor of return risk. It was not — see Statistical Validation below.
2. **Modeling:** Built a Random Forest binary classifier predicting true 30-day return risk, using tuned hyperparameters (`max_depth=8`, `min_samples_leaf=20`, `class_weight='balanced'`) rather than untuned defaults.
3. **Evaluation:** Evaluated using a held-out 20% test split, prioritizing recall over raw accuracy for the reasons outlined below.

## Statistical Validation
Before finalizing the model's feature set, logistic regression was used to test each candidate feature's *independent* effect on return risk, controlling for the others:

* **Age** is a real, independent risk factor (p < 0.001) — each additional year of age increases return odds, holding length of stay and breed constant.
* **Species** is the strongest single effect found (p < 0.001) — dogs have roughly 2.7x the return odds of cats.
* **Length of stay** has a real but small independent effect, and it points in the *opposite* direction from what the original (flawed-target) Random Forest suggested: once age is controlled for, longer stays are associated with slightly *lower*, not higher, return risk.
* **Intake reason (medical)** is associated with lower return odds relative to routine intakes (p < 0.001).
* **Breed group** mostly doesn't differentiate risk among dogs, with one exception: toy breeds show significantly lower return odds than other groups.

## Evaluation Results
The Random Forest model achieved **55% accuracy**, prioritizing recall to minimize missed at-risk cases:
* **Recall (Class 1 - Returned): 76%** — the model correctly flags roughly three-quarters of adoptions that will actually be returned within 30 days.
* **Precision (Class 1 - Returned): 11%** — given the low 6.8% base rate, most "high risk" flags will be false alarms; this is an expected and accepted tradeoff (see Metric Prioritization below), not a modeling error.
* **Feature Importance:** age_at_first_visit (41%), spp_k9 (26%), los_days (18%), all other features individually under 3%.

## Technical Decision Rationale

### Data Engineering & SQL Join Rationale
The core datasets required structural restructuring to trace individual animal timelines over multiple shelter stays. To execute this safely without generating duplicating rows or invalid cartesian logic, an explicit multi-layer SQL extraction architecture was built directly within Python using SQLite:

* **Chronological Stay Mapping:** Utilized an algorithmic inner join restricting `o.datetime_out > i.datetime_in`. This ensured outcomes only mapped to relevant, future-facing operational events.
* **Window Partitioning:** Implemented Common Table Expressions (CTEs) combining analytical window ranking functions (`ROW_NUMBER() OVER (PARTITION BY i.apt_id ORDER BY o.datetime_out ASC)`). This isolated the strict, immediate chronological outcome for every specific stay record.
* **True Adoption Filtering:** The raw `outcome_type` is preserved (`outcome_type_raw`) prior to being collapsed into broader categories, so "Adoption" can be distinguished from "Return to Owner" and "Rto-Adopt" — the target only includes true adoptions, matching what the model claims to predict.
* **Pipeline Isolation:** Structured the infrastructure cleanly into dedicated `Extraction`, `Engineering`, and `Evaluation` stages, handling object-to-datetime type-casting systematically to enforce raw schema integrity prior to feature engineering.

### Model Selection & Progression
A separate, standalone analysis modeled `return_time_days` (how long a return takes, *among animals that already returned*) using Ordinary Least Squares regression. This yielded an **R-squared of ~0.002**, meaning length of stay, age, species, and intake reason together explain almost none of the variance in *how fast* a return happens once one occurs. This is a distinct question from *whether* a return happens at all — the classification model above answers the latter, and is the basis for this project's actionable recommendations.

### Recidivism Window Definition
The target threshold for pet recidivism is a 30-day post-adoption horizon. Adoptions within 30 days of the dataset's observed cutoff date are excluded rather than labeled as "did not return" — since a recent adoption may not have had the full 30-day window elapse yet, labeling it a negative would be a guess, not a confirmed outcome. Only adoptions with a fully-observed 30-day window are used for training and evaluation.

### Metric Prioritization
Model optimization intentionally prioritized Recall over strict global Accuracy. In this operational context, a False Negative—failing to flag an animal that will ultimately be returned—results in zero intervention and a failed adoption. Prioritizing Recall ensures the shelter proactively catches the maximum number of true-risk cases, at the accepted cost of a high false-positive rate given the low base rate of returns.

### Confusion Matrix Insights
On a held-out test set of 10,882 adoptions (744 real returns):
* **True Positives (approx.):** ~566 real returns correctly flagged.
* **False Negatives (approx.):** ~178 real returns missed by the model.
* **False Positives:** the majority of "high risk" flags — an accepted tradeoff given the model's recall-first design and the low 6.8% base rate.

### Project Directory Structure:
```text
├── data/               # Raw and processed datasets
├── images/             # Generated analytical visualizations
├── src/                # Modular Python production scripts
│   ├── recidivism-data-extraction.ipynb
│   ├── recidivism-data-engineering.ipynb
│   └── recidivism-model-and-evaluation.ipynb
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
