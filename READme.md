# Data-Driven Insights into Animal Shelter Recidivism: Predicting Pet Returns at Austin Animal Center

## Executive Summary
Animal shelter data from the Austin, TX area was cleaned and examined with the intention of reducing recidivism. This project applies Random Forest classification models to identify the key risk factors that lead to pets being returned to the shelter within 30 days of adoption.

* **Business Goal:** Help shelter staff proactively identify high-risk adoptions and optimize resource allocation to prevent pet returns.
* **Key Result:** Utilizing Random Forest modeling, "Length of Stay" was identified as a critical driver, showing a 55% impact on canine recidivism within a 30-day window. 
* **Actionable Recommendation:** Implement structured follow-up check-ins with new families during the first month to proactively address adjustment challenges for both the adopter and the animal.

## The Data
* **Source:** Austin Animal Center Intake and Outcome records.
* **Dataset Size:** **4,916 total historical records** tracking unique animal trajectories.
* **Key Features:** Features include `los_days` (Length of Stay), `age_at_first_visit`, `return_time_days`, and categorical group identifiers like `spp` (species) and `akc_group` (breed groups).
* **Target Audience Focus:** The species distribution primarily focuses on **canines** (`can` with 4,005 records) and **felines** (`fel` with 905 records).

## Methodology & Architecture
1. **Exploratory Data Analysis (EDA):** Visualized Austin shelter intake trends and identified that an animal's length of stay strongly correlates with rapid return rates.
2. **Modeling:** Developed a Random Forest binary classification model to predict whether an animal would experience recidivism within a 30-day window.
3. **Evaluation:** Evaluated performance metrics using Scikit-Learn to assess real-world viability, prioritizing the model's ability to catch true risk cases.

## Evaluation Results
The Random Forest model achieved a baseline **accuracy of 55%** and demonstrated a strong orientation toward minimizing missed interventions:
* **Recall (Class 1 - Returned): 61%** — The model successfully identified 340 out of 557 actual recidivism cases, allowing the shelter to proactively target more than half of the highest-risk adoptions.
* **Precision (Class 1 - Returned): 60%** — When the model marks an adoption as "high risk," it is accurate 60% of the time, ensuring shelter resources are deployed efficiently without excessive false alarms.

### Confusion Matrix Insights
* **True Positives:** 340 animals correctly flagged as high return risks.
* **False Negatives:** 217 animals missed by the model. Future iterations will focus on reducing this number further by experimenting with gradient boosting methods.

### Project Directory Structure
```text
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for EDA and modeling
├── src/                # Python scripts for data cleaning and training
├── requirements.txt    # Library dependencies
└── README.md
```


## Key Findings & Visualizations
<!-- TIP: Open GitHub editor, drag your Kaggle chart image files, and drop them right on the lines below! -->

* **Insight 1:** Length of stay reveals a clear "Shelter Return Risk Tipping Point." Animals staying at the shelter longer face a steep, continuing upward trend in rapid return rates upon adoption.
* **Insight 2:** The historical dataset is heavily unbalanced toward canines (over 81% of data), indicating that shelter interventions and prediction boundaries should primarily target dog adoption protocols.

## Getting Started & Installation

### Prerequisites
Set up your virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Usage
Run the pipeline files from your terminal:
```bash
# Step 1: Extract and clean raw data records
python src/recidivism_data_extraction.py

# Step 2: Perform SQL joining and feature transformation
python src/recidivism_data_engineering.py

# Step 3: Execute statistical exploration and train Random Forest model
python src/recidivism_model_and_evaluation.py
```

## License & Contact
Distributed under the MIT License. For questions or collaboration, reach out via [[LinkedIn](https://www.linkedin.com/in/micah-luftig/)].
