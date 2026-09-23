# Customer Type Analysis: Predicting & Profiling Campaign Acceptance

Predicts which customers are likely to accept a marketing campaign
and profiles the distinct customer segments among those who do,
using the [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) dataset.

> Originally built in R for a graduate Machine Learning course group
> project; this is a full Python port using pandas, scikit-learn, and
> scipy in place of `caret`, `factoextra`, and base R.

## Goal

1. **Which customer attributes predict campaign acceptance?** —
   bivariate significance tests (chi-square for categorical variables,
   one-way ANOVA for numeric ones).
2. **Which model best predicts acceptance?** — compare six classifiers
   via 10-fold cross-validated ROC AUC, then evaluate the best one on
   a held-out test set.
3. **What distinct customer types exist among those who accept?** —
   K-means clustering with an elbow-method check on cluster count,
   followed by cluster profiling.

## Data pipeline

`src/data_processing.ipynb`:
- Merges rare marital-status labels (`Alone`, `Absurd`, `YOLO`) into `Single`
- Recodes education levels into plain-language categories
- Derives `age` from birth year (drops 3 implausible outliers > 100)
- Builds a single `Accepted_Cmp` target from the five historical
  campaign flags plus the most recent response
- Derives customer `duration` (tenure) and `children` (kids + teens)
- Drops redundant identifier/outlier columns

## Findings

**Bivariate tests** — nearly every purchasing and demographic variable
is significantly associated with campaign acceptance (spending across
all product categories, income, purchase channel, marital status,
education, number of children). Only `age` and `Complain` are not
significant at α = 0.05 — consistent with the original R analysis.

**Model comparison** (10-fold CV, ROC AUC):

| Model | Mean ROC AUC |
|---|---|
| Random Forest | ~0.86 |
| Gradient Boosting | ~0.84 |
| Logistic Regression (L1) | ~0.78 |
| Logistic Regression | ~0.78 |
| KNN | ~0.73 |
| Decision Tree | ~0.71 |

Random Forest wins and is evaluated on the test set at a 0.19
probability threshold (chosen, as in the original analysis, to favor
sensitivity — catching likely acceptors matters more than avoiding
false positives in a marketing context). Top predictive features are
wine spend, income, meat spend, and gold-product spend.

**Cluster analysis** — among customers who accepted a campaign,
K-means (k=3, chosen via the elbow method) finds three groups:
a high-spending, high-income segment with few children; a
lower-income segment with more children and lower spend across every
category; and a middle segment with above-average wine spend and the
highest average age.
## Setup

```bash
pip install -r requirements.txt
```

Then open and run the notebooks in order:

```
src/data_processing.ipynb     # clean + engineer features
src/statistical_tests.ipynb   # bivariate significance tests
src/model_comparison.ipynb    # compare classifiers, evaluate the best
src/clustering.ipynb          # cluster + profile campaign acceptors
```

Each notebook can also be run as a standalone script if preferred:

```bash
python src/data_processing.py
python src/statistical_tests.py
python src/model_comparison.py
python src/clustering.py
```

Plots are saved to `output/` and also shown interactively.

## Project structure

```
Customer Analysis Machine Learning Project/
├── data/
│   ├── processed/
│   │   └── customer_personality_cleaned.csv
│   └── customer_personality.csv
├── output/                          # generated plots
├── src/
│   ├── data_processing.ipynb        # cleaning + feature engineering
│   ├── data_processing.py
│   ├── statistical_tests.ipynb      # chi-square + ANOVA tests
│   ├── statistical_tests.py
│   ├── model_comparison.ipynb       # classifier comparison + evaluation
│   ├── model_comparison.py
│   ├── clustering.ipynb             # K-means + cluster profiling
│   └── clustering.py
├── requirements.txt
└── README.md
```