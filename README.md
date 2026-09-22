# Freight Rate Prediction

Predicts freight load rates from route, equipment, weight, date, and market features.

## Setup

pip install -r requirements.txt

## Run

1. Open `exploration.ipynb` and run all cells top to bottom. This:
   - Cleans `data/train_test.csv` (fixes sign-flipped weights, imputes missing weight/market_index)
   - Engineers features (date parts, one-hot equipment, target-encoded pickup/delivery)
   - Trains a Gradient Boosting model on a log-transformed target
   - Validates using a time-based split (train: Jan–Aug 2025, test: Sep–Oct 2025)
   - Generates `validation_predictions.csv` and fills `data/december_chart_inputs.csv`

2. Run the scorer:

python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

Output: `scorer_results/candidate_december.png`

## Approach summary

- **Model:** HistGradientBoostingRegressor (sklearn), trained on `log1p(posted_rate)`
- **Why:** Tree-based boosting handles the non-linear, outlier-heavy rate distribution better than linear regression; log-target reduces the influence of a small number of very high-rate loads
- **Validation:** Time-based split (not random) since the target task (validation.csv) is out-of-time prediction
- **Data quality fixes:** sign-flip correction on `weight`, median imputation for `weight`/`market_index` (fit on train only, applied to test/validation)
