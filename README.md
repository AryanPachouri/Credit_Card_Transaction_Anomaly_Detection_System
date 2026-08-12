# Anomaly Detection in Credit Card Transactions

A statistical anomaly detection system that flags potentially fraudulent credit card
transactions by fitting a multivariate Gaussian density over the transaction feature
space and thresholding on likelihood, rather than training a supervised classifier on
labeled fraud examples.

**[Live demo →](#)** &nbsp;·&nbsp; runs entirely in the browser, no backend required

---

## Why anomaly detection, not classification

The dataset is extremely imbalanced: 492 fraudulent transactions out of 284,807 total
(0.17%). A standard classifier trained on this would either collapse to predicting
"authentic" for everything, or need heavy resampling that risks overfitting to a
handful of fraud patterns. It's also unrealistic to assume future fraud will look like
past fraud.

The approach here instead models what a **normal** transaction looks like, and flags
anything that deviates significantly from that model — which generalizes better to
fraud patterns the model has never seen, and doesn't require balancing the classes.

## Method

1. **Feature engineering** — `Time` (seconds since first transaction) is decomposed
   into `Hour`, `Minute`, `Second`, `Day`; only `Hour` turned out informative. `Amount`
   is heavily right-skewed, so it's log-transformed. The 28 `V1`–`V28` columns are
   already PCA components from the original dataset.
2. **Feature selection** — anomaly detection degrades in high dimensions (the "curse
   of dimensionality" makes density estimates unreliable), so features are ranked by
   how differently they're distributed between authentic and fraudulent transactions
   in the validation set. The 9 most discriminative are kept:
   `V4, V11, V12, V14, V16, V17, V18, V19, Hour`.
3. **Density model** — each selected feature is modeled as an independent univariate
   normal, fit on the training data (authentic transactions only):

   ```
   p(x) = Π f(xᵢ; μᵢ, σᵢ)     for i = 1..9
   ```

   A transaction is flagged as anomalous when `p(x) < ε`.
4. **Threshold tuning** — ε is parameterized as `α⁹`, and α is swept over
   `[0.001, 0.05]` on the validation set, selecting the value that maximizes
   **F2-score** — recall weighted higher than precision, because a missed fraud case
   (false negative) is costlier than a false alarm (false positive) that gets
   manually reviewed and cleared.

## Results

| Split | Metric | Value |
|---|---|---|
| Validation | F2-score | 0.835 |
| Test | Accuracy | 0.997 |
| Test | Precision | 0.798 |
| Test | Recall | 0.821 |
| Test | F1-score | 0.810 |
| Test | **F2-score** | **0.816** |
| Test | Matthews correlation coefficient | 0.808 |

Optimal threshold: α = 0.009 (ε = α⁹ ≈ 3.87 × 10⁻¹⁹)

In plain terms: on transactions the model has never seen, it correctly catches
**82% of actual fraud**, and when it flags something as fraud, it's **right about
80% of the time** — a strong result for a model that never needed a single labeled
fraud example to fit its parameters.

## Repository structure

```
.
├── notebook/
│   └── anomaly_detection_training.ipynb   # full EDA, feature engineering,
│                                           # feature selection, model fitting,
│                                           # threshold tuning, evaluation
├── model_params.json                      # trained parameters (μ, σ, threshold,
│                                           # metrics) — the single source of truth
│                                           # used by both predict.py and index.html
├── predict.py                             # CLI inference script (single transaction
│                                           # or batch CSV scoring)
├── index.html                             # static, self-contained demo UI
├── requirements.txt
└── README.md
```

## Running it yourself

### Web demo
`index.html` is fully self-contained — the model parameters are embedded directly
as JavaScript constants, and density scoring happens client-side. Open the file in
any browser, or deploy it as-is.

**Deploying to Vercel:**
1. Push this repo to GitHub.
2. In Vercel, "Add New Project" → import the repo.
3. Framework preset: **Other** (no build step needed).
4. Deploy — `index.html` is served as the site root automatically.

### Command line
```bash
pip install -r requirements.txt

# score a single transaction (order: V4 V11 V12 V14 V16 V17 V18 V19 Hour)
python predict.py --values 1.2 -0.5 0.3 -6.2 -1.1 -2.0 -0.8 0.4 3

# score a batch from CSV (must contain the 9 feature columns)
python predict.py --csv transactions.csv --out predictions.csv
```

### Training notebook
The full analysis — EDA, feature engineering, feature selection rationale, threshold
tuning curve, confusion matrices — is in `notebook/anomaly_detection_training.ipynb`.
It expects the [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
at `../input/creditcardfraud/creditcard.csv` (Kaggle's default input path); update
the path if running elsewhere.

## Tech stack

Python, pandas, NumPy, scikit-learn (train/test split), Matplotlib/Seaborn/Plotly
(EDA visualization), vanilla HTML/CSS/JS (demo UI, no framework or build step).

## Limitations & possible extensions

- The independent-Gaussian assumption ignores correlation between features; a full
  multivariate Gaussian with a covariance matrix, or a Gaussian Mixture Model, could
  capture more structure.
- The demo UI expects PCA-transformed feature values (`V4`, `V11`, ...) since that's
  the form of the public dataset — a production system would compute these from raw
  transaction fields upstream.
- No online/streaming update to μ, σ as transaction patterns drift over time.

## Acknowledgements

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle, ULB Machine Learning Group)
