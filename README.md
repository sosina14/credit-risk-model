# Credit Risk Probability Model for Alternative Data

**Bati Bank × eCommerce Partnership — Buy-Now-Pay-Later Credit Scoring System**

> An end-to-end implementation for building, deploying, and automating a credit risk model using behavioral transaction data.

---

## Project Overview

Bati Bank is partnering with an eCommerce platform to enable a buy-now-pay-later (BNPL) service. This project builds a Credit Scoring Model that assigns a risk probability score to each customer, informing loan approvals, credit limits, and loan terms in real time.

**Author:** Sosina Ayele  
**GitHub:** [github.com/sosina14/credit-risk-model](https://github.com/sosina14/credit-risk-model)

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/
│   ├── raw/                       # Raw data (git-ignored)
│   └── processed/                 # Processed data (git-ignored)
├── notebooks/
│   └── eda.ipynb                  # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering pipeline
│   ├── train.py                   # Model training & MLflow tracking
│   ├── predict.py                 # Inference
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord requires financial institutions to hold capital reserves proportional to their credit risk exposure. To satisfy this requirement, banks must demonstrate that their risk models are:

- **Measurable** — producing quantitative probability of default (PD) estimates that can be validated and back-tested against historical outcomes
- **Transparent** — regulators and internal audit teams must be able to understand how the model produces each score, not just that it does
- **Documented** — every modeling choice (variable selection, preprocessing, threshold setting) must be recorded with a business or statistical justification

This has direct implications for model design. A black-box model like a deep neural network may produce superior predictive accuracy but fails the interpretability requirement — regulators cannot audit what it has learned or why it makes a particular decision. In contrast, a Logistic Regression model with Weight of Evidence (WoE) transformations produces monotonic, auditable risk relationships per feature, and a scorecard that translates directly into a points-based credit score that loan officers can explain to applicants.

Basel II also mandates **model validation** — an independent review of the model's assumptions, data quality, and ongoing performance. This requires the model to be fully reproducible, which informs practices like setting random seeds, versioning datasets with DVC, and tracking experiments with MLflow.

In summary: Basel II does not just encourage interpretability — it makes it a regulatory compliance requirement with capital adequacy consequences for non-compliance.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The Xente eCommerce dataset contains transaction records with a fraud flag but no direct credit default label — that is, no field indicating whether a customer failed to repay a loan. This is common in alternative credit scoring: the platform has behavioral data on customers, but no formal credit history.

**Why a proxy is necessary:**

Supervised machine learning requires a target variable. Without one, we cannot train a classification model to predict default. A proxy variable is an observable behavioral signal that correlates strongly enough with default risk to serve as a substitute label. In this project, we use RFM (Recency, Frequency, Monetary) analysis to identify disengaged customers — those who transact infrequently, with low monetary value, and have not transacted recently. These customers exhibit behaviors consistent with financial disengagement, which credit research associates with elevated default probability.

**Business risks introduced by proxy-based prediction:**

- **Label noise:** The proxy is not a true default label. A customer classified as high-risk by RFM may simply have stopped using the platform for reasons unrelated to creditworthiness (moved, changed platform preference). This introduces false positives that could unfairly deny credit to creditworthy customers.
- **Distribution shift:** The behavioral patterns that define the high-risk proxy today may not predict actual default in the future, especially if the eCommerce platform's user base or product mix changes.
- **Regulatory challenge:** Basel II requires PD estimates to be validated against actual default outcomes. A proxy-derived target cannot be validated this way without downstream loan performance data, creating a documentation gap that regulators may challenge.
- **Feedback loops:** If the model denies credit to certain behavioral segments, those segments never get the opportunity to demonstrate creditworthiness, potentially entrenching bias over time.

These risks must be documented transparently in the model card and revisited as actual loan performance data becomes available from the BNPL portfolio.

---

### 3. What are the key trade-offs between a simple, interpretable model (Logistic Regression with WoE) and a high-performance model (Gradient Boosting) in a regulated financial context?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost/LightGBM) |
|---|---|---|
| **Predictive Performance** | Moderate — linear decision boundary limits ability to capture complex interactions | High — captures non-linear patterns and feature interactions automatically |
| **Interpretability** | High — coefficients map directly to a scorecard; each feature's contribution is transparent | Low — feature importance available but individual predictions are not easily explained |
| **Regulatory Compliance** | Strong — satisfies Basel II interpretability requirements; auditors can follow the logic | Weak — requires additional tools (SHAP) to approximate explanations; may not satisfy strict regulators |
| **Scorecard Conversion** | Direct — WoE + logistic coefficients produce a standard points-based scorecard | Indirect — requires post-hoc calibration and additional engineering |
| **Monotonicity** | Enforced — WoE binning ensures intuitive direction (e.g., higher income always reduces risk score) | Not guaranteed — model may learn non-monotonic patterns that are hard to justify to applicants |
| **Development Time** | Lower — well-understood pipeline with few hyperparameters | Higher — requires extensive tuning, cross-validation, and interpretability post-processing |
| **Maintenance** | Simpler — model behavior is predictable when data distribution shifts | Complex — performance may degrade unpredictably; requires monitoring infrastructure |

**Recommendation for Bati Bank:**

In a regulated context, the preferred approach is a **two-model strategy**: use Logistic Regression with WoE as the primary production model for compliance and auditability, while maintaining a Gradient Boosting model as a challenger model to benchmark performance. When the performance gap justifies the interpretability cost and regulators can be satisfied with SHAP-based explanations, the challenger model can be promoted to production.

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/sosina14/credit-risk-model.git
cd credit-risk-model

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Run the API

```bash
# Build and run with Docker
docker-compose up --build

# API will be available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

## Run Tests

```bash
pytest tests/ -v
```

## MLflow Tracking

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Data

Data source: [Xente Challenge on Kaggle](https://www.kaggle.com/c/xente-fraud-detection)

The dataset contains transaction-level records including customer ID, transaction amount, product category, channel, and a fraud flag. No direct default label exists — a proxy target variable is engineered using RFM-based clustering (see Task 4).

> Note: Raw data files are excluded from version control via `.gitignore`. Download the data from Kaggle and place it in `data/raw/`.