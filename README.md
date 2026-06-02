# Lead Prioritisation in CRM Systems

**From Prediction to Decision**  
2nd Semester Project — Group AAE  
AAU Business School, Aalborg University

This repository contains the code, notebooks, model artifacts, and Streamlit prototype for a Business Data Science project on **lead prioritisation in CRM systems**.

The project explores how behavioural lead data can be transformed into segmentation insights, conversion predictions, priority labels, and CRM-oriented recommendations for marketing and sales decision-making.

**Live Streamlit prototype:**  
https://lead-prioritisation-in-crm-systems-group-aae.streamlit.app/

---

## Project Overview

Many CRM systems collect behavioural and engagement data about potential customers, but this data is not always translated into clear operational decisions. This project addresses that gap by developing a proof-of-concept CRM decision-support workflow that moves from raw lead data to actionable recommendations.

The system combines:

- Exploratory data analysis
- Behavioural feature engineering
- K-means clustering and PCA-based segmentation
- Predictive modelling with Logistic Regression, Random Forest, and XGBoost
- Lead priority classification
- CRM recommendation logic
- Human-in-the-loop decision support
- A Streamlit-based operational prototype
- Basic logging and monitoring

The goal is not to build a fully production-ready CRM platform, but to demonstrate how data-driven methods can support lead prioritisation and decision-making on a proof-of-concept level.

---

## Main Research Question

**How can data-driven methods support lead prioritisation and decision-making in CRM-based systems?**

### Sub-Research Questions

**RQ1:** Are there distinct patterns or segments among leads that influence prioritisation strategies?

**RQ2:** How do different machine learning approaches influence lead prioritisation and conversion prediction performance?

**RQ3:** How can intelligent CRM recommendation logic utilise model predictions and lead insights to support marketing and sales decision-making?

---

## Analytical Workflow

The project follows a structured workflow:

```text
Lead data
→ Data preprocessing
→ Feature engineering
→ Exploratory data analysis
→ RQ1: Behavioural segmentation with K-means and PCA
→ RQ2: Predictive model comparison
→ XGBoost-based conversion prediction
→ Priority assignment
→ RQ3: CRM recommendation logic
→ Streamlit CRM prototype
→ Logging and monitoring
```

---

## Methods Used

### RQ1 — Behavioural Segmentation and Clustering

RQ1 investigates whether leads form meaningful behavioural groups based on website engagement.

Methods used:

- Feature engineering
- Standardisation
- K-means clustering
- Elbow Method
- Silhouette Score
- Principal Component Analysis (PCA)
- Cluster profiling
- Cluster-based prioritisation strategy

The clustering analysis identifies distinct behavioural segments, such as low-engagement leads, high-intent/time-intensive leads, and browsing-heavy/research-oriented leads.

### RQ2 — Predictive Modelling and Model Comparison

RQ2 compares different supervised machine learning models for conversion prediction and lead prioritisation.

Models used:

- Logistic Regression
- Random Forest
- XGBoost

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

XGBoost is selected as the final predictive model because it achieved the strongest overall performance across the main evaluation metrics.

### RQ3 — CRM Recommendation and Decision Support

RQ3 translates predictive outputs into actionable CRM recommendations.

The recommendation workflow uses:

- XGBoost conversion probabilities
- Priority thresholds
- Contact permission constraints
- Recommendation categories
- Marketing and sales decision logic
- Human-in-the-loop review status

Priority logic:

```text
High Priority   = probability >= 0.70
Medium Priority = probability between 0.40 and 0.69
Low Priority    = probability < 0.40
```

Recommendation categories include:

- Immediate Sales Follow-up
- Targeted Nurturing
- Automated Low-Intensity Marketing
- Compliant Alternative Follow-up

The notebook also demonstrates an LLM-assisted judge-agent concept on selected examples, while the deployed Streamlit prototype uses scalable rule-based recommendation logic.

---

## Streamlit Prototype

The Streamlit app demonstrates how the analytical workflow can be translated into a CRM-style decision-support interface.

The app allows users to:

- Select an existing lead from the dataset
- Modify behavioural input variables
- Generate a predicted conversion probability
- Assign a lead score and priority level
- View CRM recommendation output
- Check whether human review is required
- Log predictions
- Monitor prediction behaviour and potential drift

To run the app locally:

```bash
streamlit run streamlit_app.py
```

---

## Repository Structure

```text
Lead-Prioritisation-in-CRM-Systems/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── .streamlit/
│   └── secrets.toml.example
│
├── artifacts/
│   ├── metrics/
│   │   └── train_stats.json
│   └── model/
│       └── xgboost_model.pkl
│
├── data/
│   └── Lead Scoring.csv
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── RQ1_Clustering.ipynb
│   └── RQ2_+_RQ3.ipynb
│
├── output/
│   ├── rq3_agent_recommendations-*.csv
│   └── rq3_multi_agent_recommendations-*.csv
│
├── .gitignore
├── monitor.py
├── README.md
├── requirements.txt
└── streamlit_app.py
```

---

## File and Folder Description

| Path | Description |
|---|---|
| `data/Lead Scoring.csv` | Original lead scoring dataset used for modelling and the Streamlit prototype |
| `Notebooks/EDA.ipynb` | Exploratory data analysis and initial feature understanding |
| `Notebooks/RQ1_Clustering.ipynb` | K-means clustering, PCA visualisation, cluster profiling, and RQ1 analysis |
| `Notebooks/RQ2_+_RQ3.ipynb` | Predictive model comparison, XGBoost selection, prioritisation logic, and recommendation workflow |
| `artifacts/model/xgboost_model.pkl` | Saved XGBoost model used by the Streamlit app |
| `artifacts/metrics/train_stats.json` | Training statistics used for basic drift detection |
| `output/` | Generated RQ3 recommendation outputs |
| `streamlit_app.py` | Main Streamlit CRM decision-support application |
| `monitor.py` | Logging and simple drift detection functions |
| `.streamlit/secrets.toml.example` | Example Streamlit secrets file; real secrets should not be committed |
| `.devcontainer/devcontainer.json` | Optional GitHub Codespaces / VS Code development container setup |

---

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/eminagracanin-cloud/Lead-Prioritisation-in-CRM-Systems.git
cd Lead-Prioritisation-in-CRM-Systems
```

### 2. Create and activate a virtual environment

For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```


---

## Monitoring

The prototype includes basic monitoring functionality through `monitor.py`.

It supports:

- Prediction logging
- Conversion probability tracking
- Basic drift detection using training statistics
- A monitoring section inside the Streamlit app

Prediction logs are stored in:

```text
logs.json
```

This file is generated automatically when predictions are made in the Streamlit app.


---

## Project Scope

This project is a proof-of-concept academic prototype. It demonstrates how behavioural lead data can support CRM decision-making through segmentation, prediction, prioritisation, and recommendation logic.

The prototype is not intended to replace a production CRM system. A production-level system would require stronger data governance, authentication, real CRM integration, continuous monitoring, automated retraining, and more extensive evaluation in a real organisational context.

---

## Limitations

Important limitations include:

- The dataset represents a simplified CRM environment.
- The model is trained primarily on behavioural engagement features.
- Zero-imputation may not fully capture the meaning of missing values.
- K-means clustering may oversimplify complex behavioural structures.
- The deployed recommendation logic is simplified and rule-based.
- The LLM-assisted judge-agent concept is demonstrated in the notebook, not fully deployed for all leads in the Streamlit app.
- The system requires further validation using real organisational CRM data.

---

## Future Work

Possible extensions include:

- Integration with a real CRM system
- Real-time lead ingestion
- Richer demographic, transactional, and firmographic features
- More advanced clustering methods
- Hyperparameter optimisation
- Explainable AI techniques such as SHAP
- Automated retraining
- Full feedback-loop integration
- Production-grade monitoring and alerting
- User authentication and role-based access

---

## Group Members

Group AAE:

- Amalie Hougaard Lang
- Ali Moghadas
- Emina Gracanin

Supervisor:

- Hamid Bekamiri

---

## Project Context

This project was developed as part of the **2nd Semester Project** in the Business Data Science programme at **Aalborg University, AAU Business School**.

The project title is:

**Lead Prioritisation in CRM Systems: From Prediction to Decision**
