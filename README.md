# 🎯 Lead Prioritisation & CRM Decision Support

> Machine learning decision-support system that transforms behavioural lead data into **conversion predictions, lead priorities, customer segments, and actionable CRM recommendations**.

🌐 **[Try the Live Application →](https://lead-prioritisation-in-crm-systems-group-aae.streamlit.app/)**

This project explores how machine learning can move beyond prediction and support real **sales and marketing decisions**.

The solution combines **behavioural segmentation, predictive modelling, lead scoring, recommendation logic, and human-in-the-loop decision support** in an interactive CRM-style application.

---

## 🏗️ System Overview

```text
                 Behavioural Lead Data
                          │
                          ▼
                  Data Preprocessing
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          K-Means + PCA      Predictive Models
          Segmentation       LR · RF · XGBoost
                │                   │
                │                   ▼
                │          Conversion Probability
                │                   │
                └─────────┬─────────┘
                          ▼
                    Lead Priority
                          │
                          ▼
               CRM Recommendation
                          │
                          ▼
                 Human-in-the-Loop
                          │
                          ▼
                Streamlit Application
```

---

## ✨ What We Built

The project covers the full journey from raw behavioural data to an actionable business recommendation:

- Engineered behavioural features from lead engagement data
- Identified lead segments using **K-Means clustering**
- Used **PCA** to analyse and visualise behavioural patterns
- Compared **Logistic Regression, Random Forest, and XGBoost**
- Evaluated models using multiple classification metrics
- Selected **XGBoost** as the final conversion prediction model
- Converted predicted probabilities into operational lead priorities
- Developed CRM recommendation logic for sales and marketing actions
- Incorporated contact-permission constraints into recommendations
- Explored **human-in-the-loop** decision support
- Built and deployed an interactive **Streamlit application**
- Added prediction logging and basic drift monitoring

---

## 🧠 Machine Learning Approach

### 1. Behavioural Segmentation

**K-Means clustering** was used to identify groups of leads with similar behavioural patterns.

**PCA** supported dimensionality reduction and visualisation of the resulting segments.

```text
Behavioural Features
        ↓
Standardisation
        ↓
K-Means Clustering
        ↓
PCA Analysis
        ↓
Lead Segments
```

This provides an unsupervised perspective on how different types of leads interact with the business.

### 2. Conversion Prediction

Three supervised learning approaches were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Models were compared using:

`Accuracy` · `Precision` · `Recall` · `F1-score` · `ROC-AUC`

**XGBoost was selected as the final model** based on its overall predictive performance.

---

## 💡 From Prediction to Business Action

A key focus of the project was translating model output into something that could actually support a sales team.

```text
Lead
 ↓
Conversion Probability
 ↓
Priority
 ↓
Recommended Action
 ↓
Human Review
 ↓
CRM Action
```

Predicted probabilities are converted into operational priority levels:

| Conversion Probability | Priority |
|---:|---|
| ≥ 70% | 🔴 High |
| 40–69% | 🟡 Medium |
| < 40% | 🟢 Low |

The priority score then contributes to recommendation logic that can suggest actions such as:

- Immediate sales follow-up
- Targeted nurturing
- Automated marketing
- Alternative follow-up based on contact permissions

This turns the ML model from a prediction tool into a **decision-support system**.

---

## 🖥️ Live Application

The analytical workflow was implemented as an interactive **Streamlit CRM prototype**.

The application allows users to:

- Select and inspect leads
- Modify behavioural variables
- Generate conversion probabilities
- View lead priority levels
- Receive recommended CRM actions
- Identify cases requiring human review
- Log predictions
- Monitor prediction behaviour

🌐 **[Open the Live Application →](https://lead-prioritisation-in-crm-systems-group-aae.streamlit.app/)**

---

## 🛠️ Tech Stack

| Area | Technologies |
|---|---|
| **Data Analysis** | Python, Pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Segmentation** | K-Means, PCA |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Application** | Streamlit |
| **Monitoring** | Prediction logging, drift detection |
| **Development** | Jupyter, Git, GitHub |

---

## 📁 Project Structure

```text
Lead-Prioritisation-in-CRM-Systems/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── RQ1_Clustering.ipynb
│   └── RQ2_+_RQ3.ipynb
│
├── artifacts/
│   ├── model/
│   └── metrics/
│
├── data/
├── output/
├── streamlit_app.py
├── monitor.py
├── requirements.txt
└── README.md
```

---

## 🎯 Skills Demonstrated

`Python` · `Data Science` · `Machine Learning` · `XGBoost` · `Random Forest` · `K-Means` · `PCA` · `Feature Engineering` · `Model Evaluation` · `Lead Scoring` · `Decision Support` · `Streamlit` · `Model Monitoring`

---

## 🚀 Key Takeaway

This project demonstrates how machine learning can be applied across the full path from **behavioural data to business decision support**:

**Segment → Predict → Prioritise → Recommend → Review**

Rather than stopping at model performance, the project translates predictions into a working CRM-style application that connects **data science with practical sales and marketing decisions**.

---

### 👥 Project Context

Developed as a group project for the **Business Data Science programme at Aalborg University**.

**Group AAE:**  
Amalie Hougaard Lang · Ali Moghadas · Emina Gracanin
