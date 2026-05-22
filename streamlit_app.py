import streamlit as st
import joblib
import pandas as pd
import os

from monitor import log_prediction, check_drift

MODEL_VERSION = "v1.0"
MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Lead Scoring.csv"

st.set_page_config(
    page_title="Lead Prioritisation CRM Tool",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f6f8fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #dbeafe;
}

.card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0px 8px 22px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

.high {
    border-left: 8px solid #16a34a;
}

.medium {
    border-left: 8px solid #f59e0b;
}

.low {
    border-left: 8px solid #dc2626;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background-color: #e0f2fe;
    color: #075985;
    font-size: 13px;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}

.badge-green {
    background-color: #dcfce7;
    color: #166534;
}

.badge-yellow {
    background-color: #fef3c7;
    color: #92400e;
}

.badge-red {
    background-color: #fee2e2;
    color: #991b1b;
}

.small-text {
    font-size: 13px;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_PATH)

required_columns = [
    "Prospect ID",
    "Lead Source",
    "Do Not Email",
    "Do Not Call",
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "Converted"
]

df = df[required_columns].dropna()

df["EngagementScore"] = df["TotalVisits"] * df["Page Views Per Visit"]
df["TimePerVisit"] = df["Total Time Spent on Website"] / (df["TotalVisits"] + 1)

# -------------------------------
# LEAD SEGMENT LOGIC
# -------------------------------
def assign_segment(engagement_score, time_per_visit):
    if engagement_score >= 20:
        return "High Engagement Segment"
    elif time_per_visit >= 150 or engagement_score >= 5:
        return "Medium Engagement Segment"
    else:
        return "Low Engagement Segment"

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("CRM Control Panel")
st.sidebar.caption(f"Model version: {MODEL_VERSION}")

selected_index = st.sidebar.selectbox(
    "Select lead from dataset",
    df.index
)

selected = df.loc[selected_index]

st.sidebar.markdown("---")
st.sidebar.write("**Lead Source:**", selected["Lead Source"])
st.sidebar.write("**Do Not Email:**", selected["Do Not Email"])
st.sidebar.write("**Do Not Call:**", selected["Do Not Call"])

# -------------------------------
# HERO
# -------------------------------
st.markdown("""
<div class="hero">
    <h1>Lead Prioritisation in CRM Systems</h1>
    <p>CRM decision-support prototype for predicting conversion potential, assigning lead priority, and generating marketing/sales recommendations.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# MAIN LAYOUT
# -------------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Lead Behaviour Input")

    total_visits = st.number_input(
        "Total Visits",
        value=float(selected["TotalVisits"]),
        format="%.0f"
    )

    time_spent = st.number_input(
        "Total Time Spent on Website",
        value=float(selected["Total Time Spent on Website"]),
        format="%.0f"
    )

    page_views = st.number_input(
        "Page Views Per Visit",
        value=float(selected["Page Views Per Visit"]),
        format="%.2f"
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    engagement_score = total_visits * page_views
    time_per_visit = time_spent / (total_visits + 1)
    lead_segment = assign_segment(engagement_score, time_per_visit)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Lead Context")

    st.write(f"**Prospect ID:** `{selected['Prospect ID']}`")
    st.write(f"**Lead Source:** {selected['Lead Source']}")
    st.write(f"**Do Not Email:** {selected['Do Not Email']}")
    st.write(f"**Do Not Call:** {selected['Do Not Call']}")

    st.markdown("---")

    if lead_segment == "High Engagement Segment":
        st.markdown('<span class="badge badge-green">High Engagement Segment</span>', unsafe_allow_html=True)
    elif lead_segment == "Medium Engagement Segment":
        st.markdown('<span class="badge badge-yellow">Medium Engagement Segment</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">Low Engagement Segment</span>', unsafe_allow_html=True)

    st.markdown(f'<span class="badge">Source: {selected["Lead Source"]}</span>', unsafe_allow_html=True)

    if selected["Do Not Email"] == "Yes":
        st.markdown('<span class="badge badge-red">Email Restricted</span>', unsafe_allow_html=True)

    if selected["Do Not Call"] == "Yes":
        st.markdown('<span class="badge badge-red">Call Restricted</span>', unsafe_allow_html=True)

    st.write(f"**Engagement Score:** {engagement_score:.2f}")
    st.write(f"**Time Per Visit:** {time_per_visit:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PRIORITY + RECOMMENDATION
# -------------------------------
def assign_priority(probability):
    if probability >= 0.70:
        return "High Priority"
    elif probability >= 0.40:
        return "Medium Priority"
    else:
        return "Low Priority"


def generate_recommendation(priority, source, do_not_email, do_not_call, lead_segment):

    if priority == "High Priority":

        if do_not_call == "Yes" and do_not_email == "Yes":
            return (
                f"This is a high-priority lead from {source} in the {lead_segment.lower()}. "
                "However, both phone and email contact are restricted. Avoid direct outreach and use compliant alternative CRM follow-up."
            )

        elif do_not_call == "Yes":
            return (
                f"This is a high-priority lead from {source} in the {lead_segment.lower()}. "
                "Phone contact is restricted. Use personalised email or digital follow-up."
            )

        elif do_not_email == "Yes":
            return (
                f"This is a high-priority lead from {source} in the {lead_segment.lower()}. "
                "Email contact is restricted. Direct phone follow-up is recommended."
            )

        else:
            return (
                f"This is a high-priority lead from {source} in the {lead_segment.lower()}. "
                "Immediate sales follow-up is recommended."
            )

    elif priority == "Medium Priority":

        return (
            f"This is a medium-priority lead from {source} in the {lead_segment.lower()}. "
            "Use targeted nurturing campaigns and monitor future activity."
        )

    else:
        return (
            f"This is a low-priority lead from {source} in the {lead_segment.lower()}. "
            "Use low-intensity automated marketing rather than direct sales engagement."
        )


# -------------------------------
# EVALUATE
# -------------------------------
if st.button("🚀 Evaluate Lead", use_container_width=True):

    features = pd.DataFrame([{
        "TotalVisits": total_visits,
        "Total Time Spent on Website": time_spent,
        "Page Views Per Visit": page_views,
        "EngagementScore": engagement_score,
        "TimePerVisit": time_per_visit
    }])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    score = int(probability * 100)
    priority = assign_priority(probability)

    recommendation = generate_recommendation(
        priority,
        selected["Lead Source"],
        selected["Do Not Email"],
        selected["Do Not Call"],
        lead_segment
    )

    st.markdown("---")
    st.subheader("CRM Decision Output")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Lead Score", f"{score}/100")

    with result_col2:
        st.metric("Conversion Probability", f"{probability:.2%}")

    with result_col3:
        if priority == "High Priority":
            st.success(priority)
        elif priority == "Medium Priority":
            st.warning(priority)
        else:
            st.error(priority)

    if priority == "High Priority":
        card_class = "card high"
    elif priority == "Medium Priority":
        card_class = "card medium"
    else:
        card_class = "card low"

    st.markdown(f"""
    <div class="{card_class}">
        <h3>Agent-Based Recommendation</h3>
        <p>{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)

    log_prediction(features, prediction, probability)

    drift = check_drift(features)
    if drift:
        st.warning(f"⚠️ Potential drift detected: {', '.join(drift)}")

# -------------------------------
# MONITORING
# -------------------------------
st.markdown("---")
st.subheader("Monitoring Dashboard")

if os.path.exists("logs.json") and os.path.getsize("logs.json") > 0:

    logs = pd.read_json("logs.json", lines=True)

    m1, m2 = st.columns(2)

    with m1:
        st.metric("Total Predictions", len(logs))

    with m2:
        if "probability" in logs.columns:
            st.metric("Average Probability", f"{logs['probability'].mean():.2%}")

    chart1, chart2 = st.columns(2)

    with chart1:
        if "probability" in logs.columns:
            st.line_chart(logs["probability"])

    with chart2:
        if "prediction" in logs.columns:
            st.bar_chart(logs["prediction"].value_counts())

else:
    st.info("No predictions logged yet.")