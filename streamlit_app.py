import os
import joblib
import pandas as pd
import streamlit as st

from monitor import log_prediction, check_drift

# -------------------------------
# CONFIG
# -------------------------------
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

.dark-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 24px;
    border-radius: 18px;
    color: white;
    box-shadow: 0px 8px 22px rgba(15, 23, 42, 0.15);
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
# LOGIC FUNCTIONS
# -------------------------------
def assign_segment(engagement_score, time_per_visit):
    if engagement_score >= 20:
        return "High Engagement Segment"
    elif time_per_visit >= 150 or engagement_score >= 5:
        return "Medium Engagement Segment"
    else:
        return "Low Engagement Segment"


def assign_priority(probability):
    if probability >= 0.70:
        return "High Priority"
    elif probability >= 0.40:
        return "Medium Priority"
    else:
        return "Low Priority"


def get_contact_status(do_not_email, do_not_call):
    if do_not_email == "Yes" and do_not_call == "Yes":
        return "No direct contact allowed"
    elif do_not_email == "Yes":
        return "Email restricted"
    elif do_not_call == "Yes":
        return "Phone restricted"
    else:
        return "Direct contact allowed"


def assign_recommendation_category(priority, contact_status):
    if priority == "High Priority" and contact_status == "Direct contact allowed":
        return "Immediate Sales Follow-up"
    elif priority == "High Priority" and contact_status != "Direct contact allowed":
        return "Compliant Alternative Follow-up"
    elif priority == "Medium Priority":
        return "Targeted Nurturing"
    else:
        return "Automated Low-Intensity Marketing"


def generate_recommendation(recommendation_category):
    if recommendation_category == "Immediate Sales Follow-up":
        return "Contact lead immediately through sales outreach."
    elif recommendation_category == "Targeted Nurturing":
        return "Place lead in targeted nurturing campaign."
    elif recommendation_category == "Compliant Alternative Follow-up":
        return "Use compliant alternative communication strategy."
    else:
        return "Use automated low-intensity marketing workflow."


def human_review_status(priority, contact_status):
    if priority == "High Priority":
        return "Required"
    elif contact_status != "Direct contact allowed":
        return "Required"
    else:
        return "Not Required"


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
    <p>AI-assisted CRM prototype for predicting conversion potential, assigning lead priority, and supporting marketing/sales decisions.</p>
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
    contact_status = get_contact_status(
        selected["Do Not Email"],
        selected["Do Not Call"]
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Lead Context")

    st.write(f"**Prospect ID:** `{selected['Prospect ID']}`")
    st.write(f"**Lead Source:** {selected['Lead Source']}")
    st.write(f"**Do Not Email:** {selected['Do Not Email']}")
    st.write(f"**Do Not Call:** {selected['Do Not Call']}")

    st.markdown("---")

    if lead_segment == "High Engagement Segment":
        st.markdown(
            '<span class="badge badge-green">High Engagement Segment</span>',
            unsafe_allow_html=True
        )
    elif lead_segment == "Medium Engagement Segment":
        st.markdown(
            '<span class="badge badge-yellow">Medium Engagement Segment</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="badge badge-red">Low Engagement Segment</span>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<span class="badge">Source: {selected["Lead Source"]}</span>',
        unsafe_allow_html=True
    )

    if contact_status != "Direct contact allowed":
        st.markdown(
            f'<span class="badge badge-red">{contact_status}</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="badge badge-green">Direct Contact Allowed</span>',
            unsafe_allow_html=True
        )

    st.write(f"**Engagement Score:** {engagement_score:.2f}")
    st.write(f"**Time Per Visit:** {time_per_visit:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

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

    recommendation_category = assign_recommendation_category(
        priority,
        contact_status
    )

    recommendation = generate_recommendation(recommendation_category)

    human_review = human_review_status(
        priority,
        contact_status
    )

    st.markdown("---")
    st.subheader("CRM Decision Output")

    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

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

    with result_col4:
        st.info(recommendation_category)

    review_col1, review_col2 = st.columns(2)

    with review_col1:
        st.markdown("""
        <div class="dark-card">
            <h3>AI Recommendation</h3>
        """, unsafe_allow_html=True)

        st.write(recommendation)

        st.markdown("</div>", unsafe_allow_html=True)

    with review_col2:
        st.markdown("""
        <div class="dark-card">
            <h3>Human-in-the-Loop</h3>
        """, unsafe_allow_html=True)

        if human_review == "Required":
            st.warning("Marketing or sales approval required before action.")
        else:
            st.success("Can be processed automatically.")

        st.markdown("</div>", unsafe_allow_html=True)

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
