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
# CSS
# -------------------------------
st.markdown("""
<style>
.main { background-color: #f6f8fb; }

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 35px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 14px 35px rgba(15, 23, 42, 0.25);
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
    border-radius: 20px;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

.dark-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 24px;
    border-radius: 20px;
    color: white;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.18);
    margin-bottom: 18px;
}

.high { border-left: 8px solid #16a34a; }
.medium { border-left: 8px solid #f59e0b; }
.low { border-left: 8px solid #dc2626; }

.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background-color: #e0f2fe;
    color: #075985;
    font-size: 13px;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 6px;
}

.badge-green { background-color: #dcfce7; color: #166534; }
.badge-yellow { background-color: #fef3c7; color: #92400e; }
.badge-red { background-color: #fee2e2; color: #991b1b; }
.badge-purple { background-color: #ede9fe; color: #5b21b6; }

.section-title {
    font-size: 26px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 12px;
}

.small-text {
    font-size: 13px;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL + DATA
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
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

    return df

model = load_model()
df = load_data()

MODEL_FEATURES = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "EngagementScore",
    "TimePerVisit"
]

# -------------------------------
# LOGIC FUNCTIONS
# -------------------------------
def assign_priority(probability):
    if probability >= 0.70:
        return "High Priority"
    elif probability >= 0.40:
        return "Medium Priority"
    else:
        return "Low Priority"


def assign_segment(engagement_score, time_per_visit):
    if engagement_score >= 20:
        return "High Engagement Segment"
    elif time_per_visit >= 150 or engagement_score >= 5:
        return "Medium Engagement Segment"
    else:
        return "Low Engagement Segment"


def contact_permission_status(do_not_email, do_not_call):
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


def generate_recommendation(priority, source, probability, contact_status, recommendation_category, segment):
    return (
        f"{recommendation_category}: {priority} lead from {source} "
        f"with predicted conversion probability of {probability:.1%}. "
        f"Segment: {segment}. Contact status: {contact_status}."
    )


@st.cache_data
def build_recommendation_export(data):
    export_df = data.copy()

    probabilities = model.predict_proba(export_df[MODEL_FEATURES])[:, 1]
    export_df["PredictedProbability"] = probabilities
    export_df["Priority"] = export_df["PredictedProbability"].apply(assign_priority)

    export_df["LeadSegment"] = export_df.apply(
        lambda row: assign_segment(row["EngagementScore"], row["TimePerVisit"]),
        axis=1
    )

    export_df["ContactPermissionStatus"] = export_df.apply(
        lambda row: contact_permission_status(row["Do Not Email"], row["Do Not Call"]),
        axis=1
    )

    export_df["RecommendationCategory"] = export_df.apply(
        lambda row: assign_recommendation_category(
            row["Priority"],
            row["ContactPermissionStatus"]
        ),
        axis=1
    )

    export_df["AgentRecommendation"] = export_df.apply(
        lambda row: generate_recommendation(
            row["Priority"],
            row["Lead Source"],
            row["PredictedProbability"],
            row["ContactPermissionStatus"],
            row["RecommendationCategory"],
            row["LeadSegment"]
        ),
        axis=1
    )

    return export_df


recommendation_df = build_recommendation_export(df)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("CRM Control Panel")
st.sidebar.caption(f"Model version: {MODEL_VERSION}")

page = st.sidebar.radio(
    "Navigation",
    [
        "Lead Evaluation",
        "Recommendation Overview",
        "Monitoring Dashboard"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Pipeline")
st.sidebar.write("Lead Data → XGBoost → Priority → Recommendation")

# -------------------------------
# HERO
# -------------------------------
st.markdown("""
<div class="hero">
    <h1>Lead Prioritisation in CRM Systems</h1>
    <p>CRM decision-support prototype using XGBoost predictions, priority assignment, contact constraints, and agent-style recommendations.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PAGE 1: LEAD EVALUATION
# =========================================================
if page == "Lead Evaluation":

    selected_index = st.sidebar.selectbox(
        "Select lead from dataset",
        df.index
    )

    selected = df.loc[selected_index]

    st.sidebar.markdown("---")
    st.sidebar.write("**Lead Source:**", selected["Lead Source"])
    st.sidebar.write("**Do Not Email:**", selected["Do Not Email"])
    st.sidebar.write("**Do Not Call:**", selected["Do Not Call"])

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
        contact_status = contact_permission_status(
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
            st.markdown('<span class="badge badge-green">High Engagement Segment</span>', unsafe_allow_html=True)
        elif lead_segment == "Medium Engagement Segment":
            st.markdown('<span class="badge badge-yellow">Medium Engagement Segment</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-red">Low Engagement Segment</span>', unsafe_allow_html=True)

        st.markdown(f'<span class="badge">Source: {selected["Lead Source"]}</span>', unsafe_allow_html=True)

        if contact_status != "Direct contact allowed":
            st.markdown(f'<span class="badge badge-red">{contact_status}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-green">Direct Contact Allowed</span>', unsafe_allow_html=True)

        st.write(f"**Engagement Score:** {engagement_score:.2f}")
        st.write(f"**Time Per Visit:** {time_per_visit:.2f}")

        st.markdown('</div>', unsafe_allow_html=True)

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

        recommendation = generate_recommendation(
            priority,
            selected["Lead Source"],
            probability,
            contact_status,
            recommendation_category,
            lead_segment
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

        if priority == "High Priority":
            card_class = "card high"
        elif priority == "Medium Priority":
            card_class = "card medium"
        else:
            card_class = "card low"

        st.markdown(f"""
        <div class="{card_class}">
            <h3>Agent-Based Recommendation Output</h3>
            <p>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)

        log_prediction(features, prediction, probability)

        drift = check_drift(features)
        if drift:
            st.warning(f"⚠️ Potential drift detected: {', '.join(drift)}")

# =========================================================
# PAGE 2: RECOMMENDATION OVERVIEW
# =========================================================
elif page == "Recommendation Overview":

    st.markdown('<div class="section-title">Recommendation Overview Dashboard</div>', unsafe_allow_html=True)

    high_count = (recommendation_df["Priority"] == "High Priority").sum()
    medium_count = (recommendation_df["Priority"] == "Medium Priority").sum()
    low_count = (recommendation_df["Priority"] == "Low Priority").sum()
    restricted_high = (
        (recommendation_df["Priority"] == "High Priority") &
        (recommendation_df["ContactPermissionStatus"] != "Direct contact allowed")
    ).sum()

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("High Priority Leads", f"{high_count:,}")

    with k2:
        st.metric("Medium Priority Leads", f"{medium_count:,}")

    with k3:
        st.metric("Low Priority Leads", f"{low_count:,}")

    with k4:
        st.metric("Restricted High Priority", f"{restricted_high:,}")

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Priority Distribution")
        priority_counts = recommendation_df["Priority"].value_counts()
        st.bar_chart(priority_counts)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recommendation Categories")
        category_counts = recommendation_df["RecommendationCategory"].value_counts()
        st.bar_chart(category_counts)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("CRM Recommendation Export")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        priority_filter = st.multiselect(
            "Filter by Priority",
            options=sorted(recommendation_df["Priority"].unique()),
            default=list(recommendation_df["Priority"].unique())
        )

    with filter_col2:
        category_filter = st.multiselect(
            "Filter by Recommendation Category",
            options=sorted(recommendation_df["RecommendationCategory"].unique()),
            default=list(recommendation_df["RecommendationCategory"].unique())
        )

    with filter_col3:
        source_filter = st.multiselect(
            "Filter by Lead Source",
            options=sorted(recommendation_df["Lead Source"].unique()),
            default=list(recommendation_df["Lead Source"].unique())[:5]
        )

    filtered_export = recommendation_df[
        (recommendation_df["Priority"].isin(priority_filter)) &
        (recommendation_df["RecommendationCategory"].isin(category_filter)) &
        (recommendation_df["Lead Source"].isin(source_filter))
    ]

    display_cols = [
        "Prospect ID",
        "Lead Source",
        "PredictedProbability",
        "Priority",
        "ContactPermissionStatus",
        "RecommendationCategory",
        "AgentRecommendation"
    ]

    st.dataframe(
        filtered_export[display_cols].sort_values(
            "PredictedProbability",
            ascending=False
        ),
        use_container_width=True,
        height=420
    )

    csv = filtered_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CRM Recommendation Export",
        data=csv,
        file_name="crm_recommendation_export.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================================================
# PAGE 3: MONITORING
# =========================================================
elif page == "Monitoring Dashboard":

    st.markdown('<div class="section-title">Monitoring Dashboard</div>', unsafe_allow_html=True)

    st.caption(
        "This section tracks prediction activity and supports basic monitoring of model behaviour over time."
    )

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
                st.subheader("Prediction Probability Over Time")
                st.line_chart(logs["probability"])

        with chart2:
            if "prediction" in logs.columns:
                st.subheader("Prediction Class Distribution")
                st.bar_chart(logs["prediction"].value_counts())

    else:
        st.info("No predictions logged yet.")
