# ===============================
# STEP 1: IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ===============================
# STEP 2: PAGE CONFIG
# ===============================
st.set_page_config(page_title="EduShield AI", layout="centered")

st.title("🎓 EduShield AI Dashboard")
st.subheader("Early Student Dropout & Mental Health Risk Prediction")


# ===============================
# STEP 3: LOAD DATA & TRAIN MODEL
# ===============================
data = pd.read_csv("hackathon.csv")

X = data.drop(["student_id", "dropout"], axis=1)
y = data["dropout"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)


# ===============================
# STEP 4: FILE UPLOAD + SAMPLE CSV
# ===============================
st.write("### 📂 Upload Student Data (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

st.download_button(
    label="📥 Download Sample CSV Format",
    data=data.to_csv(index=False),
    file_name="sample_student_data.csv",
    mime="text/csv"
)

st.info("""
🟢 Low Risk: Student is safe  
🟡 Medium Risk: Needs monitoring  
🔴 High Risk: Immediate intervention required
""")


# ===============================
# STEP 5: PREDICTION LOGIC
# ===============================
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)

    # Safe drop & feature alignment
    X_input = input_data.drop(columns=["student_id"], errors="ignore")
    X_input = X_input[X.columns]

    # Risk Score (0–100)
    probabilities = model.predict_proba(X_input)[:, 1] * 100
    input_data["Risk_Score"] = probabilities

    # Risk Level
    def risk_label(score):
        if score < 40:
            return "🟢 Low Risk"
        elif score < 70:
            return "🟡 Medium Risk"
        else:
            return "🔴 High Risk"

    input_data["Risk_Level"] = input_data["Risk_Score"].apply(risk_label)


    # ===============================
    # STEP 6: RECOMMENDATION ENGINE
    # ===============================
    def recommend_action(row):
        stress = row["stress_level"] if "stress_level" in row else 3

        if row["Risk_Score"] >= 70:
            if stress >= 4:
                return "Immediate counseling + mentor assigned"
            else:
                return "Academic support + attendance monitoring"
        elif row["Risk_Score"] >= 40:
            return "Regular mentoring and stress check"
        else:
            return "No action needed"

    input_data["Recommended_Action"] = input_data.apply(recommend_action, axis=1)


    # ===============================
    # STEP 7: EXPLAINABLE AI (WHY?)
    # ===============================
    def risk_reason(row):
        reasons = []
        if row["attendance_percentage"] < 60:
            reasons.append("Low attendance")
        if "stress_level" in row and row["stress_level"] >= 4:
            reasons.append("High stress")
        if row["internal_marks"] < 40:
            reasons.append("Poor academic performance")
        return ", ".join(reasons) if reasons else "No major risk factors"

    input_data["Risk_Reason"] = input_data.apply(risk_reason, axis=1)


    # ===============================
    # STEP 8: SUMMARY + RESULTS
    # ===============================
    high_risk_count = (input_data["Risk_Level"] == "🔴 High Risk").sum()
    st.warning(f"⚠️ {high_risk_count} students require immediate attention.")

    st.write("### 📊 Prediction Results")
    st.dataframe(input_data)


# ===============================
# FOOTER
# ===============================
st.divider()
st.caption("EduShield AI • Team VisionX • Hackathon MVP")
