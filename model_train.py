# ===============================
# STEP 1: IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ===============================
# STEP 2: LOAD DATASET
# ===============================
data = pd.read_csv("hackathon.csv")

# Features and Target
X = data.drop(["student_id", "dropout"], axis=1)
y = data["dropout"]


# ===============================
# STEP 3: TRAIN–TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# STEP 4: TRAIN MODEL
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ===============================
# STEP 5: EXPLAINABLE AI (WHY?)
# ===============================
feature_names = X.columns
coefficients = model.coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": np.abs(coefficients)
}).sort_values(by="Importance", ascending=False)

print("\nTop Risk Factors:")
print(importance_df.head(5))


# ===============================
# STEP 6: MODEL EVALUATION
# ===============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ===============================
# STEP 7: RISK SCORE CALCULATION
# ===============================
risk_scores = model.predict_proba(X_test)[:, 1] * 100

risk_df = X_test.copy()
risk_df["Risk_Score"] = risk_scores

risk_df["Risk_Level"] = risk_df["Risk_Score"].apply(
    lambda x: "Low" if x < 40 else "Medium" if x < 70 else "High"
)


# ===============================
# STEP 8: RECOMMENDATION ENGINE
# ===============================
def recommend_action(row):
    if row["Risk_Score"] >= 70:
        if row["stress_level"] >= 4:
            return "Immediate counseling + mentor assigned"
        else:
            return "Academic support + attendance monitoring"
    elif row["Risk_Score"] >= 40:
        return "Regular mentoring and stress check"
    else:
        return "No action needed"

risk_df["Recommended_Action"] = risk_df.apply(recommend_action, axis=1)

print("\nSample Risk Predictions:")
print(risk_df[["Risk_Score", "Risk_Level", "Recommended_Action"]].head())


# ===============================
# STEP 9: DATA VISUALIZATION (NO SEABORN)
# ===============================

# 1️⃣ Dropout Distribution
plt.figure()
data["dropout"].value_counts().plot(kind="bar")
plt.title("Dropout Distribution (0 = No, 1 = Yes)")
plt.xlabel("Dropout")
plt.ylabel("Number of Students")
plt.show()

# 2️⃣ Attendance vs Dropout
plt.figure()
plt.boxplot([
    data[data["dropout"] == 0]["attendance_percentage"],
    data[data["dropout"] == 1]["attendance_percentage"]
])
plt.xticks([1, 2], ["No Dropout", "Dropout"])
plt.title("Attendance vs Dropout")
plt.ylabel("Attendance Percentage")
plt.show()

# 3️⃣ Stress Level vs Dropout
plt.figure()
plt.boxplot([
    data[data["dropout"] == 0]["stress_level"],
    data[data["dropout"] == 1]["stress_level"]
])
plt.xticks([1, 2], ["No Dropout", "Dropout"])
plt.title("Stress Level vs Dropout")
plt.ylabel("Stress Level")
plt.show()
