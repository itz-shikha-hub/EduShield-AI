import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("hackathon.csv")

# 1. Dropout Distribution
plt.figure()
sns.countplot(x="dropout", data=data)
plt.title("Dropout Distribution (0 = No, 1 = Yes)")
plt.show()

# 2. Attendance vs Dropout
plt.figure()
sns.boxplot(x="dropout", y="attendance_percentage", data=data)
plt.title("Attendance vs Dropout")
plt.show()

# 3. Stress Level vs Dropout
plt.figure()
sns.boxplot(x="dropout", y="stress_level", data=data)
plt.title("Stress Level vs Dropout")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
