import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
print("Gym Membership Churn Prediction")
print("--------------------------------")

# Loading dataset
df = pd.read_csv("data/gym_churn_us.csv")

# Validating dataset structure
required_columns = [
    "gender",
    "Near_Location",
    "Partner",
    "Promo_friends",
    "Phone",
    "Contract_period",
    "Group_visits",
    "Age",
    "Avg_additional_charges_total",
    "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total",
    "Avg_class_frequency_current_month",
    "Churn"]
missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
if not set(df["Churn"].unique()).issubset({0, 1}):
    raise ValueError("Churn must contain only 0 and 1.")
print("Dataset validation successful!")
print("\nDataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# Target distribution
print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Churn distribution visualization
churn_counts = df["Churn"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(["Stayed", "Churned"], churn_counts)
plt.title("Gym Membership Churn Distribution")
plt.xlabel("Membership Status")
plt.ylabel("Number of Members")
# plt.show()

# Churn rate by contract period
churn_by_contract = df.groupby("Contract_period")["Churn"].mean() * 100

print("\nChurn rate by contract period:")
print(churn_by_contract)

plt.figure(figsize=(7, 4))
plt.bar(
    churn_by_contract.index.astype(str),
    churn_by_contract.values
)
plt.title("Churn Rate by Contract Period")
plt.xlabel("Contract Period (Months)")
plt.ylabel("Churn Rate (%)")
# plt.show()

# Churn rate by selected member characteristics
binary_features = [
    "gender",
    "Near_Location",
    "Partner",
    "Promo_friends",
    "Phone",
    "Group_visits"
]

for feature in binary_features:
    churn_rate = df.groupby(feature)["Churn"].mean() * 100

    print(f"\nChurn rate by {feature}:")
    print(churn_rate)

# Visualize churn rate by selected member characteristics
for feature in binary_features:
    churn_rate = df.groupby(feature)["Churn"].mean() * 100

    plt.figure(figsize=(6, 4))
    plt.bar(
        churn_rate.index.astype(str),
        churn_rate.values
    )
    plt.title(f"Churn Rate by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Churn Rate (%)")
    # plt.show()

# Age distribution by churn status
print("\nAverage age by churn status:")
print(df.groupby("Churn")["Age"].mean())

df.boxplot(column="Age", by="Churn")
plt.title("Age Distribution by Churn Status")
plt.suptitle("")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Age")
# plt.show()

# Lifetime distribution by churn status
print("\nAverage lifetime by churn status:")
print(df.groupby("Churn")["Lifetime"].mean())

df.boxplot(column="Lifetime", by="Churn")
plt.title("Lifetime Distribution by Churn Status")
plt.suptitle("")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Lifetime")
# plt.show()

# Additional charges by churn status
print("\nAverage additional charges by churn status:")
print(df.groupby("Churn")["Avg_additional_charges_total"].mean())

df.boxplot(column="Avg_additional_charges_total", by="Churn")
plt.title("Additional Charges by Churn Status")
plt.suptitle("")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Average Additional Charges")
# plt.show()

# Months to end of contract by churn status
print("\nAverage months to end of contract by churn status:")
print(df.groupby("Churn")["Month_to_end_contract"].mean())

df.boxplot(column="Month_to_end_contract", by="Churn")
plt.title("Months to End of Contract by Churn Status")
plt.suptitle("")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Months to End of Contract")
# plt.show()

# Overall class frequency by churn status
print("\nAverage class frequency by churn status:")
print(df.groupby("Churn")["Avg_class_frequency_total"].mean())

df.boxplot(column="Avg_class_frequency_total", by="Churn")
plt.title("Overall Class Frequency by Churn Status")
plt.suptitle("")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Average Class Frequency")
# plt.show()

# Current month class frequency by churn status
print("\nAverage current-month class frequency by churn status:")
print(df.groupby("Churn")["Avg_class_frequency_current_month"].mean())

df.boxplot(column="Avg_class_frequency_current_month", by="Churn")
plt.title("Current-Month Class Frequency by Churn Status")
plt.suptitle("")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Current-Month Class Frequency")
# plt.show()

# Creating activity level groups
df["Activity_Level"] = pd.qcut(
    df["Avg_class_frequency_current_month"],
    q=4,
    labels=["Low", "Medium-Low", "Medium-High", "High"]
)

# Churn rate by activity level
churn_by_activity = df.groupby(
    "Activity_Level",
    observed=True
)["Churn"].mean() * 100

print("\nChurn rate by activity level:")
print(churn_by_activity)

plt.figure(figsize=(7, 4))
plt.bar(
    churn_by_activity.index,
    churn_by_activity.values
)
plt.title("Churn Rate by Current-Month Activity Level")
plt.xlabel("Activity Level")
plt.ylabel("Churn Rate (%)")
# plt.show()

# Creating activity change
df["Activity_Change"] = (
    df["Avg_class_frequency_current_month"]
    - df["Avg_class_frequency_total"]
)

print("\nAverage activity change by churn status:")
print(df.groupby("Churn")["Activity_Change"].mean())

df.boxplot(column="Activity_Change", by="Churn")
plt.title("Activity Change by Churn Status")
plt.suptitle("")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Activity Change")
# plt.show()

# Correlation matrix
correlation_matrix = df.corr(numeric_only=True)

print("\nCorrelation with Churn:")
print(
    correlation_matrix["Churn"]
    .sort_values()
)

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Matrix of Gym Membership Variables")
# plt.show()

# Selecting features and target
features = [
    "gender",
    "Near_Location",
    "Partner",
    "Promo_friends",
    "Phone",
    "Contract_period",
    "Group_visits",
    "Age",
    "Avg_additional_charges_total",
    "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total",
    "Avg_class_frequency_current_month"
]

X = df[features]
y = df["Churn"]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Standardizing the features for Logistic Regression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)

# Training the Logistic Regression model
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(random_state=42)

logistic_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")

# Generating predictions
y_pred = logistic_model.predict(X_test_scaled)

print("\nLogistic Regression predictions generated successfully!")
print("Number of predictions:", len(y_pred))

# Evaluating the Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC calculation
from sklearn.metrics import roc_auc_score

y_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

print("\nLogistic Regression ROC-AUC:", roc_auc)

# ROC curve plot
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
# plt.show()

# Training Random Forest model
from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# Generating Random Forest predictions
y_pred_rf = random_forest_model.predict(X_test)

print("\nRandom Forest predictions generated successfully!")
print("Number of predictions:", len(y_pred_rf))

# Random Forest evaluation
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Calculating Random Forest ROC-AUC
y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]

roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\nRandom Forest ROC-AUC:", roc_auc_rf)

# machine learning models comparison
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, y_pred_rf)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, y_pred_rf)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred_rf)
    ],
    "ROC-AUC": [roc_auc, roc_auc_rf]
})

print("\nModel Comparison:")
print(model_comparison.round(4))

# saving model comparison
import os

os.makedirs("outputs", exist_ok=True)

model_comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")

# Random Forest feature importance
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

# Visualization of feature importance
feature_importance.plot(
    kind="barh",
    x="Feature",
    y="Importance",
    legend=False,
    figsize=(8, 6)
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
# plt.show()

# Logistic Regression coefficients
logistic_coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": logistic_model.coef_[0]
})

logistic_coefficients = logistic_coefficients.sort_values(
    by="Coefficient"
)

print("\nLogistic Regression Coefficients:")
print(logistic_coefficients)

# Visualizing coefficients
logistic_coefficients.plot(
    kind="barh",
    x="Feature",
    y="Coefficient",
    legend=False,
    figsize=(8, 6)
)

plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.axvline(x=0)
plt.tight_layout()
plt.show()

# Creating churn insights
insights = pd.DataFrame({
    "Factor": [
        "Lifetime",
        "Current Month Activity",
        "Age",
        "Contract Period",
        "Months to End of Contract",
        "Group Visits",
        "Friend Promotion",
        "Partner Status",
        "Near Location",
        "Gender",
        "Phone Availability"
    ],
    "Finding": [
        "Churned members had shorter recorded lifetimes",
        "Churned members showed lower current-month activity",
        "Churned members were younger on average",
        "Shorter contracts had higher observed churn",
        "Churned members had fewer months remaining",
        "Members without group visits had higher observed churn",
        "Members without friend promotion had higher observed churn",
        "Non-partner members had higher observed churn",
        "Members not near the gym had higher observed churn",
        "Churn rates were almost identical by gender",
        "Churn rates were almost identical by phone availability"
    ]
})

insights.to_csv("outputs/churn_insights.csv", index=False)

print("\nChurn insights saved successfully!")

# creating business recommendations
recommendations = pd.DataFrame({
    "Finding": [
        "Low member activity is associated with higher churn",
        "Short contracts have higher observed churn",
        "Churned members have shorter recorded lifetimes",
        "Members without group visits have higher observed churn",
        "Members without friend promotion have higher observed churn",
        "Members not near the gym have higher observed churn"
    ],
    "Recommended Action": [
        "Identify low-activity members and offer engagement or follow-up programs",
        "Provide stronger engagement and retention support for short-term members",
        "Focus early-stage members on onboarding and continued engagement",
        "Promote group sessions and community-based activities",
        "Encourage referral and friend-based membership programs",
        "Consider targeted engagement for members who face greater location barriers"
    ]
})

recommendations.to_csv(
    "outputs/churn_recommendations.csv",
    index=False
)

print("\nBusiness recommendations saved successfully!")

# output files verification
import os

output_files = [
    "outputs/model_comparison.csv",
    "outputs/churn_insights.csv",
    "outputs/churn_recommendations.csv"
]

print("\nVerifying output files:")

for file in output_files:
    if os.path.exists(file):
        print(f"{file} - Found")
    else:
        print(f"{file} - Missing")

