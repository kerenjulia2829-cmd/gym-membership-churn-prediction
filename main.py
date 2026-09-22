import pandas as pd
df=pd.read_csv("data/gym_churn_us.csv")
# print(df.head())
# print(df.shape)
# df.info()
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df["Churn"].value_counts())
# print(df["Churn"].value_counts(normalize=True)*100)
# print(df["gender"].unique())
# binary_columns=["gender","Near_Location","Partner","Promo_friends","Phone",
#                 "Group_visits","Churn"]
# for column in binary_columns:
#     print(column,":",df[column].unique())
# print(df["Contract_period"].unique())
# print(df["Age"].describe())
# print(df.describe().T)
# for column in binary_columns:
#     print(column,":",df[column].value_counts().to_dict())
# print(df["Contract_period"].value_counts().sort_index())

# print(df.min())
# print(df.max())
import matplotlib.pyplot as plt
# df["Lifetime"].plot(kind="box")
# plt.show()
# df["Avg_additional_charges_total"].plot(kind="box")
# plt.show()
# df["Avg_class_frequency_total"].plot(kind="box")
# plt.show()
# df["Avg_class_frequency_current_month"].plot(kind="box")
# plt.show()

# df["Churn"].value_counts().plot(kind="bar")
# plt.title("Gym Membership Churn Distribution")
# plt.xlabel("Churn")
# plt.ylabel("Number od Members")
# plt.show()

# churn_by_contract=df.groupby("Contract_period")["Churn"].mean()*100
# churn_by_contract.plot(kind="bar")
# plt.title("Churn Rate by Contract Period")
# plt.xlabel("Contract Period (Months")
# plt.ylabel("Churn Rate (%)")
# plt.show()

# churn_by_gender=df.groupby("gender")["Churn"].mean()*100
# churn_by_gender.plot(kind="bar")
# plt.title("Churn Rate by Gender")
# plt.xlabel("Gender")
# plt.ylabel("Churn Rate (%)")
# plt.show()

# churn_by_location=df.groupby("Near_Location")["Churn"].mean()*100
# churn_by_location.plot(kind="bar")
# plt.title("Churn Rate by Location")
# plt.xlabel("Near Location")
# plt.ylabel("Churn Rate (%)")
# plt.show()

# churn_by_partner=df.groupby("Partner")["Churn"].mean()*100
# churn_by_partner.plot(kind="bar")
# plt.title("Churn Rate by Partner Status")
# plt.xlabel("Partner")
# plt.ylabel("Churn Rate (%)")
# plt.show()

# churn_by_promo=df.groupby("Promo_friends")["Churn"].mean()*100
# churn_by_promo.plot(kind="bar")
# plt.title("Churn Rate by Friend Promotion")
# plt.xlabel("Joined Through Friend Promotion")
# plt.ylabel("Churn Rate (%)")
# plt.show()

# churn_by_phone = df.groupby("Phone")["Churn"].mean() * 100
# churn_by_phone.plot(kind="bar")
# plt.title("Churn Rate by Phone Availability")
# plt.xlabel("Phone Provided")
# plt.ylabel("Churn Rate (%)")
# plt.show()

# churn_by_group = df.groupby("Group_visits")["Churn"].mean() * 100
# churn_by_group.plot(kind="bar")
# plt.title("Churn Rate by Group Visits")
# plt.xlabel("Participates in Group Visits")
# plt.ylabel("Churn Rate (%)")
# plt.show()

# age_by_churn = df.groupby("Churn")["Age"].mean()
# df.boxplot(column="Age", by="Churn")
# plt.title("Age Distribution by Churn Status")
# plt.suptitle("")
# plt.xlabel("Churn Status")
# plt.ylabel("Age (Years)")
# plt.show()

# lifetime_by_churn = df.groupby("Churn")["Lifetime"].mean()
# print(lifetime_by_churn)
# df.boxplot(column="Lifetime", by="Churn")
# plt.title("Lifetime Distribution by Churn Status")
# plt.suptitle("")
# plt.xlabel("Churn Status")
# plt.ylabel("Lifetime (Months)")
# plt.show()

# charges_by_churn = df.groupby("Churn")["Avg_additional_charges_total"].mean()
# df.boxplot(column="Avg_additional_charges_total", by="Churn")
# plt.title("Additional Charges Distribution by Churn Status")
# plt.suptitle("")
# plt.xlabel("Churn Status")
# plt.ylabel("Average Additional Charges")
# plt.show()

# months_to_end_by_churn = df.groupby("Churn")["Month_to_end_contract"].mean()
# df.boxplot(column="Month_to_end_contract", by="Churn")
# plt.title("Months to End of Contract by Churn Status")
# plt.suptitle("")
# plt.xlabel("Churn Status")
# plt.ylabel("Months to End of Contract")
# plt.show()

# class_frequency_total_by_churn = df.groupby("Churn")["Avg_class_frequency_total"].mean()
# df.boxplot(column="Avg_class_frequency_total", by="Churn")
# plt.title("Total Class Frequency by Churn Status")
# plt.suptitle("")
# plt.xlabel("Churn Status")
# plt.ylabel("Average Class Frequency")
# plt.show()

# class_frequency_current_by_churn = df.groupby("Churn")["Avg_class_frequency_current_month"].mean()
# df.boxplot(column="Avg_class_frequency_current_month", by="Churn")
# plt.title("Current Month Class Frequency by Churn Status")
# plt.suptitle("")
# plt.xlabel("Churn Status")
# plt.ylabel("Average Class Frequency")
# plt.show()

# contract_member_count = df["Contract_period"].value_counts().sort_index()
# print(contract_member_count)

churned_by_contract = df[df["Churn"] == 1].groupby("Contract_period").size()
# print(churned_by_contract)

# contract_analysis = pd.DataFrame({
#     "Total Members": contract_member_count,
#     "Churned Members": churned_by_contract
# })

# contract_analysis.plot(kind="bar")
# plt.title("Total and Churned Members by Contract Period")
# plt.xlabel("Contract Period (Months)")
# plt.ylabel("Number of Members")
# plt.xticks(rotation=0)
# plt.legend()
# plt.show()

# churned_share_by_contract = (churned_by_contract / df["Churn"].sum()) * 100
# churned_share_by_contract.plot(kind="bar")
# plt.title("Share of Churned Members by Contract Period")
# plt.xlabel("Contract Period (Months)")
# plt.ylabel("Share of All Churned Members (%)")
# plt.xticks(rotation=0)
# plt.show()

# print(df["Avg_class_frequency_current_month"].describe())
df["Activity_Level"] = pd.qcut(
    df["Avg_class_frequency_current_month"],
    q=4,
    labels=["Low", "Medium-Low", "Medium-High", "High"]
)
# print(df["Activity_Level"].value_counts().sort_index())
churn_by_activity = df.groupby("Activity_Level", observed=True)["Churn"].mean() * 100
# churn_by_activity.plot(kind="bar")
# plt.title("Churn Rate by Activity Level")
# plt.xlabel("Activity Level")
# plt.ylabel("Churn Rate (%)")
# plt.xticks(rotation=0)
# plt.show()

# df["Activity_Change"] = (
#     df["Avg_class_frequency_current_month"]
#     - df["Avg_class_frequency_total"]
# )
# print(df["Activity_Change"].describe())

# activity_change_by_churn = df.groupby("Churn")["Activity_Change"].mean()
# print(activity_change_by_churn)

# df.boxplot(column="Activity_Change", by="Churn")
# plt.title("Activity Change by Churn Status")
# plt.suptitle("")
# plt.xlabel("Churn Status")
# plt.ylabel("Activity Change")
# plt.show()

import seaborn as sns
# correlation_matrix = df.corr(numeric_only=True)
# # print(correlation_matrix)
# plt.figure(figsize=(12, 8))
# sns.heatmap(
#     correlation_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm"
# )
# plt.title("Correlation Matrix of Gym Membership Variables")
# # plt.show()
#
# strong_correlations = correlation_matrix[
#     (correlation_matrix.abs() > 0.80) &
#     (correlation_matrix.abs() < 1.00)
# ]
# print(strong_correlations.dropna(how="all").dropna(axis=1, how="all"))



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
# print("X shape:", X.shape)
# print("y shape:", y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)
# print(X_train.describe().T[["min", "max"]])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print("X_train_scaled shape:", X_train_scaled.shape)
# print("X_test_scaled shape:", X_test_scaled.shape)



from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_scaled, y_train)
# print("Logistic Regression model trained successfully!")

y_pred = logistic_model.predict(X_test_scaled)
# print(y_pred[:20])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)

from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
y_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
# print("ROC-AUC:", roc_auc)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.show()

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
random_forest_model.fit(X_train, y_train)
# print("Random Forest model trained successfully!")

y_pred_rf = random_forest_model.predict(X_test)
# print(y_pred_rf[:20])

from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred_rf))

y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
# print("Random Forest ROC-AUC:", roc_auc_rf)

model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [0.93, 0.92],
    "Precision": [0.88, 0.88],
    "Recall": [0.83, 0.83],
    "F1-Score": [0.85, 0.85],
    "ROC-AUC": [0.9774, 0.9684]
})
# print(model_comparison)
model_comparison.to_csv("outputs/model_comparison.csv", index=False)

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": random_forest_model.feature_importances_
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
# print(feature_importance)

feature_importance.plot(
    kind="barh",
    x="Feature",
    y="Importance",
    legend=False
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
# plt.show()

logistic_coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": logistic_model.coef_[0]
})
logistic_coefficients = logistic_coefficients.sort_values(
    by="Coefficient"
)
# print(logistic_coefficients)

logistic_coefficients.plot(
    kind="barh",
    x="Feature",
    y="Coefficient",
    legend=False
)
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.axvline(x=0)
# plt.show()

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
        "Churned members had much shorter recorded lifetimes",
        "Churned members showed lower current-month activity",
        "Churned members were younger on average",
        "Shorter contracts had higher observed churn",
        "Churned members had fewer months remaining",
        "Members without group visits had higher observed churn",
        "Members without friend promotion had higher observed churn",
        "Non-partner members had higher observed churn",
        "Members not near the gym had higher observed churn",
        "Churn rates were almost identical",
        "Churn rates were almost identical"
    ]
})
# print(insights)
insights.to_csv("outputs/churn_insights.csv", index=False)

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
# print(recommendations)
recommendations.to_csv("outputs/churn_recommendations.csv", index=False)