import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

#loading the trained model
model_data = joblib.load("models/churn_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

#loading gym membership data
df = pd.read_csv("data/gym_churn_us.csv")

#calculating dashboard metrics
total_members = len(df)
churned_members = (df["Churn"] == 1).sum()
retained_members = (df["Churn"] == 0).sum()
from decimal import Decimal, ROUND_HALF_UP
churn_rate = (Decimal(str(churned_members * 100 / total_members))
    .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

#dashboard title
st.set_page_config(
    page_title="Gym Churn Prediction",
    page_icon="🏋️",
    layout="wide")
st.title("Gym Membership Churn Prediction")
st.write("Explore gym membership churn patterns, predict individual member "
    "churn risk, and identify opportunities to improve member retention.")

#tabs creation
tab1, tab2, tab3 = st.tabs([
    "Overview & Analytics",
    "Churn Prediction",
    "Business Insights"])

with tab1:
    # gym overview dashboard
    st.subheader("Gym Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Members", f"{total_members:,}")
    col2.metric("Churned Members", f"{churned_members:,}")
    col3.metric("Retained Members", f"{retained_members:,}")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")
    st.caption("A snapshot of membership status and overall churn performance.")

    #member churn distribution
    st.subheader("Member Churn Distribution")
    churn_counts = df["Churn"].value_counts().sort_index()
    churn_chart_data = pd.DataFrame({
        "Member Status": ["Stayed", "Churned"],
        "Members": [
            churn_counts.get(0, 0),
            churn_counts.get(1, 0)]})
    st.plotly_chart(
        px.pie(
            churn_chart_data,
            names="Member Status",
            values="Members",
            title="Stayed vs Churned Members",
            hole=0.4),
        use_container_width=True)

    #churn analysis by contract period
    st.subheader("Churn Rate by Contract Period")
    contract_churn = (df.groupby("Contract_period")["Churn"].mean() * 100).reset_index()
    contract_churn["Contract_period"] = (contract_churn["Contract_period"].astype(str))
    contract_churn.columns = ["Contract Period", "Churn Rate"]
    fig = px.bar(
        contract_churn,
        x="Contract Period",
        y="Churn Rate",
        title="Churn Rate by Membership Contract",
        text="Churn Rate",
        labels={
            "Contract Period": "Contract Period (Months)",
            "Churn Rate": "Churn Rate (%)"})
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=["1", "6", "12"])
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    #churn analysis by activity level
    st.subheader("Churn Rate by Activity Level")
    activity_df = df.copy()
    activity_df["Activity_Level"] = pd.qcut(
        activity_df["Avg_class_frequency_current_month"],
        q=4,
        labels=["Low", "Medium-Low", "Medium-High", "High"])
    activity_churn = (
        activity_df.groupby("Activity_Level", observed=True)["Churn"]
        .mean()
        .mul(100)
        .reset_index())
    activity_churn.columns = ["Activity Level", "Churn Rate"]
    fig = px.bar(
        activity_churn,
        x="Activity Level",
        y="Churn Rate",
        title="Churn Rate by Member Activity Level",
        text="Churn Rate",
        labels={"Churn Rate": "Churn Rate (%)"})
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside")
    fig.update_layout(
        yaxis_range=[0, 60])
    st.plotly_chart(fig, width="stretch")

    #churn analysis by group visits
    st.subheader("Churn Rate by Group Visits")
    group_churn = (
        df.groupby("Group_visits")["Churn"]
        .mean()
        .mul(100)
        .reset_index())
    group_churn["Group_visits"] = group_churn["Group_visits"].map({
        0: "No Group Visits",
        1: "Group Visits"})
    group_churn.columns = ["Group Visits", "Churn Rate"]
    fig = px.bar(
        group_churn,
        x="Group Visits",
        y="Churn Rate",
        title="Churn Rate by Group Visits",
        text="Churn Rate")
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside")
    st.plotly_chart(fig, width="stretch")

    #age distribution by churn status
    st.subheader("Age Distribution by Churn Status")
    age_df = df.copy()
    age_df["Member Status"] = age_df["Churn"].map({
        0: "Stayed",
        1: "Churned"})
    fig = px.box(
        age_df,
        x="Member Status",
        y="Age",
        title="Age Distribution: Stayed vs Churned",
        points="outliers")
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.header("Member Churn Prediction")
    st.subheader("Enter Member Details")
    st.caption("Enter the member's membership, activity, and personal details "
        "to estimate their churn risk.")

    #member information
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18, max_value=100, value=29)
    near_location = st.selectbox("Lives Near Gym?", ["Yes", "No"])
    partner = st.selectbox("Partner Company Employee?", ["Yes", "No"])
    promo_friends = st.selectbox("Joined Through Friend Promotion?", ["Yes", "No"])
    phone = st.selectbox("Phone Number Provided?", ["Yes", "No"])
    group_visits = st.selectbox("Participates in Group Visits?", ["Yes", "No"])

    #membership information
    contract_period = st.selectbox("Contract Period (Months)", [1, 6, 12])
    month_to_end_contract = st.number_input(
        "Months Remaining in Contract",
        min_value=1,
        max_value=12,
        value=6)
    lifetime = st.number_input(
        "Membership Lifetime (Months)",
        min_value=0,
        max_value=100,
        value=3)

    #activity information
    avg_additional_charges_total = st.number_input(
        "Average Additional Charges",
        min_value=0.0,
        value=100.0)
    avg_class_frequency_total = st.number_input(
        "Average Class Frequency (Overall)",
        min_value=0.0,
        value=2.0)
    avg_class_frequency_current_month = st.number_input(
        "Average Class Frequency (Current Month)",
        min_value=0.0,
        value=2.0)

    import pandas as pd

    if st.button("Predict Churn"):
        #convert text inputs into numerical values
        gender_value = 0 if gender == "Female" else 1
        near_location_value = 1 if near_location == "Yes" else 0
        partner_value = 1 if partner == "Yes" else 0
        promo_friends_value = 1 if promo_friends == "Yes" else 0
        phone_value = 1 if phone == "Yes" else 0
        group_visits_value = 1 if group_visits == "Yes" else 0

        #preparing member data
        member_data = pd.DataFrame([[
            gender_value,
            near_location_value,
            partner_value,
            promo_friends_value,
            phone_value,
            contract_period,
            group_visits_value,
            age,
            avg_additional_charges_total,
            month_to_end_contract,
            lifetime,
            avg_class_frequency_total,
            avg_class_frequency_current_month
        ]], columns=features)

        #scaling input data
        member_data_scaled = scaler.transform(member_data)

        #generating prediction
        prediction = model.predict(member_data_scaled)[0]

        #generating churn probability
        probability = model.predict_proba(member_data_scaled)[0][1]

        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("This member is likely to churn.")
        else:
            st.success("This member is likely to stay.")
        churn_percentage = probability * 100
        st.metric(
            label="Churn Probability",
            value=f"{churn_percentage:.2f}%")
        st.progress(float(probability))
        if churn_percentage < 30:
            risk_level = "Low Risk"
        elif churn_percentage < 60:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        st.write(f"**Churn Risk Level:** {risk_level}")

        #generating retention recommendations
        recommendations = []
        if avg_class_frequency_current_month < 1.0:
            recommendations.append(
                "Low recent activity: Contact the member and offer a personalised fitness plan.")
        if contract_period == 1:
            recommendations.append(
                "Short-term membership: Offer suitable long-term membership benefits.")
        if group_visits_value == 0:
            recommendations.append(
                "No group participation: Encourage the member to attend group fitness sessions.")
        if promo_friends_value == 0:
            recommendations.append(
                "No friend referral: Introduce referral benefits or buddy workout programs.")
        if near_location_value == 0:
            recommendations.append(
                "Location barrier: Explore flexible schedules or other ways to make gym visits easier.")
        st.subheader("Personalised Retention Recommendations")
        if recommendations:
            for recommendation in recommendations:
                st.write("•", recommendation)
        else:
            st.success(
                "Continue regular engagement and monitor the member's activity.")

with tab3:
    #overall business insights
    st.divider()
    st.header("Overall Gym Business Insights")
    st.caption("Explore the key factors associated with member churn "
        "and practical recommendations to improve retention.")

    #loading saved analysis results
    insights_df = pd.read_csv("outputs/churn_insights.csv")
    recommendations_df = pd.read_csv("outputs/churn_recommendations.csv")

    #display of key findings
    st.subheader("Key Churn Findings")
    st.dataframe(insights_df, use_container_width=True, hide_index=True)

    #display of business recommendations
    st.subheader("Business Recommendations")
    st.dataframe(
        recommendations_df,
        use_container_width=True,
        hide_index=True)
