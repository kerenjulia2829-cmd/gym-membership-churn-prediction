# Gym Membership Churn Prediction

A machine learning project that analyzes gym membership data to identify patterns associated with member churn and predict whether a member is likely to churn.

The project follows a complete data science workflow:

**Data Understanding → Data Cleaning → Exploratory Data Analysis → Feature Preparation → Machine Learning → Model Evaluation → Feature Interpretation → Business Recommendations**

The goal is not only to predict churn, but also to understand the member characteristics and activity patterns associated with churn.

## Problem Statement

Gym membership churn refers to members discontinuing their gym membership.

For a gym, identifying members who are more likely to churn can help in planning timely retention efforts. However, understanding churn requires analyzing member demographics, membership details, and activity patterns.

This project investigates the factors associated with gym membership churn and develops a machine learning model to predict whether a member is likely to churn.

## Business Objective

The main objective of this project is to:

- Identify patterns and factors associated with gym membership churn.
- Analyze member demographics, membership details, and activity patterns.
- Predict members who are more likely to churn using machine learning.
- Provide data-driven insights that can support member retention strategies.

## Dataset

This project uses a gym membership dataset containing information about member demographics, membership details, and activity patterns.

### Dataset Overview

- **Number of records:** 4,000
- **Number of columns:** 14
- **Target variable:** `Churn`
- **Churned members:** 1,061
- **Members who stayed:** 2,939

### Target Variable

The `Churn` column is the prediction target:

- `0` → Member stayed
- `1` → Member churned

### Features

The dataset contains the following member attributes:

| Feature | Description |
|---|---|
| `gender` | Member gender |
| `Near_Location` | Whether the member lives near the gym |
| `Partner` | Whether the member is associated with a partner company |
| `Promo_friends` | Whether the member joined through a friend promotion |
| `Phone` | Whether a phone number was provided |
| `Contract_period` | Membership contract duration |
| `Group_visits` | Whether the member participates in group sessions |
| `Age` | Member age |
| `Avg_additional_charges_total` | Average additional charges |
| `Month_to_end_contract` | Months remaining in the contract |
| `Lifetime` | Length of the member's recorded membership |
| `Avg_class_frequency_total` | Average class attendance frequency over the recorded period |
| `Avg_class_frequency_current_month` | Average class attendance frequency in the current month |
| `Churn` | Target variable indicating whether the member churned |

### Dataset Source

The dataset was obtained from Kaggle:

**Gym customers features and churn — Adrian Vinueza**

Dataset: https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn

The dataset is used for educational and portfolio purposes. The source dataset is licensed under **CC BY-NC-SA 4.0**.

## Approach

### 1. Data Understanding

**Objective:**  
Understand the structure, variables, data types, and target variable before performing analysis.

**What was done:**
- Loaded the dataset using Pandas.
- Checked the number of rows and columns.
- Examined data types and non-null values.
- Checked for missing values.
- Checked for duplicate records.
- Examined the distribution of the target variable.
- Verified the values of binary and categorical variables.

**Output:**
- 4,000 records
- 14 columns
- No missing values
- No duplicate records
- `Churn` identified as the target variable

### 2. Data Cleaning

**Objective:**  
Check the dataset for invalid, inconsistent, or unusual values before analysis.

**What was done:**
- Checked numerical variables for unreasonable ranges.
- Checked for negative values where they would not be meaningful.
- Examined statistical outliers using box plots.
- Reviewed unusual values without automatically removing them.

**Output:**
- No clearly invalid values were identified.
- No rows were removed.
- Statistical outliers were retained because they were not confirmed to be data errors.

### 3. Exploratory Data Analysis (EDA)

**Objective:**  
Explore the data to understand the distribution of churn and identify patterns associated with member churn.

**What was done:**
- Analyzed the overall churn distribution.
- Compared churn rates across contract periods.
- Compared churn rates across demographic and membership-related variables.
- Compared age, lifetime, additional charges, and activity levels between churned and retained members.
- Analyzed current-month activity using activity quartiles.

**Key observations:**
- Overall churn rate was approximately **26.53%**.
- Members with shorter contracts had higher observed churn.
- Churned members showed lower recent activity.
- Churned members had shorter recorded lifetimes on average.
- Churn rates were very similar across gender and phone availability.

**Visualizations:**
- Churn distribution bar chart
- Churn rate by contract period
- Churn rate by demographic and membership variables
- Box plots for numerical variables
- Churn rate by activity level

The interactive Streamlit dashboard presents a focused selection
of these visualizations to provide a clear and concise overview
of the most important churn patterns.

### 4. Correlation Analysis

**Objective:**  
Examine relationships between numerical variables and identify variables that show stronger associations with churn.

**What was done:**
- Calculated the correlation matrix for numerical variables.
- Visualized the correlations using a heatmap.
- Identified strong relationships between features.

**Key observations:**
- `Lifetime` showed a strong negative correlation with `Churn`.
- `Avg_class_frequency_current_month` showed a strong negative correlation with `Churn`.
- `Age` and `Contract_period` also showed notable negative correlations with `Churn`.
- `Contract_period` and `Month_to_end_contract` were highly correlated.
- `Avg_class_frequency_total` and `Avg_class_frequency_current_month` were highly correlated.

These correlations describe associations in the dataset and should not be interpreted as evidence of causation.

### 5. Feature Preparation

**Objective:**  
Prepare the data for machine learning while maintaining a consistent training and testing process.

**What was done:**
- Selected 13 features as model inputs.
- Separated the features (`X`) from the target (`y`).
- Split the data into training and testing sets using an 80:20 ratio.
- Used stratified splitting to preserve the churn distribution in both sets.
- Standardized the features for Logistic Regression.
- Fit the scaler only on the training data and applied it to the test data.

**Output:**
- Training set: 3,200 records
- Test set: 800 records
- Features used: 13

### 6. Machine Learning Models

Two classification models were trained to predict gym membership churn.

#### Logistic Regression

Logistic Regression was used as a baseline classification model.

**Process:**
- Trained using the standardized training data.
- Generated churn predictions for the test set.
- Generated churn probabilities for ROC-AUC evaluation.

#### Random Forest

Random Forest was used as a tree-based classification model.

**Process:**
- Trained using the original, unscaled training data.
- Generated churn predictions for the test set.
- Calculated feature importance to understand which variables contributed most to the model's predictions.

### 7. Model Evaluation

The models were evaluated on the test set using multiple classification metrics rather than accuracy alone.

**Metrics used:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Results:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.93 | 0.88 | 0.83 | 0.85 | 0.9774 |
| Random Forest | 0.92 | 0.88 | 0.83 | 0.85 | 0.9684 |

A confusion matrix and ROC curve were also used to examine the classification performance of the models.

### 8. Feature Importance & Interpretation

To understand which variables contributed most to the model's predictions, both Random Forest feature importance and Logistic Regression coefficients were examined.

**Random Forest:**
- `Lifetime` was the most important feature.
- `Avg_class_frequency_current_month` was among the most important activity-related features.
- `Age` and `Avg_class_frequency_total` also showed relatively high importance.

**Logistic Regression:**
- Coefficients were examined to understand the direction of the model's associations.
- Features with negative coefficients were associated with lower predicted churn probability, while positive coefficients were associated with higher predicted churn probability, holding the other model inputs constant.

Because some features are highly correlated, individual coefficient values should be interpreted carefully. Feature importance and coefficients indicate model associations, not causal effects.

### 9. Key Findings

The analysis identified several patterns associated with gym membership churn:

- Members with **shorter contract periods** had higher observed churn rates.
- Members with **lower current-month activity** had higher observed churn.
- Churned members had **shorter recorded lifetimes** on average.
- Members without **group visits** had higher observed churn.
- Members who did not join through **friend promotions** had higher observed churn.
- Members who were **not near the gym** had higher observed churn.
- Churn rates were nearly identical across **gender** and **phone availability**.

These findings describe patterns observed in this dataset and do not establish that any factor directly causes churn.

### 10. Business Recommendations

Based on the patterns observed in the analysis, the following actions could support member retention:

- Identify members with **low recent activity** and consider targeted engagement programs.
- Provide additional engagement support for members with **short-term contracts**.
- Focus on **early-stage members** to encourage continued participation.
- Promote **group sessions** to increase member engagement.
- Encourage **friend-based membership and referral programs**.
- Consider targeted engagement for members who face **greater location barriers**.

These recommendations are based on observed associations in the dataset and should be validated with real-world business data before implementation.

## Project Structure

```text
gym-membership-churn-prediction/
│
├── data/
│   └── gym_churn_us.csv
│
├── models/
│   └── churn_model.pkl
│
├── outputs/
│   ├── churn_insights.csv
│   ├── churn_recommendations.csv
│   └── model_comparison.csv
│
├── .gitignore
├── app.py
├── main.py
├── requirements.txt
└── README.md
```
**Note:** The dataset CSV is not included in the GitHub repository.
Download it separately and place it inside the `data/` folder before running the project.
## Technologies Used

- **Python** — Programming language
- **Pandas** — Data loading, manipulation, and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Scikit-learn** — Machine learning and model evaluation
- **Streamlit** — Interactive dashboard development
- **Plotly** — Interactive charts and visualizations
- **Joblib** — Saving and loading the trained machine learning model
- **PyCharm** — Development environment
- **Git & GitHub** — Version control and project hosting

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/kerenjulia2829-cmd/gym-membership-churn-prediction.git
cd gym-membership-churn-prediction
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the Gym Customers Features and Churn dataset from Kaggle:

[Gym Customers Features and Churn – Kaggle](https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn)

**Note:** The dataset is not included in this GitHub repository. You must download it separately and place the CSV file in the specified location.

Place the downloaded `gym_churn_us.csv` file inside the `data/` folder.

The expected file location is:

```text
data/gym_churn_us.csv
```

### 5. Run the Machine Learning Workflow

```bash
python main.py
```

This executes the data analysis, model training, evaluation, and generation of output files.

### 6. Launch the Interactive Dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the interactive dashboard.

The dashboard uses the saved Logistic Regression model to estimate
individual member churn probability. It also provides rule-based
retention recommendations based on the member's details.

The predictions are estimates and should not be treated as
guaranteed outcomes.


## Outputs

The project generates the following output files:

| File | Description |
|---|---|
| `outputs/churn_insights.csv` | Key findings identified from the churn analysis |
| `outputs/churn_recommendations.csv` | Business recommendations based on the observed churn patterns |
| `outputs/model_comparison.csv` | Performance comparison of the trained machine learning models |
| `models/churn_model.pkl` | Saved Logistic Regression model and scaler used for individual churn predictions in the Streamlit dashboard |

## Limitations

- The dataset contains 4,000 records and may not represent the behavior of gym members in all locations or business settings.
- The source dataset's real-world provenance is not fully documented, so the results should be treated as an educational analysis rather than a general representation of gym members.
- The analysis identifies associations between variables and churn but does not establish causation.
- Some features are highly correlated, which can affect the interpretation of individual model coefficients.
- Model performance was evaluated using a single train-test split rather than cross-validation.
- The project does not include real-time member data or deployment in a production environment.

## Future Improvements

- Use cross-validation to obtain more robust model performance estimates.
- Experiment with additional classification models and hyperparameter tuning.
- Investigate feature selection and dimensionality reduction for highly correlated variables.
- Add more detailed model interpretability techniques.
- Evaluate the model on real-world gym membership data.
- Deploy the model as a web application for practical use.

## Disclaimer

This project is created for educational and portfolio purposes.

The analysis and machine learning results are based on the selected dataset and should not be interpreted as definitive evidence of real-world gym member behavior or as a production-ready churn prediction system.

The findings describe patterns and associations present in the dataset and do not establish causal relationships.

