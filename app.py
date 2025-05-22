import streamlit as st
import pickle
import pandas as pd

# --- Unique values dictionary for some categorical sliders ---
unique_values_dict = {
    'RevolvingUtilizationOfUnsecuredLines': [],
    'age': [],
    'NumberOfTime30-59DaysPastDueNotWorse': [0, 1, 2, 3, 4, 5],
    'DebtRatio': [],
    'MonthlyIncome': [],
    'NumberOfOpenCreditLinesAndLoans': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'NumberOfTimes90DaysLate': [0, 1, 2],
    'NumberRealEstateLoansOrLines': [0, 1, 2, 3],
    'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 2, 3, 4],
    'NumberOfDependents': [0, 1, 2, 3]
}

# --- Input UI Function ---
def user_input_features():
    data = {}
    features = list(unique_values_dict.keys())

    st.markdown("""
    <style>
    .header-text { color: #004466; font-weight: 700; font-size: 32px; margin-bottom: 15px; }
    .subheader-text { color: #007acc; font-weight: 600; font-size: 20px; margin-top: 20px; margin-bottom: 10px; }
    .stNumberInput > label { font-weight: 600; color: #004466; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header-text">Applicant Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        uniques = unique_values_dict.get(feature, [])
        container = col1 if i % 2 == 0 else col2
        label = feature.replace('_', ' ')
        
        with container:
            if uniques and len(uniques) <= 12:
                data[feature] = st.select_slider(label=label, options=uniques, value=uniques[0], key=feature)
            else:
                if feature == 'RevolvingUtilizationOfUnsecuredLines':
                    data[feature] = st.number_input('Revolving Utilization Of Unsecured Lines', min_value=0.0, max_value=10.0, value=0.5, step=0.01, format="%.2f", key=feature)
                elif feature == 'age':
                    data[feature] = st.number_input('Age', min_value=18, max_value=100, value=30, step=1, key=feature)
                elif feature == 'DebtRatio':
                    data[feature] = st.number_input('Debt Ratio', min_value=0.0, max_value=10.0, value=0.0, step=0.01, format="%.2f", key=feature)
                elif feature == 'MonthlyIncome':
                    data[feature] = st.number_input('Monthly Income', min_value=0.0, max_value=1_000_000.0, value=5000.0, step=100.0, format="%.2f", key=feature)
                else:
                    data[feature] = st.number_input(label, min_value=0, max_value=1_000_000, value=0, step=1, key=feature)

    return pd.DataFrame(data, index=[0])

# --- Feature Engineering Function ---
def apply_feature_engineering(df):
    # Prevent division by zero
    dependents = df['NumberOfDependents'].replace(0, 1)
    loans = df['NumberOfOpenCreditLinesAndLoans'].replace(0, 1)

    # Feature 1: Income per person
    df['IncomePerPerson'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)

    # Feature 2: Estimated total debt
    df['EstimatedDebt'] = df['DebtRatio'] * df['MonthlyIncome']

    # Feature 3: Income to DebtRatio
    df['Income_to_DebtRatio'] = df['MonthlyIncome'] / (df['DebtRatio'] + 0.01)

    # Feature 4: Income per credit line
    df['Income_per_CreditLine'] = df['MonthlyIncome'] / (loans + 1)

    # Feature 5: Total late payments
    df['Total_PastDue'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )

    # Feature 6: Late payments to open loans
    df['LatePayment_to_Loans'] = df['Total_PastDue'] / (loans + 1)

    # Feature 7: Has late payment flag
    df['Has_LatePayment'] = df['Total_PastDue'].apply(lambda x: 1 if x > 0 else 0)

    return df


# --- Load Models ---
@st.cache_data(show_spinner=False)
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

model, preprocessor = load_model()

# --- Styling and Title ---
st.markdown("""
<style>
.main { background-color: #f0f8ff; padding: 25px 40px 40px 40px; border-radius: 12px; }
.stButton > button {
    background-color: #007acc;
    color: white;
    font-weight: 700;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 16px;
    transition: background-color 0.3s ease;
}
.stButton > button:hover { background-color: #005f99; }
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Risk Prediction App")
st.markdown("Enter applicant details below to predict the probability of **serious delinquency within 2 years**.")

# --- Input ---
input_df = user_input_features()
st.markdown("<br>", unsafe_allow_html=True)

# --- Prediction Button ---
if st.button("Predict Risk"):
    try:
        # 1. Apply Feature Engineering
        engineered_df = apply_feature_engineering(input_df)

        # 2. Transform with preprocessor
        processed_input = preprocessor.transform(engineered_df)

        # 3. Predict
        risk_prob = model.predict_proba(processed_input)[:, 1][0]

        st.markdown('<div class="subheader-text">Prediction Result</div>', unsafe_allow_html=True)

        # 4. Result Display
        if risk_prob > 0.5:
            st.markdown(f"""
            <div style="background-color:#f8d7da; padding:20px; border-radius:15px;
                        color:#721c24; font-weight:700; font-size:18px;
                        box-shadow: 3px 3px 10px rgba(0,0,0,0.1);">
                ⚠️ <b>High Risk!</b> Probability of Serious Delinquency: {risk_prob:.4f}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:#d4edda; padding:20px; border-radius:15px;
                        color:#155724; font-weight:700; font-size:18px;
                        box-shadow: 3px 3px 10px rgba(0,0,0,0.1);">
                ✅ <b>Low Risk</b> Probability of Serious Delinquency: {risk_prob:.4f}
            </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
