import streamlit as st
import joblib
import pandas as pd


# ---------------- Load Model and Preprocessing Tools ----------------

model = joblib.load('model_5.pkl')
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')


# Selected RFE features used during training
selected_features = [
    'savings_rate',
    'credit_score',
    'debt_to_income_ratio',
    'cash_flow_status',
    'financial_stress_level',
    'emergency_fund_coverage'
]


# ---------------- Recommendation Function ----------------

def generate_recommendations(score):

    if score >= 70:
        return (
            "Excellent Financial Health!\n\n"
            "• Continue your disciplined financial habits.\n"
            "• Diversify your investment portfolio.\n"
            "• Increase long-term investments (Mutual Funds, SIPs, Stocks).\n"
            "• Review insurance and retirement planning regularly."
        )

    elif score >= 50:
        return (
            "Good Financial Health.\n\n"
            "• Increase monthly savings.\n"
            "• Build an emergency fund covering 6 months of expenses.\n"
            "• Reduce unnecessary spending.\n"
            "• Consider increasing investment contributions."
        )

    elif score >= 30:
        return (
            "Moderate Financial Health.\n\n"
            "• Prioritize paying off high-interest debts.\n"
            "• Create and follow a monthly budget.\n"
            "• Increase your savings rate.\n"
            "• Avoid unnecessary loans and credit card spending."
        )

    else:
        return (
            "Poor Financial Health.\n\n"
            "• Focus on essential expenses only.\n"
            "• Prepare a strict monthly budget.\n"
            "• Build an emergency fund.\n"
            "• Reduce outstanding debt quickly.\n"
            "• Seek professional financial advice if needed."
        )


# ---------------- Streamlit UI ----------------

st.set_page_config(
    page_title="Financial Health Score Predictor",
    layout="centered"
)


st.title("📊 Financial Health Score Predictor")

st.write(
    "Enter your financial details to predict your Financial Health Score."
)

st.markdown("---")


input_data = {}


col1, col2 = st.columns(2)


# -------- Numerical Inputs --------

with col1:

    input_data['savings_rate'] = st.number_input(
        "Savings Rate (Example: 0.20 = 20%)",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )


    input_data['credit_score'] = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700
    )


    input_data['debt_to_income_ratio'] = st.number_input(
        "Debt To Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01
    )


# -------- Categorical Inputs --------

with col2:

    # Cash Flow Status Dropdown
    cash_flow_labels = encoders['cash_flow_status'].classes_

    selected_cash_flow = st.selectbox(
        "Cash Flow Status",
        cash_flow_labels
    )

    # Encode only for model input
    input_data['cash_flow_status'] = encoders['cash_flow_status'].transform(
        [selected_cash_flow]
    )[0]


    # Financial Stress Level Dropdown
    stress_labels = encoders['financial_stress_level'].classes_

    selected_stress = st.selectbox(
        "Financial Stress Level",
        stress_labels
    )

    # Encode only for model input
    input_data['financial_stress_level'] = encoders['financial_stress_level'].transform(
        [selected_stress]
    )[0]


    # Numerical Feature
    input_data['emergency_fund_coverage'] = st.number_input(
        "Emergency Fund Coverage (Months)",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

st.markdown("---")


# ---------------- Prediction ----------------

if st.button(
    "📈 Get Financial Health Score",
    type="primary"
):

    # Convert input into dataframe
    input_df = pd.DataFrame(
        [input_data]
    )


    # Keep same order as training
    input_df = input_df[selected_features]


    # Apply scaler
    scaled_input = scaler.transform(
        input_df
    )


    # Prediction
    prediction = model.predict(
        scaled_input
    )[0]


    st.subheader(
        f"Your Predicted Financial Health Score: {prediction:.2f}/100"
    )


    st.write("---")


    st.subheader(
        "Personalized Recommendations"
    )


    st.info(
        generate_recommendations(prediction)
    )


st.markdown("---")

st.caption(
    "Powered by Gradient Boosting Regressor trained on financial dataset."
)