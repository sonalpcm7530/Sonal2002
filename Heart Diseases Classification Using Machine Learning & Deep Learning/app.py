import streamlit as st
import pandas as pd
import joblib


# ---------------- Load Model and Encoders ----------------

model = joblib.load("xgboost_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")


# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# ---------------- Selected Features ----------------

selected_features = [
    "age",
    "sex",
    "hdl",
    "ldl",
    "hba1c",
    "max_heart_rate_achieved",
    "chest_pain_type",
    "exercise_induced_angina",
    "st_depression",
    "smoker_status"
]


categorical_columns = [
    "sex",
    "chest_pain_type",
    "exercise_induced_angina",
    "smoker_status"
]


# ---------------- Recommendation Function ----------------

def get_recommendation(probability):

    if probability >= 0.90:

        risk_level = "Very High Risk"

        recommendations = [
            "Seek immediate consultation with a cardiologist.",
            "Undergo ECG, Echocardiogram, and blood tests.",
            "Monitor blood pressure and blood sugar regularly.",
            "Avoid strenuous physical activity.",
            "Follow prescribed medications."
        ]


    elif probability >= 0.75:

        risk_level = "High Risk"

        recommendations = [
            "Schedule a cardiology appointment.",
            "Check cholesterol, blood pressure, and sugar levels.",
            "Follow a heart healthy diet.",
            "Exercise after medical advice.",
            "Avoid smoking and alcohol."
        ]


    elif probability >= 0.60:

        risk_level = "Moderate Risk"

        recommendations = [
            "Arrange routine health check-up.",
            "Monitor cholesterol and glucose levels.",
            "Exercise regularly.",
            "Eat fruits and vegetables.",
            "Manage stress and sleep properly."
        ]


    elif probability >= 0.40:

        risk_level = "Low Risk"

        recommendations = [
            "Continue healthy lifestyle.",
            "Exercise regularly.",
            "Avoid tobacco products.",
            "Perform periodic health check-ups."
        ]


    else:

        risk_level = "Very Low Risk"

        recommendations = [
            "Maintain healthy eating habits.",
            "Exercise regularly.",
            "Maintain normal cholesterol levels.",
            "Continue preventive screenings."
        ]


    return risk_level, recommendations



# ---------------- Prediction Function ----------------

def run_prediction(input_data):

    df = pd.DataFrame([input_data])


    # Encode categorical columns
    for col in categorical_columns:

        try:
            df[col] = label_encoders[col].transform(df[col])

        except ValueError as e:
            st.error(f"Invalid value for {col}: {e}")
            return None


    # Keep same order as training
    df = df[selected_features]


    # Prediction
    prediction = int(model.predict(df)[0])


    # Probability
    probability = float(
        model.predict_proba(df)[0][1]
    )


    if prediction == 1:
        prediction_label = "Heart Disease Detected"
    else:
        prediction_label = "No Heart Disease"


    risk_level, recommendations = get_recommendation(probability)


    return {
        "prediction_label": prediction_label,
        "probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "recommendations": recommendations
    }



# ---------------- Streamlit UI ----------------


st.title("❤️ Heart Disease Prediction System")

st.write(
    "Enter patient details to predict heart disease risk."
)



# Numerical Inputs

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=45
)


hdl = st.number_input(
    "HDL Level",
    min_value=0.0,
    value=50.0
)


ldl = st.number_input(
    "LDL Level",
    min_value=0.0,
    value=120.0
)


hba1c = st.number_input(
    "HbA1c",
    min_value=0.0,
    value=5.5
)


max_hr = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)


st_depression = st.number_input(
    "ST Depression",
    min_value=0.0,
    value=1.0
)



# Categorical Inputs

sex = st.selectbox(
    "Sex",
    label_encoders["sex"].classes_
)


chest_pain_type = st.selectbox(
    "Chest Pain Type",
    label_encoders["chest_pain_type"].classes_
)


exercise_induced_angina = st.selectbox(
    "Exercise Induced Angina",
    label_encoders["exercise_induced_angina"].classes_
)


smoker_status = st.selectbox(
    "Smoker Status",
    label_encoders["smoker_status"].classes_
)



# Prediction Button

if st.button("Predict Heart Disease"):


    patient_data = {

        "age": age,
        "sex": sex,
        "hdl": hdl,
        "ldl": ldl,
        "hba1c": hba1c,
        "max_heart_rate_achieved": max_hr,
        "chest_pain_type": chest_pain_type,
        "exercise_induced_angina": exercise_induced_angina,
        "st_depression": st_depression,
        "smoker_status": smoker_status
    }


    result = run_prediction(patient_data)


    if result:


        st.subheader("Prediction Result")


        if result["prediction_label"] == "Heart Disease Detected":

            st.error(
                result["prediction_label"]
            )

        else:

            st.success(
                result["prediction_label"]
            )


        st.metric(
            "Risk Probability",
            f'{result["probability"]}%'
        )


        st.info(
            f'Risk Level: {result["risk_level"]}'
        )


        st.subheader(
            "Recommendations"
        )


        for item in result["recommendations"]:

            st.write(
                "✅ " + item
            )