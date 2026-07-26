from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("xgboost_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict Heart Disease Classification Using Machine Learning"
)

# Allow the frontend to call this API even if it's ever served from a
# different origin/port. Tighten allow_origins before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves style.css (and anything else placed in this folder) at /static/...
# e.g. index.html should link to href="/static/style.css"
app.mount("/static", StaticFiles(directory="."), name="static")


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

class Patient(BaseModel):
    age: int
    sex: str
    hdl: float
    ldl: float
    hba1c: float
    max_heart_rate_achieved: int
    chest_pain_type: str
    exercise_induced_angina: str
    st_depression: float
    smoker_status: str


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/categories")
def categories():
    """
    Diagnostic endpoint: returns the exact category strings each
    LabelEncoder was fit on, straight from label_encoders.pkl.
    Use this to fix mismatched dropdown values (e.g. "female" vs "Female").
    """
    categorical_columns = [
        "sex",
        "chest_pain_type",
        "exercise_induced_angina",
        "smoker_status"
    ]
    return {
        col: label_encoders[col].classes_.tolist()
        for col in categorical_columns
    }


def get_recommendation(probability):

    if probability >= 0.90:

        risk_level = "Very High Risk"

        recommendations = [
            "Seek immediate consultation with a cardiologist.",
            "Undergo ECG, Echocardiogram, and blood tests as advised.",
            "Monitor blood pressure and blood sugar regularly.",
            "Avoid strenuous physical activity until medically evaluated.",
            "Follow prescribed medications and seek emergency medical attention if symptoms such as chest pain or shortness of breath occur."
        ]

    elif probability >= 0.75:

        risk_level = "High Risk"

        recommendations = [
            "Schedule a cardiology appointment within the next few days.",
            "Get cholesterol, blood pressure, and blood sugar evaluated.",
            "Adopt a heart-healthy diet (low salt and low saturated fat).",
            "Engage in light to moderate exercise only after medical advice.",
            "Avoid smoking and limit alcohol consumption."
        ]

    elif probability >= 0.60:

        risk_level = "Moderate Risk"

        recommendations = [
            "Arrange a routine health check-up.",
            "Monitor blood pressure, cholesterol, and glucose levels.",
            "Exercise for at least 150 minutes per week.",
            "Maintain a balanced diet rich in fruits and vegetables.",
            "Manage stress and ensure adequate sleep."
        ]

    elif probability >= 0.40:

        risk_level = "Low Risk"

        recommendations = [
            "Continue a healthy lifestyle.",
            "Exercise regularly and maintain a healthy weight.",
            "Avoid tobacco products.",
            "Have periodic medical check-ups."
        ]

    else:

        risk_level = "Very Low Risk"

        recommendations = [
            "Continue healthy eating habits.",
            "Exercise regularly.",
            "Maintain normal blood pressure and cholesterol levels.",
            "Attend routine preventive health screenings."
        ]

    return risk_level, recommendations


def run_prediction(patient: Patient) -> dict:
    """Shared prediction logic used by both the JSON and form-based endpoints."""

    patient_data = patient.dict()

    df = pd.DataFrame([patient_data])
    categorical_columns = [
        "sex",
        "chest_pain_type",
        "exercise_induced_angina",
        "smoker_status"
    ]

    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col])

    df = df[selected_features]

    # Prediction
    prediction = int(model.predict(df)[0])

    # Probability
    probability = float(model.predict_proba(df)[0][1])

    # Prediction Label
    if prediction == 1:
        prediction_label = "Heart Disease Detected"
    else:
        prediction_label = "No Heart Disease"

    # Risk Level + Recommendations
    risk_level, recommendations = get_recommendation(probability)

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "recommendations": recommendations
    }


@app.post("/predict")
def predict(patient: Patient):

    try:
        return run_prediction(patient)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value supplied for categorical feature: {e}"
        )


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    age: int = Form(...),
    sex: str = Form(...),
    hdl: float = Form(...),
    ldl: float = Form(...),
    hba1c: float = Form(...),
    max_heart_rate_achieved: int = Form(...),
    chest_pain_type: str = Form(...),
    exercise_induced_angina: str = Form(...),
    st_depression: float = Form(...),
    smoker_status: str = Form(...),
):
    """
    No-JavaScript entry point. Accepts a classic HTML form submission
    (application/x-www-form-urlencoded) and returns a rendered HTML
    page instead of JSON, via FastAPI's HTMLResponse.
    """

    patient = Patient(
        age=age,
        sex=sex,
        hdl=hdl,
        ldl=ldl,
        hba1c=hba1c,
        max_heart_rate_achieved=max_heart_rate_achieved,
        chest_pain_type=chest_pain_type,
        exercise_induced_angina=exercise_induced_angina,
        st_depression=st_depression,
        smoker_status=smoker_status,
    )

    try:
        result = run_prediction(patient)

    except ValueError as e:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <title>Prediction Error</title>
          <link rel="stylesheet" href="/static/style.css" />
        </head>
        <body>
          <div class="container">
            <h1>Heart Disease Prediction</h1>
            <div class="error-box" style="display:block;">
              Invalid value supplied for categorical feature: {e}
            </div>
            <p><a href="/">Back to form</a></p>
          </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)

    recommendations_html = "".join(
        f"<li>{item}</li>" for item in result["recommendations"]
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Prediction Result</title>
      <link rel="stylesheet" href="/static/style.css" />
    </head>
    <body>
      <div class="container">
        <h1>Heart Disease Prediction</h1>
        <p class="subtitle">Result for the submitted patient details.</p>

        <div class="result-box" style="display:block;">
          <div class="result-item">
            <span class="label">Prediction:</span>
            <span>{result["prediction_label"]}</span>
          </div>
          <div class="result-item">
            <span class="label">Probability:</span>
            <span>{result["probability"]}%</span>
          </div>
          <div class="result-item">
            <span class="label">Risk level:</span>
            <span>{result["risk_level"]}</span>
          </div>
          <div class="result-item recommendations">
            <span class="label">Recommendations:</span>
            <ul>{recommendations_html}</ul>
          </div>
        </div>

        <p><a href="/">Back to form</a></p>
      </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html)