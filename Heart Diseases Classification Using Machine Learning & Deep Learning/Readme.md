# Heart Disease Classification using Machine Learning and Deep Learning

Predicting heart disease risk from clinical and lifestyle data, with a probability-based health recommendation system built on top of the best-performing model.

## Overview

This project analyzes patient health records to predict the likelihood of heart disease and provide tiered, actionable health recommendations based on the predicted risk probability. Multiple machine learning and deep learning models were trained and compared, and the best model was deployed as a real-time prediction app.

## Dataset

- **File:** `heart_disease_risk_2026.csv`
- **Features:** 24+ clinical and lifestyle attributes, including:
  - Demographics: `age`, `sex`
  - Clinical measurements: `resting_bp_systolic`, `resting_bp_diastolic`, `cholesterol_total`, `hdl`, `ldl`, `triglycerides`, `fasting_blood_sugar`, `bmi`
  - Lifestyle factors: `smoker_status`, `alcohol_units_per_week`, `exercise_minutes_per_week`, `sleep_hours`, `stress_score`, `daily_steps`, `diet_quality_score`
  - Other: `family_history`, `chest_pain_type`, `wearable_owner`
- **Target variable:** `has_heart_disease` (binary: 0 = No, 1 = Yes)

## Project Workflow

1. **Data Preprocessing**
   - Checked for missing values and duplicate records
   - Encoded categorical features using Label Encoding

2. **Exploratory Data Analysis (EDA)**
   - Analyzed distribution of the target variable
   - Explored categorical features (sex, chest pain type, smoker status) and numerical features (age, blood pressure, cholesterol, BMI)
   - Visualized relationships between key features and heart disease outcome

3. **Feature Engineering & Selection**
   - Feature scaling via Label Encoding
   - Feature importance analysis using Random Forest
   - Recursive Feature Elimination (RFE) to select the top 10 predictive features for the final model

4. **Model Selection Strategy**
   - Used Stratified K-Fold cross-validation (10 folds) to evaluate model input selection and ensure balanced, representative folds across classes

5. **Model Training**

   Trained and evaluated the following models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Naive Bayes
   - K-Nearest Neighbors (KNN)
   - Artificial Neural Network (ANN)
   - XGBoost Classifier

6. **Hyperparameter Tuning**
   - Applied GridSearchCV (5-fold cross-validation) to tune Decision Tree, Random Forest, and XGBoost models

7. **Model Evaluation**
   - Compared models using accuracy, precision, confusion matrix, and classification report
   - **XGBoost** achieved the best overall performance with **89.44% accuracy**

8. **Health Recommendation System**

   Built a 5-tier risk classification system based on the XGBoost model's predicted probability:

   | Risk Level      | Probability Range |
   |-----------------|--------------------|
   | Very High Risk  | ≥ 0.90             |
   | High Risk       | 0.75 – 0.89        |
   | Moderate Risk   | 0.60 – 0.74        |
   | Low Risk        | 0.40 – 0.59        |
   | Very Low Risk   | < 0.40             |

   Each tier comes with tailored health guidance (e.g., cardiologist consultation, lifestyle changes, routine check-ups).

9. **Model Deployment**
   - Trained model and label encoders saved using `joblib` (`xgboost_model.pkl`, `label_encoders.pkl`)
   - Deployed as a real-time prediction web app using **Streamlit**, allowing users to input health data through an interactive form and instantly receive a classification result with a personalized health recommendation

## Tech Stack

- **Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow/Keras (ANN)
- **Model Persistence:** joblib
- **Deployment:** Streamlit

## Results

| Model               | Notes                                                |
|---------------------|-------------------------------------------------------|
| Logistic Regression | Baseline model, evaluated with CV                     |
| Decision Tree       | Tuned via GridSearchCV                                |
| Random Forest       | Tuned via GridSearchCV, used for feature importance   |
| Naive Bayes         | Evaluated with CV                                     |
| KNN                 | Evaluated across k = 1 to 40                          |
| ANN                 | Trained for 50 epochs                                 |
| **XGBoost**         | **Best model — 89.44% accuracy**, tuned with GridSearchCV + RFE |

## Project Structure

```
Heart-Disease-Classification/
│
├── 📓 Heart_Diseases_Classification.ipynb   # Main notebook (EDA, modeling, evaluation)
├── 📊 heart_disease_risk_2026.csv           # Dataset
├── 🧠 xgboost_model.pkl                     # Saved best model (XGBoost)
├── 🔤 label_encoders.pkl                    # Saved label encoders
├── 🚀 app.py                                # Streamlit frontend + inference
├── 📦 requirements.txt                      # Python dependencies
└── 📄 README.md                             # Project documentation
```

## Future Improvements
- Expand hyperparameter search space for further tuning
- Explore advanced feature engineering and additional data sources
- Add model monitoring and periodic retraining pipeline
- Containerize the Streamlit application with Docker for easier deployment
