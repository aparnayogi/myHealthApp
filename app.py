# # Training Script (Colab)
# import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Load dataset
# df = pd.read_csv("lifestyle_health_data.csv")

# # Encode categorical variables
# df_encoded = pd.get_dummies(df, drop_first=True)

# # Split features and target
# X = df_encoded.drop("Health_Risk", axis=1)
# y = df_encoded["Health_Risk"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Predictions and evaluation
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# # Save model and feature names
# model_path = "health_risk_model.pkl"
# features_path = "features.pkl"
# joblib.dump(model, model_path)
# joblib.dump(X.columns.tolist(), features_path)

# print("✅ Model Trained")
# print("🎯 Accuracy:", accuracy)
# print("📄 Classification Report:\n", report)


import streamlit as st
import joblib
import numpy as np

# ✅ This must be first Streamlit command
st.set_page_config(page_title="Lifestyle Health Risk Predictor", layout="centered")

# Load model and feature list
model = joblib.load("health_risk_model.pkl")
features = joblib.load("features.pkl")

# Title and description
st.title("🏥 Lifestyle Health Risk Predictor")
st.write("Fill in your basic health and lifestyle habits to predict your health risk.")

# Input form
with st.form("health_form"):
    age = st.slider("Age", 18, 80, 25)
    gender = st.radio("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=22.0)
    physical_activity = st.slider("Physical Activity (hrs/day)", 0.0, 5.0, 1.0, step=0.1)
    diet_type = st.selectbox("Diet Type", ["Balanced", "Junk", "Mixed"])
    smoking = st.radio("Do you smoke?", ["Yes", "No"])
    alcohol = st.radio("Do you consume alcohol?", ["Yes", "No"])
    family_history = st.radio("Family history of health issues?", ["Yes", "No"])
    sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0, step=0.5)
    submitted = st.form_submit_button("Predict Risk")

# On submit
if submitted:
    user_data = {
        "Age": age,
        "BMI": bmi,
        "Physical_Activity": physical_activity,
        "Sleep_Hours": sleep_hours,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Diet_Type_Junk": 1 if diet_type == "Junk" else 0,
        "Diet_Type_Mixed": 1 if diet_type == "Mixed" else 0,
        "Smoking_Yes": 1 if smoking == "Yes" else 0,
        "Alcohol_Yes": 1 if alcohol == "Yes" else 0,
        "Family_History_Yes": 1 if family_history == "Yes" else 0,
    }

    # Ensure all expected features are present
    for feat in features:
        if feat not in user_data:
            user_data[feat] = 0

    # Make prediction
    input_values = [user_data[feat] for feat in features]
    prediction = model.predict([input_values])[0]
    probability = model.predict_proba([input_values])[0][prediction]

    # Show result
    if prediction == 1:
        st.error(f"⚠️ You are **likely at risk** of lifestyle-related health issues.\n\n(Risk Score: {probability:.2f})")
    else:
        st.success(f"✅ You are **not at significant risk**.\n\n(Risk Score: {probability:.2f})")
