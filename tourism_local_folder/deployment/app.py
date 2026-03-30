import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# =========================
# LOAD MODEL FROM HF
# =========================
try:
    model_path = hf_hub_download(
        repo_id="shreyackdeshpande/tourism-model",
        filename="best_tourism_model.joblib"
    )
    model = joblib.load(model_path)
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")

# =========================
# STREAMLIT UI
# =========================
st.title("🌍 Tourism Conversion Prediction")

# NUMERICAL INPUTS
Age = st.slider("Age", 18, 70, 30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration Of Pitch", 5, 60, 20)
NumberOfPersonVisiting = st.slider("Number Of Persons Visiting", 1, 10, 2)
NumberOfFollowups = st.slider("Number Of Followups", 0, 10, 3)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
NumberOfTrips = st.slider("Number Of Trips", 0, 10, 2)
Passport = st.selectbox("Has Passport", [0, 1])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.slider("Children Visiting", 0, 5, 0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=30000)

# CATEGORICAL INPUTS
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Manager", "Senior Manager", "Executive", "AVP", "VP"])

# =========================
# CREATE INPUT DATA
# =========================
input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation
}])

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    try:
        # Ensure correct column order
        input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = 0

        if prediction == 1:
            st.success(f"✅ Customer is likely to PURCHASE (Probability: {probability:.2f})")
        else:
            st.error(f"❌ Customer is NOT likely to purchase (Probability: {probability:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
