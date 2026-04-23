import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Load model
# ===============================
model = joblib.load("unemployment_model.pkl")

# ===============================
# Title & UI
# ===============================
st.set_page_config(page_title="Unemployment Predictor", layout="wide")

st.title("📊 Unemployment Rate Prediction App")
st.markdown("### Predict unemployment rate based on various factors")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("🔧 Input Parameters")

# Example mappings (adjust if needed)
state_list = ["Andhra Pradesh", "Delhi", "Maharashtra", "Tamil Nadu", "Karnataka"]
area_list = ["Rural", "Urban"]

state = st.sidebar.selectbox("Select State", state_list)
area = st.sidebar.selectbox("Select Area", area_list)

employed = st.sidebar.number_input("Estimated Employed", min_value=0)
labour_rate = st.sidebar.slider("Labour Participation Rate (%)", 0.0, 100.0)

year = st.sidebar.selectbox("Year", [2019, 2020, 2021, 2022, 2023])
month = st.sidebar.selectbox("Month", list(range(1, 13)))

# ===============================
# Encode Inputs (IMPORTANT)
# ===============================
# You must match training encoding
state_mapping = {name: i for i, name in enumerate(state_list)}
area_mapping = {"Rural": 0, "Urban": 1}

state_encoded = state_mapping[state]
area_encoded = area_mapping[area]

# ===============================
# Prediction
# ===============================
if st.sidebar.button("🚀 Predict"):

    input_data = pd.DataFrame([[
        state_encoded,
        employed,
        labour_rate,
        area_encoded,
        year,
        month
    ]], columns=[
        'State',
        'Employed',
        'Labour_Participation_Rate',
        'Area',
        'Year',
        'Month'
    ])

    prediction = model.predict(input_data)[0]

    # ===============================
    # Display Result
    # ===============================
    st.subheader("📈 Prediction Result")
    st.success(f"Predicted Unemployment Rate: {prediction:.2f}%")

    # ===============================
    # Chart: Input Overview
    # ===============================
    st.subheader("📊 Input Overview")

    chart_data = pd.DataFrame({
        "Feature": ["Employed", "Labour Rate", "Month"],
        "Value": [employed, labour_rate, month]
    })

    fig, ax = plt.subplots()
    ax.bar(chart_data["Feature"], chart_data["Value"])
    st.pyplot(fig)

    # ===============================
    # Insight Message
    # ===============================
    if prediction > 10:
        st.warning("⚠️ High unemployment rate predicted")
    else:
        st.info("✅ Moderate/Low unemployment rate")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown("💡 Built using Machine Learning + Streamlit")