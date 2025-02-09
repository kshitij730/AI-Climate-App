import streamlit as st
import joblib
import numpy as np
import requests

# Load trained AI models
emission_model = joblib.load("models/emission_predictor.pkl")
disaster_model = joblib.load("models/disaster_predictor.pkl")
scaler = joblib.load("models/scaler.pkl")

# Custom CSS for advanced UI with animated background
st.markdown("""
    <style>
        @keyframes moveBackground {
            0% { background-position: 0% 0%; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0% 0%; }
        }
        body {
            background: url('https://i.pinimg.com/736x/ee/04/74/ee04741edc0f0a7754aef288cb0b588e.jpg') no-repeat center center fixed;
            background-size: cover;
            animation: moveBackground 20s infinite linear;
            font-family: 'Poppins', sans-serif;
        }
        .stApp {
            background: rgba(0, 0, 0, 0.6);
            padding: 30px;
            border-radius: 15px;
            color: #ffffff;
            text-align: center;
        }
        .stTextInput label, .stNumberInput label {
            color: white !important;
            font-size: 18px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #ff9800;
            color: white;
            font-size: 20px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 12px;
        }
        .stButton>button:hover {
            background-color: #e68900;
        }
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: white;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
            font-family: 'Montserrat', sans-serif;
        }
        .subheader {
            text-align: center;
            font-size: 24px;
            color: white;
            font-family: 'Roboto', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="stApp">', unsafe_allow_html=True)
st.markdown('<div class="title">🌍 AI Climate Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Predict environmental risks and monitor climate impact using AI</div>', unsafe_allow_html=True)

# Predict Emissions Section
st.markdown("### 📊 <span style='color: white;'>Predict Emissions</span>", unsafe_allow_html=True)
industry_type = st.text_input("Industry Type", help="Enter the industry category.")
energy_consumption = st.number_input("Energy Consumption (MWh)", min_value=0.0, step=0.1)
transport_emissions = st.number_input("Transport Emissions (tons)", min_value=0.0, step=0.1)
waste_generated = st.number_input("Waste Generated (tons)", min_value=0.0, step=0.1)
carbon_sequestration = st.number_input("Carbon Sequestration (tons)", min_value=0.0, step=0.1)

if st.button("Predict Emissions"):
    input_data = np.array([energy_consumption, transport_emissions, waste_generated, carbon_sequestration]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = emission_model.predict(input_scaled)[0]
    st.success(f"Predicted Emissions: {prediction:.2f} tons")

# Predict Disaster Risk Section
st.markdown("### ⚠️ <span style='color: white;'>Predict Disaster Risk</span>", unsafe_allow_html=True)
temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

if st.button("Predict Disaster Risk"):
    input_data = np.array([temperature, humidity, wind_speed, rainfall]).reshape(1, -1)
    prediction = disaster_model.predict(input_data)[0]
    risk_level = "High" if prediction == 1 else "Low"
    st.success(f"🌪 Disaster Risk: {risk_level}")

# Get Weather Info Section
st.markdown("### ☁️ <span style='color: white;'>Get Weather Info</span>", unsafe_allow_html=True)
city = st.text_input("Enter City Name", help="Enter a city to fetch real-time weather data.")
if st.button("Get Weather"):
    api_key = "9375fcbe724ace8fc40037234edab9b8"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    if "main" in response:
        st.info(f"🌡 Temperature: {response['main']['temp']} °C")
        st.info(f"💧 Humidity: {response['main']['humidity']} %")
        st.info(f"🌬 Wind Speed: {response['wind']['speed']} m/s")
        st.info(f"☀ Weather: {response['weather'][0]['description']}")
    else:
        st.error("❌ City not found. Please enter a valid city name.")

st.markdown('</div>', unsafe_allow_html=True)
