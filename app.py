import streamlit as st
import joblib
import pandas as pd
import base64

# Load the trained model and scaler
loaded_model = joblib.load("crop_yield_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

# Define thresholds for yield classification
LOW_THRESHOLD = 300
HIGH_THRESHOLD = 400

# Function to encode a local image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert background image to Base64
background_image = get_base64_image("BGimage.jpg")

# Apply custom styling with CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{background_image}") no-repeat center center fixed;
        background-size: cover;
    }}
    .main-container {{
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 10px;
    }}
    h1 {{
        color: #2E86C1;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }}
    h2 {{
        color: #148F77;
        font-family: 'Arial', sans-serif;
        border-bottom: 2px solid #148F77;
        padding-bottom: 5px;
    }}
    .stButton>button {{
        background-color: #148F77;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        width: 100%;
    }}
    .stButton>button:hover {{
        background-color: #117A65;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App Title
st.title("🌾 Crop Yield Prediction App")

# Input Fields for User
st.header("Input Features")
crop_type = st.selectbox("Crop Type", ["Wheat", "Corn", "Rice", "Soybean"])
irrigation = st.selectbox("Irrigation", ["No", "Yes"])

col1, col2 = st.columns(2)
with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    potassium = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
    sunlight = st.number_input("Sunlight (hours/day)", min_value=0.0, max_value=24.0, value=6.0)
    fertilizer_used = st.number_input("Fertilizer Used (kg/ha)", min_value=0, max_value=500, value=100)

# Convert categorical inputs to numerical values
irrigation = 1 if irrigation == "Yes" else 0
crop_type_mapping = {"Wheat": 0, "Corn": 1, "Rice": 2, "Soybean": 3}
crop_type = crop_type_mapping[crop_type]

# Prepare input data as DataFrame
input_data = pd.DataFrame({
    "Nitrogen": [nitrogen],
    "Phosphorus": [phosphorus],
    "Potassium": [potassium],
    "pH": [ph],
    "Temperature": [temperature],
    "Rainfall": [rainfall],
    "Humidity": [humidity],
    "Sunlight": [sunlight],
    "Irrigation": [irrigation],
    "Fertilizer_Used": [fertilizer_used],
    "Crop_Type": [crop_type]
})

# Prediction Button
if st.button("Predict Yield"):
    # Scale the input data
    input_data_scaled = loaded_scaler.transform(input_data)

    # Predict the yield using the trained model
    predicted_yield = loaded_model.predict(input_data_scaled)[0]

    # Classify the predicted yield
    if predicted_yield < LOW_THRESHOLD:
        yield_class = "Low"
    elif LOW_THRESHOLD <= predicted_yield < HIGH_THRESHOLD:
        yield_class = "Medium"
    else:
        yield_class = "High"

    # Display prediction results
    st.success(f"Predicted Crop Yield: **{predicted_yield:.2f} kg/ha**")
    st.success(f"Yield Classification: **{yield_class}**")

st.markdown('</div>', unsafe_allow_html=True)  # Close container
