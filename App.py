import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("housePrediciting.pkl")

# Set up UI
st.set_page_config(page_title="California House Price Predictor", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>🏡 California House Price Prediction</h1>", unsafe_allow_html=True)
st.write("Enter the house details below to predict its price in California.")

# Sidebar Inputs
st.sidebar.image("https://cdn.pixabay.com/photo/2017/06/10/07/24/house-2382713_1280.jpg", use_container_width=True)
st.sidebar.title("🔍 Input House Features")

med_inc = st.sidebar.number_input("💰 Median Income (scaled)", min_value=0.0, max_value=15.0, value=3.9, step=0.1)
house_age = int(st.sidebar.slider("🏠 House Age (years)", min_value=1, max_value=100, value=30))
ave_rooms = int(st.sidebar.number_input("🛏️ Avg. Rooms per House", min_value=1, max_value=10, value=5, step=1))
ave_bedrooms = int(st.sidebar.number_input("🚪 Avg. Bedrooms per House", min_value=1, max_value=5, value=1, step=1))
population = int(st.sidebar.number_input("👨‍👩‍👧‍👦 Population in Area", min_value=100, max_value=50000, value=3000))
ave_occup = st.sidebar.number_input("👥 Avg. Occupants per House", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
latitude = st.sidebar.number_input("📍 Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.01)
longitude = st.sidebar.number_input("📍 Longitude", min_value=-125.0, max_value=-114.0, value=-118.0, step=0.01)

# House Age in Years (Additional Display)
st.sidebar.write(f"📆 House Built in: **{2025 - house_age}**")

# Prediction Button
if st.sidebar.button("💰 Predict Price"):
    # Convert inputs into NumPy array
    input_features = np.array([[med_inc, house_age, ave_rooms, ave_bedrooms, population, ave_occup, latitude, longitude]])
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Display result
    st.markdown(f"<h2 style='text-align: center; color: green;'>🏡 Estimated Price: ${prediction:,.2f}</h2>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Developed with ❤️ using Streamlit</h5>", unsafe_allow_html=True)
