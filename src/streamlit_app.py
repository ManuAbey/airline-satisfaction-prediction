"""
Airline Passenger Satisfaction Prediction
FDM Mini Project 2025
Team: Y3.S1.DS.01.01
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Airline Passenger Satisfaction Prediction System</h1>', unsafe_allow_html=True)
st.markdown("**FDM Mini Project 2025 | Team Y3.S1.DS.01.01**")

st.markdown('<h2 class="sub-header">Passenger Satisfaction Prediction</h2>', unsafe_allow_html=True)
st.markdown("Enter passenger details to predict satisfaction level")

# Input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 7, 85, 40)
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    
    with col2:
        st.subheader("Travel Details")
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
        flight_distance = st.number_input("Flight Distance (miles)", 31, 4983, 1000)
    
    with col3:
        st.subheader("Delays")
        departure_delay = st.number_input("Departure Delay (minutes)", 0, 1592, 0)
        arrival_delay = st.number_input("Arrival Delay (minutes)", 0, 1584, 0)
    
    st.subheader("Service Ratings (0-5 scale)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wifi = st.slider("Inflight WiFi Service", 0, 5, 3)
        booking = st.slider("Ease of Online Booking", 0, 5, 3)
        gate = st.slider("Gate Location", 0, 5, 3)
        food = st.slider("Food and Drink", 0, 5, 3)
        boarding = st.slider("Online Boarding", 0, 5, 3)
    
    with col2:
        seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
        entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
        onboard = st.slider("On-board Service", 0, 5, 3)
        legroom = st.slider("Leg Room Service", 0, 5, 3)
        baggage = st.slider("Baggage Handling", 0, 5, 3)
    
    with col3:
        checkin = st.slider("Check-in Service", 0, 5, 3)
        inflight = st.slider("Inflight Service", 0, 5, 3)
        cleanliness = st.slider("Cleanliness", 0, 5, 3)
        time_convenient = st.slider("Departure/Arrival Time Convenient", 0, 5, 3)
    
    submitted = st.form_submit_button("Predict Satisfaction")
    
    if submitted:
        # Encode inputs
        gender_encoded = 1 if gender == "Male" else 0
        customer_type_encoded = 0 if customer_type == "Loyal Customer" else 1
        travel_type_encoded = 0 if travel_type == "Business travel" else 1
        class_mapping = {"Business": 0, "Eco": 1, "Eco Plus": 2}
        class_encoded = class_mapping[travel_class]
        
        # Calculate aggregate scores
        service_scores = [wifi, time_convenient, booking, gate, food, boarding,
                        seat_comfort, entertainment, onboard, legroom, baggage,
                        checkin, inflight, cleanliness]
        total_service = sum(service_scores)
        avg_service = np.mean(service_scores)
        
        frontline = [seat_comfort, entertainment, onboard, inflight, cleanliness]
        logistics = [checkin, baggage, gate, booking]
        frontline_score = sum(frontline)
        logistics_score = sum(logistics)
        
        # Simple prediction logic
        if avg_service >= 3.5 and class_encoded == 0:
            prediction = "Satisfied"
            confidence = 0.85
        elif avg_service >= 3.0 and travel_type_encoded == 0:
            prediction = "Satisfied"
            confidence = 0.75
        elif avg_service < 2.5:
            prediction = "Neutral or Dissatisfied"
            confidence = 0.80
        else:
            prediction = "Neutral or Dissatisfied" if avg_service < 3.0 else "Satisfied"
            confidence = 0.70
        
        # Display result
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Prediction Result</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with col3:
            st.metric("Average Service Score", f"{avg_service:.2f}/5")
        
        # Visualization
        if prediction == "Satisfied":
            st.success("This passenger is predicted to be SATISFIED")
        else:
            st.warning("This passenger is predicted to be NEUTRAL OR DISSATISFIED")
        
        # Service breakdown
        st.markdown("### Service Ratings Breakdown")
        service_df = pd.DataFrame({
            'Service': ['WiFi', 'Booking', 'Gate', 'Food', 'Boarding',
                       'Seat Comfort', 'Entertainment', 'Onboard', 'Legroom',
                       'Baggage', 'Check-in', 'Inflight', 'Cleanliness', 'Time'],
            'Rating': service_scores
        })
        
        fig = px.bar(service_df, x='Service', y='Rating', 
                    color='Rating', color_continuous_scale='RdYlGn',
                    title='Service Ratings Overview')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Airline Passenger Satisfaction Prediction System</p>
        <p>FDM Mini Project 2025 | Team Y3.S1.DS.01.01</p>
        <p>Developed by: IT23185616, IT23409446, IT23398252, IT23409514</p>
    </div>
    """, unsafe_allow_html=True)