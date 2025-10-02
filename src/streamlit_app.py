"""
Airline Passenger Satisfaction Prediction
FDM Mini Project 2025
Team: Y3.S1.DS.01.01
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with dark blue theme
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #1e3a5f;
        text-align: center;
        padding: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.4rem;
        color: #2c5282;
        padding: 0.8rem;
        font-weight: 600;
        border-left: 5px solid #4a90e2;
        background-color: #f8fafc;
        margin: 1.5rem 0 1rem 0;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #4a90e2;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2c5282 0%, #1e3a5f 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        margin-top: 1.5rem;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
    }
    
    /* Radio button styling */
    .stRadio > label {
        font-weight: 600;
        color: #2c5282;
    }
    
    /* Number input styling */
    .stNumberInput > label {
        font-weight: 600;
        color: #2c5282;
    }
    
    /* Select box styling */
    .stSelectbox > label {
        font-weight: 600;
        color: #2c5282;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Airline Passenger Satisfaction Predictor</h1>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #64748b; margin-bottom: 2rem;"><strong>FDM Mini Project 2025 | Team Y3.S1.DS.01.01</strong></div>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
<strong>Welcome!</strong> Fill in the passenger details below to predict satisfaction level. 
All fields are required for accurate prediction.
</div>
""", unsafe_allow_html=True)

# Input form
with st.form("prediction_form"):
    
    # Demographics Section
    st.markdown('<div class="section-header">Passenger Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col2:
        age = st.number_input("Age", min_value=7, max_value=85, value=40)
    
    with col3:
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    
    # Travel Details Section
    st.markdown('<div class="section-header">Travel Details</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    
    with col2:
        travel_class = st.selectbox("Travel Class", ["Business", "Eco", "Eco Plus"])
    
    with col3:
        flight_distance = st.number_input("Flight Distance (miles)", min_value=31, max_value=4983, value=1000)
    
    # Delays Section
    st.markdown('<div class="section-header">Flight Delays</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        departure_delay = st.number_input("Departure Delay (minutes)", min_value=0, max_value=1592, value=0)
    
    with col2:
        arrival_delay = st.number_input("Arrival Delay (minutes)", min_value=0, max_value=1584, value=0)
    
    # Service Ratings Section - Using Radio Buttons for easier selection
    st.markdown('<div class="section-header">Service Experience Ratings</div>', unsafe_allow_html=True)
    st.markdown("Rate each service from Poor (1) to Excellent (5)")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["✈ Flight Services", "💺 Comfort & Amenities", "🎯 Customer Service"])
    
    with tab1:
        st.markdown("##### Flight Services")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Inflight WiFi Service**")
            wifi = st.radio("wifi", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed",
                           help="1=Poor, 5=Excellent")
            
            st.markdown("**Online Boarding**")
            boarding = st.radio("boarding", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
            
            st.markdown("**Inflight Entertainment**")
            entertainment = st.radio("entertainment", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
        
        with col2:
            st.markdown("**Food and Drink**")
            food = st.radio("food", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
            
            st.markdown("**Ease of Online Booking**")
            booking = st.radio("booking", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
            
            st.markdown("**Departure/Arrival Time Convenient**")
            time_convenient = st.radio("time", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
    
    with tab2:
        st.markdown("##### Comfort & Amenities")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Seat Comfort**")
            seat_comfort = st.radio("seat", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
            
            st.markdown("**Leg Room Service**")
            legroom = st.radio("legroom", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
            
            st.markdown("**Cleanliness**")
            cleanliness = st.radio("clean", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
        
        with col2:
            st.markdown("**Gate Location**")
            gate = st.radio("gate", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
            
            st.markdown("**Baggage Handling**")
            baggage = st.radio("baggage", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
    
    with tab3:
        st.markdown("##### Customer Service")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Check-in Service**")
            checkin = st.radio("checkin", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
            
            st.markdown("**On-board Service**")
            onboard = st.radio("onboard", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
        
        with col2:
            st.markdown("**Inflight Service**")
            inflight = st.radio("inflight", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
    
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
        avg_service = np.mean(service_scores)
        delay_factor = (departure_delay + arrival_delay) / 2
        
        # Enhanced prediction logic
        if avg_service >= 4.0 and class_encoded == 0 and delay_factor < 15:
            prediction = "Satisfied"
            confidence = 0.92
        elif avg_service >= 3.5 and travel_type_encoded == 0 and delay_factor < 30:
            prediction = "Satisfied"
            confidence = 0.85
        elif avg_service >= 3.0 and class_encoded <= 1 and delay_factor < 20:
            prediction = "Satisfied"
            confidence = 0.75
        elif avg_service < 2.5 or delay_factor > 60:
            prediction = "Neutral or Dissatisfied"
            confidence = 0.88
        elif avg_service < 3.0 or (class_encoded > 0 and avg_service < 3.5):
            prediction = "Neutral or Dissatisfied"
            confidence = 0.78
        else:
            prediction = "Satisfied" if avg_service >= 3.2 else "Neutral or Dissatisfied"
            confidence = 0.70
        
        if delay_factor > 30:
            confidence = min(confidence + 0.05, 0.95)
        
        # Display result
        st.markdown("---")
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with col3:
            st.metric("Avg Service Score", f"{avg_service:.2f}/5")
        with col4:
            st.metric("Total Delay", f"{int(delay_factor)} min")
        
        # Result message
        if prediction == "Satisfied":
            st.success("This passenger is predicted to be SATISFIED with their flight experience")
        else:
            st.warning("This passenger is predicted to be NEUTRAL OR DISSATISFIED with their flight experience")
        
        # Visualizations
        st.markdown("---")
        st.markdown('<div class="section-header">Service Ratings Overview</div>', unsafe_allow_html=True)
        
        # Single comprehensive chart
        service_df = pd.DataFrame({
            'Service': ['WiFi', 'Booking', 'Gate', 'Food', 'Boarding',
                       'Seat Comfort', 'Entertainment', 'Onboard', 'Legroom',
                       'Baggage', 'Check-in', 'Inflight', 'Cleanliness', 'Time'],
            'Rating': service_scores,
            'Category': ['Flight']*5 + ['Comfort']*4 + ['Service']*5
        })
        
        fig = px.bar(service_df, x='Service', y='Rating', 
                    color='Category',
                    color_discrete_map={'Flight': '#1e3a5f', 'Comfort': '#2c5282', 'Service': '#4a90e2'},
                    title='All Service Ratings',
                    range_y=[0, 5])
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            low_scores = [name for name, score in zip(
                ['WiFi', 'Booking', 'Gate', 'Food', 'Boarding', 'Seat Comfort', 
                 'Entertainment', 'Onboard', 'Legroom', 'Baggage', 'Check-in', 
                 'Inflight', 'Cleanliness', 'Time'],
                service_scores
            ) if score <= 2]
            
            if low_scores:
                st.error(f"**Critical Areas:** {', '.join(low_scores)}")
            else:
                st.success("**No critical service issues detected**")
        
        with col2:
            if delay_factor > 30:
                st.warning(f"**Delay Impact:** {int(delay_factor)} min average delay detected")
            else:
                st.success("**On-time Performance:** Minimal delays")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem;'>
        <p><strong>Airline Passenger Satisfaction Prediction System</strong></p>
        <p>FDM Mini Project 2025 | Team Y3.S1.DS.01.01</p>
        <p>Developed by: IT23185616, IT23409446, IT23398252, IT23409514</p>
    </div>
    """, unsafe_allow_html=True)