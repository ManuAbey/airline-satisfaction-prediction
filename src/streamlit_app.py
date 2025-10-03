"""
Airline Passenger Satisfaction Prediction - ML-Powered Version
This Streamlit app uses the trained ML model for predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #F1F1F1;
        text-align: center;
        padding: 1.5rem;
        font-weight: 700;
        background: radial-gradient(circle, #456996, #1b2a40);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.4rem;
        color: #F1F1F1;
        padding: 0.8rem;
        font-weight: 600;
        border-left: 8px solid #8c9aab;
        background: radial-gradient(circle, #456996, #1b2a40);
        margin: 1.5rem 0 1rem 0;
    }
    
    .info-box {
        text-align: center;
        color: #1e3a5f;
        padding: 1rem;
        background: #f0f8ff;
        border-radius: 8px;
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
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load the trained model and associated artifacts"""
    try:
        models_dir = Path('models')
        
        if not models_dir.exists():
            models_dir = Path('../models')
        
        if not models_dir.exists():
            st.error("**Model files not found!**")
            st.info("""
            Please follow these steps:
            1. Run your model training notebook
            2. Add the model saving code at the end
            3. This will create the `models/` directory
            4. Then restart this Streamlit app
            """)
            st.stop()
        
        # Load all artifacts
        model = joblib.load(models_dir / 'best_model.pkl')
        scaler = joblib.load(models_dir / 'scaler.pkl')
        feature_names = joblib.load(models_dir / 'feature_names.pkl')
        label_mappings = joblib.load(models_dir / 'label_mappings.pkl')
        metadata = joblib.load(models_dir / 'model_metadata.pkl')
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'label_mappings': label_mappings,
            'metadata': metadata
        }
    except Exception as e:
        st.error(f"**Error loading model:** {str(e)}")
        st.info("Please ensure the model training notebook has been run and models saved.")
        st.stop()


def preprocess_input(user_input, artifacts):
    """
    Preprocess user input to match the exact format used during training.
    This must match your feature engineering from the notebooks.
    """
    
    feature_names = artifacts['feature_names']
    label_mappings = artifacts['label_mappings']
    
    # Create a dictionary to hold all features
    features = {}
    
    # Encode categorical variables (must match training encoding)
    features['Gender'] = label_mappings['Gender'][user_input['gender']]
    features['Customer Type'] = label_mappings['Customer Type'][user_input['customer_type']]
    features['Type of Travel'] = label_mappings['Type of Travel'][user_input['travel_type']]
    features['Class'] = label_mappings['Class'][user_input['travel_class']]
    
    # Add basic numerical features
    features['Age'] = user_input['age']
    features['Flight Distance'] = user_input['flight_distance']
    features['Departure Delay in Minutes'] = user_input['departure_delay']
    features['Arrival Delay in Minutes'] = user_input['arrival_delay']
    
    # Add all service rating features
    features['Inflight wifi service'] = user_input['wifi']
    features['Departure/Arrival time convenient'] = user_input['time_convenient']
    features['Ease of Online booking'] = user_input['booking']
    features['Gate location'] = user_input['gate']
    features['Food and drink'] = user_input['food']
    features['Online boarding'] = user_input['boarding']
    features['Seat comfort'] = user_input['seat_comfort']
    features['Inflight entertainment'] = user_input['entertainment']
    features['On-board service'] = user_input['onboard']
    features['Leg room service'] = user_input['legroom']
    features['Baggage handling'] = user_input['baggage']
    features['Checkin service'] = user_input['checkin']
    features['Inflight service'] = user_input['inflight']
    features['Cleanliness'] = user_input['cleanliness']
    
    # Calculate engineered features (if they exist in your training data)
    service_values = [
        user_input['wifi'], user_input['time_convenient'], user_input['booking'],
        user_input['gate'], user_input['food'], user_input['boarding'],
        user_input['seat_comfort'], user_input['entertainment'], user_input['onboard'],
        user_input['legroom'], user_input['baggage'], user_input['checkin'],
        user_input['inflight'], user_input['cleanliness']
    ]
    
    # Only add engineered features if they exist in the trained model
    if 'TotalServiceScore' in feature_names:
        features['TotalServiceScore'] = sum(service_values)
    
    if 'AvgServiceScore' in feature_names:
        features['AvgServiceScore'] = np.mean(service_values)
    
    if 'HasDelay' in feature_names:
        features['HasDelay'] = int((user_input['departure_delay'] > 0) or (user_input['arrival_delay'] > 0))
    
    if 'AgeGroup' in feature_names:
        age = user_input['age']
        if age <= 25:
            features['AgeGroup'] = 0
        elif age <= 40:
            features['AgeGroup'] = 1
        elif age <= 60:
            features['AgeGroup'] = 2
        else:
            features['AgeGroup'] = 3
    
    if 'DistanceCategory' in feature_names:
        distance = user_input['flight_distance']
        if distance <= 500:
            features['DistanceCategory'] = 0
        elif distance <= 1500:
            features['DistanceCategory'] = 1
        else:
            features['DistanceCategory'] = 2
    
    # Create DataFrame with features in the EXACT order from training
    feature_values = []
    for fname in feature_names:
        if fname in features:
            feature_values.append(features[fname])
        else:
            st.warning(f"Feature '{fname}' not found in input processing!")
            feature_values.append(0)  # Default value
    
    df = pd.DataFrame([feature_values], columns=feature_names)
    
    return df


def predict_satisfaction(input_df, artifacts):
    """Make prediction using the trained model"""
    
    model = artifacts['model']
    metadata = artifacts['metadata']
    
    # Apply scaling if required (Random Forest doesn't need it)
    if metadata.get('requires_scaling', False):
        scaler = artifacts['scaler']
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
    else:
        prediction = model.predict(input_df.values)[0]
        prediction_proba = model.predict_proba(input_df.values)[0]
    
    # Map prediction to label (0 = Dissatisfied, 1 = Satisfied)
    satisfaction_label = "Neutral or Dissatisfied" if prediction == 0 else "Satisfied"
    confidence = prediction_proba[prediction]
    
    return satisfaction_label, confidence, prediction_proba


# ==================== MAIN APP ====================

# Load model artifacts
artifacts = load_model_artifacts()

# Header
st.markdown('<h1 class="main-header">Airline Passenger Satisfaction Predictor</h1>', unsafe_allow_html=True)

# Model info
with st.expander("ℹ️ Model Information"):
    metadata = artifacts['metadata']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Type", metadata['model_name'])
    with col2:
        st.metric("Training Accuracy", f"{metadata['accuracy']*100:.2f}%")
    with col3:
        st.metric("ROC-AUC Score", f"{metadata['roc_auc']:.4f}")
    with col4:
        st.metric("Features Used", metadata['n_features'])

# Introduction
st.markdown("""
<div class="info-box">
<strong>Welcome!</strong> Fill in the passenger details below to predict satisfaction level using our trained ML model.
</div>
""", unsafe_allow_html=True)

# Input form
with st.form("prediction_form"):
    
    # Passenger Information
    st.markdown('<div class="section-header">👤 Passenger Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col2:
        age = st.number_input("Age", min_value=7, max_value=85, value=40)
    
    with col3:
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    
    # Travel Details
    st.markdown('<div class="section-header">Travel Details</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    
    with col2:
        travel_class = st.selectbox("Travel Class", ["Business", "Eco", "Eco Plus"])
    
    with col3:
        flight_distance = st.number_input("Flight Distance (miles)", min_value=31, max_value=4983, value=1000)
    
    # Delays
    st.markdown('<div class="section-header">Flight Delays</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        departure_delay = st.number_input("Departure Delay (minutes)", min_value=0, max_value=1592, value=0)
    
    with col2:
        arrival_delay = st.number_input("Arrival Delay (minutes)", min_value=0, max_value=1584, value=0)
    
    # Service Ratings
    st.markdown('<div class="section-header">Service Experience Ratings</div>', unsafe_allow_html=True)
    st.markdown("Rate each service from 1 (Poor) to 5 (Excellent)")
    
    tab1, tab2, tab3 = st.tabs(["Flight Services", "Comfort & Amenities", "Customer Service"])
    
    with tab1:
        st.markdown("##### Flight Services")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Inflight WiFi Service**")
            wifi = st.radio("wifi", [1, 2, 3, 4, 5], index=2, horizontal=True, label_visibility="collapsed")
            
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
        with st.spinner("ML Model is analyzing the data..."):
            # Prepare user input
            user_input = {
                'gender': gender,
                'age': age,
                'customer_type': customer_type,
                'travel_type': travel_type,
                'travel_class': travel_class,
                'flight_distance': flight_distance,
                'departure_delay': departure_delay,
                'arrival_delay': arrival_delay,
                'wifi': wifi,
                'time_convenient': time_convenient,
                'booking': booking,
                'gate': gate,
                'food': food,
                'boarding': boarding,
                'seat_comfort': seat_comfort,
                'entertainment': entertainment,
                'onboard': onboard,
                'legroom': legroom,
                'baggage': baggage,
                'checkin': checkin,
                'inflight': inflight,
                'cleanliness': cleanliness
            }
            
            # Preprocess input to match training format
            input_df = preprocess_input(user_input, artifacts)
            
            # Make prediction using trained model
            prediction, confidence, proba = predict_satisfaction(input_df, artifacts)
            
            # Calculate display metrics
            service_scores = [wifi, time_convenient, booking, gate, food, boarding,
                            seat_comfort, entertainment, onboard, legroom, baggage,
                            checkin, inflight, cleanliness]
            avg_service = np.mean(service_scores)
            delay_factor = (departure_delay + arrival_delay) / 2
        
        # Display results
        st.markdown("---")
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Model Confidence", f"{confidence*100:.1f}%")
        with col3:
            st.metric("Avg Service Score", f"{avg_service:.2f}/5")
        with col4:
            st.metric("Average Delay", f"{int(delay_factor)} min")
        
        # Probabilities
        st.markdown("##### Prediction Probabilities")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Dissatisfied Probability", f"{proba[0]*100:.1f}%")
        with col2:
            st.metric("Satisfied Probability", f"{proba[1]*100:.1f}%")
        
        # Probability chart
        prob_df = pd.DataFrame({
            'Outcome': ['Dissatisfied', 'Satisfied'],
            'Probability': [proba[0]*100, proba[1]*100]
        })
        fig_prob = px.bar(prob_df, x='Outcome', y='Probability',
                         color='Outcome',
                         color_discrete_map={'Dissatisfied': '#e74c3c', 'Satisfied': '#2ecc71'},
                         title='Prediction Probability Distribution',
                         range_y=[0, 100])
        fig_prob.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Result message
        if prediction == "Satisfied":
            st.success("This passenger is predicted to be SATISFIED with their flight experience")
        else:
            st.warning("This passenger is predicted to be NEUTRAL OR DISSATISFIED with their flight experience")
        
        # Service ratings visualization
        st.markdown("---")
        st.markdown('<div class="section-header">Service Ratings Overview</div>', unsafe_allow_html=True)
        
        service_df = pd.DataFrame({
            'Service': ['WiFi', 'Time', 'Booking', 'Gate', 'Food', 'Boarding',
                       'Seat', 'Entertainment', 'Onboard', 'Legroom',
                       'Baggage', 'Check-in', 'Inflight', 'Cleanliness'],
            'Rating': service_scores,
            'Category': ['Digital']*3 + ['Logistics']*3 + ['Comfort']*4 + ['Service']*4
        })
        
        fig = px.bar(service_df, x='Service', y='Rating', 
                    color='Category',
                    color_discrete_map={
                        'Digital': '#3498db',
                        'Logistics': '#9b59b6', 
                        'Comfort': '#e67e22',
                        'Service': '#1abc9c'
                    },
                    title='Service Ratings by Category',
                    range_y=[0, 5])
        fig.update_layout(xaxis_tickangle=-45, height=400)
        fig.add_hline(y=3, line_dash="dash", line_color="red", 
                     annotation_text="Neutral Threshold", annotation_position="right")
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Areas of Concern")
            low_scores = [(name, score) for name, score in zip(
                ['WiFi', 'Time', 'Booking', 'Gate', 'Food', 'Boarding',
                 'Seat', 'Entertainment', 'Onboard', 'Legroom', 'Baggage', 
                 'Check-in', 'Inflight', 'Cleanliness'],
                service_scores
            ) if score <= 2]
            
            if low_scores:
                for name, score in low_scores:
                    st.error(f"**{name}**: {score}/5 - Critical")
            else:
                st.success("No critical issues detected")
        
        with col2:
            st.markdown("##### Strong Points")
            high_scores = [(name, score) for name, score in zip(
                ['WiFi', 'Time', 'Booking', 'Gate', 'Food', 'Boarding',
                 'Seat', 'Entertainment', 'Onboard', 'Legroom', 'Baggage', 
                 'Check-in', 'Inflight', 'Cleanliness'],
                service_scores
            ) if score >= 4]
            
            if high_scores:
                for name, score in high_scores:
                    st.success(f"**{name}**: {score}/5 - Excellent")
            else:
                st.info("Room for improvement across services")

# Footer
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #64748b; padding: 2rem;'>
        <p><strong>Airline Passenger Satisfaction Prediction System</strong></p>
        <p>Powered by {artifacts['metadata']['model_name']} | 
        Accuracy: {artifacts['metadata']['accuracy']*100:.2f}% | 
        ROC-AUC: {artifacts['metadata']['roc_auc']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)