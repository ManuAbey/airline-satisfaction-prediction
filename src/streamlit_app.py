"""
Airline Passenger Satisfaction Prediction - Streamlined Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(135deg, #1b2a40, #456996);
        background-attachment: fixed;
        color: #f1f5f9;
    }

    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        padding: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #456996, #1b2a40);
        border-radius: 10px;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.6);
    }
    
    .section-header {
        font-size: 1.3rem;
        color: #fef9c3;
        padding: 0.5rem 0;
        font-weight: 600;
        border-bottom: 3px solid #facc15;
        margin: 1.5rem 0 1rem 0;
    }

    /* Subheadings inside markdown */
    .stMarkdown h2, .stMarkdown strong {
        color: #93c5fd; /* light sky blue */
    }
    
    /* Dropdown labels */
    .stSelectbox label, .stNumberInput label {
        color: #e2e8f0 !important;  /* light gray-blue */
        font-weight: 600;
    }

    /* Radio labels */
    .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #facc15 0%, #eab308 100%);
        color: #1b2a40;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        margin-top: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #eab308 0%, #facc15 100%);
        color: #0f172a;
        transform: scale(1.03);
    }

    .stRadio label span  {
        color: white !important;
        font-weight: 600;
    }

    .stRadio [role="radio"] {
        background-color: #1b2a40 !important;   /* background */
        border: 2px solid #facc15 !important;   /* golden border */
        border-radius: 50% !important;
    }

    .stRadio [role="radio"][aria-checked="true"] {
        background-color: #facc15 !important;   /* filled circle */
        border-color: #eab308 !important;
    }

    
    /* Columns spacing */
    div[data-testid="stHorizontalBlock"] > div {
        padding: 0.5rem;
    }

    [data-testid="stMetricValue"] {
        color: #38bdf8; /* bright cyan */
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        color: #fef9c3;
    }

    .stSuccess {
        color: #22c55e; /* green */
        font-weight: 600;
    }
    .stWarning {
        color: #facc15; /* yellow */
        font-weight: 600;
    }
    .stError {
        color: #ef4444; /* bright red */
        font-weight: 600;
    }
    .stInfo {
        color: #60a5fa; /* light blue */
        font-weight: 600;
    }
    
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load the trained model and artifacts"""
    try:
        models_dir = Path('models')
        if not models_dir.exists():
            models_dir = Path('../models')
        
        if not models_dir.exists():
            st.error("**Model files not found!** Please run your training notebook first.")
            st.stop()
        
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
        st.stop()


def preprocess_input(user_input, artifacts):
    """Preprocess user input to match training format"""
    
    feature_names = artifacts['feature_names']
    label_mappings = artifacts['label_mappings']
    
    # Create feature dictionary
    features = {}
    
    # Encode categorical variables
    features['Gender'] = label_mappings['Gender'][user_input['gender']]
    features['Customer Type'] = label_mappings['Customer Type'][user_input['customer_type']]
    features['Type of Travel'] = label_mappings['Type of Travel'][user_input['travel_type']]
    features['Class'] = label_mappings['Class'][user_input['travel_class']]
    
    # Numerical features
    features['Age'] = user_input['age']
    features['Flight Distance'] = user_input['flight_distance']
    features['Departure Delay in Minutes'] = user_input['departure_delay']
    features['Arrival Delay in Minutes'] = user_input['arrival_delay']
    
    # Service ratings
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
    
    # Engineered features (if they exist)
    service_values = [
        user_input['wifi'], user_input['time_convenient'], user_input['booking'],
        user_input['gate'], user_input['food'], user_input['boarding'],
        user_input['seat_comfort'], user_input['entertainment'], user_input['onboard'],
        user_input['legroom'], user_input['baggage'], user_input['checkin'],
        user_input['inflight'], user_input['cleanliness']
    ]
    
    if 'TotalServiceScore' in feature_names:
        features['TotalServiceScore'] = sum(service_values)
    
    if 'AvgServiceScore' in feature_names:
        features['AvgServiceScore'] = np.mean(service_values)
    
    if 'DistanceCategory' in feature_names:
        distance = user_input['flight_distance']
        if distance <= 500:
            features['DistanceCategory'] = 0
        elif distance <= 1500:
            features['DistanceCategory'] = 1
        else:
            features['DistanceCategory'] = 2
    
    # Create DataFrame in exact training order
    feature_values = [features.get(fname, 0) for fname in feature_names]
    df = pd.DataFrame([feature_values], columns=feature_names)
    
    return df


def predict_satisfaction(input_df, artifacts):
    """Make prediction using the trained model"""
    
    model = artifacts['model']
    
    # Random Forest doesn't need scaling
    prediction = model.predict(input_df.values)[0]
    prediction_proba = model.predict_proba(input_df.values)[0]
    
    satisfaction_label = "Neutral or Dissatisfied" if prediction == 0 else "Satisfied"
    confidence = prediction_proba[prediction]
    
    return satisfaction_label, confidence, prediction_proba


# ==================== MAIN APP ====================

artifacts = load_model_artifacts()

st.markdown('<h1 class="main-header">Airline Passenger Satisfaction Predictor</h1>', unsafe_allow_html=True)

# Model metrics
col1, col2, col3 = st.columns(3)
metadata = artifacts['metadata']

# Input form
with st.form("prediction_form"):
    
    st.markdown('<div class="section-header">Passenger Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age = st.number_input("Age", min_value=7, max_value=85, value=40)
    with col3:
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    
    st.markdown('<div class="section-header">Travel Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    with col2:
        travel_class = st.selectbox("Travel Class", ["Business", "Eco", "Eco Plus"])
    with col3:
        flight_distance = st.number_input("Flight Distance (miles)", min_value=31, max_value=4983, value=1000)
    
    st.markdown('<div class="section-header">Flight Delays</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        departure_delay = st.number_input("Departure Delay (minutes)", min_value=0, max_value=1592, value=0)
    with col2:
        arrival_delay = st.number_input("Arrival Delay (minutes)", min_value=0, max_value=1584, value=0)
    
    st.markdown('<div class="section-header">Service Ratings (1=Poor, 5=Excellent)</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Digital Services**")
        wifi = st.radio("WiFi Service", [1, 2, 3, 4, 5], index=2, horizontal=True)
        booking = st.radio("Online Booking", [1, 2, 3, 4, 5], index=2, horizontal=True)
        boarding = st.radio("Online Boarding", [1, 2, 3, 4, 5], index=2, horizontal=True)
        entertainment = st.radio("Entertainment", [1, 2, 3, 4, 5], index=2, horizontal=True)
        gate = st.radio("Gate Location", [1, 2, 3, 4, 5], index=2, horizontal=True)
    
    with col2:
        st.markdown("**Comfort**")
        seat_comfort = st.radio("Seat Comfort", [1, 2, 3, 4, 5], index=2, horizontal=True)
        legroom = st.radio("Leg Room", [1, 2, 3, 4, 5], index=2, horizontal=True)
        cleanliness = st.radio("Cleanliness", [1, 2, 3, 4, 5], index=2, horizontal=True)
        food = st.radio("Food & Drink", [1, 2, 3, 4, 5], index=2, horizontal=True)
        time_convenient = st.radio("Time Convenient", [1, 2, 3, 4, 5], index=2, horizontal=True)
    
    with col3:
        st.markdown("**Service Quality**")
        checkin = st.radio("Check-in Service", [1, 2, 3, 4, 5], index=2, horizontal=True)
        onboard = st.radio("On-board Service", [1, 2, 3, 4, 5], index=2, horizontal=True)
        inflight = st.radio("Inflight Service", [1, 2, 3, 4, 5], index=2, horizontal=True)
        baggage = st.radio("Baggage Handling", [1, 2, 3, 4, 5], index=2, horizontal=True)
    
    submitted = st.form_submit_button("🔮 Predict Satisfaction")
    
    if submitted:
        user_input = {
            'gender': gender, 'age': age, 'customer_type': customer_type,
            'travel_type': travel_type, 'travel_class': travel_class,
            'flight_distance': flight_distance, 'departure_delay': departure_delay,
            'arrival_delay': arrival_delay, 'wifi': wifi, 'time_convenient': time_convenient,
            'booking': booking, 'gate': gate, 'food': food, 'boarding': boarding,
            'seat_comfort': seat_comfort, 'entertainment': entertainment,
            'onboard': onboard, 'legroom': legroom, 'baggage': baggage,
            'checkin': checkin, 'inflight': inflight, 'cleanliness': cleanliness
        }
        
        with st.spinner("Analyzing..."):
            input_df = preprocess_input(user_input, artifacts)
            prediction, confidence, proba = predict_satisfaction(input_df, artifacts)
        
        st.markdown("---")
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with col3:
            service_scores = [wifi, time_convenient, booking, gate, food, boarding,
                            seat_comfort, entertainment, onboard, legroom, baggage,
                            checkin, inflight, cleanliness]
            avg_service = np.mean(service_scores)
            st.metric("Avg Service Score", f"{avg_service:.2f}/5")
        
        # Probability visualization
        prob_df = pd.DataFrame({
            'Outcome': ['Dissatisfied', 'Satisfied'],
            'Probability': [proba[0]*100, proba[1]*100]
        })
        fig = px.bar(prob_df, x='Outcome', y='Probability',
                    color='Outcome',
                    color_discrete_map={'Dissatisfied': '#e74c3c', 'Satisfied': '#2ecc71'},
                    title='Prediction Probabilities')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        if prediction == "Satisfied":
            st.success("This passenger is predicted to be SATISFIED")
        else:
            st.warning("This passenger is predicted to be NEUTRAL OR DISSATISFIED")
        
        # Key Insights
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        service_names = ['WiFi', 'Time Convenient', 'Booking', 'Gate', 'Food', 'Boarding',
                        'Seat Comfort', 'Entertainment', 'Onboard', 'Legroom', 
                        'Baggage', 'Check-in', 'Inflight', 'Cleanliness']
        
        with col1:
            st.markdown("**Areas of Concern** (Score ≤ 2)")
            low_scores = [(name, score) for name, score in zip(service_names, service_scores) if score <= 2]
            
            if low_scores:
                for name, score in low_scores:
                    st.error(f"**{name}**: {score}/5 - Critical")
            else:
                st.success("No critical issues detected")
        
        with col2:
            st.markdown("**Strong Points** (Score ≥ 4)")
            high_scores = [(name, score) for name, score in zip(service_names, service_scores) if score >= 4]
            
            if high_scores:
                for name, score in high_scores:
                    st.success(f"**{name}**: {score}/5 - Excellent")
            else:
                st.info("ℹ️ Room for improvement across services")

st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        Powered by {metadata['model_name']} | Accuracy: {metadata['accuracy']*100:.2f}%
    </div>
""", unsafe_allow_html=True)