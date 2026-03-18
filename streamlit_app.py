# streamlit_app.py
import streamlit as st
import requests
import json

# --- Page Config -----------------------------------
st.set_page_config(
    page_title = "Health Risk Predictor",
    page_icon  = "🏥",
    layout     = "wide" 
)


# --- FastAPI endpoint ------------------------------
# Learning note:
# streamlit calls FastAPI - this is a microservice pattern
# streamlit = frontend (UI layer)
# FastAPI   = backend (business logic + model)
# They are SEPARATE service - can be scaled independently
API_URL = "http://localhost:8000/predict"

# --- Header ------------------------------------------
st.title("Health Risk Predictor")
st.markdown("""
            **Diabetes Risk Assessment** powered by XGBoost  
            *Trained on 253,680 CDC patient records | ROC-AUC: 0.8242*
        """)
st.divider()

# --- Sidebar - model info --------------------------------
with st.sidebar:
    st.header("Model Information")
    st.metric("ROC-AUC",          "0.8242")
    st.metric("Recall",           "79.0%")
    st.metric("F1 Score",         "0.438")
    st.metric("Training Samples", "253,680")
    st.divider()
    st.caption("Model: XGBoost (GridSearchCV tuned)")
    st.caption("Dataset: CDC BRFSS 2015")
    st.caption("Strategy: scale_pos_weight=6.18")
    
    
# --- Define format function once --------------------------
def yes_no(x): return "Yes" if x ==1 else "No"    

# --- Bianry field definitions ------------------------------
# (variable_name, display_label)
# Adding a new field = one line here, loop handles the rest
binary_col1 = [
    ("HighBP",               "High Blood Pressure"),
    ("HighChol",             "High Cholesterol"),
    ("Stroke",               "Ever had Stroke"),
    ("HeartDiseaseorAttack", "Heart Disease / Attack"),
    ("DiffWalk",             "Difficulty Walking")
]

binary_col2 = [
    ("Smoker",            "Smoker (100+ cigarettes)"),
    ("PhysActivity",      "Physical Activity (past 30 days)"),
    ("Fruits",            "Eats Fruit Daily"),
    ("Veggies",           "Eats Vegetables Daily"),
    ("HvyAlcoholConsump", "Heavy Alcohol Consumption"),
]

# --- Input form ---------------------------------------------
st.subheader("Patient Health Information")
st.info("file in the patient's health indicators below")

col1, col2, col3 = st.columns(3)

inputs={} # All input values will stored in input

with col1:
    st.markdown("**Medical Conditions**")
    for var_name, label in binary_col1:
        inputs[var_name] = st.selectbox(
            label,
            options=[0,1],
            format_func=yes_no,
            key = var_name
        )
        
with col2:
    st.markdown("**Lifestyle Factors**")
    inputs['BMI'] = st.slider(            
        "BMI", 10.0, 60.0, 27.0, 0.5)
    for var_name, label in binary_col2:
        inputs[var_name] = st.selectbox(
            label,
            options=[0,1],
            format_func=yes_no,
            key = var_name
        )
        
        
with col3:
    st.markdown("**Demographics & General Health**")
    inputs['Age'] = st.selectbox(
        "Age Group",
        options=list(range(1,14)),
        format_func=lambda x: {
            1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39",
            5:"40-44", 6:"45-49", 7:"50-54", 8:"55-59",
            9:"60-64", 10:"65-69", 11:"70-74",
            12:"75-79", 13:"80+" 
        }[x]
    )
    inputs['GenHlth'] = st.selectbox(
        "General Health",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1:"Excellent", 2:"Very Good",
            3:"Good", 4:"Fair", 5:"Poor"
            }[x]
    )
    inputs['MentHlth'] = st.slider(
        "Bad Mental Health Days (past 30)", 0, 30, 0
    )
    inputs['PhysHlth'] = st.slider(
        "Bad Physical Health Days (past 30)", 0, 30, 0
    )
    inputs['Education'] = st.selectbox(
        "Education Level",
        options=list(range(1,7)),
        format_func= lambda x: {
            1:"<$10K",  2:"$10-15K", 3:"$15-20K",
            4:"$20-25K", 5:"$25-35K", 6:"$35-50K",
            7:"$50-75K", 8:"$75K+"
            }[x]
    )
    inputs['Income']    = st.selectbox(
        "Income Level",
        options=list(range(1, 9)),
        format_func=lambda x: {
            1:"<$10K",   2:"$10-15K", 3:"$15-20K",
            4:"$20-25K", 5:"$25-35K", 6:"$35-50K",
            7:"$50-75K", 8:"$75K+"
        }[x]
    )
    
# --- Prediction button ------------------------------------
st.divider()
if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    try:
        with st.spinner("Analyzing health data...."):
            #inputs dist is the payload
            response = requests.post(API_URL, json=inputs)
            if response.status_code != 200:
                st.error(f"API Error {response.status_code}: {response.text}")
                st.stop()

            result = response.json()

        if 'risk_level' not in result:
            st.error(f"Unexpected response: {result}")
            st.stop()
            
        st.divider()
        col_result, col_detail = st.columns([1,2])
        
        with col_result:
            risk = result['risk_level']
            prob = result['probability']
            
            if risk == 'High':
                st.error(f"⚠️ {result['label']}")
            elif risk == 'Moderate':
                st.warning(f"⚡ {result['label']}")
            else:
                st.success(f"✅ {result['label']}")
                
            st.metric("Diabetes Risk Probability", f"{prob}%")
            st.metric("Risk Level", risk)
            
        with col_detail:
            st.markdown("**Risk Interpretation**")
            if risk == 'High':
                st.error("""
                **High Risk (≥60% probability)**
                Patient shows multiple diabetes risk factors.
                Recommend: Clinical glucose testing,
                HbA1c screening, lifestyle intervention.
                """)
            elif risk == 'Moderate':
                st.warning("""
                **Moderate Risk (30-60% probability)**
                Some risk factors present.
                Recommend: Annual screening,
                dietary consultation.
                """)
            else:
                st.success("""
                **Low Risk (<30% probability)**
                Patient shows few diabetes risk factors.
                Recommend: Continue healthy lifestyle,
                routine annual checkup.
                """)

            with st.expander("🔧 API Response (JSON)"):
                st.json(result)

    except requests.exceptions.ConnectionError:
        st.error("""
        ❌ Cannot connect to FastAPI.
        Run: `uvicorn src.api.fast_api:app --port 8000`
        """)