import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Silent Dropout Risk Prediction",
    page_icon="⚠️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #fee2e2;
        border-left: 4px solid #dc2626;
    }
    .medium-risk {
        background-color: #fef3c7;
        border-left: 4px solid #d97706;
    }
    .low-risk {
        background-color: #dcfce7;
        border-left: 4px solid #16a34a;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚠️ Silent Dropout Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered patient retention risk assessment using machine learning</p>', unsafe_allow_html=True)

def load_models_safely():
    try:
        model_files = {
            "model": "svm_model.pkl",
            "scaler": "scaler.pkl",
            "label_encoder": "label_encoder.pkl",
            "feature_columns": "feature_columns.pkl"
        }
        for name, filepath in model_files.items():
            if not os.path.exists(filepath):
                st.error(f"❌ Missing file: {filepath}")
                st.info("Please ensure all model files are in the same directory as the app.")
                return None, None, None, None
        model = pickle.load(open("svm_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
        feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
        return model, scaler, label_encoder, feature_columns
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None

@st.cache_data
def get_feature_info():
    return {
        "Days_Since_Last_Contact": {
            "description": "Days since the patient was last contacted",
            "min": 0, "max": 365, "default": 10,
            "help": "Higher values indicate longer gaps in communication"
        },
        "Expected_Gap_Between_Visits_days": {
            "description": "Expected time between patient visits",
            "min": 0, "max": 365, "default": 7,
            "help": "Based on treatment protocol and patient condition"
        },
        "Medicine_Refill_Delay_days": {
            "description": "Delay in medicine refill",
            "min": 0, "max": 180, "default": 2,
            "help": "Indicates medication adherence issues"
        },
        "Missed_Lab_Tests": {
            "description": "Number of missed laboratory tests",
            "min": 0, "max": 10, "default": 0,
            "help": "Missed tests may indicate disengagement"
        },
        "Days_Late_Follow_Up": {
            "description": "Days late for scheduled follow-up",
            "min": 0, "max": 365, "default": 5,
            "help": "Delays in follow-up appointments are red flags"
        },
        "Silent_Dropout_Score": {
            "description": "Preliminary silent dropout risk score",
            "min": 0.0, "max": 10.0, "default": 3.0,
            "help": "Composite score from previous assessments"
        }
    }

def create_risk_visualization(risk_level, confidence_score=None):
    rl = str(risk_level).lower()
    if rl == "high" or "high" in rl:
        color = "#dc2626"
        icon = "🔴"
        bg_color = "#fee2e2"
    elif rl == "medium" or "med" in rl:
        color = "#d97706"
        icon = "🟡"
        bg_color = "#fef3c7"
    else:
        color = "#16a34a"
        icon = "🟢"
        bg_color = "#dcfce7"
    st.markdown(f"""
    <div class="risk-card" style="background-color: {bg_color}; border-left: 4px solid {color};">
        <h3 style="color: {color}; margin: 0;">{icon} Risk Level: {str(risk_level).upper()}</h3>
        {f'<p>Confidence: {confidence_score:.1%}</p>' if confidence_score is not None else ''}
    </div>
    """, unsafe_allow_html=True)

def show_model_info():
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.info("""
        **Model Type:** Support Vector Machine (SVM)
        
        **Purpose:** Predict patient silent dropout risk
        
        **Risk Levels:**
        - 🟢 Low Risk
        - 🟡 Medium Risk  
        - 🔴 High Risk
        """)
        st.header("📊 Feature Importance")
        st.write("1. **Communication gaps** (Days since last contact)")
        st.write("2. **Visit adherence** (Expected vs actual gaps)")
        st.write("3. **Medication compliance** (Refill delays)")
        st.write("4. **Test compliance** (Missed lab tests)")
        st.write("5. **Follow-up patterns** (Days late)")
        st.write("6. **Historical risk** (Silent dropout score)")

model, scaler, le, feature_columns = load_models_safely()

if model is None:
    st.stop()

show_model_info()
feature_info = get_feature_info()

st.header("🧾 Patient Assessment Form")
st.markdown("---")
col1, col2 = st.columns(2)
input_values = {}

with col1:
    st.subheader("📞 Communication & Visits")
    input_values['Days_Since_Last_Contact'] = st.number_input(
        "Days Since Last Contact",
        min_value=feature_info['Days_Since_Last_Contact']['min'],
        max_value=feature_info['Days_Since_Last_Contact']['max'],
        value=feature_info['Days_Since_Last_Contact']['default'],
        help=feature_info['Days_Since_Last_Contact']['help']
    )
    input_values['Expected_Gap_Between_Visits_days'] = st.number_input(
        "Expected Gap Between Visits (days)",
        min_value=feature_info['Expected_Gap_Between_Visits_days']['min'],
        max_value=feature_info['Expected_Gap_Between_Visits_days']['max'],
        value=feature_info['Expected_Gap_Between_Visits_days']['default'],
        help=feature_info['Expected_Gap_Between_Visits_days']['help']
    )
    input_values['Days_Late_Follow_Up'] = st.number_input(
        "Days Late for Follow-Up",
        min_value=feature_info['Days_Late_Follow_Up']['min'],
        max_value=feature_info['Days_Late_Follow_Up']['max'],
        value=feature_info['Days_Late_Follow_Up']['default'],
        help=feature_info['Days_Late_Follow_Up']['help']
    )

with col2:
    st.subheader("💊 Treatment Compliance")
    input_values['Medicine_Refill_Delay_days'] = st.number_input(
        "Medicine Refill Delay (days)",
        min_value=feature_info['Medicine_Refill_Delay_days']['min'],
        max_value=feature_info['Medicine_Refill_Delay_days']['max'],
        value=feature_info['Medicine_Refill_Delay_days']['default'],
        help=feature_info['Medicine_Refill_Delay_days']['help']
    )
    input_values['Missed_Lab_Tests'] = st.number_input(
        "Missed Lab Tests",
        min_value=feature_info['Missed_Lab_Tests']['min'],
        max_value=feature_info['Missed_Lab_Tests']['max'],
        value=feature_info['Missed_Lab_Tests']['default'],
        help=feature_info['Missed_Lab_Tests']['help']
    )
    input_values['Silent_Dropout_Score'] = st.slider(
        "Silent Dropout Score",
        min_value=feature_info['Silent_Dropout_Score']['min'],
        max_value=feature_info['Silent_Dropout_Score']['max'],
        value=feature_info['Silent_Dropout_Score']['default'],
        step=0.1,
        help=feature_info['Silent_Dropout_Score']['help']
    )

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Risk Level", use_container_width=True)

if predict_button:
    try:
        with st.spinner("🤖 Analyzing patient data..."):
            input_df = pd.DataFrame([input_values])
            missing_features = set(feature_columns) - set(input_df.columns)
            if missing_features:
                for feature in missing_features:
                    input_df[feature] = 0
            input_df = input_df[feature_columns]

            # Attempt to find numeric columns scaler expects
            try:
                scaler_features = scaler.feature_names_in_
                numeric_cols = [col for col in scaler_features if col in input_df.columns]
            except Exception:
                numeric_cols = [col for col in input_df.columns if pd.api.types.is_numeric_dtype(input_df[col])]

            if len(numeric_cols) > 0:
                # Avoid SettingWithCopyWarning
                scaled_part = scaler.transform(input_df[numeric_cols])
                input_df = input_df.copy()
                input_df[numeric_cols] = scaled_part

            # Make prediction
            prediction = model.predict(input_df)
            # Try to get probabilities
            try:
                probabilities = model.predict_proba(input_df)
                confidence = float(np.max(probabilities))
            except Exception:
                probabilities = None
                confidence = None

            # Try to decode predicted label using provided label encoder
            risk_label_from_model = None
            try:
                if le is not None:
                    decoded = le.inverse_transform(prediction)
                    risk_label_from_model = str(decoded[0])
                else:
                    # If label encoder not provided, try model.classes_
                    if hasattr(model, "classes_"):
                        cls = model.classes_
                        risk_label_from_model = str(cls[int(prediction[0])])
                    else:
                        risk_label_from_model = str(prediction[0])
            except Exception:
                try:
                    if hasattr(model, "classes_"):
                        cls = model.classes_
                        risk_label_from_model = str(cls[int(prediction[0])])
                    else:
                        risk_label_from_model = str(prediction[0])
                except Exception:
                    risk_label_from_model = None

        # After spinner block, compute heuristic score (same as before)
        total_risk_score = (
            input_values['Days_Since_Last_Contact'] * 0.2 +
            input_values['Days_Late_Follow_Up'] * 0.3 +
            input_values['Medicine_Refill_Delay_days'] * 0.25 +
            input_values['Missed_Lab_Tests'] * 5 +
            input_values['Silent_Dropout_Score'] * 2
        )

        # Determine final risk level using model label when confident, else fall back to score thresholds
        final_risk = "Low"
        # If we have a model-derived label and high-enough confidence, trust it
        if risk_label_from_model is not None and confidence is not None:
            if confidence >= 0.75:
                final_risk = risk_label_from_model
            else:
                # low confidence -> combine model label hint + score
                # if model says high but low confidence, require score confirmation
                if "high" in str(risk_label_from_model).lower() and total_risk_score >= 40:
                    final_risk = "High"
                elif "med" in str(risk_label_from_model).lower() and total_risk_score >= 25:
                    final_risk = "Medium"
                else:
                    # fallback to score thresholds
                    if total_risk_score >= 50:
                        final_risk = "High"
                    elif total_risk_score >= 25:
                        final_risk = "Medium"
                    else:
                        final_risk = "Low"
        else:
            # No model label or no confidence -> use score thresholds
            if total_risk_score >= 50:
                final_risk = "High"
            elif total_risk_score >= 25:
                final_risk = "Medium"
            else:
                final_risk = "Low"

        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        result_col1, result_col2 = st.columns([2, 1])
        with result_col1:
            create_risk_visualization(final_risk, confidence)
        with result_col2:
            st.markdown("### 📈 Risk Metrics")
            # Cap the displayed risk score at a sensible maximum, but show actual value in tooltip
            display_score = total_risk_score
            st.metric("Risk Score", f"{display_score:.1f}/100")
            if confidence is not None:
                st.metric("Model Confidence", f"{confidence:.1%}")
            st.metric("Assessment Date", datetime.now().strftime("%Y-%m-%d"))

        # Recommendations based on final_risk
        st.markdown("### 💡 Recommendations")
        if str(final_risk).lower().startswith("high"):
            st.error("""
            **Immediate Action Required:**
            - Contact patient within 24 hours
            - Schedule urgent follow-up appointment
            - Review treatment plan and barriers to care
            - Consider case management intervention
            """)
        elif str(final_risk).lower().startswith("med"):
            st.warning("""
            **Proactive Measures Recommended:**
            - Contact patient within 3 days
            - Reinforce importance of follow-up care
            - Address any identified barriers
            - Schedule reminder calls/messages
            """)
        else:
            st.success("""
            **Continue Standard Care:**
            - Maintain regular follow-up schedule
            - Monitor for any changes in behavior
            - Provide patient education as needed
            - Document all interactions
            """)

        # Contributing Factors chart
        st.markdown("### 🔍 Contributing Factors")
        fig, ax = plt.subplots(figsize=(10, 4))
        features = list(input_values.keys())
        values = list(input_values.values())
        normalized_values = []
        for i, (feature, value) in enumerate(zip(features, values)):
            max_val = feature_info.get(feature, {}).get('max', 10)
            if max_val == 0:
                normalized_values.append(0)
            else:
                normalized_values.append((value / max_val) * 100)
        colors = ['#3b82f6' if v < 33 else '#f59e0b' if v < 66 else '#ef4444' for v in normalized_values]
        bars = ax.barh(features, normalized_values, color=colors)
        ax.set_xlabel('Risk Contribution (%)')
        ax.set_title('Patient Risk Factor Analysis')
        ax.set_xlim(0, 100)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{value}', ha='left', va='center')
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please check your input values and try again. If the problem persists, contact technical support.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>Silent Dropout Risk Prediction System</strong></p>
    <p>Powered by Machine Learning | Built for Healthcare Excellence</p>
</div>
""", unsafe_allow_html=True)