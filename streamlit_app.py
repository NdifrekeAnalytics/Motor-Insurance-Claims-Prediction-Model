# streamlit_app.py (corrected & Docker/Render compatible)

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import os

# --------------------------------------------------
# API Configuration (uses env var from docker-compose)
# --------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# For local testing without Docker you can temporarily override:
# API_URL = "http://localhost:8000/predict"

# In production (Render), docker-compose sets: http://claims-api:8000/predict
# When deployed separately on Render → set as environment variable in Render dashboard
# --------------------------------------------------

# App directory & assets
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
favicon_path = ASSETS_DIR / "favicon.png"
logo_path = ASSETS_DIR / "guadian_new_logo.png"

# --------------------------------------------------
# Page configuration (MUST BE FIRST Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="Motor Insurance Ultimate Claim Predictor",
    page_icon=str(favicon_path) if favicon_path.exists() else "🛡️",
    layout="centered"
)

# --------------------------------------------------
# Centered Logo
# --------------------------------------------------
if logo_path.exists():
    logo = Image.open(logo_path)
    st.markdown(
        """
        <div style="text-align: center; margin-top: 25px;">
        """,
        unsafe_allow_html=True
    )
    st.image(logo, width=320)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("Logo not found – running without branding.")

# --------------------------------------------------
# Centered Title & Caption
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-top: 10px;">
        <h1 style="margin-bottom: 6px;">
            🚗 Motor Insurance Ultimate Claim Predictor
        </h1>
        <p style="
            font-size: 18px;
            font-style: italic;
            color: #0E2841;
            margin-top: 0;
        ">
            Predicting ultimate claims using early claim indicators & risk drivers
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# Enums (mirroring FastAPI for consistency)
# --------------------------------------------------
from enum import Enum

class Gender(str, Enum):
    Male = "Male"
    Female = "Female"

class ClaimType(str, Enum):
    Theft = "Theft"
    Collision = "Collision"
    Weather = "Weather"
    Fire = "Fire"
    Vandalism = "Vandalism"
    Other = "Other"

class VehicleType(str, Enum):
    Coupe = "Coupe"
    Hatchback = "Hatchback"
    Motorcycle = "Motorcycle"
    Sedan = "Sedan"
    SUV = "SUV"
    Van = "Van"

class Region(str, Enum):
    Newcastle = "Newcastle"
    Liverpool = "Liverpool"
    Cardiff = "Cardiff"
    Edinburgh = "Edinburgh"
    Leeds = "Leeds"
    Glasgow = "Glasgow"
    Manchester = "Manchester"
    Birmingham = "Birmingham"
    London = "London"
    Bristol = "Bristol"

class ClaimStatus(str, Enum):
    Open = "Open"
    Settled = "Settled"

class CreditScoreBand(str, Enum):
    Poor = "Poor"
    Fair = "Fair"
    Good = "Good"
    Excellent = "Excellent"

class Occupation(str, Enum):
    Retired = "Retired"
    Unemployed = "Unemployed"
    Employed = "Employed"
    Self_Employed = "Self-Employed"
    Student = "Student"

# --------------------------------------------------
# UI Inputs
# --------------------------------------------------
st.subheader("Initial Claim Benchmark")
estimated_claim_amount = st.number_input(
    "Initial Claim Estimate ($)",
    min_value=0.0,
    value=2500.0,
    step=100.0,
    help="Initial reserve / early business estimate"
)

severity_score = st.slider(
    "Severity Score (0 = None, 5 = Fatal)",
    min_value=0,
    max_value=5,
    value=2
)

st.subheader("Driver & Vehicle Information")
age = st.number_input("Driver Age", min_value=16, max_value=100, value=35)
experience = st.number_input("Driving Experience (Years)", min_value=0, max_value=80, value=10)
vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=5)
annual_mileage = st.number_input("Annual Mileage", min_value=0, value=12000)

st.subheader("Claim Context")
status = st.radio("Claim Status", [e.value for e in ClaimStatus])
gender = st.radio("Gender", [e.value for e in Gender], horizontal=True)
credit_score = st.radio("Credit Score Band", [e.value for e in CreditScoreBand])
vehicle_type = st.selectbox("Vehicle Type", [v.value for v in VehicleType])
region = st.selectbox("Region", [r.value for r in Region])
occupation = st.selectbox("Occupation", [o.value for o in Occupation])
claim_type = st.selectbox("Claim Type", [c.value for c in ClaimType])

st.subheader("Third-Party Involvement")
num_pedestrians = st.number_input("Number of Pedestrians", min_value=0, value=0)
num_passengers = st.number_input("Number of Passengers", min_value=0, value=0)
num_drivers = st.number_input("Number of Other Drivers", min_value=0, value=0)

st.subheader("Risk Flags")
fraud_flag = st.checkbox("Fraud Flag")
litigation_flag = st.checkbox("Litigation Flag")

# --------------------------------------------------
# Build payload
# --------------------------------------------------
payload = {
    "Estimated_Claim_Amount": estimated_claim_amount,
    "Severity_Score": severity_score,
    "Age_of_Driver": age,
    "Driving_Experience_Years": experience,
    "Vehicle_Age": vehicle_age,
    "Annual_Mileage": annual_mileage,
    "Vehicle_Type": vehicle_type,
    "Gender": gender,
    "Claim_Type": claim_type,
    "Occupation": occupation,
    "Region": region,
    "Credit_Score_Band": credit_score,
    "Status": status,
    "Fraud_Flag": int(fraud_flag),
    "Litigation_Flag": int(litigation_flag),
    "Num_Pedestrians": num_pedestrians,
    "Num_Passengers": num_passengers,
    "Num_Drivers": num_drivers
}

# --------------------------------------------------
# Prediction button
# --------------------------------------------------
if st.button("🔮 Predict Ultimate Claim Amount"):
    with st.spinner("Scoring Claim..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()  # Raise exception for bad status codes

            result = response.json()

            st.success(f"💰 Predicted Ultimate Claim Amount: **${result['prediction']:,.2f}**")

            if "top_drivers" in result and result["top_drivers"]:
                st.subheader("🔍 Explainability – Key Claim Drivers")

                def highlight_impact(val):
                    color = '#90EE90' if val > 0 else '#FFB6C1'
                    return f'background-color: {color}'

                # ── SHAP ────────────────────────────────────────────────────────────────
                if "shap" in result["top_drivers"] and result["top_drivers"]["shap"]:
                    st.markdown("### SHAP Top Drivers")
                    shap_df = pd.DataFrame(result["top_drivers"]["shap"])

                    # Ensure feature is string
                    shap_df["feature"] = shap_df["feature"].astype(str)

                    # Rename if needed
                    if "shap_value" in shap_df.columns:
                        shap_df = shap_df.rename(columns={"shap_value": "impact"})

                    # Sort by absolute impact
                    shap_df["abs_impact"] = shap_df["impact"].abs()
                    shap_df = shap_df.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")

                    st.dataframe(
                        shap_df.style
                        .format(precision=4)
                        .map(highlight_impact, subset=["impact"]),
                        use_container_width=True
                    )

                    # ── SHAP Waterfall Plot ─────────────────────────────────────────────
                    features = shap_df["feature"].tolist()
                    impacts = shap_df["impact"].tolist()
                    predicted_amount = result['prediction']

                    fig = go.Figure(go.Waterfall(
                        name="SHAP Contribution",
                        orientation="h",
                        measure=["relative"] * len(features) + ["total"],
                        y=features + ["Prediction f(x)"],
                        x=impacts + [predicted_amount],
                        text=[f"{imp:+,.2f}" for imp in impacts] + [f"${predicted_amount:,.2f}"],
                        textposition="outside",
                        connector={"line": {"color": "rgb(150, 150, 150)", "dash": "dot"}},
                        increasing={"marker": {"color": "#FF0051"}},
                        decreasing={"marker": {"color": "#0000FF"}},
                        totals={"marker": {"color": "rgba(0,0,0,0)"}},
                    ))

                    fig.update_layout(
                        title="SHAP Waterfall – How features drive the prediction",
                        showlegend=False,
                        height=700,
                        margin=dict(l=220, r=100, t=80, b=60),
                        xaxis_title="Impact on Predicted Claim Amount ($)",
                        yaxis_title="Features",
                        xaxis=dict(
                            zeroline=True,
                            zerolinecolor="black",
                            zerolinewidth=2,
                            range=[min(impacts + [0]) * 1.2, max(impacts + [0]) * 1.2],
                            gridcolor="lightgray",
                            gridwidth=1
                        ),
                        yaxis=dict(autorange="reversed"),
                        font=dict(size=12),
                        waterfallgap=0.35,
                    )

                    fig.update_traces(hovertemplate="%{y}<br>Impact: %{x:,.2f} $<extra></extra>")

                    st.plotly_chart(fig, use_container_width=True)

                # ── LIME ────────────────────────────────────────────────────────────────
                if "lime" in result["top_drivers"] and result["top_drivers"]["lime"]:
                    st.markdown("### LIME Top Drivers")
                    lime_df = pd.DataFrame(result["top_drivers"]["lime"])

                    lime_df["abs_impact"] = lime_df["impact"].abs()
                    lime_df = lime_df.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")

                    st.dataframe(
                        lime_df.style
                        .format(precision=4)
                        .map(highlight_impact, subset=["impact"]),
                        use_container_width=True
                    )

                    st.bar_chart(
                        lime_df.set_index("feature")["impact"],
                        use_container_width=True
                    )

            else:
                st.info("No explainability data returned from the API.")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to prediction service: {str(e)}")
        except ValueError as e:
            st.error(f"Invalid response from API: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.exception(e)  # Shows full traceback in UI (useful for debugging)