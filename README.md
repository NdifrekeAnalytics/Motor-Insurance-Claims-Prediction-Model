### Motor Insurance Ultimate Claims Prediction Model


![Guardian Logo](assets/guardian-logo.png) <!-- Optional: add if you have a logo in assets/ -->


Predicting **ultimate claim amounts** for motor insurance using early FNOL (First Notification of Loss) indicators, with **SHAP** and **LIME** explainability.



## 🎯 Project Objective



Develop a machine learning model and web application that:

- Predicts the **ultimate claim cost** shortly after an accident is reported (FNOL stage)

- Provides **interpretable explanations** (using SHAP and LIME) to help underwriters, claims adjusters, and fraud teams understand key risk drivers

- Enables fast, data-driven decisions to improve reserving accuracy, detect potential fraud, and optimize claims handling



**Business value**:

- Better financial reserving

- Early identification of high-severity / fraudulent claims

- Improved customer experience through faster settlements for low-risk cases



## 🛠️ Tech Stack



- **Backend**: FastAPI (Python)  

- **Frontend**: Streamlit  

- **Model**: Gradient Boosting Regressor (scikit-learn)  

- **Explainability**: SHAP (TreeExplainer), LIME (tabular)  

- **Containerization & Orchestration**: Docker + Docker Compose  

- **Deployment options tested**: Render, Railway.app, Oracle Cloud Always Free VM



## 📁 Project Structure _(Core Deployment Files)_



### **Motor Insurance FNOL Claims Prediction/**

│

├── app/                       # FastAPI app

│   ├── main.py                      	# FastAPI application with /predict endpoint

│   ├── schemas.py                   	# Pydantic models

│   ├── explainability.py            	# SHAP + LIME logic

│   └── artifacts/                   	# preprocessor.joblib, model.joblib, shap_background.pkl

├── streamlit_app.py		# Streamlit frontend UI

├── assets/			# Images, logos, static files

│

├── Dockerfile.api		      	# FastAPI backend image

├── Dockerfile.streamlit	      	# Streamlit frontend image

├── docker-compose.yml			# Orchestrates API + Streamlit

│

├── requirements.api.txt

├── requirements.streamlit.txt

├── wheels/

│

├── .dockerignore

└── README.md

### 📊 **Model Performance & Results**
* **Model:** Gradient Boosting Regressor
* **Target:** Ultimate Claim Amount (log-transformed during training, exponentiated for final output)
* **Key metrics** _(on hold-out/test set – update with your actual numbers):_
&nbsp;	- **MAE:** 1,315.26 USD
&nbsp;	- **RMSE:** 3534.04 USD
&nbsp;	- **R²:** 0.9913
&nbsp;	- **MAPE:** 11.37%

* **Explainability Highlights:**
&nbsp;	- SHAP Explainabilty: Primary for global & local explanations. Top global drivers using SHAP summary for features like Estimated Claim Amount, Severity Score, Days to Settlement, Fraud Flag, Litigation Flag, Credit Score Band

&nbsp;	- LIME: Local confirmation. LIME often highlights non-linear thresholds (e.g., Severity Score > 3, Claim Duration < 30 days)


**Example Prediction Output:**
* **Input:** Moderate severity, young driver, theft claim → Predicted ~NGN 1,961,300
* **SHAP** shows strong negative impact from low severity + short settlement time
