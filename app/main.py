from fastapi import FastAPI, HTTPException
from app.schemas import ClaimInput, PredictionResponse
from app.explainability import explain_with_shap, explain_with_lime
import joblib
import pandas as pd
import numpy as np
import traceback
import logging
import sklearn

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Motor Insurance Ultimate Claims API",
    description="Predicts Ultimate Claim Amount with SHAP & LIME explainability",
    version="1.0.0",
)

# --------------------------------------------------
# Global artifacts (loaded at startup)
# --------------------------------------------------
preprocessor = None
model = None
X_train_shap = None

# --------------------------------------------------
# Runtime version guard (RELAXED – major.minor only)
# --------------------------------------------------
# IMPORTANT: These are now major.minor prefixes.
# Update them only when you intentionally upgrade to a new major/minor series.
EXPECTED = {
    "numpy": "2.",       # ← this line changed
    "sklearn": "1.",
    "joblib": "1.",
}

def verify_runtime_environment():
    import numpy as np
    import sklearn
    import joblib

    def check_version(lib, name, expected_prefix):
        actual = lib.__version__
        if not actual.startswith(expected_prefix):
            raise AssertionError(
                f"{name} major.minor version mismatch: "
                f"expected {expected_prefix}.x, got {actual}"
            )
        logger.info(f"{name} version OK: {actual} (expected prefix: {expected_prefix}.x)")

    logger.info("Verifying runtime environment (relaxed major.minor check)...")
    check_version(np, "NumPy", EXPECTED["numpy"])
    check_version(sklearn, "scikit-learn", EXPECTED["sklearn"])
    check_version(joblib, "joblib", EXPECTED["joblib"])
    logger.info("All runtime versions verified successfully (major.minor match).")

# --------------------------------------------------
# Startup event (SAFE loading)
# --------------------------------------------------
@app.on_event("startup")
def load_artifacts():
    global preprocessor, model, X_train_shap
    logger.info("Loading model artifacts...")
    try:
        # Updated paths and filenames to match re-saved artifacts in Docker
        preprocessor = joblib.load("app/artifacts/preprocessor.joblib")
        model = joblib.load("app/artifacts/model.joblib")
        X_train_shap = joblib.load("app/artifacts/shap_background.pkl")
        logger.info("Artifacts loaded successfully.")
    except Exception as e:
        logger.error("Failed to load one or more model artifacts")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Artifact loading failed: {str(e)}")

    # Run version check AFTER artifacts are loaded (but before endpoint is ready)
    verify_runtime_environment()

# --------------------------------------------------
# /predict Endpoint
# --------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Ultimate Claim Amount",
    description="Returns prediction with SHAP & LIME explanations",
)
def predict(payload: ClaimInput):
    try:
        # 1. Input → DataFrame
        raw_df = pd.DataFrame([payload.dict()])
        # Ensure required date columns exist
        for col in ["Accident_Date", "FNOL_Date", "Settlement_Date"]:
            if col not in raw_df:
                raw_df[col] = pd.Timestamp.now()
        # 2. Preprocess
        X_processed = preprocessor.transform(raw_df)
        # 3. Predict (log → natural scale)
        prediction_log = model.predict(X_processed)[0]
        prediction = float(np.expm1(prediction_log))
        # 4. Explainability
        shap_drivers = explain_with_shap(
            model=model,
            X_processed=X_processed,
            background=X_train_shap,
            top_k=10,
        )
        lime_drivers = explain_with_lime(
            model=model,
            X_train_processed=X_train_shap,
            X_processed=X_processed,
            top_k=10,
        )
        return {
            "prediction": prediction,
            "top_drivers": {
                "shap": shap_drivers,
                "lime": lime_drivers,
            },
        }
    except AssertionError as ae:
        logger.error("Runtime environment validation failed")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Environment mismatch: {str(ae)}")
    except Exception:
        logger.error("Prediction failed")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")