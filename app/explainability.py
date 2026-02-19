# app/explainability.py

import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from app.utils import strip_column_prefixes
from lime.lime_tabular import LimeTabularExplainer


from app.utils import strip_column_prefixes


# 🔹 SHAP (model-safe)

def explain_with_shap(
    model,
    X_processed: pd.DataFrame,
    background: pd.DataFrame,
    top_k: int = 10
):
    print("===== SHAP EXPLAINER STARTED =====")
    print("Input X_processed shape:", X_processed.shape)
    print("Input X_processed columns:", X_processed.columns.tolist())
    
    explainer = shap.Explainer(model, background)
    shap_values = explainer(X_processed)

    X_explain = strip_column_prefixes(X_processed)

    # Temporary fix: map integer strings to real names (update this list!)
    feature_map = {
        '0': 'Claim_Complexity',
        '1': 'Credit_Score_Band',
        '2': 'Claim_Type_Fire',
        '3': 'Claim_Type_Other',
        '4': 'Claim_Type_Theft',
        '5': 'Claim_Type_Vandalism',
        '6': 'Claim_Type_Weather',
        '7': 'Occupation_Retired',
        '8': 'Occupation_Self-Employed',
        '9': 'Occupation_Student',
        '10': 'Occupation_Unemployed',
        '11': 'Bristol',
        '12': 'Cardiff',
        '13': 'Edinburgh',
        '14': 'Glasgow',
        '15': 'Leeds',
        '16': 'Liverpool',
        '17': 'London',
        '18': 'Manchester',
        '19': 'Newcastle',
        '20': 'Estimated_Claim_Amount',
        '21': 'Age_of_Driver',
        '22': 'Driving_Experience_Years',
        '23': 'Vehicle_Age',
        '24': 'Annual_Mileage',
        '25': 'Vehicle_Risk',
        '26': 'Driver_Risk',
        '27': 'Days_To_FNOL',
        '28': 'Claim_Duration',
        '29': 'Days_To_Settlement',
        '30': 'Severity_Score',
        '31': 'Fraud_Flag',
        '32': 'Litigation_Flag',
        '33': 'Gender',
        '34': 'Status',
        '35': 'Num_of_Third_Parties',
        '36': 'Num_Pedestrians_3rd Party',
        '37': 'Num_Passengers_3rd Party',
        '38': 'Num_Drivers_3rd Party',
        '39': '',
    }

    X_explain.columns = [feature_map.get(str(col), str(col)) for col in X_explain.columns]
    
    # Debug: print column names before creating df
    print("SHAP feature names:", X_explain.columns.tolist())
    
    shap_row = shap_values.values[0]

    df = (
        pd.DataFrame({
            "feature": X_explain.columns,
            "value": X_explain.iloc[0].values,
            "shap_value": shap_row
        })
        .assign(abs_shap=lambda d: d.shap_value.abs())
        .sort_values("abs_shap", ascending=False)
        .head(top_k)
        .drop(columns="abs_shap")
    )

    return df.to_dict(orient="records")



# 🔹 LIME (model-safe)

def explain_with_lime(
    model,
    X_train_processed: pd.DataFrame,
    X_processed: pd.DataFrame,
    top_k: int = 10
):
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_processed.values,
        feature_names=strip_column_prefixes(X_train_processed).columns.tolist(),
        mode="regression",
        discretize_continuous=True
    )

    exp = lime_explainer.explain_instance(
        X_processed.iloc[0].values,
        model.predict,
        num_features=top_k
    )

    return [
        {"feature": f, "impact": v}
        for f, v in exp.as_list()
    ]




# -------------------------
# SHAP
# -------------------------
# def build_shap_explainer(model, X_background):
#    return shap.Explainer(model, X_background)


# def shap_explain_single(
#    explainer,
#    preprocessor,
#    model,
#    raw_df,
#    top_k=10
#):
#    X_proc = preprocessor.transform(raw_df)
#    shap_vals = explainer(X_proc)

#    X_clean = strip_column_prefixes(X_proc)

#    df = (
#        pd.DataFrame({
#            "feature": X_clean.columns,
#            "value": X_clean.iloc[0],
#            "shap_value": shap_vals.values[0]
#        })
#        .assign(abs_shap=lambda d: d["shap_value"].abs())
#        .sort_values("abs_shap", ascending=False)
#        .head(top_k)
#        .drop(columns="abs_shap")
#    )

#    return df.to_dict(orient="records")


# -------------------------
# LIME
# -------------------------
# def build_lime_explainer(X_train_processed, feature_names):
#    return LimeTabularExplainer(
#        training_data=X_train_processed.values,
#        feature_names=feature_names,
#        mode="regression"
#    )


# def lime_explain_single(
#    lime_explainer,
#    preprocessor,
#    model,
#    raw_df,
#    num_features=10
#):
#    X_proc = preprocessor.transform(raw_df)

#    exp = lime_explainer.explain_instance(
#        X_proc.iloc[0].values,
#        model.predict,
#        num_features=num_features
#    )

#    return exp.as_list()
