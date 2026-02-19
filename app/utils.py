# app/utils.py

import pandas as pd

def insurance_risk_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internalized Risk Engineering for Vehicle and Driver profiles.
    This function MUST live in a real module for pickle safety.
    """
    df = df.copy()

    # 🔒 Ensure inference-safe defaults
    # -----------------------------
    if "Claim_Complexity" not in df.columns:
        df["Claim_Complexity"] = "Medium"   # neutral default

    if "Num_of_Third_Parties" not in df.columns:
        df["Num_of_Third_Parties"] = (
            df.get("Num_Pedestrians", 0)
            + df.get("Num_Passengers", 0)
            + df.get("Num_Drivers", 0)
        )


    # --- Vehicle Risk --- (# Custom function for Vehicle_Risk)
    vehicle_type_risk = {
        'Motorcycle': 6, 'Coupe': 5, 'Sedan': 4,
        'Hatchback': 3, 'Van': 2, 'SUV': 1
    }
    df['Vehicle_Type_Risk'] = df['Vehicle_Type'].map(vehicle_type_risk).fillna(3)
    df['Vehicle_Age_Risk'] = df['Vehicle_Age'].clip(upper=20) / 20 * 5
    df['Vehicle_Risk'] = ((df['Vehicle_Type_Risk'] + df['Vehicle_Age_Risk']) / 2).round(1)

    # --- Driver Risk --- (# Custom function for Driver_Risk)
    # We use manual binning logic to avoid pd.cut issues in production
    def get_age_risk(age):
        if age < 20: return 6
        elif age < 25: return 5
        elif age < 35: return 4
        elif age < 45: return 3
        elif age < 55: return 2
        elif age < 65: return 1
        elif age < 75: return 2
        else: return 3

    def get_exp_risk(exp):
        if exp < 2: return 5
        elif exp < 10: return 4
        elif exp < 20: return 2
        else: return 1

    df['Driver_Age_Risk'] = df['Age_of_Driver'].apply(get_age_risk)
    df['Driver_Exp_Risk'] = df['Driving_Experience_Years'].apply(get_exp_risk)
    df['Driver_Risk'] = ((df['Driver_Exp_Risk'] * 0.6) + (df['Driver_Age_Risk'] * 0.4)).round(1)

    # --- Compute Key Date differences ---
    # Ensure columns are datetime before math
    for col in ['Accident_Date', 'FNOL_Date', 'Settlement_Date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['Days_To_FNOL'] = (df['FNOL_Date'] - df['Accident_Date']).dt.days
    df['Claim_Duration'] = (df['Settlement_Date'] - df['Accident_Date']).dt.days
    df['Days_To_Settlement'] = (df['Settlement_Date'] - df['FNOL_Date']).dt.days

    # 4. --- Handle Missing/Future Values ---
    # In production, if dates are missing, we fill with -1 (your model's indicator for "Not Applicable/Open")
    fill_cols = ['Days_To_FNOL', 'Claim_Duration', 'Days_To_Settlement']
    df[fill_cols] = df[fill_cols].fillna(-1)

    return df


def binary_mapping_func(df: pd.DataFrame) -> pd.DataFrame:
    binary_mapping = {
        'Fraud_Flag': {False: 0, True: 1},
        'Litigation_Flag': {False: 0, True: 1},
        'Status': {'open': 0, 'settled': 1},
        'Gender': {'Female': 0, 'Male': 1}
    }
    for col, mapping in binary_mapping.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)  # Fill any NaN with 0
    return df


def drop_columns_func(df: pd.DataFrame) -> pd.DataFrame:
    # 🔒 Convert numpy array → DataFrame if needed
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    drop_cols = [
        'Claim_ID', 'Policy_ID', 'Customer_ID',
        'Accident_Date', 'FNOL_Date', 'Settlement_Date',
        'Claim_Complexity', 'Severity_Band', 'Credit_Score_Band'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert bool → int
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def strip_column_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Critical: Force ALL column names to strings first (fixes int columns)
    df.columns = [str(col) for col in df.columns]
    
    # Now strip prefixes safely
    df.columns = [
        col.split("__", 1)[1] if "__" in col else col
        for col in df.columns
    ]
    
    return df
