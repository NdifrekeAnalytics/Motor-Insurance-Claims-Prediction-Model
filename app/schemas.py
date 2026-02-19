from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum
from typing import Literal


# 🔹 ENUMS (Swagger Dropdowns)

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



# 🔹 INPUT SCHEMA (ClaimInput)

class ClaimInput(BaseModel):
    # ----------------------------------
    # Initial business benchmark inputs
    # ----------------------------------
    Estimated_Claim_Amount: float = Field(
        ...,
        ge=0,
        description="Initial claim estimate / early reserve"
    )

    Severity_Score: int = Field(
        ...,
        ge=0,
        le=5,
        description="Overall injury severity score (0=None, 5=Fatal)"
    )

    # ----------------------------------
    # Driver & vehicle information
    # ----------------------------------
    Age_of_Driver: int = Field(..., ge=16, le=100)
    Driving_Experience_Years: int = Field(..., ge=0, le=80)
    Vehicle_Age: int = Field(..., ge=0, le=30)
    Annual_Mileage: float = Field(..., ge=0)

    Vehicle_Type: VehicleType
    Gender: Gender

    # ----------------------------------
    # Claim context
    # ----------------------------------
    Credit_Score_Band: CreditScoreBand
    Claim_Type: ClaimType
    Occupation: Occupation
    Region: Region
    Status: ClaimStatus

    # ----------------------------------
    # Binary categorical drivers
    # ----------------------------------
    Fraud_Flag: int = Field(..., ge=0, le=1)
    Litigation_Flag: int = Field(..., ge=0, le=1)

    # ----------------------------------
    # Third-party involvement
    # ----------------------------------
    Num_Pedestrians: int = Field(0, ge=0)
    Num_Passengers: int = Field(0, ge=0)
    Num_Drivers: int = Field(0, ge=0)

    class Config:
        schema_extra = {
            "example": {
                "Estimated_Claim_Amount": 25000,
                "Severity_Score": 3,
                "Age_of_Driver": 42,
                "Driving_Experience_Years": 18,
                "Vehicle_Age": 6,
                "Annual_Mileage": 12000,
                "Vehicle_Type": "Sedan",
                "Gender": "Male",
                "Credit_Score_Band": "Good",
                "Claim_Type": "Collision",
                "Occupation": "Employed",
                "Region": "London",
                "Status": "Open",
                "Fraud_Flag": 0,
                "Litigation_Flag": 0,
                "Num_Pedestrians": 1,
                "Num_Passengers": 0,
                "Num_Drivers": 1
            }
        }


    # ----------------------------------
    # Validators (business sanity checks)
    # ----------------------------------
    @validator("Driving_Experience_Years")
    def experience_cannot_exceed_age(cls, v, values):
        age = values.get("Age_of_Driver")
        if age is not None and v > age:
            raise ValueError("Driving experience cannot exceed driver age")
        return v

    class Config:
        extra = "forbid"


# 🔹 RESPONSE SCHEMA (PredictionResponse)
from typing import Dict, List, Optional

class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted Ultimate Claim Amount")
    
    top_drivers: Optional[Dict[str, List[dict]]] = Field(
        default=None,
        description="Top SHAP and LIME drivers"
    )

from typing import Dict, List, Any

class ExplainabilityDrivers(BaseModel):
    shap: List[Dict[str, Any]]
    lime: List[Dict[str, Any]]




# 🔧 Bug Fix Validations

# 1. Prevents impossible cases (e.g. 30 years driving experience for a 20-year-old)
# 2. Protects your model from garbage inputs in production


# Swagger UX

# 1. Enums give dropdowns
# 2. Field descriptions explain business meaning
# 3. extra = "forbid" prevents silent feature drift