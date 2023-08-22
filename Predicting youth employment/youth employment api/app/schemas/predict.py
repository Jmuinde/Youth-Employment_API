from typing import Any, List, Optional

from pydantic import BaseModel
from regression_model.processing.validation import YOUTHDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleYOUTHDataInputs(BaseModel):
    inputs: List[YOUTHDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {


                        "ref_area": "GHA",
                        "sex": "SEX_M",
                        "age_bracket": "15-24",
                        "population": 3211620.0,
                        "total_inactive_population": 891931,
                        "total_unemployed_population": 120628,
                        "total_employed_population": 2199060,
                    
                    }
                ]
            }
        }
