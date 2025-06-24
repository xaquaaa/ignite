from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load trained ML model
model = joblib.load("model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, change this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define request body
class SurveyResponse(BaseModel):
    responses: list[int]

@app.post("/predict")
async def predict(response: SurveyResponse):
    try:
        # Convert response to numpy array
        input_data = np.array(response.responses).reshape(1, -1)
        
        # Predict learner type
        prediction = model.predict(input_data)[0]
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server (use `uvicorn filename:app --reload` to run)
