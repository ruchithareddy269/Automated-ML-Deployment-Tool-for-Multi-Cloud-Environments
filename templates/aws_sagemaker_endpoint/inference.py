from fastapi import FastAPI, HTTPException,Response
from pydantic import BaseModel
import joblib
import numpy as np
import os
from http import HTTPStatus

# Load the trained model
MODEL_DIR = '/opt/ml/model'
model_file = None

# Find the .pkl model file
for file in os.listdir(MODEL_DIR):
    if file.endswith('.pkl'):
        model_file = file
        break

if model_file is None:
    raise ValueError("No .pkl model file found in /opt/ml/model")

MODEL_PATH = os.path.join(MODEL_DIR, model_file)
model = joblib.load(MODEL_PATH)

# Create a FastAPI instance
app = FastAPI()

# Define the input data model
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Health check endpoint
@app.get("/ping")
def ping():
    return Response(content="ping", status_code=200)

# Prediction endpoint
@app.post("/invocations")
def predict(iris_input: IrisInput):
    input_data = np.array([[iris_input.sepal_length, iris_input.sepal_width,
                            iris_input.petal_length, iris_input.petal_width]])

    try:
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"predicted_class": predicted_class}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
