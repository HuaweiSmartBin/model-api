from typing import Optional

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from ultralytics import YOLO

app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Initialize the model
        model = YOLO("./models/best.pt")
        labels = model.names

        # Read image file
        image_data = await image.read()

        # Convert the bytes to numpy array
        image_data = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Use the model to predict on the image
        results = model.predict(img)

        # Get the class of the prediction
        classified_class = results[0].probs.top1

        # Get the confidence of the prediction
        confidence = results[0].probs.top1conf.tolist()

        return {"prediction": labels[classified_class], "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))