from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# 1. Initialize FastAPI app
app = FastAPI(title="Digit Recognition API")

# 2. Add CORS middleware
# PRECAUTION: This allows your frontend (which will run on a different port/domain) to talk to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://digit-recognizer-web-blond.vercel.app/"],  # In production, change "*" to your actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the model globally so it only loads once when the server starts
# PRECAUTION: Never load the model inside the prediction function, or it will crash your server under load.
try:
    model = tf.keras.models.load_model('digit_recognition_model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 4. Define the expected data structure using Pydantic
class ImageData(BaseModel):
    image_base64: str

def preprocess_image(base64_str: str) -> np.ndarray:
    """Decodes the base64 string and preps it for the CNN."""
    # Strip the HTML canvas metadata prefix if it exists
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
        
    # Decode base64 to image
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale (1 channel)
    image = image.convert('L')
    
    # Resize to 28x28 (Standard MNIST size)
    image = image.resize((28, 28))
    img_array = np.array(image)
    
    # PRECAUTION: Invert colors. 
    # Web canvases usually have a white background with black drawing. 
    # MNIST models are trained on black backgrounds with white digits.
    img_array = 255 - img_array
    
    # Normalize pixel values to 0-1
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch and channel dimensions: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

# 5. Define the prediction endpoint
@app.post("/predict")
async def predict_digit(data: ImageData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # Preprocess
        processed_image = preprocess_image(data.image_base64)
        
        # Predict
        predictions = model.predict(processed_image)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return {
            "digit": predicted_digit,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))