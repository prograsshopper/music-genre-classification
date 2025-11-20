import pickle
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict


MODEL_PATH = "model.bin"

app = FastAPI(title="Music Genre Classification API")

# Load model on startup
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']


class MusicFeatures(BaseModel):
    """Input features for music genre prediction"""

    features: Dict[str, float] = Field(
        ...,

        description="Dictionary of audio features extracted from music file",
        example={
            "length": 66149,
            "chroma_stft_mean": 0.335,
            "chroma_stft_var": 0.091,
            "rms_mean": 0.130,
            "rms_var": 0.003,
            "spectral_centroid_mean": 1773.065,
            "spectral_centroid_var": 167541.63,
            "spectral_bandwidth_mean": 1972.74,
            "spectral_bandwidth_var": 117335.77,
            "rolloff_mean": 3741.98,
            "rolloff_var": 389220.11,
            "zero_crossing_rate_mean": 0.081,
            "zero_crossing_rate_var": 0.005,
            "harmony_mean": 0.195,
            "harmony_var": 0.089,
            "perceptr_mean": -0.065,
            "perceptr_var": 0.001,
            "tempo": 149.976,
            "mfcc1_mean": -193.81,
            "mfcc1_var": 2283.52,
            "mfcc2_mean": 109.95,
            "mfcc2_var": 1330.76,
            "mfcc3_mean": 5.52,
            "mfcc3_var": 938.69,
            "mfcc4_mean": 26.88,
            "mfcc4_var": 607.06,
            "mfcc5_mean": -1.18,
            "mfcc5_var": 555.58,
            "mfcc6_mean": 11.80,
            "mfcc6_var": 421.83,
            "mfcc7_mean": -7.43,
            "mfcc7_var": 407.48,
            "mfcc8_mean": 4.27,
            "mfcc8_var": 337.18,
            "mfcc9_mean": -5.21,
            "mfcc9_var": 296.44,
            "mfcc10_mean": 2.63,
            "mfcc10_var": 295.31,
            "mfcc11_mean": -4.89,
            "mfcc11_var": 228.33,
            "mfcc12_mean": 0.40,
            "mfcc12_var": 234.77,
            "mfcc13_mean": -3.52,
            "mfcc13_var": 187.88,
            "mfcc14_mean": -0.92,
            "mfcc14_var": 172.40,
            "mfcc15_mean": -0.53,
            "mfcc15_var": 144.42,
            "mfcc16_mean": -2.14,
            "mfcc16_var": 39.69,
            "mfcc17_mean": -3.24,
            "mfcc17_var": 36.49,
            "mfcc18_mean": 0.72,
            "mfcc18_var": 38.10,
            "mfcc19_mean": -5.05,
            "mfcc19_var": 33.62,
            "mfcc20_mean": -0.24,
            "mfcc20_var": 43.77
        }
    )


class PredictionResponse(BaseModel):
    """Prediction response"""
    genre: str
    confidence: float
    probabilities: Dict[str, float]


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Music Genre Classification API",
        "model_metrics": model_data['metrics'],
        "available_genres": list(label_encoder.classes_)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: MusicFeatures):
    """
    Predict music genre from audio features
    Returns the predicted genre, confidence score, and probabilities for all genres
    """

    try:
        df = pd.DataFrame([input_data.features])
        # Ensure all required features are present
        missing_features = set(feature_names) - set(df.columns)

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing_features}"
            )

        # Reorder columns to match training data
        df = df[feature_names]

        # Handle missing values (same as training)
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)

        # Predict
        dmatrix = xgb.DMatrix(df)
        y_pred_proba = model.predict(dmatrix)

        # Get predicted class and confidence
        predicted_class = y_pred_proba.argmax()
        confidence = float(y_pred_proba[0][predicted_class])
        genre = label_encoder.inverse_transform([predicted_class])[0]

        # Get probabilities for all genres
        probabilities = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(y_pred_proba[0])
        }

        return PredictionResponse(
            genre=genre,
            confidence=confidence,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)