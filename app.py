from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

model = joblib.load('model.pkl')
CLASSES = ['setosa', 'versicolor', 'virginica']

# Pydantic valida automáticamente que lleguen los 4 campos con tipos correctos
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get('/')
def home():
    return {
        'name': 'Iris ML API',
        'version': '1.0',
        'algorithm': 'RandomForestClassifier',
        'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'example': {
            'url': '/predict',
            'method': 'POST',
            'body': {
                'sepal_length': 5.1,
                'sepal_width': 3.5,
                'petal_length': 1.4,
                'petal_width': 0.2
            }
        }
    }

@app.post('/predict')
def predict(data: IrisInput):
    try:
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        prediction_index = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        return {
            'prediction': CLASSES[prediction_index],
            'prediction_index': prediction_index,
            'probabilities': {
                'setosa':     round(float(probabilities[0]), 4),
                'versicolor': round(float(probabilities[1]), 4),
                'virginica':  round(float(probabilities[2]), 4)
            },
            'confidence': round(float(probabilities.max()), 4),
            'status': 'success'
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/health')
def health():
    return {'status': 'healthy'}
