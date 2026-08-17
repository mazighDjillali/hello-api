from fastapi import FastAPI
import joblib
import pandas as pd
app = FastAPI()

 # Load the model and features
saved = joblib.load("model.joblib")
model = saved["model"]
features = saved["features"]

@app.get("/")
def hello():
    return {"message": "Hello, World!"}

@app.get("/predict")
def predict():

    # Load the data to predict
    df_predict = pd.read_csv("data/raw/predict.csv")

    # Ensure the input data has the same features as the model expects
    X_predict = df_predict[features]

    # Make predictions
    predictions = model.predict(X_predict)

    # Return predictions as a list
    return {"predictions": predictions.tolist()}