from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi import Form


FORM_HTML = """
<form method="post" action="/predict">
  <label>Unemployment: <input name="unemployment" type="number" step="any" required></label><br>
  <label>Inflation (current year): <input name="inflation" type="number" step="any" required></label><br>
  <label>Oil rents: <input name="oil_rents" type="number" step="any" required></label><br>
  <label>GDP growth: <input name="gdp_growth" type="number" step="any" required></label><br>
  <label>Exchange rate: <input name="exchange_rate" type="number" step="any" required></label><br>
  <button type="submit">Predict</button>
</form>
"""

app = FastAPI()

 # Load the model and features
saved = joblib.load("model.joblib")
model = saved["model"]
features = saved["features"]

@app.get("/")
def hello():
    return {"message": "Hello, World!"}


@app.get("/predict", response_class=HTMLResponse)
def predict_form():
    return FORM_HTML


@app.post("/predict", response_class=HTMLResponse)
def predict(unemployment: float = Form(...), inflation: float = Form(...),
            oil_rents: float = Form(...), gdp_growth: float = Form(...),
            exchange_rate: float = Form(...)):
    # your job: build the feature vector in `features` order, call model.predict(),
    # return some HTML string showing the number
    # Load the data to predict
    df_predict = pd.DataFrame({
        'unemployment': [unemployment],
        'inflation': [inflation],
        'oil_rents': [oil_rents],
        'gdp_growth': [gdp_growth],
        'exchange_rate': [exchange_rate]
    })

    # Ensure the input data has the same features as the model expects
    X_predict = df_predict[features]

    # Make predictions
    predictions = model.predict(X_predict)
    html = f"<p>Predicted value: {predictions[0]}</p>"
    # Return predictions as a list
    return html









    