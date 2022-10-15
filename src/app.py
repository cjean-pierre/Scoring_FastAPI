import pandas as pd
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Body
from src.predict_model import PredictScore

# Load new applications
path = Path(__file__).parent

# load trained classifier
classifier = PredictScore()

# create app object
app = FastAPI()

# 3. Index route, opens automatically on http://127.0.0.1:8000


@app.get('/')
def index():
    return {'message': 'Hello, stranger'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere


@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

# Route for model prediction, make a prediction from the passed
#    new apps data and return the lgbm prediction and shap values


@app.post('/predict')
def predict(new_app=Body(...)):

    app_df = pd.read_json(new_app, orient='records')
    prediction = classifier.predict_default(app_df)
    app_df['PREDS'] = prediction

    return {
        'new_apps_prediction': app_df.to_json(orient='records'),
    }

# Route for model prediction, make a prediction from the passed
#    new apps data and return shap values


@app.post('/shap')
def shap(new_app=Body(...)):

    app_df = pd.read_json(new_app, orient='records')
    shap_values, exp_values = classifier.predict_shap(app_df)

    return {
        'shap_values': pd.DataFrame(shap_values).to_json(orient='index'),
        'expectation_values': pd.DataFrame(exp_values).to_json(orient='index'),
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
