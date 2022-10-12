import uvicorn
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from new_apps_preprocessing import new_apps_preprocessing
from predict_model import PredictScore

# Load new applications
path = Path(__file__).parent
app_test_df = pd.read_csv(path / "Data" / "application_test.csv")


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
def predict(app_test_df):
    new_apps = new_apps_preprocessing(app_test_df)
    prediction = classifier.predict_default(new_apps)
    shap_values, exp_values = classifier.predict_shap(new_apps)
    image = classifier.shap_summary(new_apps)

    new_apps['PREDS'] = prediction

    return {
        'new_apps_prediction': new_apps,
        'shap_values': shap_values,
        'expectation_values': exp_values,
        'shap_summary': image,
        'new_apps_file': app_test_df
    }
# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
