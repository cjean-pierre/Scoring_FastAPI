from pathlib import Path
from joblib import load
from joblib import dump
import pandas as pd
from new_apps_preprocessing import new_apps_preprocessing

path = Path(__file__).parent.parent

# prepare a sample of old apps for Heroku app
app_train_df = load(path / 'resources' / 'app_train.joblib')
train_df = load(path / 'resources' / 'train_set.joblib')

old_apps = train_df.merge(app_train_df[["SK_ID_CURR", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"]], how='left', on='SK_ID_CURR')

old_apps_samp = old_apps.sample(30000)

dump(old_apps_samp, path / 'resources' / 'old_apps.joblib')

# prepare a new apps file for Heroku app
app_test_df = pd.read_csv(path/'Data'/'application_test.csv')
test_df = new_apps_preprocessing(app_test_df)

new_apps = test_df.merge(app_test_df[["SK_ID_CURR", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"]], how='left', on='SK_ID_CURR')
new_apps_samp = new_apps.sample(10000)

dump(new_apps_samp, path / 'resources' / 'new_apps.joblib')

