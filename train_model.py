from joblib import dump
from joblib import load
from pathlib import Path
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from preprocess import preprocess


def train_model():

    """ train a lgbm classifier and save it as a joblib file"""

    path = Path(__file__).parent
    try:
        train_df = load(path / 'resources' / 'train_set.joblib')
    except FileNotFoundError:
        print("Preprocessing has to be done : it will take about one minute")
        preprocess()
        train_df = load(path / 'resources' / 'train_set.joblib')

    feats = load(path / 'resources' / 'feats.joblib')
    train_x, valid_x, train_y, valid_y = train_test_split(train_df[feats], train_df['TARGET'],
                                                          stratify=train_df['TARGET']
                                                          )
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'n_estimators': 5000,
        'learning_rate': 0.02,
        'num_leaves': 34,
        'colsample_bytree': 0.9497036,
        'max_depth': 8,
        'reg_alpha': 0.041545473,
        'reg_lambda': 0.0735294,
        'min_split_gain': 0.0222415,
        'min_child_weight': 39.3259775,
        'class_weight': {0: 1, 1: 6}
    }
    clf = LGBMClassifier()
    clf.set_params(**params)

    clf.fit(train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc',
            callbacks=[early_stopping(200)])

    dump(clf, path / 'resources' / 'classifier.joblib')
