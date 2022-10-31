from pathlib import Path
import re
from application import *
from joblib import load
from preprocess import preprocess


def new_apps_preprocessing(app_test_df):
    """ perform preprocessing of new application files"""

    # read and load source files
    path = Path(__file__).parent.parent

    try:
        train_df = load(path / 'resources' / 'train_set.joblib')
    except FileNotFoundError:
        print("Preprocessing has to be done : it will take about one minute")
        preprocess()
        train_df = load(path / 'resources' / 'train_set.joblib')

    bureau = load(path / 'resources' / 'bureau.joblib')
    prev_app = load(path / 'resources' / 'prev_app.joblib')
    pos = load(path / 'resources' / 'pos.joblib')
    ins = load(path / 'resources' / 'installments.joblib')
    cc = load(path / 'resources' / 'credit_card.joblib')

    # preprocess application test file
    df = application_train_test(app_test_df)

    # add extra information to application test file
    df = df.merge(bureau, how='left', on='SK_ID_CURR')
    del bureau
    df = df.merge(prev_app, how='left', on='SK_ID_CURR')
    del prev_app
    df = df.merge(pos, how='left', on='SK_ID_CURR')
    del pos
    df = df.merge(ins, how='left', on='SK_ID_CURR')
    del ins
    df = df.merge(cc, how='left', on='SK_ID_CURR')
    del cc

    # format harmonization
    df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x), inplace=True)

    # ensure test and train files have same number of columns
    cols = list(set(train_df.columns) - set(df.columns))
    if cols is not None:
        df[cols] = np.NAN

    return df
