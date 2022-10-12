from pathlib import Path
import re
from application import *
from bureau import *
from credit_card import *
from prev_app import *
from pos import *
from installments import *
from joblib import dump


def preprocess():
    """ perform preprocessing of files and saves them as .joblib files"""

    # read source files
    path = Path(__file__).parent
    app_train_df = pd.read_csv(path / "Data" / "application_train.csv")
    bureau_balance_df = pd.read_csv(path / "Data" / "bureau_balance.csv")
    bureau_df = pd.read_csv(path / "Data" / "bureau.csv")
    ccb_df = pd.read_csv(path / "Data" / "credit_card_balance.csv")
    ins_df = pd.read_csv(path / "Data" / "installments_payments.csv")
    pos_df = pd.read_csv(path / "Data" / "POS_CASH_balance.csv")
    prev_app_df = pd.read_csv(path / "Data" / "previous_application.csv")

    df = application_train_test(app_train_df)
    dump(app_train_df, path / 'resources' / 'app_train.joblib')

    bureau = bureau_and_balance(bureau_df, bureau_balance_df)
    df = df.merge(bureau, how='left', on='SK_ID_CURR')
    dump(bureau, path / 'resources' / 'bureau.joblib')
    del bureau

    prev_app = previous_applications(prev_app_df)
    df = df.merge(prev_app, how='left', on='SK_ID_CURR')
    dump(prev_app, path / 'resources' / 'prev_app.joblib')
    del prev_app

    pos = pos_cash(pos_df)
    df = df.merge(pos, how='left', on='SK_ID_CURR')
    dump(pos, path / 'resources' / 'pos.joblib')
    del pos

    ins = installments_payments(ins_df)
    df = df.merge(ins, how='left', on='SK_ID_CURR')
    dump(ins, path / 'resources' / 'installments.joblib')
    del ins

    cc = credit_card_balance(ccb_df)
    df = df.merge(cc, how='left', on='SK_ID_CURR')
    dump(cc, path / 'resources' / 'credit_card.joblib')
    del cc

    df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x), inplace=True)
    dump(df, path / 'resources' / 'train_set.joblib')

    feats = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR']]
    dump(feats, path / 'resources' / 'feats.joblib')

