from pathlib import Path
from joblib import load
from lightgbm import LGBMClassifier

class PredictScore:

    def __init__(self):
        self.path = Path(__file__).parent
        self.prediction = None
        self.shap_values = None
        self.exp_values = None
        self.clf = load(self.path / 'resources' / 'classifier.joblib')
        self.feats = load(self.path / 'resources' / 'feats.joblib')

    def predict_default(self, test_df):
        self.prediction = self.clf.predict_proba(test_df[self.feats],
                                                 num_iteration=self.clf.best_iteration_)[:, 1]
        return self.prediction

    def predict_shap(self, test_df):
        contribs = self.clf.predict_proba(test_df[self.feats],
                                          num_iteration=self.clf.best_iteration_, pred_contrib=True)
        self.shap_values = contribs[:, :-1]
        self.exp_values = contribs[:, -2:-1]

        return self.shap_values, self.exp_values
