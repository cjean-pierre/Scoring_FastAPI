from pathlib import Path
from joblib import load
from src.train_model import train_model
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import shap
from PIL import Image


class PredictScore:

    def __init__(self):
        self.path = Path(__file__).parent
        self.prediction = None
        self.shap_values = None
        self.exp_values = None
        self.image = None

        try:
            self.clf = load(self.path / 'resources' / 'classifier.joblib')
        except FileNotFoundError:
            print("Model Training has to be done, you have time to offer a coffee to your client")
            train_model()
            self.clf = load(self.path / 'resources' / 'classifier.joblib')

        self.train_df = load(self.path / 'resources' / 'train_set.joblib')
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

    def shap_summary(self, test_df):

        feat_names = [feat.capitalize() for feat in self.feats]
        feat_values = np.array(test_df[self.feats])

        hl_colors = ['#CADAE6', '#A6C1CF', '#86A6B0', '#546E7A', '#37474F']
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('hl_cmap', colors=hl_colors, N=100)

        shap.summary_plot(self.shap_values, feat_values, feature_names=feat_names, max_display=20,
                          cmap=cmap,
                          plot_size=0.35,
                          show=False)

        plt.savefig(fname=self.path / 'resources' / 'shap_summary.png')

        self.image = Image.open(self.path / 'resources' / 'shap_summary.png')

        return self.image
