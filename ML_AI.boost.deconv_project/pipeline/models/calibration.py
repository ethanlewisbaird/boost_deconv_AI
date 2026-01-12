import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

class Calibrator:
    def __init__(self, method='linear'):
        self.method = method
        self.models = {}

    def fit(self, y_true, y_pred, labels):
        # y_true and y_pred: arrays shape (n_samples, n_celltypes)
        self.labels = labels
        if self.method == 'linear':
            self.models = {}
            for i, lab in enumerate(labels):
                lr = LinearRegression()
                lr.fit(y_pred[:, i].reshape(-1,1), y_true[:, i])
                self.models[lab] = lr
        elif self.method == 'isotonic':
            self.models = {}
            for i, lab in enumerate(labels):
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(y_pred[:, i], y_true[:, i])
                self.models[lab] = ir

    def predict(self, y_pred):
        y_corr = np.zeros_like(y_pred)
        for i, lab in enumerate(self.labels):
            m = self.models[lab]
            if self.method == 'linear':
                y_corr[:, i] = m.predict(y_pred[:, i].reshape(-1,1))
            else:
                y_corr[:, i] = m.transform(y_pred[:, i])
        return y_corr
