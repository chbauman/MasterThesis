
import numpy as np

from statsmodels.tsa.arima_model import ARIMA

# AR Time Series Model
class AR_Model(object):

    def __init__(self, lag = 1):
        self.lag = lag

    def fit(self, data):

        self.model = ARIMA(data, order=(self.lag, 0, 0))
        self.model_fit = self.model.fit(trend='nc', disp=False)

    def predict(self, data):

        ar_coef = self.model_fit.arparams
        sig = sqrt(self.model_fit.sigma2)
        yhat = 0.0
        for i in range(1, len(ar_coef) + 1):
            yhat += ar_coef[i-1] * data[-i]
        return yhat + np.random.normal(0, sig)
    pass

