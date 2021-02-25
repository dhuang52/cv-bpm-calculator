from dataprocessor.smoothing import BaseSmoother
import numpy as np


class DoubleExponential(BaseSmoother):

    def __init__(self):
        super().__init__()
        self.alpha = 0.6   # may change later
        self.gamma = 0.5    # may change later

    def smooth(self, data):
        S = np.zeros_like(data)
        b = np.zeros_like(data)

        # initial values
        S[0] = data[0]
        b[0] = data[1] - data[0]

        for t in range(1, len(data)):
            S[t] = self.alpha * data[t] + (1 - self.alpha) * (S[t-1] + b[t - 1])
            b[t] = self.gamma * (S[t] - S[t - 1]) + (1 - self.gamma) * b[t - 1]

        return S

