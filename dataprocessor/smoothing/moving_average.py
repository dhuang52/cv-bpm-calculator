from dataprocessor.smoothing import  BaseSmoother
import numpy as np


class MovingAverage(BaseSmoother):
    '''
    Centered moving average smoothing
    '''

    def __init__(self, num_points=3):
        super().__init__()
        while num_points % 2 == 0:
            num_points =  input("Please enter an odd number of points to perform a moving average: ")
        self._num_points = num_points

    def smooth(self, data):
        ind = self._num_points // 2
        for i in range(ind, ind + len(data[ind:-ind])):
            data[i] = np.mean(data[i - ind: i + ind + 1], axis=0)
        return data

