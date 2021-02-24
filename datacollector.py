# only for testing

import numpy as np

class DataCollector:
    def __init__(self):
        self.time = []
        self.data = []

    def add(self, time, data):
        self.time.append(time)
        self.data.append(data)

    def write(self):
        time = np.array(self.time)
        data = np.array([*self.data]).astype(float)
        with open('test_position_data.npy', 'wb') as f:
            np.save(f, time)
            np.save(f, data)
