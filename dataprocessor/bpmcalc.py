from dataprocessor.smoothing.doubleexponential import DoubleExponential
import numpy as np
from scipy.optimize import curve_fit
import math


def project_velocities(data, direction):
    elt_wise_product = data * direction
    dots = np.array([sum(elt_wise_product[i]) for i in range(len(elt_wise_product))])
    projections = dots / np.linalg.norm(direction)
    return projections


def lsrl_vector(data):
    A = np.array([data[:, 0], np.ones(len(data))]).T
    m, _ = np.linalg.lstsq(A, data[:, 1], rcond=None)[0]
    mag = np.sqrt(1 + m ** 2)
    return np.array([np.sign(m) / mag, np.abs(m) / mag])  # want positive y to be positive in this axis


def one_sided_derivative(time, data):
    return (data[1] - data[0]) / (time[1] - time[0])


def derivative_helper(time, data):
    # converges to centered differences when h1 = h2
    derivs = [one_sided_derivative(time[i: i + 2], data[i: i + 2]) for i in range(len(time) - 1)]
    return np.average(derivs)


def derivative(time, data):
    one_sided_derivs = [one_sided_derivative(time[i:i+2], data[i:i+2]) for i in range(len(time) - 1)]
    deriv = np.zeros_like(data)

    deriv[0] = one_sided_derivs[0]
    for i in range(1, len(data) - 1):
        deriv[i] = np.average(one_sided_derivs[i - 1:i + 1])
    deriv[-1] = one_sided_derivs[-1]

    return deriv


def sinfunc(x, a, w, phi, c):
    return a * np.sin(w * x + phi) + c


def clean_data(data):
    data = data.T
    if len(data.shape) == 1:
        clean_data_helper(data)
    else:
        for dimension in data:
            clean_data_helper(dimension)
    return data.T


def clean_data_helper(data):
    N = 0
    total = 0
    bad_indexes = []
    for i in range(len(data)):
        if math.isnan(data[i]) or math.isinf(data[i]):
            bad_indexes.append(i)
            continue
        N += 1
        total += data[i]
    avg_val = total / N
    for i in bad_indexes:
        data[i] = avg_val


def bpm_from_ang_freq(angular_frequency):
    freq = np.abs(angular_frequency) / (2 * np.pi)
    return freq * 60


class BPMCalc:

    def __init__(self):
        self.smoother = DoubleExponential()
        self.time = []
        self.data = []
        self.velocity_data = None
        self.minimum_points = 10
        self.maximum_points = 200

    def calculate_bpm(self):
        if len(self.time) < self.minimum_points:
            return None

        if len(self.time) > self.maximum_points:
            self.time = self.time[-self.maximum_points:]
            self.data = self.data[-self.maximum_points:]

        smoothed_data = self.smoother.smooth(np.array([*self.data]).astype(float))
        time = np.array(self.time)
        self.process_data(time, smoothed_data)

        if not np.any(self.velocity_data):
            print("Processed data, but got None!")
            return None

        # initial guess -- amplitude 10, 60bpm (6.28 rad/sec), phase shift 1 rad, 0 vertical offset
        p0 = [10, 6.28, 1, 0]
        popt, _ = curve_fit(sinfunc, time[2: -2], self.acceleration_data[2: -2], p0=p0)
        return bpm_from_ang_freq(popt[1])

    def process_data(self, time, smoothed_data):
        head_velocity = derivative(time, smoothed_data)
        bop_vector = lsrl_vector(smoothed_data)
        head_velocity = project_velocities(head_velocity, bop_vector)
        self.velocity_data = clean_data(head_velocity)

    def send_data(self, time, coord):
        if len(self.time) == 0 or time and coord and self.time[-1] != time:
            self.data.append(coord)
            self.time.append(time)

    def save_data(self):
        np.save('test_acceleration_data', np.array([self.time, self.acceleration_data]))
