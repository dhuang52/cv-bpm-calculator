import numpy as np
import math


def one_sided_derivative(time, data):
    return (data[1] - data[0]) / (time[1] - time[0])


def derivative_helper(time, data):
    # converges to centered differences when h1 = h2
    derivs = [one_sided_derivative(time[i: i + 2], data[i: i + 2]) for i in range(len(time) - 1)]
    return np.average(derivs)


def derivative(time, data):
    one_sided_derivs = [one_sided_derivative(time[i:i + 2], data[i:i + 2]) for i in range(len(time) - 1)]
    deriv = np.zeros_like(data)

    deriv[0] = one_sided_derivs[0]
    for i in range(1, len(data) - 1):
        deriv[i] = np.mean(one_sided_derivs[i - 1:i + 1], axis=0)
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
