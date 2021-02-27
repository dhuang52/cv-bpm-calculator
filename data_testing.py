import matplotlib.pyplot as plt

from dataprocessor.bpmcalc import *
from dataprocessor.smoothing import DoubleExponential, MovingAverage


def sinfunc(x, a, w, phi, c):
    return a * np.sin(w * x + phi) + c


if __name__ == '__main__':
    time = None
    data = None

    with open('test_position_data.npy', 'rb') as f:
        time = np.load(f)
        data = np.load(f)

    smoother = MovingAverage(3)
    # smoother = DoubleExponential()
    bpmcalc = BPMCalc(smoother)
    velocity = bpmcalc.process_data(time, data)

    popt, _ = curve_fit(sinfunc, time[2: -2], velocity[2: -2], p0=[10, 6.28, 1, 0])
    bpm = bpmcalc.bpm_from_ang_freq(popt[1])
    print(bpm)

    plt.plot(time[1:len(time)-1], velocity[1:len(time) - 1], label='processed unsmoothed data')
    plt.plot(time[2:-2], sinfunc(time[2:-2], *popt), label='fit for unsmoothed data')
    plt.plot(time, smoother.smooth(velocity), label='smoothed velocity data')

    bpmcalc.time = time.copy()
    bpmcalc.data = data.copy()
    print(bpmcalc.calculate_bpm())

    popt, _ = curve_fit(sinfunc, bpmcalc.fit[2][2: -2], bpmcalc.fit[3][2: -2], p0=[10, 6.28, 1, 0])

    plt.plot(bpmcalc.fit[2][1:len(bpmcalc.time)-1], bpmcalc.fit[3][1:len(bpmcalc.time) - 1], label='processed smoothed data')
    plt.plot(bpmcalc.fit[2][2:-2], sinfunc(bpmcalc.fit[2][2:-2], *popt), label='fit for smoothed data')

    plt.legend()
    plt.show()

    plt.plot(time, data, label='raw position data')
    plt.legend()
    plt.show()

    plt.plot(time, smoother.smooth(data), label='smoothed position data')
    plt.legend()
    plt.show()

    plt.plot(time, derivative(time, data), label='raw velocity data')
    plt.legend()
    plt.show()

    plt.plot(time, smoother.smooth(derivative(time, data)), label='smoothed raw velocity data')
    plt.legend()
    plt.show()
