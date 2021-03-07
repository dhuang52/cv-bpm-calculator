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

    label = ['Raw x values', 'Raw y values']
    plot = plt.plot(time, data)
    plt.legend(plot, label)
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.show()

    label = ['Smoothed x values', 'Smoothed y values']
    plot = plt.plot(time, smoother.smooth(data))
    plt.legend(plot, label)
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.show()

    label = ['Raw x velocity', 'Raw y velocity']
    plot = plt.plot(time, derivative(time, data))
    plt.legend(plot, label)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.show()

    label = ['Smoothed x velocity', 'Smoothed y velocity']
    plt.plot(time, smoother.smooth(derivative(time, data)))
    plt.legend(plot, label)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.show()
