import matplotlib.pyplot as plt

from dataprocessor.bpmcalc import *


def sinfunc(x, a, w, phi, c):
    return a * np.sin(w * x + phi) + c


if __name__ == '__main__':
    time = None
    data = None

    with open('test_position_data.npy', 'rb') as f:
        time = np.load(f)
        data = np.load(f)

    bpmcalc = BPMCalc()
    bpmcalc.process_data(time, data)

    popt, _ = curve_fit(sinfunc, time[2: -2], bpmcalc.acceleration_data[2: -2], p0=[10, 6.28, 1, 0])
    bpm = bpm_from_ang_freq(popt[1])
    print(bpm)

    plt.plot(time[1:len(time)-1], bpmcalc.acceleration_data[1:len(time)-1], label='processed unsmoothed data')
    plt.plot(time[2:-2], sinfunc(time[2:-2], *popt), label='fit for unsmoothed data')

    bpmcalc.time = time
    bpmcalc.data = data
    print(bpmcalc.calculate_bpm())

    popt, _ = curve_fit(sinfunc, time[2: -2], bpmcalc.acceleration_data[2: -2], p0=[10, 6.28, 1, 0])

    plt.plot(time[1:len(time)-1], bpmcalc.acceleration_data[1:len(time)-1], label='processed smoothed data')
    plt.plot(time[2:-2], sinfunc(time[2:-2], *popt), label='fit for smoothed data')

    plt.legend()
    plt.show()
