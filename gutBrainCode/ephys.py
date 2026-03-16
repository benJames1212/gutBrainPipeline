import sys
import numpy as np

def getchannelnumber(ep_file_path):
    ch_num = int(ep_file_path.split('.')[-1][0:2])
    print('number of channels: {0}'.format(ch_num))
    return ch_num

def getversion(ep_file_path):
    print(ep_file_path)
    version_test = (ep_file_path.split('.')[-1]).split('-v')
    if len(version_test)>1:
#         software_version = int(version_test[-1][1:])
        software_version = int(version_test[-1])
    else:
        software_version = 0
    print('version: {0}'.format(software_version))
    return software_version


def load(in_file, num_channels=10, memmap=False):
    """Load multichannel binary data from disk, return as a [channels,samples] sized numpy array
    """
    from numpy import fromfile, float32

    if memmap:
        from numpy import memmap

        data = memmap(in_file, dtype=float32)
    else:
        with open(in_file, "rb") as fd:
            data = fromfile(file=fd, dtype=float32)
    trim = data.size % num_channels
    # transpose to make dimensions [channels, time]
    data = data[: (data.size - trim)].reshape(data.size // num_channels, num_channels).T
    if trim > 0:
        print("Data needed to be truncated!")
    return data



def detect_LStimes(data, LS_channel, th = [3.9,100], duration = 20, sr = 6000):
    print('detecting acquisition times')
    camLS_times = []
    camLS_times = estimate_onset(data[LS_channel], th=th, duration=20)
    if len(camLS_times)>1:
        print('# time points = : %d' % len(camLS_times))
        stLS_rate = 1/(np.diff(camLS_times).mean()/sr)
        print('# acquisition rate = : %.2f Hz' % stLS_rate)
    else:
        stLS_rate = 0
    return np.array(camLS_times), np.array(stLS_rate)



def estimate_onset(signal, th = [0.3, 100], duration = 100):
    """
    Find indices in a vector when the values first cross a threshold. Useful when e.g. finding onset times for
    a ttl signal.
    Parameters
    ----------
    signal : numpy array, 1-dimensional
        Vector of values to be processed.

    threshold : instance of numeric data type contained in signal
        Onsets are counted as indices where the signal first crosses this value.

    duration : instance of numeric data type contained in signal
        Minimum distance between consecutive onsets.
    """
    thmax = th[1]
    th = th[0]
    from numpy import where, diff, concatenate
    inits = where((signal[:-1] < th) * (signal[1:] > th) *(signal[1:] < thmax) )[0]
    if len(inits) == 0:
        print("no value found")
        valid = []
        inits = []
    else:
        valid = concatenate([where(diff(inits) > duration)[0],[-1]])
    if len(valid) == 0:
        inits = []
        print("no valid points found")
    else:
        inits = inits[valid]  + 1
    return np.array(inits)

def estimate_pulsetrain_onset(signal, th, durationpulse, durationtrain):
    from numpy import where, diff, concatenate
    thmax = th[1]
    th = th[0]
    inits = where((signal[:-1] < th) * (signal[1:] > th) *(signal[1:] < thmax) )[0]
    if len(inits) == 0:
        print("no value found")
        valid = []
        inits = []
    else:
        valid = concatenate([where(diff(inits) > durationpulse)[0],[-1]])
    if len(valid) == 0:
        inits = []
        print("no valid points found")
    else:
        inits = inits[valid]  + 1
  
    valid2 = concatenate([[0], where(np.diff(inits)>durationtrain)[0]+1])
    if len(valid2) <2:
        inits2 = []
        print("no valid2 points found")
    else:
      
        inits2 = inits[valid2]
    return np.array(inits), np.array(inits2)


def find_frame_by_LS(LS_inits, signal_inits):
    import numpy as np
    from numpy import append, where, zeros
    signal_fr = np.zeros_like(signal_inits)
    for ind, time in enumerate(LS_inits[:-1]):
        tmp = np.argwhere(np.logical_and(signal_inits>=time,signal_inits<LS_inits[ind+1]))
        if tmp.shape[0]>0:
            signal_fr[tmp[0]] = ind
    return np.array(signal_fr)


# def find_onset_offset_pulsetimeseries(data, LED_channel, th = [.5,100], durationpulse=40, durationtrain = 1000):
#     LED_times = []
#     LEDp_times = []
#     LEDoffsetp_times, LEDoffset_times = (len(data[LED_channel])-estimate_pulsetrain_onset(data[LED_channel][::-1],th,durationpulse, durationtrain))[::-1]
#     LEDonsetp_times, LEDonset_times = (estimate_pulsetrain_onset(data[LED_channel],th,durationpulse, durationtrain))
#     print('# onsets = : %d' % len(LEDonset_times))
#     print('# offsets = : %d' % len(LEDoffset_times))
#     LED_times = list(zip(LEDonset_times, LEDoffset_times))
#     LEDp_times = list(zip(LEDonsetp_times, LEDoffsetp_times))
#     return LEDp_times, LED_times


def find_onset_offset_pulsetimeseries(data, LED_channel, th = [.5,100], durationpulse=40, durationtrain = 1000):
    LED_times = []
    LEDp_times = []
    LEDonsetp_times, LEDonset_times = estimate_pulsetrain_onset(data[LED_channel],th,durationpulse, durationtrain)
    LEDoffsetp_times, LEDoffset_times = estimate_pulsetrain_onset(data[LED_channel][::-1],th,durationpulse, durationtrain)
    LEDoffsetp_times = (len(data[LED_channel]) - LEDoffsetp_times)[::-1]
    LEDoffset_times = (len(data[LED_channel]) - LEDoffset_times)[::-1]
    
    print('# onsets = : %d' % len(LEDonset_times))
    print('# offsets = : %d' % len(LEDoffset_times))
    LED_times = list(zip(LEDonset_times, LEDoffset_times))
    LEDp_times = list(zip(LEDonsetp_times, LEDoffsetp_times))
    return np.array(LEDp_times), np.array(LED_times)

def find_onset_offset_timeseries(data, LED_channel, th = [.5,100], duration = 40):
    LED_times = []
    try:
        LEDoffset_times = (len(data[LED_channel])-estimate_onset(data[LED_channel][::-1], th=th, duration=duration))[::-1]
        LEDonset_times = (estimate_onset(data[LED_channel], th=th, duration = duration))
        print('# onsets = : %d' % len(LEDonset_times))
        print('# offsets = : %d' % len(LEDoffset_times))
        LED_times = list(zip(LEDonset_times, LEDoffset_times))
    except:
        pass
    return np.array(LED_times)

def find_start_end_frame(camLS_times, LED_times, var = 2):
    if var == 2:
        tmp = [find_frame_by_LS(camLS_times, [i[j] for i in LED_times]) for j in range(2)]
        LED_stack_time = list(zip(tmp[0], tmp[1]+1))
    if var ==1:
        LED_stack_time = [find_frame_by_LS(camLS_times, [i[0] for i in LED_times])]
    return np.array(LED_stack_time)



def get_intervals(inds):
    return np.array([inds[ind+1][0] - i[1] for ind, i in enumerate(inds[:-1])])

def get_durations(inds):
    return np.array([(i[1] - i[0]) for ind, i in enumerate(inds)]).squeeze()


def windowed_variance(signal, kw = 0.04, fs=6000):
    """
    Estimate smoothed sliding variance of the input signal

    signal : numpy array

    kern_mean : numpy array
        kernel to use for estimating baseline

    kern_var : numpy array
        kernel to use for estimating variance

    fs : int
        sampling rate of the data
    """
    from scipy.signal import gaussian, fftconvolve

    # set the width of the kernels to use for smoothing
    kw = int(kw * fs)


    kern_mean = gaussian(kw, kw // 10)
    kern_mean /= kern_mean.sum()

    kern_var = gaussian(kw, kw // 10)
    kern_var /= kern_var.sum()

    mean_estimate = fftconvolve(signal, kern_mean, "same")
    var_estimate = (signal - mean_estimate) ** 2
    fltch = fftconvolve(var_estimate, kern_var, "same")

    return fltch, var_estimate, mean_estimate