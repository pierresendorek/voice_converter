from scipy.io import wavfile
import numpy as np

def get_mono_left_channel_sound_and_sampling_frequency(filename):
    sampling_frequency, sound = wavfile.read(filename)
    if len(sound.shape) > 1:
        return sound[: ,0], sampling_frequency
    else:
        return sound, sampling_frequency


def get_segment(sound=None, i_segment=None, params=None):

    samples_per_segment = params["segment_len"]
    n_gap = params["n_gap"]

    i_start = n_gap * i_segment
    i_end = i_start + samples_per_segment
    if i_start >= 0 and i_end < len(sound):
        return sound[i_start: i_end]
    else:
        return np.zeros(i_end - i_start)

def count_segments(sound=None, params=None):
    samples_per_segment = params["segment_len"]
    n_gap = params["n_gap"]
    return int(np.floor((len(sound) - samples_per_segment)/n_gap))



if __name__ == "__main__":
    filename = "/home/monsieur/projets/voice_conversion/data/bruce_willis/Studio/18.wav"
    sound, sampling_frequency =  get_mono_left_channel_sound_and_sampling_frequency(filename)






