from feature_extraction.spectral_enveloppe_extractor import SpectralEnvelopeExtractor
from database_tools.sound_file_loader import get_mono_left_channel_sound_and_sampling_frequency
from alignment.dynamic_time_warping import get_dtw_matrix
from params.params import get_params
import numpy as np

import matplotlib.pyplot as plt

params = get_params()

params["n_triangle_function"] = 15
rough_spectral_envelope_extractor = SpectralEnvelopeExtractor(params=params)

file_1 = "/home/monsieur/projets/voice_conversion/data/bruce_willis/Studio/1.wav"
file_2 = "/home/monsieur/projets/voice_conversion/data/pierre_sendorek/Studio/1.wav"

sound_1, _ = get_mono_left_channel_sound_and_sampling_frequency(file_1)
sound_2, _ = get_mono_left_channel_sound_and_sampling_frequency(file_2)

voice_1_features = rough_spectral_envelope_extractor.get_spectral_envelope_from_sound(sound=sound_1)
voice_2_features = rough_spectral_envelope_extractor.get_spectral_envelope_from_sound(sound=sound_2)



def get_element_from_list(i_element, L):
    if i_element < 0 or i_element >= len(L):
        return np.zeros(L[0].shape)
    else:
        return L[i_element]


def derivative_on_list(voice_features):
    derivative_voice_features = []
    for i in range(len(voice_features)):
        derivative_voice_features.append(get_element_from_list(i, voice_features) -
                                         get_element_from_list(i-1, voice_features))
    return derivative_voice_features


def distance(a, b):
    return np.linalg.norm(np.log(a) - np.log(b))




M, corresponding_segments, local_distance_matrix = get_dtw_matrix(voice_1_features,
                                           voice_2_features,
                                           distance_function=distance)


a, b = zip(*corresponding_segments)



print(corresponding_segments)

plt.figure(1)

def derivative(a):
    return a[:, 1:] - a[:, :-1]

f, axarr = plt.subplots(3, 1)
axarr[0].imshow(np.log(np.array(voice_1_features)).T)
for i in range(len(corresponding_segments)):
    j1, j2 = corresponding_segments[i]
    axarr[1].plot([j1, j2], [1, 0], color='k')

axarr[2].imshow(np.log(np.array(voice_2_features)).T)

plt.show()

plt.figure(3)
plt.imshow(local_distance_matrix)
plt.plot(b, a, color="red")
plt.show()



