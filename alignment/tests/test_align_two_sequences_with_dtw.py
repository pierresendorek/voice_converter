from feature_extraction.voice_feature_extractor import voice_feature_extractor
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




def distance(a, b):
    return np.linalg.norm(np.log(a) - np.log(b))




M, corresponding_segments, local_distance_matrix = get_dtw_matrix(voice_1_features,
                                           voice_2_features,
                                           distance_function=distance)


a, b = zip(*corresponding_segments)



print(corresponding_segments)



plt.figure(1)
plt.imshow(voice_1_features)
plt.show()

plt.figure(2)
plt.imshow(voice_2_features)
plt.show()

plt.figure(3)
plt.imshow(local_distance_matrix)
plt.plot(b, a)
plt.show()



