from params.params import get_params
from database_tools.sound_file_loader import get_mono_left_channel_sound_and_sampling_frequency
from feature_extraction.pitch_estimator import PitchEstimator
from feature_extraction.periodic_and_noise_separator import PeriodicAndNoiseSeparator
from feature_extraction.spectral_enveloppe_extractor import SpectralEnvelopeExtractor
from common.count_segments import count_segments
from database_tools.sound_file_loader import get_segment
import matplotlib.pyplot as plt
import numpy as np
from feature_extraction.voice_feature_extractor import voice_feature_extractor


params = get_params()

filename = "/home/monsieur/projets/voice_conversion/data/bruce_willis/Studio/1.wav"

sound, sampling_frequency = get_mono_left_channel_sound_and_sampling_frequency(filename)


params = get_params()

params["n_triangle_function"] = 40
rough_spectral_envelope_extractor = SpectralEnvelopeExtractor(params=params)



spectrum_as_list = rough_spectral_envelope_extractor.get_spectral_envelope_from_sound(sound)


spectrum = np.zeros([40, len(spectrum_as_list)])

for it in range(len(spectrum_as_list)):
    spectrum[:, it] = spectrum_as_list[it]



plt.imshow(np.log(spectrum + 1E-7))
plt.gca().invert_yaxis()
plt.show()
