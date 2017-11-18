from params.params import get_params

from database_tools.sound_file_loader import get_mono_left_channel_sound_and_sampling_frequency
import os
from feature_extraction.pitch_estimator import PitchEstimator
from feature_extraction.periodic_and_noise_separator import PeriodicAndNoiseSeparator
from feature_extraction.spectral_enveloppe_extractor import SpectralEnvelopeExtractor

from common.count_segments import count_segments
from database_tools.sound_file_loader import get_segment

import matplotlib.pyplot as plt
import numpy as np

params = get_params()

filename = os.path.join(params["project_base_path"], "data/bruce_willis/Studio/1.wav")
#filename = "/home/monsieur/maria.wav"
sound, sampling_frequency = get_mono_left_channel_sound_and_sampling_frequency(filename)


from feature_extraction.voice_feature_extractor import VoiceFeatureExtractor

voice_feature_extractor = VoiceFeatureExtractor(params)

sound_features = voice_feature_extractor.extract(sound=sound)


spectral_envelope_coeffs_noise_list = sound_features["spectral_envelope_coeffs_noise_list"]
spectral_envelope_coeffs_harmonic_list = sound_features["spectral_envelope_coeffs_harmonic_list"]
period_list = sound_features["period_list"]
# period_list_bzz = [sampling_frequency/220]*len(period_list)


frequency_list = [sampling_frequency/p for p in period_list]

from synthesis.voice_synthesizer import generate_periodic_sound, generate_periodic_filtered_sound, \
    generate_filtered_noise
from scipy.io.wavfile import write

out_sound_noise = generate_filtered_noise(spectral_envelope_coeffs_noise_list, params)
out_sound_periodic = generate_periodic_sound(segment_period_expressed_in_sample_list=period_list, params=params)
out_sound_periodic_filtered =generate_periodic_filtered_sound(segment_period_expressed_in_sample_list=period_list,
                                                              spectral_envelope_coeffs_list=spectral_envelope_coeffs_harmonic_list,
                                                              params=params)


reconstruction = out_sound_noise + out_sound_periodic_filtered
reconstruction = reconstruction / max(abs(reconstruction))

base_path = "/Users/pierresendorek/"

write(base_path + "temp/out.wav", 44100, out_sound_noise / np.max(np.abs(out_sound_noise)))
write(base_path + "temp/out_periodic.wav", 44100, out_sound_periodic / np.max(np.abs(out_sound_periodic)))
write(base_path + "temp/out_periodic_filt.wav", 44100, out_sound_periodic_filtered / np.max(np.abs(out_sound_periodic_filtered)))
write(base_path + "temp/reconstruction.wav", 44100, reconstruction)

plt.plot(frequency_list, linewidth=5)
plt.show()





