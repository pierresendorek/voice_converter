from params.params import get_params

from database_tools.sound_file_loader import get_mono_left_channel_sound_and_sampling_frequency

from feature_extraction.pitch_estimator import PitchEstimator
from feature_extraction.periodic_and_noise_separator import PeriodicAndNoiseSeparator
from feature_extraction.spectral_enveloppe_extractor import SpectralEnvelopeExtractor

from database_tools.sound_file_loader import count_segments
from database_tools.sound_file_loader import get_segment

import matplotlib.pyplot as plt
import numpy as np

params = get_params()

filename = "/home/monsieur/projets/voice_conversion/data/bruce_willis/Studio/18.wav"
sound, sampling_frequency = get_mono_left_channel_sound_and_sampling_frequency(filename)


pitch_extractor = PitchEstimator(params=params)
period_and_noise_separator = PeriodicAndNoiseSeparator(params=params)
spectral_envelope_extractor = SpectralEnvelopeExtractor(params=params)

apowin2 = params["apowin2"]

n_segment = count_segments(sound=sound, params=params)

period_list = []
spectral_envelope_coeffs_noise_list = []
spectral_envelope_coeffs_periodic_list = []

sound_out = np.zeros(sound.shape)

for i_segment in range(n_segment):
    x = get_segment(sound=sound,  i_segment=i_segment, params=params)
    x_apodized = x * apowin2
    period = pitch_extractor.estimate_period(x_apodized)
    periodic, noise = period_and_noise_separator.separate_components(x_apodized=x_apodized, period=period)
    spectral_envelope_coeffs_periodic = spectral_envelope_extractor.get_spectral_envelope_coeffs(periodic)
    spectral_envelope_coeffs_noise = spectral_envelope_extractor.get_spectral_envelope_coeffs(noise)

    spectral_envelope_coeffs_noise_list.append(spectral_envelope_coeffs_noise)
    spectral_envelope_coeffs_periodic_list.append(spectral_envelope_coeffs_periodic)
    period_list.append(period)



from synthesis.synth import generate_filtered_noise
from synthesis.synth import generate_periodic_sound
from synthesis.synth import generate_periodic_filtered_sound
from scipy.io.wavfile import write

out_sound_noise = generate_filtered_noise(spectral_envelope_coeffs_noise_list, params)
out_sound_periodic = generate_periodic_sound(segment_period_expressed_in_sample_list=period_list, params=params)
out_sound_periodic_filtered =generate_periodic_filtered_sound(segment_period_expressed_in_sample_list=period_list,
                                                              spectral_envelope_coeffs_list=spectral_envelope_coeffs_periodic_list,
                                                              params=params)


write("/home/monsieur/temp/18.wav", 44100, out_sound_noise / np.max(np.abs(out_sound_noise)))
write("/home/monsieur/temp/18_periodic.wav", 44100, out_sound_periodic / np.max(np.abs(out_sound_periodic)))
write("/home/monsieur/temp/18_periodic_filt.wav", 44100, out_sound_periodic_filtered / np.max(np.abs(out_sound_periodic_filtered)))

#plt.plot(period_list)
#plt.show()





