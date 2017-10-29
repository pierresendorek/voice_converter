from feature_extraction.pitch_estimator import PitchEstimator
from feature_extraction.spectral_enveloppe_extractor import SpectralEnvelopeExtractor
from feature_extraction.periodic_and_noise_separator import PeriodicAndNoiseSeparator
from common.count_segments import count_segments
from database_tools.sound_file_loader import get_segment
import numpy as np


def voice_feature_extractor(params=None, sound=None):

    pitch_extractor = PitchEstimator(params=params)
    period_and_noise_separator = PeriodicAndNoiseSeparator(params=params)
    spectral_envelope_extractor = SpectralEnvelopeExtractor(params=params)

    apowin2 = params["apowin2"]

    n_segment = count_segments(sound=sound, params=params)

    period_list = []
    spectral_envelope_coeffs_noise_list = []
    spectral_envelope_coeffs_harmonic_list = []

    sound_out = np.zeros(sound.shape)

    for i_segment in range(n_segment):
        x = get_segment(sound=sound,  i_segment=i_segment, params=params)
        x_apodized = x * apowin2
        period = pitch_extractor.estimate_period(x_apodized)
        periodic, noise = period_and_noise_separator.separate_components(x_apodized=x_apodized, period=period)
        spectral_envelope_coeffs_periodic = spectral_envelope_extractor.get_spectral_envelope_coeffs(periodic)
        spectral_envelope_coeffs_noise = spectral_envelope_extractor.get_spectral_envelope_coeffs(noise)

        spectral_envelope_coeffs_noise_list.append(spectral_envelope_coeffs_noise)
        spectral_envelope_coeffs_harmonic_list.append(spectral_envelope_coeffs_periodic)
        period_list.append(period)

    return {"period_list": period_list,
            "spectral_envelope_coeffs_harmonic_list":spectral_envelope_coeffs_harmonic_list,
            "spectral_envelope_coeffs_noise_list":spectral_envelope_coeffs_noise_list}