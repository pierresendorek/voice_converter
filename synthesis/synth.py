import numpy as np
from database_tools.segment_setter import add_to_segment
from feature_extraction.spectral_enveloppe_extractor import SpectralEnvelopeExtractor

def saw(x):
    return x - np.round(x)


def generate_sawtooth_sound(instantaneous_frequency_in_hertz=None, params=None):
    # TODO : la division par 2 est un quickfix : voir comment l'enlever
    frequency = instantaneous_frequency_in_hertz / 2
    sampling_frequency = params["sampling_frequency"]
    phase = np.cumsum(frequency / sampling_frequency)
    sawtooth = saw(phase)
    return sawtooth


def get_instantenous_frequency_array(segment_period_in_sample_list, params=None):
    sampling_frequency = params["sampling_frequency"]
    segment_len = params["segment_len"]
    n_gap = params["n_gap"]
    apowin2 = params["apowin2"]

    inst_fq = np.zeros(n_gap * len(segment_period_in_sample_list) + segment_len)


    for i_segment in range(len(segment_period_in_sample_list)):
        segment_period_in_seconds = segment_period_in_sample_list[i_segment] / sampling_frequency
        frequency_in_hertz = 1.0 / segment_period_in_seconds

        add_to_segment(i_segment=i_segment,
                       source=apowin2 * frequency_in_hertz / 2,
                       dest=inst_fq,
                       n_gap=n_gap)

    return inst_fq




def generate_periodic_sound(segment_period_expressed_in_sample_list=None,
                            params=None):

    inst_fq = get_instantenous_frequency_array(segment_period_in_sample_list=segment_period_expressed_in_sample_list,
                                               params=params)

    sawtooth = generate_sawtooth_sound(instantaneous_frequency_in_hertz=inst_fq, params=params)

    return sawtooth

from database_tools.sound_file_loader import get_segment

def generate_periodic_filtered_sound(segment_period_expressed_in_sample_list=None,
                                     spectral_envelope_coeffs_list=None,
                                     params=None):

    sawtooth = generate_periodic_sound(segment_period_expressed_in_sample_list=segment_period_expressed_in_sample_list,
                                      params=params)

    filtered_sawtooth = np.zeros(sawtooth.shape[0])

    spectral_envelope_extractor = SpectralEnvelopeExtractor(params)

    apowin2 = params["apowin2"]
    n_gap = params["n_gap"]

    for i_segment in range(len(spectral_envelope_coeffs_list)):
        seg = get_segment(sound=sawtooth, i_segment=i_segment, params=params)

        fft_seg = np.fft.fft(seg * apowin2)
        coeffs = spectral_envelope_coeffs_list[i_segment]
        spectral_envelope = spectral_envelope_extractor.get_full_spectral_envelope_from_coeffs(coeffs)
        filtered_seg = np.real(np.fft.ifft(fft_seg * spectral_envelope))
        add_to_segment(i_segment=i_segment, source=filtered_seg, dest=filtered_sawtooth, n_gap=n_gap)

    return filtered_sawtooth




def generate_filtered_noise(spectral_envelope_coeffs_list=None, params=None):

    n_gap = params["n_gap"]
    segment_len = params["segment_len"]
    apowin2 = params["apowin2"]

    spectral_envelope_extractor = SpectralEnvelopeExtractor(params)

    sound_len = len(spectral_envelope_coeffs_list) * n_gap + segment_len

    sound = np.zeros(sound_len)


    for i_segment in range(len(spectral_envelope_coeffs_list)):

        coeffs = spectral_envelope_coeffs_list[i_segment]
        full_spectral_envelope = spectral_envelope_extractor.get_full_spectral_envelope_from_coeffs(coeffs)


        segment = np.random.randn(segment_len) * apowin2
        fft_segment = np.fft.fft(segment)
        fft_filtered_segment = fft_segment * full_spectral_envelope
        filtered_segment = np.real(np.fft.ifft(fft_filtered_segment))
        add_to_segment(i_segment=i_segment, source=filtered_segment, dest=sound, n_gap=n_gap)


    return sound











