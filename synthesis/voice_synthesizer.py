import numpy as np
from scipy.signal import fftconvolve

from database_tools.segment_setter import add_to_segment
from database_tools.sound_file_loader import get_segment
from feature_extraction.common_arrays import CommonArrays
from feature_extraction.spectral_envelope import SpectralEnvelopeExtractor
from common.parameters import Parameters



def generate_sawtooth_sound(instantaneous_frequency_in_hertz:float, params:Parameters):
    frequency = instantaneous_frequency_in_hertz
    sampling_frequency = params.sampling_frequency
    # TODO: ne pas utiliser cumsum
    #phase = np.cumsum(frequency / sampling_frequency)
    sawtooth = cumulative_saw(frequency / sampling_frequency)
    return sawtooth


def cumulative_saw(normalized_frequency:np.ndarray):
    exp_phase = np.exp(2j * np.pi * 0.0)
    phases = []
    for f in normalized_frequency:
        exp_phase *= np.exp(1j * 2 * np.pi * f)
        phases.append(np.real(np.log(exp_phase) / (1j * np.pi))) # so the signal is between -1 and 1
    return np.array(phases)
        


def get_instantenous_frequency_array(segment_period_in_sample, params:Parameters, common_arrays:CommonArrays):
    sampling_frequency = params.sampling_frequency
    segment_len = params.segment_len
    n_gap = params.n_gap
    triangle = common_arrays.triangle

    inst_fq = np.zeros(n_gap * segment_period_in_sample.shape[0] + segment_len)


    for i_segment in range(segment_period_in_sample.shape[0]):
        segment_period_in_seconds = segment_period_in_sample[i_segment] / sampling_frequency
        frequency_in_hertz = 1.0 / segment_period_in_seconds

        add_to_segment(i_segment=i_segment,
                       source=triangle * frequency_in_hertz,
                       dest=inst_fq,
                       n_gap=n_gap)

    return inst_fq


def generate_periodic_sound(segment_period_expressed_in_sample:np.ndarray,
                            common_arrays:CommonArrays,
                            params:Parameters):

    inst_fq = get_instantenous_frequency_array(segment_period_in_sample=segment_period_expressed_in_sample,
                                               common_arrays=common_arrays,
                                               params=params)


    sawtooth = generate_sawtooth_sound(instantaneous_frequency_in_hertz=inst_fq, params=params)

    return sawtooth


def generate_periodic_filtered_sound(segment_period_expressed_in_sample:np.ndarray,
                                     spectral_envelope_coeffs:np.ndarray,
                                     common_arrays:CommonArrays,
                                     params:Parameters):

    sawtooth = generate_periodic_sound(segment_period_expressed_in_sample=segment_period_expressed_in_sample,
                                        common_arrays=common_arrays,
                                      params=params)

    filtered_sawtooth = np.zeros(sawtooth.shape[0])

    spectral_envelope_extractor = SpectralEnvelopeExtractor(params)

    apowin2 = common_arrays.apowin2
    n_gap = params.n_gap
    smoother = np.ones(params.segment_len // 256)

    for i_segment in range(spectral_envelope_coeffs.shape[0]):
        seg = get_segment(sound=sawtooth, i_segment=i_segment, params=params)

        fft_seg = np.fft.fft(seg * apowin2)
        fft_seg_envelope = fftconvolve(np.abs(fft_seg),  smoother, "same")
        white_fft_seg = fft_seg / fft_seg_envelope
        coeffs = spectral_envelope_coeffs[i_segment]
        spectral_envelope = spectral_envelope_extractor.get_full_spectral_envelope_from_coeffs(coeffs)

        filtered_seg = np.real(np.fft.ifft(white_fft_seg * spectral_envelope))
        add_to_segment(i_segment=i_segment, source=filtered_seg, dest=filtered_sawtooth, n_gap=n_gap)

    return filtered_sawtooth


def generate_filtered_noise(spectral_envelope_coeffs:np.ndarray, params:Parameters, common_arrays:CommonArrays):

    sqrt_triangle = np.sqrt(common_arrays.triangle)
    spectral_envelope_extractor = SpectralEnvelopeExtractor(params)
    sound_len = spectral_envelope_coeffs.shape[0] * params.n_gap + params.segment_len
    sound = np.zeros(sound_len)
    smoother = np.ones(params.segment_len // 256)


    for i_segment in range(spectral_envelope_coeffs.shape[0]):
        coeffs = spectral_envelope_coeffs[i_segment]
        full_spectral_envelope = spectral_envelope_extractor.get_full_spectral_envelope_from_coeffs(coeffs)


        segment = np.random.randn(params.segment_len) * sqrt_triangle
        fft_segment = np.fft.fft(segment)
        white_fft_segment = fft_segment / fftconvolve(np.abs(fft_segment), smoother, "same")
        fft_filtered_segment = white_fft_segment * full_spectral_envelope
        filtered_segment = np.real(np.fft.ifft(fft_filtered_segment))

        add_to_segment(i_segment=i_segment, 
                       source=filtered_segment * sqrt_triangle, 
                       dest=sound, 
                       n_gap=params.n_gap)

    return sound


def synthesize_voice(features:np.ndarray, params:Parameters, common_arrays:CommonArrays, normalize:bool):
    """
    features: np.ndarray of shape (n_segments, n_features = 1+2 * n_triangle_function)
    """
    
    periods = features[:, 0]
    spectral_envelope_coeffs_harmonic = features[:,1:1 + params.n_triangle_function]
    spectral_envelope_coeffs_noise = features[:,1+ params.n_triangle_function:]
    
    noise_filtered = generate_filtered_noise(spectral_envelope_coeffs=spectral_envelope_coeffs_noise,
                                             common_arrays=common_arrays,
                                             params=params)

    periodic_filtered = generate_periodic_filtered_sound(segment_period_expressed_in_sample=periods,
                                                                   spectral_envelope_coeffs=spectral_envelope_coeffs_harmonic,
                                                                   common_arrays=common_arrays,
                                                                   params=params)

    reconstruction = noise_filtered + periodic_filtered
    if normalize:
        reconstruction = reconstruction / max(abs(reconstruction))

    return reconstruction






if __name__ == "__main__":

    from argparse import ArgumentParser
    from pathlib import Path
    from scipy.io import wavfile

    parser = ArgumentParser()
    parser.add_argument("input_features_file", type=Path)
    parser.add_argument("output_sound_file", type=Path)
    args = parser.parse_args()

    features = np.load(args.input_features_file)
    
    params = Parameters()
    common_arrays = CommonArrays(params)

    
    
    reconstructed_sound = synthesize_voice(features=features, 
                                           params=params, 
                                           common_arrays=common_arrays, 
                                           normalize=True)

    wavfile.write(args.output_sound_file, 44100, reconstructed_sound)
    

    