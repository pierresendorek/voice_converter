from copy import deepcopy

from params.params import get_params





def feature_vector_to_3_components(feature_vector, n_triangle_function):

    period = deepcopy(feature_vector[0])
    harmonic = deepcopy(feature_vector[1:1+n_triangle_function])
    noise = deepcopy(feature_vector[1+n_triangle_function:])

    return period, harmonic, noise



def feature_vector_array_to_feature_dict(vector_array):
    params = get_params()
    n_triangle_function = params["n_triangle_function"]
    period_list = []
    harmonic_list = []
    noise_list = []

    for it in range(vector_array.shape[0]):
        v = vector_array[it, :]
        period, harmonic, noise = feature_vector_to_3_components(feature_vector=v, n_triangle_function=n_triangle_function)
        period_list.append(period)
        harmonic_list.append(harmonic)
        noise_list.append(noise)

    return {"period_list": period_list,
            "spectral_envelope_coeffs_harmonic_list": harmonic_list,
            "spectral_envelope_coeffs_noise_list": noise_list}



