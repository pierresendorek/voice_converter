from common.get_element_from_list import get_element_from_list_zero_outside


def derivative_on_list(voice_features):
    derivative_voice_features = []
    for i in range(len(voice_features)):
        derivative_voice_features.append(get_element_from_list_zero_outside(i, voice_features) -
                                         get_element_from_list_zero_outside(i - 1, voice_features))
    return derivative_voice_features