import numpy as np


def get_element_from_list_zero_outside(i_element, L):
    if i_element < 0 or i_element >= len(L):
        return np.zeros(L[0].shape)
    else:
        return L[i_element]


def get_element_from_list_constant_outside(i_element: object, L: object) -> object:
    if i_element < 0:
        return L[0]
    elif i_element >= len(L):
        return L[-1]
    else:
        return L[i_element]