from scipy.sparse import linalg
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import lsqr
import numpy as np

from feature_extraction.new_feature_extractor import Parameters
from logging import Logger


logger = Logger("PeriodicAndNoiseSeparator")

class PeriodicAndNoiseSeparator:

    def __init__(self, params:Parameters):

        
        self.params = params
        
        # table of regularly spaced periods (expressed in number of samples)
        # Each time delay in this vector is a candidate period
        self.period_list = np.arange(params.period_min, params.period_max)

        # corresponding frequencies for the table of regularly spaced periods
        self.frequency_list = np.array(self.sampling_frequency / self.period_list)

        self.periodic_function_basis_dict = {}
        logger.info("Initializing periodic function basis for all periods...")
        for period in self.period_list:
            self.periodic_function_basis_dict[period] = self.create_apodized_periodic_function_basis_for_period(period)
        logger.info("Done.")


    def create_apodized_periodic_function_basis_for_period(self, period):
        """
        Creates a sparse matrix, say A, where each column of A is an dirac hair comb shifted by the phase phi
        and apodized by self.apowin2.

        :param period: the period expressed in number of samples
        :return: the matrix A, in sparse format
        """

        n_row = self.segment_len
        n_col = period
        # A = np.zeros((n_row, n_col))
        i = []
        j = []
        data = []
        for phi in range(period):
            k = 0
            while k < (n_row - phi) / period:
                # A[k * period + phi, phi] = 1
                i.append(k * period + phi)
                j.append(phi)
                data.append(self.apowin2[k * period + phi])
                k += 1

        # data = np.ones(len(i))
        data = np.array(data)
        ij = np.zeros((2, len(i)))
        ij[0, :] = i
        ij[1, :] = j
        return bsr_matrix((data, ij), shape=(n_row, n_col))


    def separate_components(self, x_apodized:np.ndarray, period:int):
        a = self.periodic_function_basis_dict[period]
        
        phase_amplitude = lsqr(a, x_apodized)
        periodic_component = a.dot(phase_amplitude[0])
        noise_component = x_apodized - periodic_component

        return periodic_component, noise_component

