def period_in_samples_to_period_in_seconds(period_in_samples, sampling_frequency):
    return (period_in_samples*1.0)/sampling_frequency


def period_in_samples_to_frequency_in_hertz(period_in_samples, sampling_frequency):
    period_in_seconds = period_in_samples_to_period_in_seconds(period_in_samples, sampling_frequency)
    frequency = 1.0 / period_in_seconds

