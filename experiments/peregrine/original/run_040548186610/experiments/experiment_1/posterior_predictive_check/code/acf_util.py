"""Simple ACF implementation"""
import numpy as np

def acf(x, nlags=40, fft=False):
    """
    Compute autocorrelation function

    Parameters:
    -----------
    x : array-like
        Time series data
    nlags : int
        Number of lags to compute
    fft : bool
        Not used, for compatibility

    Returns:
    --------
    acf_vals : ndarray
        Autocorrelation values from lag 0 to nlags
    """
    x = np.asarray(x).squeeze()
    x = x - np.mean(x)

    c0 = np.dot(x, x) / len(x)

    acf_vals = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        c_k = np.dot(x[:-k], x[k:]) / len(x)
        acf_vals[k] = c_k / c0

    return acf_vals
