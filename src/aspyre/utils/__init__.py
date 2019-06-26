def ensure(cond, error_message=None):
    """
    assert statements in Python are sometimes optimized away by the compiler, and are for internal testing purposes.
    For user-facing assertions, we use this simple wrapper to ensure conditions are met at relevant parts of the code.

    :param cond: Condition to be ensured
    :param error_message: An optional error message if condition is not met
    :return: If condition is met, returns nothing, otherwise raises AssertionError
    """
    if not cond:
        raise AssertionError(error_message)


CUPY_ENABLED = False
try:
    import cupy as xp
    CUPY_ENABLED = True
except ImportError:
    import numpy as xp


def get_numeric_library():
    """
    Based on configuration, import and return numpy or cupy
    """
    return xp


def asnumpy(array):
    """
    Based on configuration, return the cupy array as a numpy array
    or pass back the numpy array
    """
    if CUPY_ENABLED:
        return xp.asnumpy(array)
    else:
        return array


class CPUDevice:
    # A no-op dummy 'Device' that can be used wherever
    # a cuda.cupy.Device would have been used as a context manager
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def device(device_id):
    if CUPY_ENABLED:
        return xp.cuda.Device(device_id)
    else:
        return CPUDevice()
