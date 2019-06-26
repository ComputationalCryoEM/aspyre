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

if CUPY_ENABLED:
    import cupy
    my_erode = cupy.RawKernel(r'''
      extern "C" __global__
      void my_erode(const bool * __restrict__  i, bool * __restrict__ o, int r, int dx, int dy) {
      int x = blockDim.x * blockIdx.x + threadIdx.x;
      int y = blockDim.y * blockIdx.y + threadIdx.y;
      int tr = r * r;  // handle circular structure element
      if ((x < dx) && (y < dy)){ // thread valid check
        if ((x < r) || (dx-x <= r) || (dy-y <= r) || (y < r)) // handle border region
          o[x+dx*y] = false;
        else // handle central region
          for (int iy = y-r; iy <= y+r; iy++){
            int trr = tr - ((iy-y) * (iy-y)); // handle circular structure element
            for (int ix = x-r; ix <= x+r; ix++)
              if (trr >= ((ix-x) * (ix-x))) // handle circular structure element
                if (!(i[ix + dx*iy])) {o[x+dx*y] = false; return;}
            }
        }
      }
    ''', 'my_erode')
    def erode_func(segmentation, element):

        i = cupy.asarray(segmentation)
        o = cupy.ones((segmentation.shape[0],segmentation.shape[1]), dtype=cupy.bool)
        bdim = 32
        gdim0 = (segmentation.shape[0]//bdim)+1
        gdim1 = (segmentation.shape[1]//bdim)+1
        radius = (element.shape[0]-1) // 2
        my_erode((gdim0,gdim1), (bdim,bdim), (i,o,radius,segmentation.shape[0],segmentation.shape[1]))  # grid, block and arguments
        segmentation_o = cupy.asnumpy(o)
        return segmentation_o

else:
    from scipy.ndimage import binary_erosion
    erode_func = binary_erosion


