#---INFO-----------------------------------------------------------------------

#--Authors        |Last Updated |tattva-core-layers
#--Aditya Prakash |10-05-2023   |stable

#--Needed
#--A test to determine optimal convolution method in 'Potential'.
#--'jit' can be use in a better way to enable altering built objects. There is
#   a chance that methods might use old class variables since they were compiled
#   with those values.
#--'_convmet' can be pre-jitted although it's getting jitted internally in 
#  '__call__', compare performance differences.

#---DEPENDENCIES---------------------------------------------------------------
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from jax import vmap

from functools import partial

from .. utils.guards import ExpectationError as EXER
from .. utils.guards import ExpectationGuard as EXGU

#---CONSTANTS------------------------------------------------------------------


#---LAYERS---------------------------------------------------------------------
class CircularPadding():
    """
    Initializes an instance of the CircularPadding class.

    Args:
        kernel_shape (tuple): The shape of the kernel as a tuple of integers.
        channel_index (str, optional): The index of the channel dimension.
            Either 'last' or 'first'. Defaults to 'last'.

    Returns:
        None

    Raises:
        ValueError: If the length of input array does not match the length of 
        the kernel shape.
    """
    def __init__(self, kernel_shape: tuple, channel_index: str = 'last'):
        self.channel_index = channel_index
        self.kernel_shape = kernel_shape
        
        # determining padding widths along each axes except the channel axis
        if channel_index == 'last':
            self.pad_width = [
                [(v - 1)//2, (v - 1)//2] for v in kernel_shape[: -1]
            ] + [[0, 0]]
        elif channel_index == 'first':
            self.pad_width = [[0, 0]] + [
                [(v - 1)//2, (v - 1)//2] for v in kernel_shape[1:  ]
            ]
        
    @partial(jit, static_argnums = (0))
    def __call__(self, input_array: jnp.array):
        # checking array compatibility
        if len(input_array.shape) != len(self.kernel_shape):
            raise ValueError(
                f"Expected input array with {len(self.kernel_shape)}"
                f" dimensions, but got array with {len(input_array.shape)}"
                "dimensions")
            
        padded_array = jnp.pad(input_array, self.pad_width, mode = 'wrap')
        
        return padded_array
    
    
class Potential():
    """
    Represents a potential function that applies a convolution operation on a 
    given input array.

    Attributes:
    -----------
    kernel : jnp.array
        The kernel used for the convolution operation.

    Methods:
    --------

    __call__(input_array: jnp.array) -> jnp.array:
        Applies the convolution operation on the input_array using the kernel 
        attribute and returns the potential distribution.

        Parameters:
        -----------
        input_array : jnp.array
            The input array to apply the convolution operation on.

        Returns:
        --------
        jnp.array:
            The potential distribution resulting from the convolution operation.
    """
    def __init__(self, kernels: jnp.array, method: str = 'fft'):
        # /Guard clause
        # None
        # \Guard clause
        
        self.kernels = kernels
        self.method = method
        
    def _convmet(self, kernel: jnp.array, input_array: jnp.array):
        if self.method == 'fft':
            return jsp.signal.fftconvolve(input_array, kernel, mode = 'valid')
        elif self.method == 'direct':
            return jsp.signal.convolve(input_array, kernel, mode = 'valid')
    
    @partial(jit, static_argnums = (0))
    def __call__(self, input_array: jnp.array):
        # /Guard clause
        # None
        # \Guard clause
        
        v_convmet = vmap(self._convmet, (0, None))
        potential_distribution = v_convmet(self.kernels, input_array)
        
        return potential_distribution
        
        
class Growth():
    """
    A class for simulating growth of a system.

    Attributes:
    ----------
    growth_function : function
        A function that takes in a potential distribution and returns the growth 
        rate.
    dt : float
        The time step used for updating the output array.
    clip_function : function
        A function that takes in an array and returns a clipped version of that 
        array.

    Methods:
    -------
    __init__(self, growth_function, dt: float, clip_function, out: jnp.array):
        Constructs all the necessary attributes for the `Growth` layer.

    __call__(self, potential_distribution, out: jnp.array) -> jnp.array:
        Applies the growth function to the potential distribution, updates the 
        output array with the resulting growth, and clips the output array using 
        the provided clipping function. Returns the updated `out` array.
    """
    def __init__(self, growth_function, dt: float, clip_function):
        self.growth_function = growth_function
        self.dt = dt
        self.clip_function = clip_function
        
    @partial(jit, static_argnums = (0))
    def __call__(self, input_array: jnp.array, potential_distribution):
        # /Guard clause
        with EXGU(EXER, 1) as c:
            c(len(potential_distribution.shape) == 2)
        # \Guard clause
        
        dg = self.dt * self.growth_function(potential_distribution)
        out = jnp.add(input_array, dg)
        out = self.clip_function(out)
        
        return out
        

class Aggregate():
    """
    Performs aggregation of a multi-dimensional input array.

    Attributes:
    ----------
    weights (tuple): 
        A tuple of weights used for aggregation. The sum of the weights can be 
        set to 1 for a weighted aggregation and all 1s for a summation.

    Methods:
    -------
    __call__(input_array): 
        Performs weighted aggregation of the input array along the last axis.
    """   
    def __init__(self, weights: tuple):
        self.weights = weights
        
    @partial(jit, static_argnums = (0))
    def __call__(self, input_array: jnp.array):
        # /Guard clause
        with EXGU(EXER, 1) as c:
            c(input_array.shape[-1] == len(self.weights))
        # \Guard clause
        
        out = input_array
        for i, w in  enumerate(self.weights): 
            out = out.at[..., i].multiply(w)
        out = jnp.sum(out, axis = len(out.shape) - 1)
        
        return out
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        