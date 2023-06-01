# ---INFO-----------------------------------------------------------------------

# --Authors        |Last Updated |tattva-core-layers
# --Aditya Prakash |10-05-2023   |stable

# --Needed
# --A test to determine optimal convolution method in 'Potential'.
# --'jit' can be use in a better way to enable altering built objects. There is
#   a chance that methods might use old class variables since they were compiled
#   with those values.
# --'_convmet' can be pre-jitted although it's getting jitted internally in
#  '__call__', compare performance differences.
# --A thorough benchmarking for channel-first vs channel-last convolutions

# ---DEPENDENCIES---------------------------------------------------------------
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from jax import vmap

from functools import partial

from ..utils.guards import ExpectationError as EXER
from ..utils.guards import ExpectationGuard as EXGU


# ---CONSTANTS------------------------------------------------------------------


# ---LAYERS---------------------------------------------------------------------
class CircularPadding:
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

    def __init__(self, kernel_shape: tuple, channel_index: str = "last"):
        self.channel_index = channel_index
        self.kernel_shape = kernel_shape

        # determining padding widths along each axes except the channel axis
        if channel_index == "last":
            self.pad_width = [
                [(v - 1) // 2, (v - 1) // 2] for v in kernel_shape[:-1]
            ] + [[0, 0]]
        elif channel_index == "first":
            self.pad_width = [[0, 0]] + [
                [(v - 1) // 2, (v - 1) // 2] for v in kernel_shape[1:]
            ]

    @partial(jit, static_argnums=(0))
    def __call__(self, input_array: jnp.array):
        # checking array compatibility
        if len(input_array.shape) != len(self.kernel_shape):
            raise ValueError(
                f"Expected input array with {len(self.kernel_shape)}"
                f" axes, but got array with {len(input_array.shape)}"
                " axes"
            )

        padded_array = jnp.pad(input_array, self.pad_width, mode="wrap")
        return padded_array


class Potential:
    """
    A class for calculating the potential distribution of a CA.

    Attributes:
    ----------
    kernels : jnp.array
        A 3D array of kernels used for calculating the potential distribution.
    method : str
        The method used for calculating the potential distribution. Either
        'fft' or 'direct'.
    depthwise : bool
        A boolean value indicating whether the kernels are depthwise or not.

    Methods:
    -------
    __call__(self, input_array):
        Applies the convolution method to the input array using the kernels and
        returns the potential distribution.
    """

    def __init__(
        self, kernels: jnp.array, method: str = "fft", depthwise: bool = False
    ):
        self.method = method
        self.depthwise = depthwise
        if self.depthwise:
            self.kernels = jnp.squeeze(kernels, axis=-1)
        else:
            self.kernels = kernels

    def _convmet(self, input_array: jnp.array, kernel: jnp.array):
        if self.method == "fft":
            return jsp.signal.fftconvolve(input_array, kernel, mode="valid")
        elif self.method == "direct":
            return jsp.signal.convolve(input_array, kernel, mode="valid")

    @partial(jit, static_argnums=(0))
    def __call__(self, input_array: jnp.array):
        if self.depthwise:
            v_convmet = vmap(self._convmet, in_axes=(-1, 0), out_axes=-1)
            potential_distribution = v_convmet(input_array, self.kernels)
            return potential_distribution
        else:
            v_convmet = vmap(self._convmet, in_axes=(None, 0), out_axes=-1)
            potential_distribution = v_convmet(input_array, self.kernels)
            potential_distribution = jnp.squeeze(potential_distribution, axis=-2)
            return potential_distribution


class Growth:
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

    @partial(jit, static_argnums=(0))
    def __call__(self, input_array: jnp.array, potential_distribution):
        dg = self.dt * self.growth_function(potential_distribution)
        out = jnp.add(input_array, dg)
        out = self.clip_function(out)
        return out


class Target:
    """
    Growth based system cab be fromulated as,

    u(x, t + dt) = clip([u(x, t) + dtG(K ∗ u)]])

    where K ∗ u is a non-local kernel convolution, clip is a clipping function
    and G is a reaction term.

    An asymptotic variant can be formulated as,

    u(x, t + dt) = u(x, t) + dt(T(K ∗ u) - u)

    where T is a reaction term and is usually constructed from the reaction term
    G in a Growth based system as T = (G + 1)/2. THis does not include the clipping
    procedure. This allows the equation to be transformed into a differential equation
    of the form,

    du(x, t)/dt = T(K ∗ u) - u
    """

    def __init__(self, target_function, dt: float):
        self.target_function = target_function
        self.dt = dt

    @partial(jit, static_argnums=(0))
    def __call__(self, input_array: jnp.array, potential_distribution):
        dtr = self.dt * (self.target_function(potential_distribution) - input_array)
        out = jnp.add(input_array, dtr)
        return out


class Aggregate:
    """
    Performs aggregation of a multi-dimensional input array.

    Attributes:
    ----------
    weights (tuple):
        A tuple of weights used for aggregation.
    Methods:
    -------
    __call__(input_array):
        Performs weighted aggregation of the input array along the last axis.
    """

    def __init__(self, weights: jnp.array):
        self.weights = weights

    @partial(jit, static_argnums=(0))
    def __call__(self, input_array: jnp.array):
        out = jnp.einsum("...i,i->...", input_array, self.weights)
        return out
