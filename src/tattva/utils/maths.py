# ---DEPENDENCIES---------------------------------------------------------------
import jax.numpy as jnp
from jax import jit


# ---FUNCTIONS------------------------------------------------------------------
@jit
def hardclip(input_array: jnp.array, set_min: float = 0.0, set_max: float = 1.0):
    """
    Clips the values in the input array to lie between the set_min and set_max
    values.

    Args:
        input_array: A JAX array of numerical values.
        set_min: The minimum value to clip the array to. Defaults to 0.
        set_max: The maximum value to clip the array to. Defaults to 1.

    Returns:
        A JAX array of the same shape as the input_array, with values
        clipped to the set_min and set_max range.

    The hard clip function is a simple way to "clamp" the values in an array to
    a specific range. Values below set_min will be set to set_min, and values
    above set_max will be set to set_max. Values within the set_min and set_max
    range will be left unchanged.
    """
    return jnp.clip(input_array, a_min=set_min, a_max=set_max)


@jit
def softclip(input_array: jnp.array):
    """
    Applies a soft clip function to the input array.

    Args:
        input_array: A JAX array of numerical values.

    Returns:
        A NumPy or JAX array of the same shape as the input_array, with values
        transformed by the soft clip function.

    The soft clip function is defined as:

        y = 1 / (1 + e^(-4(x - 0.5))) -- clips to range (0, 1)

    where x is the input value, and y is the transformed value. This function
    has the effect of compressing extreme values towards the center of the
    range, while leaving moderate values relatively unchanged.
    """

    return 1 / (1 + jnp.exp(-4 * (input_array - 0.5)))
