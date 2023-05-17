# ---INFO-----------------------------------------------------------------------

# --Authors        |Last Updated |tattva-automata-cgol (Conway's game of life)
# --Aditya Prakash |17-05-2023   |stable

# ---DEPENDENCIES---------------------------------------------------------------
import jax.numpy as jnp

from .base import BaseAutomaton
from ..core import layers as ttvl


# ---CONWAY'S GAME OF-----------------------------------------------------------
class ConwayGOL(BaseAutomaton):
    def __init__(self, grid_shape: tuple, grid_0: jnp.array):
        super(ConwayGOL, self).__init__(grid_shape)
        self.kernel = jnp.array([[[1], [1], [1]], [[1], [0], [1]], [[1], [1], [1]]])
        self.init_grid()
